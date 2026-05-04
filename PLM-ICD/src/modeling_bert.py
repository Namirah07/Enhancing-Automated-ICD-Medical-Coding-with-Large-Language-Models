# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# SOURCE ATTRIBUTION - FILE LEVEL
# =============================================================================
# ADAPTED FROM:
#   (1) PLM-ICD repository by Huang et al. (ClinicalNLP 2022)
#       Paper : "PLM-ICD: Automatic ICD Coding with Pretrained Language Models"
#       URL   : https://github.com/MiuLab/PLM-ICD
#       File  : src/modeling_bert.py (BertForMultilabelClassification class)
#
#   (2) HuggingFace Transformers library
#       URL   : https://github.com/huggingface/transformers
#       File  : transformers/models/bert/modeling_bert.py
#
# CHANGES MADE BY NAMIRAH IMTIEAZ SHAIK:
#   - Renamed class from BertForMultilabelClassification
#     to BertForSingleLabelClassification
#   - Changed loss function from BCEWithLogitsLoss to CrossEntropyLoss
#     to support single-label multiclass classification
#   - Changed label format from multi-hot binary vector [B, num_labels]
#     to single integer class index [B]
#   - Removed sigmoid output activation (softmax applied externally in run_icd.py)
#   - Added safety check for labels.dim() > 1 in the forward pass
#   - Added detailed inline comments throughout explaining the single-label adaptation
# =============================================================================

"""
modeling_bert.py — BERT backbone for single-label ICD-10 classification.

Adapted from the original PLM-ICD BertForMultilabelClassification to perform
SINGLE-LABEL multiclass classification on the MIMIC-IV ICD-10 dataset.

KEY ADAPTATIONS FROM MULTI-LABEL TO SINGLE-LABEL:
  1. Loss function: BCEWithLogitsLoss → CrossEntropyLoss
     - BCE treats each of the 30 logits as an independent binary decision.
     - CrossEntropy treats the 30 logits as a mutually exclusive distribution.
  2. Label format: multi-hot binary vector [B, num_labels] → integer index [B]
     - Multi-label: labels = [0, 1, 0, 0, 1, ...] (multiple 1s allowed)
     - Single-label: labels = 7 (exactly one correct class)
  3. Output activation: sigmoid → softmax (applied at evaluation time in run_icd.py)

WHAT STAYS THE SAME:
  - BertModel backbone (Bio_ClinicalBERT weights)
  - Chunk-wise processing of long discharge notes
  - CLS-sum / CLS-max / LAAT aggregation mechanisms
  - Output shape: [batch_size, num_labels] logits (unchanged)

SUPPORTED AGGREGATION MODES:
  - cls-sum:   Sum the [CLS] token vector from each chunk → one document vector
  - cls-max:   Take element-wise max across [CLS] vectors from all chunks
  - laat:      Label-Aware Attention Pooling over all tokens concatenated
  - laat-split: LAAT applied per-chunk, then max-pooled across chunks
"""

import logging
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)


# =============================================================================
# SOURCE ATTRIBUTION - CLASS LEVEL
# =============================================================================
# ADAPTED FROM: PLM-ICD (Huang et al., ClinicalNLP 2022)
#   Original class : BertForMultilabelClassification
#   URL            : https://github.com/MiuLab/PLM-ICD/blob/main/src/modeling_bert.py
#
# ORIGINAL PARTS (written by Namirah Imtieaz Shaik):
#   - Single-label CrossEntropyLoss block in forward()
#   - labels.dim() > 1 safety check
#   - All inline comments explaining the single-label adaptation
#
# UNCHANGED FROM SOURCE:
#   - BertModel backbone instantiation
#   - CLS-sum, CLS-max, LAAT, LAAT-split aggregation logic
#   - Chunk-wise flattening and reshaping in forward()
#   - SequenceClassifierOutput return structure
# =============================================================================
class BertForSingleLabelClassification(BertPreTrainedModel):
    """
    BERT model for single-label ICD-10 classification using chunk-wise processing.

    Architecture:
      BertModel (Bio_ClinicalBERT backbone, fully fine-tuned)
        → chunk-wise forward pass
        → aggregation (CLS-sum / CLS-max / LAAT)
        → dropout
        → Linear → 30 logits
        → CrossEntropyLoss

    Inheriting from BertPreTrainedModel gives free access to:
      - from_pretrained() for loading Bio_ClinicalBERT weights
      - save_pretrained() for saving the fine-tuned model
      - init_weights() for correct weight initialization of new layers
    """

    def __init__(self, config):
        super().__init__(config)

        # Number of ICD-10 classes (30 in our MIMIC-IV benchmark)
        self.num_labels = config.num_labels

        # Aggregation mode set in config by run_icd.py:
        # "cls-sum", "cls-max", "laat", or "laat-split"
        self.model_mode = getattr(config, "model_mode", "cls-sum")

        # ---- Backbone ----
        # BertModel loads the full Bio_ClinicalBERT transformer stack.
        # All 110M parameters are trainable - full fine-tuning, no frozen layers.
        self.bert = BertModel(config)

        # Dropout applied to the pooled representation before the classifier.
        # config.hidden_dropout_prob is typically 0.1 for BERT-base.
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # ---- Classification head ----
        # The head architecture depends on the chosen aggregation mode.
        if "cls" in self.model_mode:
            # CLS-sum / CLS-max mode:
            # Simple linear projection from BERT hidden size (768) to num_labels (30).
            # Input: the aggregated [CLS] vector of shape [B, hidden_size]
            # Output: raw logits of shape [B, num_labels]
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        elif "laat" in self.model_mode:
            # LAAT mode: Label-Aware Attention Pooling.
            # Three linear layers implement the attention mechanism:
            #
            # first_linear:  projects token hidden states to an intermediate space
            #                shape: [hidden_size → hidden_size], no bias
            # second_linear: produces attention logits — one score per label per token
            #                shape: [hidden_size → num_labels], no bias
            # third_linear:  final projection from per-label context vectors to logits
            #                shape: [hidden_size → num_labels], with bias
            #
            # Together these allow each of the 30 ICD codes to attend to different
            # tokens in the discharge note, learning label-specific token weights.
            self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
            self.third_linear = nn.Linear(config.hidden_size, config.num_labels)
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")

        # Initialize new layer weights (classifier / attention linears) using
        # BERT's initialization scheme for consistency with the backbone.
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,        # [B, num_chunks, chunk_size]
        attention_mask: Optional[torch.Tensor] = None,   # [B, num_chunks, chunk_size]
        token_type_ids: Optional[torch.Tensor] = None,   # [B, num_chunks, chunk_size]
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,           # [B] — single integer per example
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceClassifierOutput:
        """
        Forward pass for single-label ICD-10 classification.

        Args:
            input_ids:      Token IDs after chunk-wise reshaping.
                            Shape: (batch_size, num_chunks, chunk_size)
                            Example: (8, 2, 256) for batch_size=8, 2 chunks of 256 tokens each.
            attention_mask: 1 for real tokens, 0 for padding.
                            Same shape as input_ids.
            token_type_ids: Segment IDs (all zeros for single-sequence classification).
                            Same shape as input_ids.
            labels:         Single integer class index per example.
                            Shape: (batch_size,), dtype: torch.long
                            Values in range [0, num_labels-1].
                            ADAPTATION: was [B, num_labels] binary vector in multi-label.

        Returns:
            SequenceClassifierOutput with:
              - loss:   CrossEntropyLoss scalar (if labels provided)
              - logits: [B, num_labels] raw unnormalized scores
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Unpack the 3D shape introduced by chunk-wise processing
        batch_size, num_chunks, chunk_size = input_ids.size()

        # ---- Step 1: Flatten chunks for BERT ----
        # BERT only accepts 2D input (batch_size, seq_len). We treat each chunk
        # as an independent sequence by merging batch and chunk dimensions.
        # (B, num_chunks, chunk_size) → (B * num_chunks, chunk_size)
        #
        # Example: (8, 2, 256) → (16, 256)
        # BERT processes 16 independent 256-token sequences, unaware they belong
        # to 8 documents with 2 chunks each.
        # SOURCE: chunk-wise flattening pattern from PLM-ICD (Huang et al., 2022)
        outputs = self.bert(
            input_ids=input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size) if attention_mask is not None else None,
            token_type_ids=token_type_ids.view(-1, chunk_size) if token_type_ids is not None else None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        # outputs.pooler_output:    [B*num_chunks, hidden_size] — [CLS] token representation
        # outputs.last_hidden_state: [B*num_chunks, chunk_size, hidden_size] — all token states

        # ---- Step 2: Aggregation - combine chunk representations ----
        # SOURCE: CLS-sum, CLS-max, LAAT aggregation strategies from PLM-ICD (Huang et al., 2022)
        if "cls" in self.model_mode:
            # CLS-SUM / CLS-MAX AGGREGATION
            # Each chunk's [CLS] token has absorbed information from all 256 tokens
            # in that chunk through BERT's self-attention. The pooler_output is the
            # [CLS] hidden state passed through an additional linear + tanh layer.
            #
            # Shape: [B*num_chunks, hidden_size] → [B, num_chunks, hidden_size]
            pooled_output = outputs.pooler_output.view(batch_size, num_chunks, -1)

            if self.model_mode == "cls-sum":
                # Sum the [CLS] vectors from all chunks element-wise.
                # This assumes the contributions of different document sections
                # are additive. Used for BERT-PLM-ICD in this thesis.
                # Shape: [B, num_chunks, H] → [B, H]
                pooled_output = pooled_output.sum(dim=1)

            elif self.model_mode == "cls-max":
                # Take element-wise maximum across all chunk [CLS] vectors.
                # Captures the strongest activation for each hidden dimension
                # across all document sections.
                # Shape: [B, num_chunks, H] → [B, H]
                pooled_output = pooled_output.max(dim=1).values
            else:
                raise ValueError(f"model_mode {self.model_mode} not recognized")

            # Apply dropout to the aggregated document vector for regularization
            pooled_output = self.dropout(pooled_output)

            # Linear projection: [B, hidden_size] → [B, num_labels]
            # Produces 30 raw logits, one per ICD-10 class.
            logits = self.classifier(pooled_output)

        elif "laat" in self.model_mode:
            # LAAT AGGREGATION - Label-Aware Attention Pooling
            # SOURCE: LAAT mechanism from PLM-ICD (Huang et al., 2022), originally
            # proposed by Mullenbach et al. (NAACL 2018) for ICD coding
            # URL: https://github.com/MiuLab/PLM-ICD
            # Instead of using only the [CLS] token, LAAT uses ALL token hidden
            # states and learns a different attention weight per ICD label per token.
            # This allows each label to focus on the specific tokens most relevant
            # to that diagnosis code.

            if self.model_mode == "laat":
                # Concatenate all chunk token states into one long sequence.
                # [B*num_chunks, chunk_size, H] → [B, num_chunks*chunk_size, H]
                # All tokens from all chunks are visible together.
                hidden_output = outputs.last_hidden_state.view(batch_size, num_chunks * chunk_size, -1)

            elif self.model_mode == "laat-split":
                # Keep chunks separate — apply LAAT within each chunk independently.
                # [B*num_chunks, chunk_size, H] stays as (B*num_chunks, chunk_size, H)
                # Max-pooling across chunks is applied after LAAT (see below).
                hidden_output = outputs.last_hidden_state.view(batch_size * num_chunks, chunk_size, -1)
            else:
                raise ValueError(f"model_mode {self.model_mode} not recognized")

            # LAAT Step 1: Non-linear transformation of token hidden states.
            # first_linear: [*, T, H] → [*, T, H]
            # tanh introduces non-linearity to the attention computation.
            weights = torch.tanh(self.first_linear(hidden_output))

            # LAAT Step 2: Compute attention scores - one per label per token.
            # second_linear: [*, T, H] → [*, T, num_labels]
            # Each of the 30 labels gets its own attention score for each token.
            att_weights = self.second_linear(weights)

            # LAAT Step 3: Normalize attention scores over the token dimension.
            # softmax(dim=1) ensures attention weights sum to 1 across all tokens
            # for each label. Then transpose to [*, num_labels, T] for matmul.
            att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)

            # LAAT Step 4: Compute weighted sum of token states for each label.
            # [*, num_labels, T] @ [*, T, H] → [*, num_labels, H]
            # Each label now has a 768-dim context vector focused on its relevant tokens.
            weighted_output = att_weights @ hidden_output

            # LAAT Step 5: Project each label's context vector to a scalar logit.
            # This uses element-wise multiplication with the weight matrix rows
            # (equivalent to a batched dot product) then sums over the H dimension.
            # third_linear.weight: [num_labels, H]
            # weighted_output:     [*, num_labels, H]
            # Result: [*, num_labels] - one logit per label
            logits = self.third_linear.weight.mul(weighted_output).sum(dim=2).add(self.third_linear.bias)

            if self.model_mode == "laat-split":
                # laat-split processes each chunk independently, so logits has shape
                # [B*num_chunks, num_labels]. Reshape and take the max across chunks
                # to get the final [B, num_labels] logits.
                logits = logits.view(batch_size, num_chunks, -1).max(dim=1).values

        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")

        # ---- Step 3: Compute loss - SINGLE-LABEL ADAPTATION ----
        # ORIGINAL CODE: Written by Namirah Imtieaz Shaik
        # The original PLM-ICD used BCEWithLogitsLoss for multi-label classification.
        # This block replaces it with CrossEntropyLoss for single-label classification
        # and changes the label format from multi-hot [B, num_labels] to integer [B].
        loss = None
        if labels is not None:
            # Safety check: squeeze labels if they accidentally have shape [B, 1]
            if labels.dim() > 1:
                labels = labels.view(-1)

            # CrossEntropyLoss for single-label multiclass classification.
            #
            # ADAPTATION FROM MULTI-LABEL (original PLM-ICD):
            #   Original: loss = BCEWithLogitsLoss()(logits, labels.float())
            #             labels shape: [B, num_labels] binary vector
            #   Adapted:  loss = CrossEntropyLoss()(logits, labels)
            #             labels shape: [B] integer class indices
            #
            # CrossEntropyLoss internally applies log-softmax to the logits and
            # computes negative log-likelihood for the correct class only.
            # It treats the 30 classes as mutually exclusive (one correct answer).
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # ---- Return output ----
        if not return_dict:
            output = (logits,) + (outputs.hidden_states, outputs.attentions)
            return ((loss,) + output) if loss is not None else output

        # Return as HuggingFace SequenceClassifierOutput for compatibility
        # with the training loop in run_icd.py which accesses outputs.loss and outputs.logits
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )