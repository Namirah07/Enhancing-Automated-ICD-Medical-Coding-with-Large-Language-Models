# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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
#       File  : src/modeling_longformer.py (LongformerForMultilabelClassification class)
#
#   (2) HuggingFace Transformers library
#       URL   : https://github.com/huggingface/transformers
#       File  : transformers/models/longformer/modeling_longformer.py
#
#   (3) Longformer paper by Beltagy et al. (2020)
#       Paper : "Longformer: The Long-Document Transformer"
#       URL   : https://arxiv.org/abs/2004.05150
#
# CHANGES MADE BY NAMIRAH IMTIEAZ SHAIK:
#   - Renamed class from LongformerForMultilabelClassification
#     to LongformerForSingleLabelClassification
#   - Changed loss function from BCEWithLogitsLoss to CrossEntropyLoss
#   - Changed label format from multi-hot binary vector [B, num_labels]
#     to single integer class index [B]
#   - Removed sigmoid output activation
#   - Added detailed inline comments throughout
#
# UNCHANGED FROM SOURCE:
#   - LongformerModel backbone with add_pooling_layer=False
#   - Global attention mask setup on position 0
#   - LAAT attention pooling mechanism (first_linear, second_linear, third_linear)
#   - Chunk flattening logic in forward()
#   - LongformerSequenceClassifierOutput return structure
# =============================================================================

"""
modeling_longformer.py - Longformer backbone for single-label ICD-10 classification.

Adapted from the original PLM-ICD LongformerForMultilabelClassification to perform
SINGLE-LABEL multiclass classification on the MIMIC-IV ICD-10 dataset.

KEY DIFFERENCE FROM BERT-PLM-ICD:
  Longformer flattens all chunks into ONE long sequence and processes them together,
  whereas BERT processes each chunk independently as a separate sequence.
  This means Longformer tokens can attend to tokens in other chunks through its
  sparse attention mechanism, enabling cross-chunk information flow.

KEY ADAPTATIONS FROM MULTI-LABEL TO SINGLE-LABEL:
  1. Loss function: BCEWithLogitsLoss → CrossEntropyLoss
  2. Label format: multi-hot binary vector [B, num_labels] → integer index [B]
  3. Output activation: sigmoid → softmax (applied at evaluation time in run_icd.py)

LONGFORMER-SPECIFIC FEATURES:
  - Sparse attention: each token attends only to nearby tokens (local window)
    plus a small set of globally attending tokens. This allows O(n) complexity
    instead of O(n²) for standard BERT attention, making long sequences feasible.
  - Global attention on [CLS] token (position 0): the first token attends to
    ALL other tokens and all tokens attend back to it. This allows information
    from the entire document to flow through this single hub token.
  - add_pooling_layer=False: no separate pooler head is created because we use
    LAAT attention pooling directly on the raw token hidden states instead.

AGGREGATION:
  LAAT (Label-Aware Attention Pooling) - same mechanism as BERT-LAAT mode,
  but operates over the full flattened sequence rather than per-chunk.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import LongformerModel
from transformers.models.longformer.modeling_longformer import (
    LongformerSequenceClassifierOutput,
    LongformerPreTrainedModel,
)


# =============================================================================
# SOURCE ATTRIBUTION - CLASS LEVEL
# =============================================================================
# ADAPTED FROM: PLM-ICD (Huang et al., ClinicalNLP 2022)
#   Original class : LongformerForMultilabelClassification
#   URL            : https://github.com/MiuLab/PLM-ICD/blob/main/src/modeling_longformer.py
#
# ORIGINAL PARTS (written by Namirah Imtieaz Shaik):
#   - Single-label CrossEntropyLoss block in forward()
#   - All inline comments explaining the single-label adaptation and
#     the architectural difference from BERT (flattened vs per-chunk)
#
# UNCHANGED FROM SOURCE:
#   - LongformerModel instantiation with add_pooling_layer=False
#   - Global attention mask logic
#   - LAAT mechanism (first_linear, second_linear, third_linear)
#   - Chunk flattening in forward()
#   - _keys_to_ignore_on_load_unexpected declaration
# =============================================================================
class LongformerForSingleLabelClassification(LongformerPreTrainedModel):
    """
    Longformer model for single-label ICD-10 classification.

    Architecture:
      LongformerModel (backbone, fully fine-tuned, no pooling layer)
        → flatten all chunks into one long sequence
        → global attention on position 0 ([CLS] token)
        → LAAT attention pooling over all tokens
        → 30 logits
        → CrossEntropyLoss

    Unlike BERT which has a built-in pooler output, Longformer uses raw token
    hidden states directly fed into LAAT, making add_pooling_layer=False correct.
    """

    # Tell HuggingFace to ignore the "pooler" key when loading weights,
    # since this model does not have a pooling layer.
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # ---- Backbone ----
        # LongformerModel with sparse + global attention.
        # add_pooling_layer=False: skip the [CLS]-based pooler output because
        # we use LAAT attention pooling on the raw hidden states instead.
        # All parameters are trainable - full fine-tuning, no frozen layers.
        # SOURCE: LongformerModel from HuggingFace Transformers
        self.longformer = LongformerModel(config, add_pooling_layer=False)

        # ---- LAAT attention head (same as BERT-LAAT) ----
        # SOURCE: LAAT mechanism from PLM-ICD (Huang et al., 2022)
        # Three linear layers implement Label-Aware Attention Pooling.
        # Operates on the full flattened sequence (all chunks concatenated).
        #
        # first_linear:  [hidden_size → hidden_size]  — non-linear transformation
        # second_linear: [hidden_size → num_labels]   — per-label attention scores
        # third_linear:  [hidden_size → num_labels]   — final logit projection
        self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.third_linear = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,              # shape: (batch_size, num_chunks, chunk_size)
        attention_mask=None,         # shape: (batch_size, num_chunks, chunk_size)
        global_attention_mask=None,  # shape: (batch_size, seq_len) - 1 for global, 0 for local
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,                 # shape: (batch_size,) - single integer class index
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass for single-label ICD-10 classification with Longformer.

        CRITICAL DIFFERENCE FROM BERT FORWARD:
          BERT: processes chunks as SEPARATE sequences (B*num_chunks, chunk_size)
          Longformer: FLATTENS all chunks into ONE long sequence (B, total_length)
                      where total_length = num_chunks * chunk_size.

        This allows tokens in different chunks to attend to each other through
        Longformer's sparse local attention, enabling cross-chunk context flow.

        Args:
            input_ids:            Chunked token IDs, shape (B, num_chunks, chunk_size).
                                  Will be flattened to (B, num_chunks*chunk_size).
            attention_mask:       Local attention mask, same shape as input_ids.
                                  1 for real tokens, 0 for padding.
            global_attention_mask: Controls which tokens have global attention.
                                  Position 0 ([CLS]) gets value 1 (global).
                                  All other positions get value 0 (local only).
                                  If None, built automatically here.
            labels:               Single integer class index per example.
                                  Shape: (B,), dtype: torch.long.
                                  ADAPTATION: was [B, num_labels] in multi-label.

        Returns:
            LongformerSequenceClassifierOutput with .loss and .logits.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)

        # ---- Step 1: Flatten chunks into one long sequence ----
        # SOURCE: Chunk flattening pattern from PLM-ICD (Huang et al., 2022)
        # Unlike BERT which processes each chunk as a separate sequence,
        # Longformer concatenates all chunks into a single long sequence.
        #
        # (B, num_chunks, chunk_size) → (B, num_chunks * chunk_size)
        # Example: (8, 2, 256) → (8, 512) - one 512-token sequence per document.
        #
        # This is the key architectural difference from BERT-PLM-ICD:
        # tokens across chunk boundaries can attend to each other.
        input_ids = input_ids.view(batch_size, -1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(batch_size, -1)

        # ---- Step 2: Set up global attention mask ----
        # SOURCE: Global attention mask pattern from PLM-ICD (Huang et al., 2022)
        # and Longformer paper (Beltagy et al., 2020)
        # Longformer has two types of attention:
        #   - LOCAL attention: each token attends to nearby tokens within a window
        #     (window size = config.attention_window = chunk_size)
        #   - GLOBAL attention: special tokens attend to ALL tokens and vice versa
        #
        # We enable global attention on position 0 ([CLS] token).
        # This makes [CLS] a global hub - it receives information from every token
        # in the full document, enabling document-level classification.
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1  # Enable global attention on [CLS] only

        # ---- Step 3: Longformer forward pass ----
        # SOURCE: LongformerModel from HuggingFace Transformers
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the full token hidden state matrix.
        # Shape: (B, seq_len, hidden_size) - one vector per token per document.
        hidden_output = outputs[0]

        # ---- Step 4: LAAT Aggregation ----
        # SOURCE: LAAT mechanism from PLM-ICD (Huang et al., 2022)
        # Label-Aware Attention Pooling operates on ALL token hidden states
        # across the full flattened sequence. Each of the 30 ICD labels
        # learns its own attention weights over all tokens.

        # LAAT Step 1: Non-linear transformation of all token hidden states.
        # [B, L, H] → [B, L, H] (tanh introduces non-linearity)
        weights = torch.tanh(self.first_linear(hidden_output))

        # LAAT Step 2: Compute per-label attention scores for each token.
        # [B, L, H] → [B, L, num_labels]
        att_weights = self.second_linear(weights)

        # LAAT Step 3: Normalize over token dimension and transpose.
        # softmax(dim=1): attention weights sum to 1 across all L tokens per label.
        # transpose: [B, L, num_labels] → [B, num_labels, L]
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)

        # LAAT Step 4: Weighted sum of token states per label.
        # [B, num_labels, L] @ [B, L, H] → [B, num_labels, H]
        weighted_output = att_weights @ hidden_output

        # LAAT Step 5: Project each label's context vector to a scalar logit.
        logits = self.third_linear.weight.mul(weighted_output).sum(dim=2).add(self.third_linear.bias)

        # ---- Step 5: Compute loss - SINGLE-LABEL ADAPTATION ----
        # ORIGINAL CODE: Written by Namirah Imtieaz Shaik
        # The original PLM-ICD used BCEWithLogitsLoss for multi-label classification.
        # This block replaces it with CrossEntropyLoss for single-label classification.
        loss = None
        if labels is not None:
            # SINGLE-LABEL CrossEntropyLoss.
            #
            # ADAPTATION FROM MULTI-LABEL (original PLM-ICD):
            #   Original: loss = BCEWithLogitsLoss()(logits, labels.float())
            #             labels shape [B, num_labels] binary vector
            #   Adapted:  loss = CrossEntropyLoss()(logits, labels)
            #             labels shape [B] single integer index
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LongformerSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )