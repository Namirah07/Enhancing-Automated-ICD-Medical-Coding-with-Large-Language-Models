# coding=utf-8
# Copyright 2018 ...

# =============================================================================
# SOURCE ATTRIBUTION - FILE LEVEL
# =============================================================================
# ADAPTED FROM:
#   (1) PLM-ICD repository by Huang et al. (ClinicalNLP 2022)
#       Paper : "PLM-ICD: Automatic ICD Coding with Pretrained Language Models"
#       URL   : https://github.com/MiuLab/PLM-ICD
#       File  : src/modeling_roberta.py (RobertaForMultilabelClassification class)
#
#   (2) HuggingFace Transformers library
#       URL   : https://github.com/huggingface/transformers
#       File  : transformers/models/roberta/modeling_roberta.py
#
# CHANGES MADE BY NAMIRAH IMTIEAZ SHAIK:
#   - Renamed class from RobertaForMultilabelClassification
#     to RobertaForSingleLabelClassification
#   - Changed loss function from BCEWithLogitsLoss to CrossEntropyLoss
#   - Changed label format from multi-hot binary vector [B, num_labels]
#     to single integer class index [B]
#   - Removed sigmoid output activation
#   - Added detailed inline comments explaining the key difference from BERT:
#     RoBERTa uses add_pooling_layer=False so position 0 of last_hidden_state
#     is used directly instead of outputs.pooler_output
#
# UNCHANGED FROM SOURCE:
#   - RobertaModel backbone instantiation with add_pooling_layer=False
#   - CLS-sum, CLS-max, LAAT, LAAT-split aggregation logic
#   - Chunk-wise flattening in forward()
#   - SequenceClassifierOutput return structure
# =============================================================================

"""
modeling_roberta.py - RoBERTa backbone for single-label ICD-10 classification.

Adapted from the original PLM-ICD RobertaForMultilabelClassification to perform
SINGLE-LABEL multiclass classification on the MIMIC-IV ICD-10 dataset.

KEY ADAPTATIONS FROM MULTI-LABEL TO SINGLE-LABEL:
  1. Loss function: BCEWithLogitsLoss → CrossEntropyLoss
  2. Label format: multi-hot binary vector [B, num_labels] → integer index [B]
  3. Output activation: sigmoid → softmax (applied at evaluation time in run_icd.py)

KEY DIFFERENCE FROM BERT-PLM-ICD (modeling_bert.py):
  RoBERTa does NOT have a built-in pooler output on top of [CLS].
  BERT's pooler_output = Linear(tanh([CLS] hidden state)).
  RoBERTa is initialized with add_pooling_layer=False, so there is no pooler.
  Therefore, for CLS-based aggregation, we manually extract position 0
  from the raw last_hidden_state instead of using outputs.pooler_output.

WHAT IS THE SAME AS BERT:
  - Chunk-wise processing: same (B, num_chunks, chunk_size) input format
  - CLS-sum / CLS-max / LAAT / LAAT-split aggregation modes
  - CrossEntropyLoss for single-label classification
  - Dropout before classifier

SUPPORTED AGGREGATION MODES:
  - cls-sum:    Sum per-chunk first-token (position 0) hidden states
  - cls-max:    Element-wise max across per-chunk first-token hidden states
  - laat:       Label-Aware Attention Pooling over all tokens concatenated
  - laat-split: LAAT applied per chunk, then max-pooled across chunks
"""

import math
import logging
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

logger = logging.getLogger(__name__)


# =============================================================================
# SOURCE ATTRIBUTION - CLASS LEVEL
# =============================================================================
# ADAPTED FROM: PLM-ICD (Huang et al., ClinicalNLP 2022)
#   Original class : RobertaForMultilabelClassification
#   URL            : https://github.com/MiuLab/PLM-ICD/blob/main/src/modeling_roberta.py
#
# ORIGINAL PARTS (written by Namirah Imtieaz Shaik):
#   - Single-label CrossEntropyLoss block in forward()
#   - Inline comments explaining the add_pooling_layer=False difference from BERT
#     and why position 0 of last_hidden_state is used instead of pooler_output
#
# UNCHANGED FROM SOURCE:
#   - RobertaModel instantiation with add_pooling_layer=False
#   - CLS-sum, CLS-max, LAAT, LAAT-split aggregation logic
#   - Chunk-wise flattening in forward()
#   - SequenceClassifierOutput return structure
# =============================================================================
class RobertaForSingleLabelClassification(RobertaPreTrainedModel):
    """
    RoBERTa model for single-label ICD-10 classification using chunk-wise processing.

    Architecture:
      RobertaModel (backbone, fully fine-tuned, no pooling layer)
        → chunk-wise forward pass
        → aggregation (CLS-sum / CLS-max / LAAT)
        → dropout
        → Linear → 30 logits
        → CrossEntropyLoss

    Inheriting from RobertaPreTrainedModel provides from_pretrained() and
    save_pretrained() support for loading/saving RoBERTa-base weights.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Aggregation mode: "cls-sum", "cls-max", "laat", or "laat-split"
        self.model_mode = config.model_mode

        # ---- Backbone ----
        # SOURCE: RobertaModel from HuggingFace Transformers
        # add_pooling_layer=False: RoBERTa does not create a pooler output.
        # This is the key difference from BertModel which does create a pooler.
        # We extract [CLS] (position 0) directly from last_hidden_state instead.
        # All parameters are trainable - full fine-tuning, no frozen layers.
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # Dropout applied before the classifier for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # ---- Classification head ----
        if "cls" in self.model_mode:
            # Simple linear projection: [hidden_size → num_labels]
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        elif "laat" in self.model_mode:
            # SOURCE: LAAT mechanism from PLM-ICD (Huang et al., 2022)
            self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
            self.third_linear = nn.Linear(config.hidden_size, config.num_labels)
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")

        self.init_weights()

    def forward(
        self,
        input_ids=None,          # (batch_size, num_chunks, chunk_size)
        attention_mask=None,     # (batch_size, num_chunks, chunk_size)
        token_type_ids=None,     # (batch_size, num_chunks, chunk_size) — unused by RoBERTa
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,             # (batch_size,) - single integer class index
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass for single-label ICD-10 classification with RoBERTa.

        Args:
            input_ids:      Token IDs after chunk-wise reshaping.
                            Shape: (batch_size, num_chunks, chunk_size).
                            Will be flattened to (B*num_chunks, chunk_size) for RoBERTa.
            attention_mask: 1 for real tokens, 0 for padding. Same shape as input_ids.
            token_type_ids: Segment IDs - RoBERTa does not use these, but they are
                            passed through from the collator for API compatibility.
            labels:         Single integer class index per example.
                            Shape: (batch_size,), dtype: torch.long.
                            ADAPTATION: was [B, num_labels] binary vector in multi-label.

        Returns:
            SequenceClassifierOutput with .loss and .logits.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Unpack the 3D chunked shape
        batch_size, num_chunks, chunk_size = input_ids.size()

        # ---- Step 1: Flatten chunks for RoBERTa ----
        # SOURCE: Chunk flattening pattern from PLM-ICD (Huang et al., 2022)
        # Same as BERT: merge batch and chunk dimensions so RoBERTa sees each
        # chunk as an independent 2D sequence.
        # (B, num_chunks, chunk_size) → (B*num_chunks, chunk_size)
        outputs = self.roberta(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size) if attention_mask is not None else None,
            token_type_ids=token_type_ids.view(-1, chunk_size) if token_type_ids is not None else None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # outputs[0] = last_hidden_state: (B*num_chunks, chunk_size, hidden_size)
        # Note: outputs.pooler_output does NOT exist because add_pooling_layer=False.

        # ---- Step 2: Aggregation ----
        # SOURCE: CLS-sum, CLS-max, LAAT aggregation from PLM-ICD (Huang et al., 2022)
        if "cls" in self.model_mode:
            # CLS-SUM / CLS-MAX AGGREGATION
            #
            # KEY DIFFERENCE FROM BERT (original PLM-ICD implementation):
            # BERT:    uses outputs.pooler_output  - [CLS] passed through linear + tanh
            # RoBERTa: extracts position 0 directly from last_hidden_state
            #          because there is no pooler layer (add_pooling_layer=False)
            last_hidden_state = outputs[0]

            # Extract the first token (position 0 = [CLS] equivalent in RoBERTa)
            cls_per_chunk = last_hidden_state[:, 0, :]               # (B*num_chunks, H)
            pooled_output = cls_per_chunk.view(batch_size, num_chunks, -1)  # (B, num_chunks, H)

            if self.model_mode == "cls-sum":
                pooled_output = pooled_output.sum(dim=1)              # (B, H)
            elif self.model_mode == "cls-max":
                pooled_output = pooled_output.max(dim=1).values      # (B, H)
            else:
                raise ValueError(f"model_mode {self.model_mode} not recognized")

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)                  # (B, num_labels)

        elif "laat" in self.model_mode:
            # LAAT AGGREGATION
            # SOURCE: LAAT mechanism from PLM-ICD (Huang et al., 2022)
            if self.model_mode == "laat":
                hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)   # (B, L, H)
            elif self.model_mode == "laat-split":
                hidden_output = outputs[0].view(batch_size * num_chunks, chunk_size, -1)   # (B*num_chunks, chunk_size, H)

            weights = torch.tanh(self.first_linear(hidden_output))
            att_weights = self.second_linear(weights)
            att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
            weighted_output = att_weights @ hidden_output
            logits = (
                self.third_linear.weight.mul(weighted_output).sum(dim=2).add(self.third_linear.bias)
            )
            if self.model_mode == "laat-split":
                logits = logits.view(batch_size, num_chunks, -1).max(dim=1).values
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")

        # ---- Step 3: Compute loss - SINGLE-LABEL ADAPTATION ----
        # ORIGINAL CODE: Written by Namirah Imtieaz Shaik
        # The original PLM-ICD used BCEWithLogitsLoss for multi-label classification.
        # This block replaces it with CrossEntropyLoss for single-label classification.
        loss = None
        if labels is not None:
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

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )