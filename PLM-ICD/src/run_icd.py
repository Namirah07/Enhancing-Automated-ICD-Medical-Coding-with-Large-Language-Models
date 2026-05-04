# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
#       File  : run_icd.py (original multi-label training script)
#
#   (2) HuggingFace example training scripts
#       URL   : https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification
#
#   (3) HuggingFace Accelerate library
#       URL   : https://github.com/huggingface/accelerate
#
# CHANGES MADE BY NAMIRAH IMTIEAZ SHAIK:
#   - Changed preprocess_function() to map LABELS column to a single integer
#     class index instead of a multi-hot binary vector
#   - Changed data_collator() to produce labels as dtype=torch.long integer tensor
#     [B] instead of float tensor [B, num_labels]
#   - Changed evaluation loop to use softmax + argmax instead of sigmoid + threshold
#   - Replaced BCEWithLogitsLoss metric interpretation with CrossEntropyLoss
#   - Added ROC-AUC weighted OVR metric alongside F1 metrics
#   - Added detailed inline comments throughout
#
# UNCHANGED FROM SOURCE:
#   - Overall script structure (parse_args, main, training loop, eval loop)
#   - Accelerator setup and distributed training boilerplate
#   - Data loading via HuggingFace datasets library
#   - Optimizer (AdamW) and LR scheduler setup
#   - Chunk-wise data collation logic
#   - Model saving via save_pretrained
#   - Logging configuration
# =============================================================================

"""
run_icd.py - Main training and evaluation script for the PLM-ICD deep learning models.

This script was adapted from the original multi-label PLM-ICD framework
(Huang et al., ClinicalNLP 2022) to perform SINGLE-LABEL multiclass ICD-10
classification on the MIMIC-IV dataset.

Key adaptation from multi-label to single-label:
  - Original: labels were a multi-hot binary vector [B, num_labels],
              loss was BCEWithLogitsLoss, output activation was sigmoid.
  - Adapted:  labels are a single integer class index [B],
              loss is CrossEntropyLoss, output activation is softmax.

Supports three backbone variants:
  - BERT-PLM-ICD   (BertForSingleLabelClassification)
  - RoBERTa-PLM-ICD (RobertaForSingleLabelClassification)
  - Longformer-PLM-ICD (LongformerForSingleLabelClassification)

All three backbones are fully fine-tuned end-to-end — no layers are frozen.
"""

import argparse
import logging
import math
import os
import random

import datasets
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
import torch
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)

from datasets import load_dataset  # type: ignore

# AdamW from torch.optim is used because transformers.AdamW is deprecated
# in recent versions of the transformers library.
from torch.optim import AdamW

# Accelerate handles distributed training across multiple GPUs automatically.
# find_unused_parameters=True is required because in LAAT mode the classifier
# linear layer (used in CLS mode) is unused, which would otherwise cause a
# DDP error about unused parameters.
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# Import the three custom single-label modeling classes adapted from PLM-ICD.
# These replace the original multi-label BertForMultilabelClassification etc.
from modeling_bert import BertForSingleLabelClassification
from modeling_roberta import RobertaForSingleLabelClassification
from modeling_longformer import LongformerForSingleLabelClassification

# Evaluation metrics — all standard sklearn functions.
# NOTE: In single-label multiclass classification, Micro Precision = Micro Recall
# = Micro F1 = Accuracy because sum(FP) = sum(FN) across all classes.
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Registry mapping model type string → model class.
# Used in main() to select the correct class based on --model_type argument.
MODELS_CLASSES = {
    "bert": BertForSingleLabelClassification,
    "roberta": RobertaForSingleLabelClassification,
    "longformer": LongformerForSingleLabelClassification,
}


# =============================================================================
# SOURCE ATTRIBUTION - parse_args()
# =============================================================================
# ADAPTED FROM: PLM-ICD (Huang et al., 2022) and HuggingFace example scripts
#   URL: https://github.com/MiuLab/PLM-ICD/blob/main/run_icd.py
#   URL: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
# All argument definitions are taken from the original PLM-ICD script.
# No changes were made to the argument list itself.
# =============================================================================
def parse_args():
    """
    Parse all command-line arguments for the training script.

    Key arguments:
      --train_file / --validation_file : paths to the CSV data files
      --code_file                      : path to label_vocab.txt (one ICD code per line)
      --model_name_or_path             : HuggingFace model name or local checkpoint path
      --model_type                     : one of bert / roberta / longformer
      --model_mode                     : aggregation strategy — cls-sum / cls-max / laat / laat-split
      --chunk_size                     : number of tokens per chunk (default 256)
      --max_length                     : maximum total token length before chunking (default 128)
      --gradient_accumulation_steps    : simulates larger batch size without extra GPU memory
      --num_train_epochs               : set to 0 for eval-only mode (loads from --output_dir)
    """
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="(Optional) Task name, not really used for ICD.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--code_file", type=str, default=None, help="A txt file containing all codes."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help=("The size of chunks that we'll split the inputs into"),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="The type of model",
        required=True,
        choices=["bert", "roberta", "longformer"],
    )
    parser.add_argument(
        "--model_mode",
        type=str,
        help="Specify how to aggregate output in the model",
        required=True,
        choices=["cls-sum", "cls-max", "laat", "laat-split"],
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the hugging face Tokenizers library).",
    )
    parser.add_argument(
        "--cased",
        action="store_true",
        help="equivalent to do_lower_case=False",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--code_50", action="store_true", help="use only top-50 codes")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()

    # Sanity checks - ensure at least one data source is provided
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # Accelerator setup
    # -------------------------------------------------------------------------
    # Accelerator wraps the training loop for multi-GPU / distributed training.
    # find_unused_parameters=True prevents DDP errors when some model parameters
    # (e.g. the cls classifier in LAAT mode) are not used in the forward pass.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Configure logging - only the main process logs at INFO level to avoid
    # duplicate log messages when running with multiple GPUs.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set random seed for reproducibility across Python, NumPy, and PyTorch.
    if args.seed is not None:
        set_seed(args.seed)

    # -------------------------------------------------------------------------
    # Load raw datasets from CSV/JSON files
    # -------------------------------------------------------------------------
    # The datasets library loads the CSV files into a DatasetDict with
    # "train" and "validation" splits. Each row has TEXT and LABELS columns.
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    # -------------------------------------------------------------------------
    # Build label vocabulary from label_vocab.txt
    # -------------------------------------------------------------------------
    # label_vocab.txt contains one ICD-10 code per line.
    # Labels are sorted alphabetically to produce a deterministic class index
    # ordering that is consistent across all model families.
    # label_to_id maps ICD code string → integer class index (e.g. "I2510" → 7)
    labels = set()
    all_codes_file = args.code_file

    with open(all_codes_file, "r") as f:
        for line in f:
            if line.strip() != "":
                labels.add(line.strip())

    # Sort to ensure deterministic ordering across all runs and models
    label_list = sorted(list(labels))
    num_labels = len(label_list)  # Should be 30 for our MIMIC-IV benchmark

    # Reverse mapping: ICD code string → integer index
    label_to_id = {v: i for i, v in enumerate(label_list)}

    # -------------------------------------------------------------------------
    # Load model config, tokenizer, and model
    # -------------------------------------------------------------------------
    # AutoConfig loads the HuggingFace config for the chosen backbone and
    # attaches our task-specific settings (num_labels, model_mode).
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )

    # Longformer uses attention_window to set the local attention window size,
    # which we align with chunk_size so each chunk is processed with the same
    # local attention scope. BERT and RoBERTa receive model_mode to select
    # the aggregation strategy (cls-sum, cls-max, laat, laat-split).
    if args.model_type == "longformer":
        config.attention_window = args.chunk_size
    elif args.model_type in ["bert", "roberta"]:
        config.model_mode = args.model_mode

    # Load the tokenizer for the chosen backbone.
    # do_lower_case=True (default) converts text to lowercase before tokenization,
    # which is appropriate for Bio_ClinicalBERT which was pretrained on lowercased text.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        do_lower_case=not args.cased,
    )

    # Select the correct model class from the registry and load pretrained weights.
    # If num_train_epochs > 0, load from the HuggingFace pretrained checkpoint.
    # If num_train_epochs == 0 (eval-only), load from the local output_dir.
    model_class = MODELS_CLASSES[args.model_type]
    if args.num_train_epochs > 0:
        # Training mode: load pretrained backbone weights from HuggingFace
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        # Eval-only mode: load previously saved fine-tuned model from output_dir
        model = model_class.from_pretrained(
            args.output_dir,
            config=config,
        )

    # Column names for the text input. sentence2_key is None because we have
    # a single text input (no sentence pair task).
    sentence1_key, sentence2_key = "TEXT", None
    padding = False  # Dynamic padding is applied in the data collator instead

    # -------------------------------------------------------------------------
    # Preprocess function - SINGLE LABEL ADAPTATION
    # -------------------------------------------------------------------------
    # ORIGINAL CODE: Written by Namirah Imtieaz Shaik
    # The original PLM-ICD preprocess_function produced a multi-hot binary list
    # for labels. This version maps each ICD code string to a single integer
    # class index. The tokenization call is unchanged from the original.
    # -------------------------------------------------------------------------
    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *texts,
            padding=padding,
            max_length=args.max_length,
            truncation=True,
            # For CLS-based models: do NOT add special tokens here because
            # [CLS] and [SEP] are inserted manually per-chunk in the collator.
            # For LAAT-based models: add special tokens normally here.
            add_special_tokens="cls" not in args.model_mode,
        )

        # SINGLE-LABEL: map each ICD code string to its integer class index.
        # Each example["LABELS"] contains a single ICD code string (e.g. "I2510").
        if "LABELS" in examples:
            label_ids = []
            for lab in examples["LABELS"]:
                if lab is None:
                    raise ValueError("Found None in LABELS; please clean your dataset.")
                code = lab.strip()
                if code not in label_to_id:
                    raise ValueError(f"Label '{code}' not found in code vocab file {all_codes_file}")
                # Append single integer index - NOT a binary vector
                label_ids.append(label_to_id[code])
            result["label_ids"] = label_ids

        return result

    # Apply preprocessing to all dataset splits.
    # remove_columns drops the original TEXT and LABELS string columns since
    # they are no longer needed after tokenization.
    remove_columns = (
        raw_datasets["train"].column_names
        if args.train_file is not None
        else raw_datasets["validation"].column_names
    )
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=remove_columns
    )

    eval_dataset = processed_datasets["validation"]

    if args.num_train_epochs > 0:
        train_dataset = processed_datasets["train"]
        # Log three random training examples to verify preprocessing is correct
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            logger.info(
                f"Original tokens: {tokenizer.decode(train_dataset[index]['input_ids'])}"
            )

    # -------------------------------------------------------------------------
    # Data collator - SINGLE LABEL with chunk-wise processing
    # -------------------------------------------------------------------------
    # ADAPTED FROM: PLM-ICD (Huang et al., 2022)
    #   URL: https://github.com/MiuLab/PLM-ICD/blob/main/run_icd.py
    # The chunk-wise padding and reshaping logic is taken directly from the
    # original PLM-ICD data_collator.
    #
    # ORIGINAL CODE (written by Namirah Imtieaz Shaik):
    # The label tensor construction at the bottom of this function is new.
    # The original produced float labels [B, num_labels] for BCEWithLogitsLoss.
    # This version produces long integer labels [B] for CrossEntropyLoss.
    # -------------------------------------------------------------------------
    def data_collator(features):
        batch = {}

        # CLS-based chunking: manually insert [CLS] and [SEP] around each chunk.
        # chunk_size - 2 content tokens per chunk to leave room for [CLS] and [SEP].
        if "cls" in args.model_mode:
            for f in features:
                new_input_ids = []
                for i in range(0, len(f["input_ids"]), args.chunk_size - 2):
                    # Each chunk: [CLS] + content_tokens + [SEP]
                    new_input_ids.extend(
                        [tokenizer.cls_token_id]
                        + f["input_ids"][i: i + (args.chunk_size) - 2]
                        + [tokenizer.sep_token_id]
                    )
                f["input_ids"] = new_input_ids
                # All positions in the chunked sequence are real tokens (mask=1)
                f["attention_mask"] = [1] * len(f["input_ids"])
                # token_type_ids are all 0 (single-sequence, not sentence pair)
                f["token_type_ids"] = [0] * len(f["input_ids"])

        # Pad all sequences in the batch to the same length, then make that
        # length a multiple of chunk_size so the tensor can be cleanly reshaped
        # into (batch_size, num_chunks, chunk_size).
        max_length = max(len(f["input_ids"]) for f in features)
        if max_length % args.chunk_size != 0:
            max_length = max_length - (max_length % args.chunk_size) + args.chunk_size

        # Build input_ids tensor: pad each sequence to max_length, then reshape
        # from (batch_size, max_length) → (batch_size, num_chunks, chunk_size).
        # The model forward methods expect this 3D shape.
        batch["input_ids"] = (
            torch.tensor(
                [
                    f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
                    for f in features
                ]
            )
            .contiguous()
            .view((len(features), -1, args.chunk_size))
        )

        # Build attention_mask tensor: 0 for padding positions, 1 for real tokens.
        # Also reshaped to (batch_size, num_chunks, chunk_size).
        if "attention_mask" in features[0]:
            batch["attention_mask"] = (
                torch.tensor(
                    [
                        f["attention_mask"] + [0] * (max_length - len(f["attention_mask"]))
                        for f in features
                    ]
                )
                .contiguous()
                .view((len(features), -1, args.chunk_size))
            )

        # token_type_ids: all zeros for single-sequence classification.
        # Only present for BERT; RoBERTa and Longformer do not use this.
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = (
                torch.tensor(
                    [
                        f["token_type_ids"] + [0] * (max_length - len(f["token_type_ids"]))
                        for f in features
                    ]
                )
                .contiguous()
                .view((len(features), -1, args.chunk_size))
            )

        # SINGLE-LABEL: labels are a 1D tensor of integer class indices [B].
        # ADAPTATION FROM MULTI-LABEL:
        #   Original: labels = torch.tensor([[0,1,0,...]], dtype=torch.float)  # [B, num_labels]
        #   Adapted:  labels = torch.tensor([7, 2, 15, ...], dtype=torch.long) # [B]
        label_ids = torch.tensor([f["label_ids"] for f in features], dtype=torch.long)
        batch["labels"] = label_ids

        return batch

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------
    # Training dataloader shuffles examples each epoch.
    # Evaluation dataloader does not shuffle to ensure deterministic results.
    if args.num_train_epochs > 0:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
        )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # -------------------------------------------------------------------------
    # Optimizer - AdamW with weight decay
    # -------------------------------------------------------------------------
    # Weight decay (L2 regularization) is applied to all parameters EXCEPT:
    #   - bias terms: applying weight decay to biases hurts performance
    #   - LayerNorm weights: these are scale parameters, not regular weights
    # This is the standard practice for fine-tuning transformer models.
    #
    # IMPORTANT: All model parameters are included here - backbone + classifier head.
    # This is what makes PLM-ICD a full fine-tuning approach, unlike the
    # discriminative LLMs which only update LoRA adapter parameters.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            # Parameters that receive weight decay (all regular weight matrices)
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            # Parameters that do NOT receive weight decay (bias, LayerNorm)
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # -------------------------------------------------------------------------
    # Prepare everything with Accelerator for distributed training
    # -------------------------------------------------------------------------
    # accelerator.prepare() wraps the model, optimizer, and dataloaders
    # so they work correctly across multiple GPUs or machines.
    model, optimizer, eval_dataloader = accelerator.prepare(
        model, optimizer, eval_dataloader
    )
    if args.num_train_epochs > 0:
        train_dataloader = accelerator.prepare(train_dataloader)

        # Calculate total number of optimizer update steps.
        # Each epoch has len(train_dataloader) batches, but with gradient
        # accumulation we only update weights every gradient_accumulation_steps batches.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            # If max_train_steps is set directly, compute the implied number of epochs
            args.num_train_epochs = math.ceil(
                args.max_train_steps / num_update_steps_per_epoch
            )

        # Learning rate scheduler - controls how the LR changes during training.
        # Linear schedule: starts at learning_rate and linearly decays to 0.
        # Warmup steps: LR ramps up from 0 to learning_rate during the first
        # num_warmup_steps steps to avoid large gradient steps at the start.
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

    # -------------------------------------------------------------------------
    # Training loop - SINGLE LABEL
    # -------------------------------------------------------------------------
    if args.num_train_epochs > 0:
        # Log training configuration for reproducibility
        total_batch_size = (
            args.per_device_train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) "
            f"= {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        progress_bar = tqdm(
            range(args.max_train_steps), disable=not accelerator.is_local_main_process
        )
        completed_steps = 0

        for epoch in tqdm(range(args.num_train_epochs)):
            # Set model to training mode - enables dropout for regularization
            model.train()
            epoch_loss = 0.0

            for step, batch in enumerate(train_dataloader):
                # Forward pass: model receives the chunked input and computes
                # CrossEntropyLoss internally using the labels in the batch.
                outputs = model(**batch)
                loss = outputs.loss

                # Divide loss by gradient_accumulation_steps so the effective
                # gradient magnitude stays correct regardless of accumulation count.
                loss = loss / args.gradient_accumulation_steps

                # Backward pass: accelerator.backward handles distributed training
                # correctly (replaces the standard loss.backward() call).
                accelerator.backward(loss)
                epoch_loss += loss.item()

                # Only update weights after accumulating enough gradients.
                # This simulates a larger effective batch size without extra memory.
                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    optimizer.step()        # Update all model parameters using accumulated gradients
                    lr_scheduler.step()     # Advance the learning rate schedule
                    optimizer.zero_grad()   # Reset gradients to zero for the next accumulation window
                    progress_bar.update(1)
                    completed_steps += 1
                    progress_bar.set_postfix(
                        loss=epoch_loss / max(completed_steps, 1)
                    )

                if completed_steps >= args.max_train_steps:
                    break

            # -----------------------------------------------------------------
            # Evaluation at the end of each epoch - SINGLE LABEL
            # -----------------------------------------------------------------
            # ORIGINAL CODE: Written by Namirah Imtieaz Shaik
            # The original PLM-ICD evaluation loop used sigmoid thresholding
            # for multi-label prediction. This version uses softmax + argmax
            # for single-label prediction and adds ROC-AUC weighted OVR.
            # The Accelerator gather pattern is adapted from HuggingFace
            # Accelerate documentation examples.
            # -----------------------------------------------------------------
            # Set model to eval mode - disables dropout for deterministic output
            model.eval()
            all_preds = []
            all_labels = []
            all_probs = []

            for step, batch in enumerate(tqdm(eval_dataloader, disable=not accelerator.is_local_main_process)):
                with torch.no_grad():
                    # Forward pass without gradient computation (faster, less memory)
                    outputs = model(**batch)

                logits = outputs.logits  # [B, num_labels] - raw unnormalized scores

                # SINGLE-LABEL ADAPTATION:
                # Apply softmax to get a probability distribution over the 30 classes.
                # Original multi-label used sigmoid applied independently to each logit.
                # Softmax ensures all 30 probabilities sum to 1 (mutually exclusive).
                probs = torch.softmax(logits.float(), dim=-1)  # [B, num_labels]

                # argmax picks the class with highest logit as the prediction.
                # This is equivalent to picking the most probable class under softmax.
                preds = logits.argmax(dim=-1)  # [B] — single predicted class per example

                # Gather predictions from all GPUs if using distributed training.
                # Safe to use even on a single GPU.
                preds = accelerator.gather_for_metrics(preds)
                labels = accelerator.gather_for_metrics(batch["labels"])
                probs = accelerator.gather_for_metrics(probs)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

            # Concatenate predictions from all batches
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs, axis=0)  # [N, num_labels]

            # Compute evaluation metrics
            acc = accuracy_score(all_labels, all_preds)
            macro_f1 = f1_score(all_labels, all_preds, average="macro")
            micro_f1 = f1_score(all_labels, all_preds, average="micro")

            macro_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
            macro_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
            weighted_prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
            weighted_rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

            # ROC-AUC: one-vs-rest scheme across all 30 classes.
            # Uses the softmax probability matrix all_probs [N, 30] as the score.
            # This is more informative than using hard predictions because it
            # captures how confidently the model ranks the correct class.
            auc_macro_ovr = None
            auc_weighted_ovr = None
            try:
                n_classes = all_probs.shape[1]
                auc_macro_ovr = roc_auc_score(
                    all_labels,
                    all_probs,
                    labels=list(range(n_classes)),
                    multi_class="ovr",
                    average="macro",      # Equal weight to each class
                )
                auc_weighted_ovr = roc_auc_score(
                    all_labels,
                    all_probs,
                    labels=list(range(n_classes)),
                    multi_class="ovr",
                    average="weighted",   # Weight by class frequency — our primary AUC metric
                )
            except Exception as e:
                logger.warning(f"ROC-AUC could not be computed: {repr(e)}")

            auc_macro_str = f"{auc_macro_ovr:.4f}" if auc_macro_ovr is not None else "NA"
            auc_weighted_str = f"{auc_weighted_ovr:.4f}" if auc_weighted_ovr is not None else "NA"

            logger.info(f"Epoch {epoch} finished")
            logger.info(
                "Evaluation metrics: "
                f"accuracy={acc:.4f}, "
                f"macro_f1={macro_f1:.4f}, micro_f1={micro_f1:.4f}, "
                f"macro_precision={macro_prec:.4f}, macro_recall={macro_rec:.4f}, "
                f"weighted_precision={weighted_prec:.4f}, weighted_recall={weighted_rec:.4f}, "
                f"roc_auc_macro_ovr={auc_macro_str}, "
                f"roc_auc_weighted_ovr={auc_weighted_str}"
            )
            # Per-class breakdown showing precision, recall, F1 for each ICD code
            logger.info("Classification report:\n" + classification_report(all_labels, all_preds))

    # -------------------------------------------------------------------------
    # Pure evaluation mode (num_train_epochs == 0)
    # -------------------------------------------------------------------------
    # When --num_train_epochs 0 is passed, skip training entirely and only
    # evaluate the model loaded from --output_dir on the validation set.
    if args.num_train_epochs == 0 and accelerator.is_local_main_process:
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            # Apply softmax for probability scores needed by ROC-AUC
            probs = torch.softmax(logits.float(), dim=-1)
            preds = logits.argmax(dim=-1)

            preds = accelerator.gather_for_metrics(preds)
            labels = accelerator.gather_for_metrics(batch["labels"])
            probs = accelerator.gather_for_metrics(probs)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs, axis=0)

        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        micro_f1 = f1_score(all_labels, all_preds, average="micro")

        macro_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        macro_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        weighted_prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        weighted_rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

        auc_macro_ovr = None
        auc_weighted_ovr = None
        try:
            n_classes = all_probs.shape[1]
            auc_macro_ovr = roc_auc_score(
                all_labels,
                all_probs,
                labels=list(range(n_classes)),
                multi_class="ovr",
                average="macro",
            )
            auc_weighted_ovr = roc_auc_score(
                all_labels,
                all_probs,
                labels=list(range(n_classes)),
                multi_class="ovr",
                average="weighted",
            )
        except Exception as e:
            logger.warning(f"ROC-AUC could not be computed: {repr(e)}")

        auc_macro_str = f"{auc_macro_ovr:.4f}" if auc_macro_ovr is not None else "NA"
        auc_weighted_str = f"{auc_weighted_ovr:.4f}" if auc_weighted_ovr is not None else "NA"

        logger.info("Evaluation finished")
        logger.info(
            "Evaluation metrics: "
            f"accuracy={acc:.4f}, "
            f"macro_f1={macro_f1:.4f}, micro_f1={micro_f1:.4f}, "
            f"macro_precision={macro_prec:.4f}, macro_recall={macro_rec:.4f}, "
            f"weighted_precision={weighted_prec:.4f}, weighted_recall={weighted_rec:.4f}, "
            f"roc_auc_macro_ovr={auc_macro_str}, "
            f"roc_auc_weighted_ovr={auc_weighted_str}"
        )
        logger.info("Classification report:\n" + classification_report(all_labels, all_preds))

    # -------------------------------------------------------------------------
    # Save the fine-tuned model
    # -------------------------------------------------------------------------
    # accelerator.wait_for_everyone() ensures all processes finish before saving,
    # preventing partial writes in distributed training.
    # accelerator.unwrap_model() removes the DDP wrapper to get the original model.
    # save_pretrained() saves in HuggingFace format so it can be loaded later
    # with model_class.from_pretrained(output_dir, config=config).
    if args.output_dir is not None and args.num_train_epochs > 0:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()