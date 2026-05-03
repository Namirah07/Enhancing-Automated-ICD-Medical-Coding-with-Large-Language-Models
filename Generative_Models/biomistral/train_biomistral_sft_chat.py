# train_biomistral_sft_chat.py
#
# =============================================================================
# SOURCE ATTRIBUTION - FILE LEVEL
# =============================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
#
# This script is entirely original work mirroring train_meditron_sft_chat.py
# for the BioMistral-7B backbone, with label mixing as the key difference.
# The same library component attributions apply:
#
#   TRL SFTTrainer (von Werra et al., 2020):
#     URL: https://github.com/huggingface/trl
#
#   LoRA / PEFT (Dettmers et al., NeurIPS 2023):
#     URL: https://github.com/huggingface/peft
#
#   HuggingFace Transformers / datasets:
#     URL: https://huggingface.co/docs/transformers
#
# WHAT IS ENTIRELY ORIGINAL (written by Namirah Imtieaz Shaik):
#   - extract_fields(): chat JSONL parsing
#   - build_head_label_set(): cumulative frequency head label identification
#   - resample_to_size(): oversample/undersample with replacement
#   - build_head_tail_mixture(): head/tail label resampling for BioMistral bias correction
#   - preprocess(): inline tokenization + label masking (different from Meditron
#     which uses TailICDCollator - here labels are masked inside the map() call)
#   - The decision to use default_data_collator instead of TailICDCollator
#     because label masking is applied upfront in preprocess()
#   - The label mixing strategy itself (40% head / 60% tail) as an original
#     contribution to correct BioMistral's head-code over-prediction bias
# =============================================================================
#
# This script trains BioMistral-7B as a generative ICD-10 coder using TRL SFTTrainer.
# It is the BioMistral counterpart of train_meditron_sft_chat.py and shares the
# same overall approach - the model is instruction-tuned to autoregressively generate
# the ICD code string rather than classifying it.
#
# The key feature that distinguishes this script from the Meditron generative training
# is LABEL MIXING. During training, BioMistral showed a tendency to over-predict the
# most frequent ICD codes in the dataset (the "head" codes) at the expense of rarer
# ones (the "tail" codes). Label mixing addresses this by resampling the training set
# so that 40% of examples come from head codes and 60% from tail codes, deliberately
# underrepresenting frequent codes and overrepresenting rare ones.
#
# Head labels are defined as the smallest set of most frequent codes that together
# account for head_coverage (default 40%) of all training examples. Everything else
# is a tail label.
#
# The second key feature is HEAD+TAIL TEXT CROPPING, same as Meditron. Discharge notes
# are far longer than the model's context window, so we keep a head slice and a tail
# slice of the note tokens and discard the middle.
#
# Unlike train_meditron_sft_chat.py, BioMistral uses a preprocessed dataset approach -
# we tokenize every example upfront in preprocess() and assign labels=-100 to all
# prompt tokens. This means we do not need a custom collator for tail supervision -
# it is handled directly during preprocessing.
#
# Prompt format (same as Meditron):
#   [SYSTEM]
#   <system instruction>
#   [/SYSTEM]
#   [USER]
#   <head+tail cropped discharge note>
#   [/USER]
#   [ICD]
#   ICD-10 code: <ICD code><eos>

import os
import math
import argparse
from typing import Dict, Any, List, Tuple

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


# =========================================================================
# SOURCE ATTRIBUTION - extract_fields (train_biomistral_sft_chat.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Simpler version than Meditron's extract_fields - assumes messages is
# always a well-formed list (no double-encoding edge case handling needed
# for BioMistral's data pipeline). Entirely original.
# =========================================================================
def extract_fields(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Extract the system prompt, user note, and assistant ICD code from a chat JSONL row.

    Each row in the chat JSONL has a messages list with three entries: one system
    message, one user message (the discharge note), and one assistant message (the ICD code).
    We loop through and pick out each role's content.

    If the assistant field is empty or missing we fall back to Z5111 rather than
    returning an empty string, because an empty ICD code would cause the label
    computation in preprocess() to produce an all -100 label tensor.
    """
    system, user, icd = "", "", ""
    for m in ex["messages"]:
        r = m.get("role", "")
        c = m.get("content", "")
        if r == "system":
            system = c
        elif r == "user":
            user = c
        elif r == "assistant":
            icd = (c or "").strip()
    if icd == "":
        icd = "Z5111"  # fallback - Z5111 is a valid code in our vocabulary
    return system, user, icd


# =========================================================================
# SOURCE ATTRIBUTION - build_head_label_set
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Cumulative frequency approach to define HEAD vs TAIL labels is entirely
# original. The head_coverage threshold and the use of sorted(counts.items())
# are original design choices for this thesis.
# =========================================================================
def build_head_label_set(train_ds: Dataset, head_coverage: float) -> Tuple[set, Dict[str, int]]:
    """
    Identify which ICD codes are HEAD labels by cumulative frequency.

    We count how many times each ICD code appears in the training set, then sort
    codes from most frequent to least frequent. We walk down the sorted list adding
    codes to the head set until the codes we have collected together account for
    head_coverage fraction of all training examples.

    Example: with head_coverage=0.40 and 16,540 training examples, we keep adding
    the most frequent codes until they collectively cover 6,616 examples (40%).
    The resulting head set is typically a small number of high-frequency codes.
    Everything not in the head set is a TAIL label.

    Returns:
        head_labels - set of ICD code strings that are considered HEAD
        counts      - dict mapping every ICD code to its training frequency
    """
    counts: Dict[str, int] = {}
    for ex in train_ds:
        _, _, icd = extract_fields(ex)
        counts[icd] = counts.get(icd, 0) + 1

    total = sum(counts.values())
    target = head_coverage * total  # we want to cover this many examples

    # Sort codes from most to least frequent
    sorted_labels = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    head_labels = set()
    running = 0
    for lab, c in sorted_labels:
        # Stop once we have covered the target fraction AND have at least one label
        if running >= target and len(head_labels) > 0:
            break
        head_labels.add(lab)
        running += c

    return head_labels, counts


# =========================================================================
# SOURCE ATTRIBUTION - resample_to_size
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Oversample/undersample with replacement logic using HuggingFace Dataset
# .select() is entirely original. The torch.randint fallback for the
# __index_level_0__ edge case is original defensive coding.
# =========================================================================
def resample_to_size(ds: Dataset, target_size: int, seed: int) -> Dataset:
    """
    Resize a dataset to exactly target_size examples using downsampling or oversampling.

    If target_size < len(ds), we shuffle and take the first target_size examples.
    If target_size > len(ds), we need to oversample with replacement. We do this
    by first repeating the full dataset as many times as needed and then sampling
    the remainder from a randomly shuffled copy.

    The fallback at the end handles an edge case in the oversampling logic where
    the index list might not come out to exactly target_size due to HuggingFace
    Dataset internals - we just use torch.randint to draw indices directly in that case.
    """
    n = len(ds)
    if n == 0:
        return ds
    if target_size == n:
        return ds
    if target_size < n:
        # Undersample: shuffle and take the first target_size rows
        return ds.shuffle(seed=seed).select(range(target_size))

    # Oversample with replacement by repeating the dataset in full cycles
    full = target_size // n     # how many complete copies of the dataset we need
    rem = target_size % n       # how many extra examples we need after the full copies
    idx = list(range(n)) * full
    if rem > 0:
        # Try to get the remainder indices from HuggingFace's internal index column
        if "__index_level_0__" in ds.column_names:
            idx += ds.shuffle(seed=seed).select(range(rem)).to_dict()["__index_level_0__"]
        else:
            idx += list(range(rem))

    # Robust fallback: if the index list is not the right size, just sample randomly
    if len(idx) != target_size:
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randint(low=0, high=n, size=(target_size,), generator=g).tolist()

    return ds.select(idx)


# =========================================================================
# SOURCE ATTRIBUTION - build_head_tail_mixture
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# The label mixing strategy (resampling head and tail groups independently
# to reach a target proportion) is entirely original. The 40/60 default
# (head_mix=0.4, tail_mix=0.6) was chosen through experimentation and is
# an original contribution of this thesis to address BioMistral's
# head-code over-prediction bias.
# =========================================================================
def build_head_tail_mixture(
    train_ds: Dataset,
    head_labels: set,
    head_mix: float,
    tail_mix: float,
    seed: int,
) -> Dataset:
    """
    Build a resampled training dataset with a controlled head/tail label ratio.

    We split the training set into two groups - examples whose ICD code is a head label,
    and examples whose ICD code is a tail label. We then resample each group independently
    to hit the target proportions (head_mix / (head_mix + tail_mix) and the complement).

    Why we need this:
    Without mixing, BioMistral tends to overfit on the most frequent codes because the
    training loss is dominated by those examples. By oversampling rare (tail) codes and
    undersampling common (head) codes, we force the model to pay more attention to the
    long tail of ICD codes it would otherwise ignore.

    The default settings (head_mix=0.4, tail_mix=0.6) produce a training set where
    40% of examples are from head codes and 60% from tail codes, regardless of their
    natural frequency in the original data. The total size of the mixed dataset is
    preserved at len(train_ds) so training time is not affected.

    We use seed+1 for the tail resample to avoid using the same shuffle for both groups.
    """
    assert head_mix > 0 and tail_mix > 0, "head_mix and tail_mix must be > 0"

    # Normalize so they sum to 1
    s = head_mix + tail_mix
    head_mix /= s
    tail_mix /= s

    # Split into head and tail groups
    head_ds = train_ds.filter(lambda ex: extract_fields(ex)[2] in head_labels)
    tail_ds = train_ds.filter(lambda ex: extract_fields(ex)[2] not in head_labels)

    # Compute target sizes that preserve the total training set size
    total = len(train_ds)
    target_head = int(round(total * head_mix))
    target_tail = total - target_head

    # Resample each group independently to hit the targets
    head_rs = resample_to_size(head_ds, target_head, seed=seed)
    tail_rs = resample_to_size(tail_ds, target_tail, seed=seed + 1)

    # Concatenate and shuffle so head and tail examples are interleaved
    mixed = concatenate_datasets([head_rs, tail_rs]).shuffle(seed=seed)
    return mixed


# =========================================================================
# SOURCE ATTRIBUTION - main (train_biomistral_sft_chat.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Full pipeline (label mixing -> tokenize with inline label masking ->
# SFTTrainer with default_data_collator) is entirely original.
# SFTTrainer usage follows TRL documentation (von Werra et al., 2020).
# URL: https://github.com/huggingface/trl
# LoraConfig with task_type=CAUSAL_LM follows PEFT documentation.
# URL: https://github.com/huggingface/peft
# =========================================================================
def main():
    """
    Entry point. Loads chat JSONL data, applies label mixing, tokenizes with
    head+tail note cropping, and runs SFTTrainer with tail-only label supervision.

    Key differences from train_meditron_sft_chat.py:
      1. Label mixing via build_head_tail_mixture() to reduce head-code bias
      2. Tokenization and label masking happen in preprocess() before training,
         not inside a custom collator - so we use default_data_collator
      3. The label mask is applied directly in preprocess() rather than via TailICDCollator
      4. No device_map used - model is loaded directly and moved to GPU

    The preprocess() function is defined inside main() so it can close over args
    and tokenizer without needing to pass them as arguments to every map() call.
    """
    parser = argparse.ArgumentParser()

    # Model and data paths
    parser.add_argument("--model_name", type=str, default="BioMistral/BioMistral-7B")
    parser.add_argument("--train_path", type=str, default="../shared/train_chat.jsonl")
    parser.add_argument("--dev_path", type=str, default="../shared/dev_chat.jsonl")
    parser.add_argument("--out_dir", type=str, default="./output/biomistral_sft_chat")

    # Sequence length and text cropping parameters
    parser.add_argument("--max_length", type=int, default=512)
    # completion_max_tokens reserves a fixed budget for the ICD code + EOS at training time
    parser.add_argument("--completion_max_tokens", type=int, default=16)
    # text_head_fraction and text_tail_fraction control the head/tail note split.
    # They are normalized to sum to 1 inside preprocess(), so passing 0.4/0.6
    # is equivalent to 40% head and 60% tail.
    parser.add_argument("--text_head_fraction", type=float, default=0.4)
    parser.add_argument("--text_tail_fraction", type=float, default=0.6)

    # Label mixing parameters
    # head_coverage: the top-N most frequent codes that together cover this fraction of training data
    parser.add_argument("--head_coverage", type=float, default=0.4)
    # head_mix/tail_mix: target proportions in the resampled training set
    parser.add_argument("--head_mix", type=float, default=0.4)
    parser.add_argument("--tail_mix", type=float, default=0.6)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load BioMistral's tokenizer - no pad token by default, so we set it to EOS
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the raw chat JSONL splits
    raw = load_dataset("json", data_files={"train": args.train_path, "validation": args.dev_path})
    train_raw = raw["train"]
    val_raw = raw["validation"]

    # Build the head label set from the full training data before any resampling
    head_labels, counts = build_head_label_set(train_raw, head_coverage=args.head_coverage)

    # Apply label mixing to produce the resampled training set
    train_mixed = build_head_tail_mixture(
        train_raw,
        head_labels=head_labels,
        head_mix=args.head_mix,
        tail_mix=args.tail_mix,
        seed=args.seed,
    )

    print(f"Train size original: {len(train_raw)} | mixed: {len(train_mixed)}")
    print(f"Unique labels train: {len(counts)} | HEAD labels: {len(head_labels)}")
    print(f"Label mix target: head={args.head_mix:.2f} tail={args.tail_mix:.2f}")
    print(f"Text crop: head={args.text_head_fraction:.2f} tail={args.text_tail_fraction:.2f}")

    # =========================================================================
    # SOURCE ATTRIBUTION - preprocess (nested function)
    # =========================================================================
    # ORIGINAL CODE: Written by Namirah Imtieaz Shaik
    # Inline label masking (labels=-100 for prompt, real IDs for completion)
    # inside the map() call is the key difference from train_meditron_sft_chat.py
    # which uses TailICDCollator. The all-zero safety fallback (labels[-1])
    # and math.floor for exact budget allocation are original choices.
    # =========================================================================
    def preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize one chat example into causal LM tensors with tail-only label supervision.

        This function does three things:
          1. Formats the example into the [SYSTEM]/[USER]/[ICD] prompt structure
          2. Crops the user note to fit within the token budget using head+tail cropping
          3. Assigns labels=-100 to all prompt tokens so only the ICD completion tokens
             contribute to the training loss

        The token budget is computed as:
          user_budget = max_length - len(prefix_ids) - len(suffix_ids) - len(comp_ids)

        This ensures the full formatted example (prompt + ICD + EOS) fits within max_length
        even before the final truncation step.

        The head+tail crop splits the available user_budget using text_head_fraction and
        text_tail_fraction, normalized to sum to 1. For the default 0.4/0.6 split, 40% of
        the budget goes to the note beginning and 60% to the note end.

        The safety fallback at the end (labels[-1] = input_ids[-1]) handles the edge case
        where all label tokens happen to be masked - this ensures at least one position
        contributes to the loss to prevent a NaN gradient.
        """
        system, user, icd = extract_fields(ex)

        # Tokenize the completion (ICD code + EOS token)
        # We cap at completion_max_tokens to match the budget reservation
        completion = icd + tokenizer.eos_token
        comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"][: args.completion_max_tokens]

        # Fixed prefix and suffix that wrap the note content
        prefix = "[SYSTEM]\n" + system + "\n[/SYSTEM]\n[USER]\n"
        suffix = "\n[/USER]\n[ICD]\nICD-10 code: "

        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]

        # Remaining budget for the user note tokens
        user_budget = args.max_length - (len(prefix_ids) + len(suffix_ids) + len(comp_ids))
        user_budget = max(0, user_budget)

        user_ids_full = tokenizer(user, add_special_tokens=False)["input_ids"]

        # Apply head+tail cropping to fit the note into user_budget
        if user_budget <= 0 or len(user_ids_full) == 0:
            # No budget left or empty note - skip the note entirely
            user_ids = []
        else:
            if len(user_ids_full) <= user_budget:
                # Note fits entirely within budget - keep all tokens
                user_ids = user_ids_full
            else:
                # Note is longer than budget - apply head+tail split
                head_frac = max(0.0, min(1.0, args.text_head_fraction))
                tail_frac = max(0.0, min(1.0, args.text_tail_fraction))
                s = head_frac + tail_frac
                # Normalize to sum to 1 in case the user passed non-normalized values
                if s <= 0:
                    head_frac, tail_frac = 0.4, 0.6
                    s = 1.0
                head_frac /= s
                tail_frac /= s

                # Use floor for head to ensure head + tail exactly equals user_budget
                h = int(math.floor(user_budget * head_frac))
                t = user_budget - h

                head_part = user_ids_full[:h] if h > 0 else []
                tail_part = user_ids_full[-t:] if t > 0 else []
                user_ids = head_part + tail_part

        # Assemble the full input: prompt tokens + ICD completion tokens
        prompt_ids = prefix_ids + user_ids + suffix_ids
        input_ids = (prompt_ids + comp_ids)[: args.max_length]
        attention_mask = [1] * len(input_ids)

        # Build labels: -100 for all prompt tokens, real token IDs for the completion
        # This is the tail-only supervision - prompt tokens contribute zero gradient
        labels = ([-100] * len(prompt_ids) + comp_ids)[: args.max_length]

        # Safety fallback: if all labels are -100 (very short sequence edge case),
        # set the last label to the last input token so we have at least one
        # supervised position and avoid a NaN loss
        if all(x == -100 for x in labels):
            labels[-1] = input_ids[-1]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # Apply preprocessing to the mixed training set and the validation set.
    # remove_columns drops the original "messages" column - we only keep
    # the tokenized fields that the model actually needs.
    train_ds = train_mixed.map(preprocess, remove_columns=train_mixed.column_names)
    val_ds = val_raw.map(preprocess, remove_columns=val_raw.column_names)

    # Load BioMistral as a causal LM (with the language modelling head)
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    # use_cache must be False during training - it is incompatible with gradient checkpointing
    base_model.config.use_cache = False

    # Apply LoRA adapters with task_type=CAUSAL_LM.
    # The same seven projection matrices are targeted as in the discriminative model,
    # but task_type=CAUSAL_LM configures PEFT for the full autoregressive generation path
    # rather than for a sequence classification head.
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()  # confirms only ~0.1% of params are trainable

    # Standard HuggingFace TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,     # keep only the 2 most recent epoch checkpoints
        report_to="none",       # disable wandb/tensorboard
        # remove_unused_columns=False is required because we pass pretokenized tensors
        # directly and do not want SFTTrainer to drop any columns based on model signature
        remove_unused_columns=False,
    )

    # SFTTrainer with default_data_collator.
    # Unlike the Meditron script which uses a custom TailICDCollator, we use
    # default_data_collator here because the label masking was already applied
    # in preprocess() - all prompt positions already have labels=-100.
    # default_data_collator just stacks the pre-built tensors into a batch.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
    )

    trainer.train()

    # Save the LoRA adapter weights and the tokenizer
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Finished training. Saved to {args.out_dir}")


if __name__ == "__main__":
    main()