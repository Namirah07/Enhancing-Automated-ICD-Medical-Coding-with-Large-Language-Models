# train_meditron_sft_chat.py
#
# =============================================================================
# SOURCE ATTRIBUTION - FILE LEVEL
# =============================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
#
# This script is entirely original work. The following library components
# are used within it:
#
#   TRL SFTTrainer:
#     Adapted from: TRL library documentation
#     (von Werra et al., 2020 - "TRL: Transformer Reinforcement Learning")
#     URL: https://github.com/huggingface/trl
#     Specifically: SFTTrainer class, processing_class parameter usage
#
#   LoRA / PEFT:
#     Adapted from: PEFT library documentation and QLoRA paper
#     (Dettmers et al., NeurIPS 2023)
#     URL: https://github.com/huggingface/peft
#     Specifically: LoraConfig, get_peft_model with task_type=CAUSAL_LM
#
#   AutoModelForCausalLM (HuggingFace Transformers):
#     URL: https://huggingface.co/docs/transformers
#
# WHAT IS ENTIRELY ORIGINAL (written by Namirah Imtieaz Shaik):
#   - TailICDCollator: tail-only supervision strategy (mask prompt tokens,
#     supervise only the last icd_max_tokens positions)
#   - extract_fields(): chat JSONL parsing with three edge-case handlers
#   - safe_label(): Z5111 fallback for empty assistant fields
#   - crop_head_tail_tokens(): head+tail note cropping for long discharge notes
#   - build_text(): full prompt assembly with token budget calculation
#   - The decision to convert chat format to plain "text" before SFTTrainer
#     to avoid TRL chat_template dependency on Meditron's tokenizer
#   - The Z5111 dummy completion for identical train/eval budget measurement
#   - device_map=None during training (required for SFTTrainer compatibility)
# =============================================================================
#
# This script trains Meditron-7B as a generative ICD-10 coder using TRL SFTTrainer.
# Unlike the discriminative approach (train_meditron_cls.py) which treats ICD coding
# as a 30-class classification problem, this script treats it as a text generation
# problem: the model is instructed to autoregressively generate the ICD code as a string.
#
# The input data is a chat JSONL file where each example has three messages:
#   - system: fixed instruction telling the model its role
#   - user: the discharge note (head+tail cropped to fit the token budget)
#   - assistant: the ICD-10 code string (e.g. "I2510")
#
# We convert the chat format into plain formatted text before training because
# TRL 0.26.2 has issues with chat templates on Meditron's tokenizer. Formatting
# everything manually into a single "text" string sidesteps those issues entirely.
#
# The most important training design choice here is TAIL-ONLY SUPERVISION:
# we only compute the loss on the last icd_max_tokens tokens (the ICD code itself).
# All prompt tokens (system + user note) are masked with -100 so they contribute
# zero gradient. This forces the model to learn to generate the correct code,
# not to memorize the prompt format.
#
# Prompt format (OpenBioLLM-style, ICD at the end):
#   [SYSTEM]
#   <system instruction>
#   [/SYSTEM]
#   [USER]
#   <cropped clinical note>
#   [/USER]
#   [ICD]
#   ICD-10 code: <ICD><eos>

import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


# =========================================================================
# SOURCE ATTRIBUTION - TailICDCollator
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# The tail-only supervision strategy (masking all prompt positions to -100
# and supervising only the last icd_max_tokens positions) is entirely
# original. The tokenizer.pad() call follows standard HuggingFace usage
# but the label masking logic above it is original work.
# =========================================================================
@dataclass
class TailICDCollator:
    """
    Custom data collator that pads a batch of tokenized examples and then masks
    the labels so that only the final icd_max_tokens positions contribute to the loss.

    This is the core of the tail-only supervision strategy. After padding, every
    prompt token gets its label set to -100, which tells CrossEntropyLoss to ignore
    that position entirely. Only the ICD code tokens at the end of each sequence
    have real label values and receive gradient signal.

    Why mask the prompt?
    If we trained on the full sequence loss, the model would spend most of its
    learning capacity memorizing the prompt format rather than learning which ICD
    code corresponds to which clinical note. Supervising only the ICD tail forces
    the model to focus on generating the right code.

    The tokenizer argument uses AutoTokenizer as the type annotation rather than
    a specific class so this collator works with any HuggingFace tokenizer.
    """

    tokenizer: AutoTokenizer
    icd_max_tokens: int = 8  # ICD codes are typically 4-7 characters, so 8 tokens is enough

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Build a padded batch and apply the tail-only label mask.

        TRL sometimes keeps the raw 'text' string field inside each feature dict
        alongside the tokenized input_ids and attention_mask. The tokenizer.pad()
        method cannot handle string values in the feature dict - it expects only
        numeric tensors - so we strip 'text' and any other non-numeric fields
        out of each feature before calling pad().

        After padding, we clone input_ids as the labels and then mask everything
        except the last k = min(icd_max_tokens, sequence_length) tokens to -100.
        """
        # Strip raw string fields - keep only what tokenizer.pad() can handle
        cleaned = []
        for f in features:
            item = {}
            if "input_ids" in f:
                item["input_ids"] = f["input_ids"]
            if "attention_mask" in f:
                item["attention_mask"] = f["attention_mask"]
            # token_type_ids may be present for BERT-style tokenizers - keep if numeric
            if "token_type_ids" in f:
                item["token_type_ids"] = f["token_type_ids"]
            cleaned.append(item)

        batch = self.tokenizer.pad(cleaned, padding=True, return_tensors="pt")

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        # Start with labels = input_ids (standard causal LM teacher forcing)
        labels = input_ids.clone()
        B, T = input_ids.shape

        for i in range(B):
            # Find the actual sequence length by summing the attention mask.
            # Everything beyond the real sequence is padding and must be masked.
            if attention_mask is not None:
                length = int(attention_mask[i].sum().item())
            else:
                # Fallback: count non-pad tokens from the right
                row = input_ids[i].tolist()
                length = T
                while length > 0 and row[length - 1] == self.tokenizer.pad_token_id:
                    length -= 1

            if length <= 0:
                labels[i, :] = -100
                continue

            # k is the number of tail tokens that will receive gradient signal.
            # Everything before position (length - k) gets masked to -100.
            k = min(self.icd_max_tokens, length)
            labels[i, : length - k] = -100

        batch["labels"] = labels
        return batch


# =========================================================================
# SOURCE ATTRIBUTION - extract_fields
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# This function is entirely original. The three edge-case handlers
# (double-encoded JSON, single dict, missing role keys) are original
# defensive coding choices. No external reference was used.
# =========================================================================
def extract_fields(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Parse a single chat JSONL example and return the system prompt, user note,
    and assistant ICD code as separate strings.

    The messages field can arrive in several forms depending on how the JSONL
    was serialized. We handle three edge cases:
      - A JSON string that needs to be parsed (if messages was double-encoded)
      - A plain dict instead of a list (single-message edge case)
      - A list of message dicts (the normal case)

    If any role is missing we return an empty string for that field so the
    caller does not have to check for None.
    """
    system, user, icd = "", "", ""
    msgs = ex.get("messages", [])

    # Handle double-encoded JSON (messages stored as a string)
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            msgs = []

    # Handle single dict instead of list
    if isinstance(msgs, dict):
        msgs = [msgs]
    if not isinstance(msgs, list):
        msgs = []

    for m in msgs:
        if not isinstance(m, dict):
            continue
        r = m.get("role", "")
        if r == "system":
            system = m.get("content", "") or ""
        elif r == "user":
            user = m.get("content", "") or ""
        elif r == "assistant":
            icd = (m.get("content", "") or "").strip()

    return system, user, icd


# =========================================================================
# SOURCE ATTRIBUTION - safe_label
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Z5111 fallback choice and the empty-string guard are entirely original.
# =========================================================================
def safe_label(icd: str) -> str:
    """
    Return the ICD code if it is non-empty, otherwise return a safe fallback code.

    Z5111 (encounter for antineoplastic chemotherapy) is used as the fallback
    because it is a real valid ICD-10 code that exists in our 30-label vocabulary.
    An empty assistant field would cause the label masking logic to misbehave,
    so we always guarantee a non-empty string here.
    """
    icd = (icd or "").strip()
    return icd if icd else "Z5111"


# =========================================================================
# SOURCE ATTRIBUTION - crop_head_tail_tokens
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# The 40/60 head/tail split strategy for long discharge notes is an
# original contribution of this thesis. The idea to keep the beginning
# (admission context) and end (discharge summary) while discarding the
# middle is original experimental design, not taken from any reference.
# =========================================================================
def crop_head_tail_tokens(token_ids: List[int], budget: int, head_frac: float) -> List[int]:
    """
    Trim a token sequence to fit within budget by keeping a head slice and a tail slice.

    Discharge notes are far longer than any model's context window. Rather than
    simply truncating from the right (which would lose the discharge summary and
    final diagnosis at the end of the note), we keep:
      - The first head_frac * budget tokens (admission reason, chief complaint)
      - The last (1 - head_frac) * budget tokens (discharge summary, final diagnosis)
    and discard the middle section which tends to contain less diagnostically
    relevant content like nursing notes and procedural details.

    With head_frac=0.40 and budget=476 (a typical value after accounting for
    the prompt wrapper tokens), we keep 190 tokens from the note start and
    286 tokens from the note end.
    """
    if budget <= 0:
        return []
    if len(token_ids) <= budget:
        return token_ids

    head_keep = int(round(budget * head_frac))
    head_keep = max(0, min(head_keep, budget))
    tail_keep = budget - head_keep

    head_part = token_ids[:head_keep] if head_keep > 0 else []
    tail_part = token_ids[-tail_keep:] if tail_keep > 0 else []
    return head_part + tail_part


# =========================================================================
# SOURCE ATTRIBUTION - build_text
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# The [SYSTEM]/[USER]/[ICD] prompt format, the token budget calculation,
# the Z5111 dummy completion for consistent budget measurement between
# training and evaluation, and the full text assembly are all entirely
# original. The OpenBioLLM-style bracket tags were adopted from the
# OpenBioLLM model card but the full prompt structure is original.
# =========================================================================
def build_text(example: Dict[str, Any], tokenizer, max_len: int, text_head_frac: float) -> str:
    """
    Render one chat example into the plain text string that SFTTrainer trains on.

    The output format is:
      [SYSTEM]
      <system instruction>
      [/SYSTEM]
      [USER]
      <head+tail cropped discharge note>
      [/USER]
      [ICD]
      ICD-10 code: <ICD code>

    The token budget for the user note is computed by subtracting the prefix,
    suffix, and estimated completion lengths from max_len. This guarantees the
    full formatted string fits within max_len tokens even before the tokenizer
    applies its own truncation.

    We measure the dummy completion footprint (up to 16 tokens) using Z5111
    as a representative ICD code, which matches what the eval script uses for
    the same budget calculation. Using the same dummy ensures train and eval
    crop the note to identical lengths.
    """
    system, user, icd = extract_fields(example)
    icd = safe_label(icd)

    prefix = "[SYSTEM]\n" + system + "\n[/SYSTEM]\n[USER]\n"
    suffix = "\n[/USER]\n[ICD]\nICD-10 code: "

    # Measure the token footprint of the fixed parts
    dummy_completion = "Z5111" + (tokenizer.eos_token or "")
    comp_ids = tokenizer(dummy_completion, add_special_tokens=False)["input_ids"][:16]

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]

    # Whatever token budget is left after prefix + suffix + completion goes to the user note
    user_budget = max_len - (len(prefix_ids) + len(suffix_ids) + len(comp_ids))
    user_budget = max(0, user_budget)

    # Tokenize the full note and crop it to the budget using head+tail strategy
    user_ids = tokenizer(user, add_special_tokens=False)["input_ids"]
    user_ids = crop_head_tail_tokens(user_ids, user_budget, head_frac=text_head_frac)

    # Decode back to text so the full string can be assembled and passed to SFTTrainer
    user_cropped = tokenizer.decode(user_ids, skip_special_tokens=True)

    return prefix + user_cropped + suffix + icd


# =========================================================================
# SOURCE ATTRIBUTION - main (train_meditron_sft_chat.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# The overall pipeline (chat JSONL -> plain text -> SFTTrainer with
# TailICDCollator) is entirely original. SFTTrainer usage follows TRL
# documentation patterns (von Werra et al., 2020).
# URL: https://github.com/huggingface/trl
# LoraConfig with task_type=CAUSAL_LM follows PEFT documentation.
# URL: https://github.com/huggingface/peft
# The device_map=None design decision (required for SFTTrainer compatibility)
# and gradient_checkpointing_enable() usage are original choices.
# =========================================================================
def main():
    """
    Entry point. Loads the chat JSONL data, converts it to plain text format,
    applies LoRA to Meditron-7B, and runs SFTTrainer with tail-only supervision.

    Key steps:
      1. Load tokenizer and raw chat JSONL dataset
      2. Map every example through build_text() to produce a "text" column
      3. Load Meditron-7B via AutoModelForCausalLM (not AutoModel - we need the LM head)
      4. Wrap with LoRA using task_type=CAUSAL_LM
      5. Run SFTTrainer with TailICDCollator as the data collator
      6. Save the LoRA adapter and tokenizer

    Why SFTTrainer instead of a manual loop?
    SFTTrainer handles tokenization, padding, gradient accumulation, logging,
    and checkpoint saving automatically. Since we control supervision via the
    custom collator (TailICDCollator), we do not need the manual loop that the
    discriminative scripts use.

    Why device_map=None during training?
    device_map="auto" uses Accelerate's model sharding which is not compatible
    with the standard SFTTrainer training loop. We load the model on CPU and
    then call model.to(device) to move it to a single GPU in one shot.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="epfl-llm/meditron-7b")
    parser.add_argument("--train_path", type=str, default="../shared/train_chat.jsonl")
    parser.add_argument("--dev_path", type=str, default="../shared/dev_chat.jsonl")
    parser.add_argument("--out_dir", type=str, default="./output/meditron_sft_chat_sfttrainer")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # icd_max_tokens controls how many tail tokens receive gradient signal.
    # ICD-10 codes are 4-7 characters and tokenize to roughly 3-6 tokens,
    # so 8 is a safe ceiling that always covers the full code.
    parser.add_argument("--icd_max_tokens", type=int, default=8)

    # text_head_frac controls the head/tail split when cropping long notes.
    # 0.40 means 40% from the note beginning and 60% from the end.
    parser.add_argument("--text_head_frac", type=float, default=0.40)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Tokenizer setup - same as discriminative scripts
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load raw chat JSONL - each row has a "messages" field
    raw = load_dataset("json", data_files={"train": args.train_path, "validation": args.dev_path})
    train_raw = raw["train"]
    val_raw = raw["validation"]

    # Convert from chat format to plain "text" strings.
    # We remove all original columns (messages, etc.) and keep only "text"
    # because SFTTrainer expects a dataset with a single text column.
    # =========================================================================
    # SOURCE ATTRIBUTION - to_text (nested function)
    # =========================================================================
    # ORIGINAL CODE: Written by Namirah Imtieaz Shaik
    # Wraps build_text() for use in HuggingFace datasets.map().
    # The decision to remove all original columns and keep only "text"
    # so SFTTrainer does not encounter the messages column is original.
    # =========================================================================
    def to_text(ex: Dict[str, Any]) -> Dict[str, str]:
        """Render one chat example into the plain training text format consumed by SFTTrainer."""
        return {"text": build_text(ex, tokenizer, args.max_length, args.text_head_frac)}

    train_ds = train_raw.map(to_text, remove_columns=train_raw.column_names)
    val_ds = val_raw.map(to_text, remove_columns=val_raw.column_names)

    # Sanity check - verify the formatted output looks correct before training
    print("Sample text (first 250 chars):")
    print(train_ds[0]["text"][:250].replace("\n", "\\n"))

    # Load base model as a causal LM (with the language modelling head attached).
    # This is different from the discriminative scripts which use AutoModel.
    # Here we need the LM head so the model can generate token probabilities.
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,  # load on CPU first, then move to single GPU
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.config.use_cache = False            # required for gradient checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()     # saves memory by recomputing activations on backward

    # Apply LoRA with task_type=CAUSAL_LM.
    # Note: the discriminative scripts use task_type=SEQ_CLS.
    # CAUSAL_LM is required here because we are fine-tuning the full
    # autoregressive generation path rather than adding a classification head.
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Standard HuggingFace TrainingArguments - same pattern as other experiments
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,          # keep only the 2 most recent epoch checkpoints
        remove_unused_columns=False, # must be False or the collator's columns will be removed
        report_to="none",            # disable wandb/tensorboard logging
    )

    # Our custom collator handles padding and tail-label masking
    collator = TailICDCollator(tokenizer=tokenizer, icd_max_tokens=args.icd_max_tokens)

    # SFTTrainer setup:
    #   - train_dataset contains only the "text" column after our mapping
    #   - data_collator is our TailICDCollator
    #   - peft_config=None because we already applied LoRA manually above
    #   - formatting_func=None because our dataset already has plain text
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        formatting_func=None,
        peft_config=None,            # LoRA already applied - do not apply it again
    )

    trainer.train()

    # Save only the LoRA adapter weights (not the full backbone)
    trainer.model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Finished training. Saved to {args.out_dir}")


if __name__ == "__main__":
    main()