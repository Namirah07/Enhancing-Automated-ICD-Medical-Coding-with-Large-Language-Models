# train_openbiollm_sft_chat.py
#
# =============================================================================
# SOURCE ATTRIBUTION - FILE LEVEL
# =============================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
#
# This script is entirely original work. It inherits the same three ideas
# from the Meditron and BioMistral generative scripts and adds QLoRA:
#
#   TRL SFTTrainer (von Werra et al., 2020):
#     URL: https://github.com/huggingface/trl
#
#   LoRA / PEFT with task_type=CAUSAL_LM (Dettmers et al., NeurIPS 2023):
#     URL: https://github.com/huggingface/peft
#
#   BitsAndBytesConfig / QLoRA (Dettmers et al., NeurIPS 2023):
#     URL: https://arxiv.org/abs/2305.14314
#     URL: https://github.com/TimDettmers/bitsandbytes
#     Specifically: NF4 quantization, double_quant, prepare_model_for_kbit_training
#
#   prepare_model_for_kbit_training (PEFT):
#     URL: https://github.com/huggingface/peft
#
#   Mean pooling pattern:
#     Standard HuggingFace community pattern.
#     Reference: https://huggingface.co/blog/how-to-train-sentence-transformers
#
# WHAT IS ENTIRELY ORIGINAL (written by Namirah Imtieaz Shaik):
#   - extract_fields(), safe_label(), crop_head_tail_tokens(): same as other scripts
#   - compute_head_labels(): Counter-based cumulative frequency approach
#   - build_head_tail_mixture(): rng.choice with replace for over/undersampling
#   - sample() nested function: replace=True logic for oversampling
#   - build_text(): full prompt assembly with comp_reserve token budget
#   - TailICDCollator: tail-only supervision (same as Meditron version)
#   - to_text() nested function: chat -> plain text conversion
#   - The combination of QLoRA (4-bit NF4) + TailICDCollator + optional
#     label mixing in one generative training pipeline is entirely original
# =============================================================================
#
# This script trains OpenBioLLM-8B as a generative ICD-10 coder using TRL SFTTrainer.
# OpenBioLLM-8B is a Llama-3 based model that was fine-tuned by Aaditya on biomedical
# text. It is the largest backbone in the generative family (8B vs 7B for Meditron
# and BioMistral), which is why it requires QLoRA 4-bit quantization to fit on a
# single GPU during training, unlike the other two generative models.
#
# Three ideas from the other generative scripts are carried over here:
#
#   1. TAIL-ONLY SUPERVISION via TailICDCollator - loss is computed only on the
#      last icd_max_tokens positions (the ICD code), not on the system prompt or
#      clinical note. This forces the model to learn what code to generate, not
#      to memorize the formatting.
#
#   2. HEAD+TAIL TEXT CROPPING - discharge notes are far longer than the 512-token
#      context window so we keep a head slice (40%) and a tail slice (60%) and
#      discard the middle, preserving the admission context and discharge summary.
#
#   3. OPTIONAL LABEL MIXING - the same head/tail label resampling strategy from
#      train_biomistral_sft_chat.py is available via --use_label_mixing, though
#      it is off by default for OpenBioLLM.
#
# The key difference from Meditron and BioMistral training is QLoRA:
#   - BitsAndBytesConfig loads the backbone in 4-bit NF4 quantization
#   - prepare_model_for_kbit_training() sets up the quantized model for gradient flow
#   - LoRA adapters are then injected on top of the quantized backbone
#   - device_map="auto" is used (unlike the other scripts) because bitsandbytes
#     manages multi-GPU placement automatically when quantizing
#
# Prompt format (same as Meditron and BioMistral):
#   [SYSTEM]
#   <system instruction>
#   [/SYSTEM]
#   [USER]
#   <head+tail cropped discharge note>
#   [/USER]
#   [ICD]
#   ICD-10 code: <ICD code><eos>

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


# =========================================================================
# SOURCE ATTRIBUTION - extract_fields (train_openbiollm_sft_chat.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Same three edge-case handlers as Meditron version. Entirely original.
# =========================================================================
def extract_fields(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Parse one chat JSONL row and return (system, user, assistant-ICD) strings.

    The messages field may arrive in several forms depending on how the JSONL
    was serialized. We handle three edge cases:
      - A JSON string that needs to be parsed (double-encoded messages)
      - A single dict instead of a list (single-message edge case)
      - A list of message dicts (the normal case)

    If any role is missing we return an empty string for that field rather
    than None so the caller never needs to check for None.
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
        role = m.get("role", "")
        if role == "system":
            system = m.get("content", "") or ""
        elif role == "user":
            user = m.get("content", "") or ""
        elif role == "assistant":
            icd = (m.get("content", "") or "").strip()

    return system, user, icd


# =========================================================================
# SOURCE ATTRIBUTION - safe_label
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik. Z5111 fallback original.
# =========================================================================
def safe_label(icd: str) -> str:
    """
    Return the ICD code string if non-empty, otherwise return Z5111 as a fallback.

    An empty assistant field would cause build_text() to construct a training
    example with no valid completion, producing an all -100 label tensor that
    contributes no gradient. Z5111 is a real code in the vocabulary so it is
    a safe placeholder for any corrupted or missing label.
    """
    icd = (icd or "").strip()
    return icd if icd else "Z5111"


# =========================================================================
# SOURCE ATTRIBUTION - crop_head_tail_tokens (train_openbiollm_sft_chat.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Identical to Meditron and BioMistral versions - must be identical so
# training and eval see the same note content.
# =========================================================================
def crop_head_tail_tokens(token_ids: List[int], budget: int, head_frac: float) -> List[int]:
    """
    Trim a token sequence to fit within budget by keeping a head and tail slice.

    Discharge notes average over 10,000 characters which far exceeds any model's
    context window. Simple right-truncation would lose the discharge summary and
    final diagnosis at the end of the note. This function keeps:
      - The first head_frac * budget tokens (admission context, chief complaint)
      - The last (1 - head_frac) * budget tokens (discharge summary, final diagnosis)
    and discards the middle which tends to contain less diagnostically relevant content.

    With head_frac=0.40 this keeps 40% from the start and 60% from the end.
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
# SOURCE ATTRIBUTION - compute_head_labels
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Counter-based cumulative frequency approach using freq.most_common() is
# entirely original. Mirrors build_head_label_set() from BioMistral but
# uses Counter instead of a manual dict for clarity.
# =========================================================================
def compute_head_labels(train_ds: Dataset, head_fraction_labels: float) -> Tuple[set, Counter]:
    """
    Identify HEAD ICD codes by cumulative training frequency.

    We count how many times each ICD code appears across the training set, sort
    codes from most to least frequent, and accumulate codes until the running total
    covers head_fraction_labels of all training examples. The resulting set is the
    HEAD - the most frequent codes that dominate the training distribution.

    Everything not in the head set is a TAIL label. With head_fraction_labels=0.40
    and 16,540 training examples, we collect codes until they account for 6,616
    examples. This is typically a small number of very common diagnoses.

    Returns:
        head - set of ICD code strings considered HEAD labels
        freq - Counter mapping every ICD code to its training frequency
    """
    labels = [safe_label(extract_fields(ex)[2]) for ex in train_ds]
    freq = Counter(labels)
    total = sum(freq.values())
    running = 0
    head = set()

    for lab, c in freq.most_common():
        head.add(lab)
        running += c
        # Stop once we have covered the target fraction of all training examples
        if total > 0 and (running / total) >= head_fraction_labels:
            break

    return head, freq


# =========================================================================
# SOURCE ATTRIBUTION - build_head_tail_mixture (train_openbiollm_sft_chat.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Uses numpy rng.choice with replace=True for oversampling (cleaner than
# the torch.randint fallback approach used in BioMistral). Entirely original.
# =========================================================================
def build_head_tail_mixture(train_ds: Dataset, head_labels: set, head_mix: float, seed: int) -> Dataset:
    """
    Resample the training set so that head_mix fraction comes from HEAD labels.

    We split all training examples into two groups based on their ICD code,
    then independently resample each group to reach the target sizes, and
    concatenate and shuffle the result. The total size of the mixed dataset
    equals the total size of the original training set so training time is unchanged.

    sampling with replacement via rng.choice handles both oversampling (when a
    group is smaller than its target) and undersampling (when it is larger).

    Default: head_mix=0.40 means 40% of examples come from high-frequency codes
    and 60% from rare codes, counteracting the natural class imbalance.
    """
    rng = np.random.default_rng(seed)
    idx_head, idx_tail = [], []

    # Partition training examples into head and tail groups by ICD code
    for i, ex in enumerate(train_ds):
        icd = safe_label(extract_fields(ex)[2])
        (idx_head if icd in head_labels else idx_tail).append(i)

    n = len(train_ds)
    target_head = int(round(n * head_mix))
    target_tail = n - target_head

    def sample(pool: List[int], k: int) -> List[int]:
        """
        Sample k indices from a pool, using replacement when the pool is smaller than k.

        replace=True is needed when we are oversampling - if a group has fewer
        examples than our target size we need to draw the same example multiple
        times. replace=False would raise an error in that case.
        """
        if k <= 0 or len(pool) == 0:
            return []
        replace = len(pool) < k  # oversample if pool is too small
        return rng.choice(pool, size=k, replace=replace).tolist()

    # Sample each group to its target size and combine
    chosen = sample(idx_head, target_head) + sample(idx_tail, target_tail)
    rng.shuffle(chosen)  # interleave head and tail examples
    return train_ds.select(chosen)


# =========================================================================
# SOURCE ATTRIBUTION - build_text (train_openbiollm_sft_chat.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Same as Meditron build_text() but adds EOS token at the very end of the
# string (icd + eos_token) since OpenBioLLM's TailICDCollator uses
# icd_max_tokens=16 (vs 8 for Meditron) to accommodate larger tokenizer.
# comp_reserve parameter name is original to this script.
# =========================================================================
def build_text(
    ex: Dict[str, Any],
    tokenizer,
    max_length: int,
    text_head_frac: float,
    comp_reserve: int,
) -> str:
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
      ICD-10 code: <ICD code><eos>

    We compute a token budget for the user note by measuring the fixed prefix
    and suffix lengths and reserving comp_reserve tokens for the ICD completion.
    This guarantees the full formatted string fits within max_length tokens.

    We use Z5111 as the dummy completion for the budget measurement - the same
    dummy must be used in the eval script's build_prompt_like_training() function
    so that training and eval crop the note to the exact same length.

    The ICD code and EOS token appear at the very end of the string. This is
    important for the TailICDCollator - it masks everything except the last
    icd_max_tokens positions, so the ICD must be at the tail.
    """
    system, user, icd = extract_fields(ex)
    icd = safe_label(icd)

    prefix = "[SYSTEM]\n" + system + "\n[/SYSTEM]\n[USER]\n"
    suffix = "\n[/USER]\n[ICD]\nICD-10 code: "

    # Measure the completion footprint to reserve tokens for it in the budget
    dummy_completion = "Z5111" + (tokenizer.eos_token or "")
    comp_ids = tokenizer(dummy_completion, add_special_tokens=False)["input_ids"][:comp_reserve]

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]

    # Remaining token budget for the clinical note content
    user_budget = max_length - (len(prefix_ids) + len(suffix_ids) + len(comp_ids))
    user_budget = max(0, user_budget)

    # Tokenize and crop the note to the budget using head+tail strategy
    user_ids = tokenizer(user, add_special_tokens=False)["input_ids"]
    user_ids = crop_head_tail_tokens(user_ids, user_budget, head_frac=text_head_frac)
    user_cropped = tokenizer.decode(user_ids, skip_special_tokens=True)

    # Assemble full training string - ICD and EOS are at the very end
    return prefix + user_cropped + suffix + icd + (tokenizer.eos_token or "")


# =========================================================================
# SOURCE ATTRIBUTION - TailICDCollator (train_openbiollm_sft_chat.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Identical to Meditron's TailICDCollator. icd_max_tokens=16 here (vs 8)
# to accommodate OpenBioLLM's tokenizer which may split ICD codes into
# more tokens. Entirely original work.
# =========================================================================
@dataclass
class TailICDCollator:
    """
    Custom data collator that pads a batch and masks labels so only the ICD tail
    tokens contribute to the training loss.

    This is the same collator used in train_meditron_sft_chat.py. All prompt tokens
    (system instruction + clinical note) get labels=-100 so CrossEntropyLoss ignores
    them entirely. Only the last icd_max_tokens positions (the ICD code itself) have
    real label values and receive gradient signal.

    Why mask the prompt?
    If we trained on the full sequence loss the model would spend most capacity
    memorizing the prompt format rather than learning which ICD code corresponds
    to which clinical presentation. Supervising only the ICD tail forces the model
    to focus on the actual coding decision.

    TRL sometimes keeps the raw "text" string field inside each feature alongside
    the tokenized input_ids and attention_mask. The tokenizer.pad() method cannot
    handle string values, so we strip any non-numeric fields before padding.
    """

    tokenizer: AutoTokenizer
    icd_max_tokens: int = 16  # 16 is more generous than Meditron's 8 since OpenBioLLM is larger

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Build a padded batch and apply the tail-only label mask.

        For each sequence in the batch we find the true sequence length by summing
        the attention mask, then mask all positions before (length - k) to -100,
        where k = min(icd_max_tokens, length). The last k positions keep their
        real token IDs as labels so the loss is computed only on the ICD code.
        """
        # Strip raw string fields - keep only what tokenizer.pad() can handle
        cleaned = []
        for f in features:
            item = {}
            if "input_ids" in f:
                item["input_ids"] = f["input_ids"]
            if "attention_mask" in f:
                item["attention_mask"] = f["attention_mask"]
            # token_type_ids may be present for some tokenizer configs - keep if numeric
            if "token_type_ids" in f:
                item["token_type_ids"] = f["token_type_ids"]
            cleaned.append(item)

        batch = self.tokenizer.pad(cleaned, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        # Start with labels equal to input_ids (standard causal LM teacher forcing)
        labels = input_ids.clone()
        B, T = input_ids.shape

        for i in range(B):
            # Find actual sequence length from the attention mask
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

            # k tail positions receive gradient; everything before gets masked
            k = min(self.icd_max_tokens, length)
            labels[i, : length - k] = -100

        batch["labels"] = labels
        return batch


# =========================================================================
# SOURCE ATTRIBUTION - main (train_openbiollm_sft_chat.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Full QLoRA pipeline (4-bit NF4 -> prepare_model_for_kbit_training ->
# LoRA -> TailICDCollator -> SFTTrainer) is entirely original.
# prepare_model_for_kbit_training usage follows PEFT documentation.
# URL: https://github.com/huggingface/peft
# BitsAndBytesConfig follows QLoRA paper (Dettmers et al., NeurIPS 2023).
# URL: https://arxiv.org/abs/2305.14314
# =========================================================================
def main():
    """
    Entry point. Loads chat JSONL, optionally applies label mixing, converts to
    plain text, applies QLoRA to OpenBioLLM-8B, and runs SFTTrainer with
    tail-only label supervision.

    QLoRA vs standard LoRA:
    OpenBioLLM-8B has 8 billion parameters. At bf16 precision that is 16 GB just
    for the weights, which exceeds single-GPU memory when combined with activations
    and optimizer states during training. QLoRA solves this by loading the backbone
    in 4-bit NF4 quantization (roughly 4 GB) and injecting trainable LoRA adapters
    on top of the frozen quantized weights. Only the adapter parameters (~0.1% of
    total) are stored and updated in full precision.

    prepare_model_for_kbit_training() does three things:
      1. Disables the KV cache (incompatible with gradient checkpointing)
      2. Casts all layer norms to float32 for numerical stability
      3. Prepares the model's embedding layers for gradient flow through LoRA

    device_map="auto" is used here rather than loading on CPU then moving to GPU.
    bitsandbytes quantization requires device_map="auto" to place layers correctly
    across available GPUs during quantization.
    """
    parser = argparse.ArgumentParser()

    # Model and data paths
    parser.add_argument("--model_name", type=str, default="aaditya/Llama3-OpenBioLLM-8B")
    parser.add_argument("--train_path", type=str, default="../shared/train_chat.jsonl")
    parser.add_argument("--dev_path", type=str, default="../shared/dev_chat.jsonl")
    parser.add_argument("--out_dir", type=str, default="./output/openbiollm_sft_sfttrainer")

    parser.add_argument("--max_length", type=int, default=512)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # Label mixing parameters - off by default for OpenBioLLM
    parser.add_argument("--use_label_mixing", action="store_true")
    # head_fraction_labels: the top-N codes that together cover this fraction of training
    parser.add_argument("--head_fraction_labels", type=float, default=0.40)
    # head_mix: what fraction of the resampled training set comes from head codes
    parser.add_argument("--head_mix", type=float, default=0.40)

    # text_head_frac controls the head/tail split for note cropping
    parser.add_argument("--text_head_frac", type=float, default=0.40)

    # icd_max_tokens: number of tail tokens that receive gradient signal.
    # Set to 16 for OpenBioLLM (vs 8 for Meditron) because OpenBioLLM's tokenizer
    # may split some ICD codes into more tokens than Meditron's.
    parser.add_argument("--icd_max_tokens", type=int, default=16)
    # comp_reserve: how many token slots to reserve for the ICD completion in the budget.
    # Must match the dummy_completion measurement in build_text().
    parser.add_argument("--comp_reserve", type=int, default=16)

    # QLoRA / bitsandbytes settings
    # load_in_4bit is True by default - required for fitting 8B params on one GPU
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    # nf4 (NormalFloat4) preserves the weight distribution better than standard int4
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    # double_quant quantizes the quantization constants too, saving ~0.4 bits per param
    parser.add_argument("--bnb_double_quant", action="store_true", default=True)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # OpenBioLLM uses a Llama-3 tokenizer which also has no pad token by default
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load raw chat JSONL
    raw = load_dataset("json", data_files={"train": args.train_path, "validation": args.dev_path})
    train_ds = raw["train"]
    val_ds = raw["validation"]

    # Optionally apply label mixing to counteract head-code dominance in the loss
    if args.use_label_mixing:
        head_labels, freq = compute_head_labels(train_ds, args.head_fraction_labels)
        train_ds = build_head_tail_mixture(train_ds, head_labels, args.head_mix, args.seed)
        print(f"Train mixed size: {len(train_ds)} | HEAD labels: {len(head_labels)} | unique labels: {len(freq)}")
        print(f"Label mix target: head={args.head_mix:.2f} tail={1-args.head_mix:.2f}")
    else:
        print(f"Train size: {len(train_ds)} (no label mixing)")

    print(f"Text crop: head={args.text_head_frac:.2f} tail={1-args.text_head_frac:.2f}")

    # =========================================================================
    # SOURCE ATTRIBUTION - to_text (nested, train_openbiollm_sft_chat.py)
    # =========================================================================
    # ORIGINAL CODE: Written by Namirah Imtieaz Shaik
    # Same as Meditron's to_text - wraps build_text() for datasets.map().
    # =========================================================================
    def to_text(ex: Dict[str, Any]) -> Dict[str, str]:
        """
        Convert one chat JSONL example to a single plain text string for SFTTrainer.

        We map the chat messages through build_text() to produce the formatted training
        string, then return it as a dict with a single "text" key. SFTTrainer expects
        either a "text" column in the dataset or a formatting_func - using the "text"
        column approach avoids TRL's chat template logic which is not compatible with
        OpenBioLLM's tokenizer out of the box.
        """
        return {
            "text": build_text(
                ex,
                tokenizer=tokenizer,
                max_length=args.max_length,
                text_head_frac=args.text_head_frac,
                comp_reserve=args.comp_reserve,
            )
        }

    # Map both splits - remove_columns drops the original messages field
    train_text = train_ds.map(to_text, remove_columns=train_ds.column_names)
    val_text = val_ds.map(to_text, remove_columns=val_ds.column_names)

    # Verify the formatted output looks reasonable before spending GPU time on training
    print("Sample text (first 250 chars):")
    print(train_text[0]["text"][:250].replace("\n", "\\n"))

    # QLoRA quantization config.
    # compute_dtype controls the precision for arithmetic operations inside the
    # quantized layers - bf16 is preferred on A100/H100, fp16 on older GPUs.
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Load OpenBioLLM-8B in 4-bit quantization.
    # device_map="auto" lets bitsandbytes distribute the quantized layers across
    # available GPUs - this is required when using quantization_config.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config if args.load_in_4bit else None,
        device_map="auto",
        torch_dtype=(torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)),
    )
    model.config.use_cache = False           # must be False for gradient checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id

    # prepare_model_for_kbit_training sets up the quantized model for LoRA training:
    # casts layer norms to float32, disables cache, and enables gradient flow for adapters.
    # This step is specific to QLoRA and not needed for the Meditron/BioMistral scripts.
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # LoRA adapters injected on top of the 4-bit quantized backbone.
    # The same seven projection matrices are targeted as in all other LLM scripts.
    # task_type=CAUSAL_LM configures PEFT for the autoregressive generation path.
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # confirms only adapter weights (~0.1%) are trainable

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
        report_to="none",            # disable wandb/tensorboard
        remove_unused_columns=False, # required so SFTTrainer does not drop our text column
        gradient_checkpointing=True, # reduces memory by recomputing activations on backward
    )

    # TailICDCollator handles padding and applies the tail-only label mask
    collator = TailICDCollator(tokenizer=tokenizer, icd_max_tokens=args.icd_max_tokens)

    # SFTTrainer setup:
    #   - train_dataset has only "text" column after our mapping step
    #   - processing_class tells TRL which tokenizer to use for the text column
    #   - peft_config=None because we already applied LoRA manually above
    #   - formatting_func=None because the dataset already has plain text
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_text,
        eval_dataset=val_text,
        processing_class=tokenizer,
        data_collator=collator,
        formatting_func=None,
        peft_config=None,  # LoRA already applied - do not apply it again
    )

    trainer.train()

    # Save only the LoRA adapter weights (not the full quantized backbone)
    trainer.model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Finished training. Saved to {args.out_dir}")


if __name__ == "__main__":
    main()