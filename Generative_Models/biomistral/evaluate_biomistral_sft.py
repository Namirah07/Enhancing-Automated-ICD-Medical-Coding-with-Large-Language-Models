# evaluate_biomistral_sft.py
#
# =============================================================================
# SOURCE ATTRIBUTION - FILE LEVEL
# =============================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
#
# This script is entirely original work mirroring evaluate_meditron_sft.py
# for BioMistral-7B. The same function-level attributions apply as in the
# Meditron eval script. Key difference: build_prompt_ids_like_training_headtail()
# returns raw token ID lists instead of a string to avoid a decode-retokenize
# round-trip that could change the token sequence.
#
# Library component attributions:
#   PeftModel: https://github.com/huggingface/peft
#   sklearn metrics: https://scikit-learn.org
#
# WHAT IS ENTIRELY ORIGINAL (written by Namirah Imtieaz Shaik):
#   - All functions listed below (same as Meditron eval but adapted for BioMistral)
#   - build_prompt_ids_like_training_headtail(): returns token ID lists directly
#     (vs Meditron eval which returns a string) - original design choice to avoid
#     round-trip tokenization differences
#   - _short_trial_name approach from ray_tune_biomistral.py influence
#   - num_beams applies to BOTH constrained and unconstrained modes (original)
# =============================================================================
#
# Evaluation script for the BioMistral generative ICD-10 model (BioMistral-G).
#
# This script mirrors evaluate_meditron_sft.py but is adapted for BioMistral's
# checkpoint format and the slightly different prompt building approach used
# in train_biomistral_sft_chat.py.
#
# Like the Meditron eval script, three things must match training exactly:
#   1. The same prompt format ([SYSTEM]..[/SYSTEM][USER]..[/USER][ICD]...)
#   2. The same head+tail token-space cropping (text_head_frac, max_length)
#   3. The same reserve_completion_tokens budget
#
# One difference from Meditron eval: build_prompt_ids_like_training_headtail()
# returns raw token ID lists rather than a formatted string, which avoids a
# round-trip decode step. The tokenized prompt is passed directly to model.generate().
#
# Two generation modes:
#   - Greedy (default, num_beams=1): fastest, one beam, highest-probability token each step
#   - Constrained via trie (--constrain): forces the model to only generate valid ICD codes
#     from label_vocab.txt. num_beams applies to this path (default 1, increase for better coverage).
#
# Evaluation strategy:
#   - If the generated string matches a code in label_vocab.txt, it is a valid prediction
#   - If the model generates something outside the vocabulary, we record y_pred=-1 (invalid)
#   - Overall metrics include all examples; invalid predictions count as wrong
#   - Valid-only metrics (diagnostic) show performance on the subset where the model
#     produced a recognizable ICD code

import os
import argparse
import re
import json
from typing import Dict, Any, List, Tuple, Set

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    roc_auc_score,
)

# Regex to detect ICD-10 like patterns in generated text.
# ICD-10 codes start with a letter (A-T or V-Z, skipping U which is reserved),
# followed by two digits, then optionally a dot and up to 4 more characters.
ICD_REGEX = re.compile(r"[A-TV-Z][0-9][0-9A-Z]\.?[0-9A-Z]{0,4}")


# =========================================================================
# SOURCE ATTRIBUTION - load_label_vocab, normalize_icd,
#                      extract_icd_from_text, build_hard_score_matrix,
#                      safe_multiclass_roc_auc, Trie, build_icd_trie,
#                      build_prompt_ids_like_training_headtail, main
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# All functions in this file are entirely original work, mirroring the
# Meditron eval script with BioMistral-specific adaptations:
#   - extract_icd_from_text() searches full generated text (not just gen_only)
#     because BioMistral may generate preamble text before the code
#   - build_prompt_ids_like_training_headtail() returns raw token ID lists
#     directly (not a string) to avoid decode-retokenize round-trip
#   - num_beams applies to both constrained and unconstrained generation modes
#   - Trie, build_icd_trie, build_hard_score_matrix, safe_multiclass_roc_auc
#     are identical to the Meditron eval script
# sklearn metrics from: https://scikit-learn.org
# PEFT library: https://github.com/huggingface/peft
# =========================================================================
def load_label_vocab(path: str):
    """
    Read label_vocab.txt and build the three standard integer/code mappings.

    Alphabetical sort ensures the class index ordering is consistent with
    all other scripts in the pipeline.
    """
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            lab = line.strip()
            if lab:
                labels.append(lab)
    labels = sorted(set(labels))
    id2label = {i: lab for i, lab in enumerate(labels)}
    label2id = {lab: i for i, lab in enumerate(labels)}
    return labels, id2label, label2id


def normalize_icd(code: str) -> str:
    """
    Strip dots, spaces, and hyphens from an ICD code and uppercase it.

    BioMistral may generate "I25.10" when the vocabulary contains "I2510",
    or use lowercase letters. Normalizing both the generated code and the
    vocabulary before comparing prevents failing to match a correct prediction
    due to minor formatting differences.
    """
    if code is None:
        return ""
    code = code.strip().upper()
    # Keep only alphanumeric characters - remove dots, hyphens, spaces
    code = "".join(ch for ch in code if ch.isalnum())
    return code


def extract_icd_from_text(text: str) -> str:
    """
    Search the generated text for the first ICD-10 like token and return it normalized.

    We apply the regex to the full generated text (not just the first token) because
    BioMistral sometimes generates a preamble like "The code is I2510" instead of
    outputting just the code directly. The regex finds the code wherever it appears.

    Returns an empty string if no ICD-like token is found, which the caller treats
    as an invalid prediction.
    """
    if not text:
        return ""
    m = ICD_REGEX.search(text.upper())
    if not m:
        return ""
    return normalize_icd(m.group(0))


def build_hard_score_matrix(y_pred_np: np.ndarray, num_labels: int) -> np.ndarray:
    """
    Convert hard integer predictions to a score matrix for ROC-AUC computation.

    For each example we create a one-hot row: 1.0 at the predicted class index,
    0.0 everywhere else. Rows where y_pred=-1 (invalid prediction) get an all-zero
    row, which means they contribute nothing to any class's AUC but do lower the
    aggregated score appropriately since they represent missed predictions.

    This is less informative than using real softmax probabilities because there
    is no notion of how confident the model was in its prediction, but it is the
    only option for generation-based models that output a single string.
    """
    scores = np.zeros((len(y_pred_np), num_labels), dtype=np.float32)
    valid_mask = (y_pred_np >= 0) & (y_pred_np < num_labels)
    valid_idx = np.where(valid_mask)[0]
    scores[valid_idx, y_pred_np[valid_idx]] = 1.0
    return scores


def safe_multiclass_roc_auc(y_true_np: np.ndarray, y_score: np.ndarray, average: str) -> float:
    """
    Compute multiclass ROC-AUC with one-vs-rest scheme, returning NaN on failure.

    The most common failure case is when the test split contains fewer than two
    unique true classes, which can happen when evaluating on a small debug subset.
    We return NaN rather than raising so the rest of the metrics are still reported.
    """
    try:
        return float(
            roc_auc_score(
                y_true_np,
                y_score,
                multi_class="ovr",
                average=average,
                labels=list(range(y_score.shape[1])),
            )
        )
    except Exception:
        return float("nan")


class Trie:
    """
    Prefix trie for constrained token generation.

    Each node maps token IDs to child nodes and marks whether it represents a
    complete valid ICD code. During generation, at each step we look up which
    token IDs are valid continuations of the prefix generated so far. Only those
    token IDs are returned by the prefix_allowed_tokens_fn callback, making it
    impossible for the model to generate a code not in the vocabulary.
    """

    def __init__(self):
        self.next = {}    # token_id -> child Trie node
        self.end = False  # True if this node completes a valid ICD code

    def insert(self, token_ids: List[int]):
        """Insert a token ID sequence representing one valid ICD code into the trie."""
        node = self
        for t in token_ids:
            if t not in node.next:
                node.next[t] = Trie()
            node = node.next[t]
        node.end = True

    def allowed_next(self, prefix_ids: List[int]) -> Tuple[Set[int], bool]:
        """
        Given the tokens generated so far, return which token IDs can legally follow
        and whether the current prefix already completes a valid code.

        If prefix_ids does not match any path in the trie (model generated something
        off-vocab), we return an empty set. The generation function then falls back
        to EOS to end generation cleanly.
        """
        node = self
        for t in prefix_ids:
            if t not in node.next:
                return set(), False
            node = node.next[t]
        return set(node.next.keys()), node.end


def build_icd_trie(tokenizer, labels: List[str]) -> Trie:
    """
    Build a trie from all valid ICD codes in label_vocab.txt.

    We insert both the raw code string and the version with dots removed because
    different tokenizers split ICD codes differently. For example, I25.10 and I2510
    may tokenize differently, so inserting both variants ensures the trie covers
    all realistic tokenizations of every valid code.
    """
    trie = Trie()
    for lab in labels:
        variants = {lab.strip(), lab.strip().replace(".", "")}
        for v in variants:
            ids = tokenizer(v, add_special_tokens=False)["input_ids"]
            if len(ids) > 0:
                trie.insert(ids)
    return trie


def build_prompt_ids_like_training_headtail(
    tokenizer,
    messages: List[Dict[str, Any]],
    max_length: int,
    text_head_frac: float,
    reserve_completion_tokens: int = 16,
) -> Tuple[List[int], List[int], str]:
    """
    Build the inference prompt using the exact same format and token budget as training.

    This function must produce prompts that look identical to what the model saw
    during training. Any mismatch - different bracket tags, different cropping
    fractions, different reservation size - will cause a distribution shift at
    inference time and degrade performance in ways that are hard to diagnose.

    Unlike build_prompt_like_training() in evaluate_meditron_sft.py which returns
    a string, this function returns raw token ID lists directly. This avoids a
    decode-then-retokenize round trip that could subtly change the token sequence
    due to HuggingFace tokenizer normalization.

    The user note is cropped with the same head/tail strategy as training:
      - If the note fits within user_budget, keep it all
      - Otherwise keep the first head_frac * user_budget tokens and the last
        (1 - head_frac) * user_budget tokens, discarding the middle

    Args:
        tokenizer               - the BioMistral tokenizer
        messages                - list of chat message dicts (role + content)
        max_length              - maximum total sequence length
        text_head_frac          - fraction of user_budget to take from the note beginning
        reserve_completion_tokens - how many token slots to reserve for the ICD generation

    Returns:
        prompt_ids      - list of token IDs for the prompt (no ICD code at the end)
        attention_mask  - list of 1s the same length as prompt_ids
        gold            - the gold ICD code string from the assistant message
    """
    system, user, gold = "", "", ""
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "system":
            system = m.get("content", "") or ""
        elif m.get("role") == "user":
            user = m.get("content", "") or ""
        elif m.get("role") == "assistant":
            gold = (m.get("content", "") or "").strip()

    prefix = "[SYSTEM]\n" + system + "\n[/SYSTEM]\n[USER]\n"
    suffix = "\n[/USER]\n[ICD]\nICD-10 code: "

    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]

    # Compute how many tokens are available for the user note
    user_budget = max_length - (len(prefix_ids) + len(suffix_ids) + int(reserve_completion_tokens))
    user_budget = max(0, user_budget)

    user_ids = tokenizer(user, add_special_tokens=False)["input_ids"]

    if user_budget > 0 and len(user_ids) > user_budget:
        # Note is longer than budget - apply head+tail crop
        head_n = int(round(user_budget * float(text_head_frac)))
        head_n = max(0, min(head_n, user_budget))
        tail_n = user_budget - head_n

        head_part = user_ids[:head_n] if head_n > 0 else []
        tail_part = user_ids[-tail_n:] if tail_n > 0 else []
        user_ids = head_part + tail_part
    else:
        # Note fits within budget or budget is zero - keep as much as we can
        user_ids = user_ids[:user_budget] if user_budget > 0 else []

    # Assemble and truncate to max_length
    prompt_ids = (prefix_ids + user_ids + suffix_ids)[:max_length]
    attention_mask = [1] * len(prompt_ids)

    return prompt_ids, attention_mask, gold


def main():
    """
    Entry point. Loads the trained BioMistral-G model and evaluates it on the test set.

    We load BioMistral without quantization by default (unlike Meditron eval which
    uses 4-bit NF4). BioMistral-7B is the same parameter count as Meditron-7B so
    it fits in a single GPU in fp16 without quantization.

    The inference loop follows the same pattern as evaluate_meditron_sft.py:
      1. Build the prompt token IDs using build_prompt_ids_like_training_headtail()
      2. Pass to model.generate() with greedy or constrained settings
      3. Decode only the newly generated tokens (after input_len)
      4. Extract the ICD code with regex or use the raw generated text in constrained mode
      5. Compare against the gold label and record the result
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="BioMistral/BioMistral-7B")
    parser.add_argument("--adapter_dir", type=str, default="./output/biomistral_sft_chat")
    parser.add_argument("--test_path", type=str, default="../shared/test_chat.jsonl")
    parser.add_argument("--label_vocab", type=str, default="../shared/label_vocab.txt")

    parser.add_argument("--max_length", type=int, default=512)
    # max_new_tokens caps how many tokens the model generates after the prompt
    parser.add_argument("--max_new_tokens", type=int, default=8)
    # reserve_completion_tokens must match what was used during training
    parser.add_argument("--reserve_completion_tokens", type=int, default=16)

    # CRITICAL: text_head_frac must match the value used during training.
    # A different value means the note is cropped at a different point,
    # so the model sees a prompt that looks different from what it trained on.
    parser.add_argument(
        "--text_head_frac",
        type=float,
        default=0.40,
        help="Fraction of USER budget from the note start; rest from the note end.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # --constrain enables trie-based constrained decoding.
    # When active, the model physically cannot generate a code outside label_vocab.txt.
    parser.add_argument("--constrain", action="store_true",
                        help="Force outputs to be one of label_vocab via trie")
    # num_beams applies to both constrained and unconstrained modes.
    # Default 1 = greedy; increase for better coverage at the cost of more compute.
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Beam count for both constrained and unconstrained generation")
    # Print a sample prediction every debug_every examples to monitor progress
    parser.add_argument("--debug_every", type=int, default=200)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    labels, id2label, label2id = load_label_vocab(args.label_vocab)
    num_labels = len(labels)
    print(f"Loaded {num_labels} labels from {args.label_vocab}")

    # Load tokenizer from the adapter directory if it was saved there,
    # otherwise fall back to the base model tokenizer
    tok_source = args.adapter_dir if os.path.isdir(args.adapter_dir) else args.base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name, torch_dtype=dtype)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    # use_cache=True enables the KV cache during generation for faster decoding
    base_model.config.use_cache = True

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.to(device)
    model.eval()

    # Build the constraint trie if requested
    trie = None
    if args.constrain:
        trie = build_icd_trie(tokenizer, labels)
        print("Constraint mode ON: generation restricted to label_vocab.")

    print("Loading test dataset...")
    raw = load_dataset("json", data_files={"test": args.test_path})
    test_ds = raw["test"]
    print(f"Test examples: {len(test_ds)}")
    print("Running inference on test set...\n")

    # use_autocast enables mixed precision during generation on GPU
    use_autocast = (args.fp16 or args.bf16) and device.type == "cuda"
    autocast_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None)

    y_true, y_pred = [], []

    for idx, ex in enumerate(test_ds):
        # Build the prompt token IDs using the same format and cropping as training
        prompt_ids, attention_mask, gold_icd = build_prompt_ids_like_training_headtail(
            tokenizer=tokenizer,
            messages=ex["messages"],
            max_length=args.max_length,
            text_head_frac=args.text_head_frac,
            reserve_completion_tokens=args.reserve_completion_tokens,
        )

        gold_norm = normalize_icd(gold_icd)

        # Skip examples whose gold label is not in the 30-code vocabulary
        if gold_norm not in label2id:
            continue

        # Convert to tensors and move to device
        input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        attn = torch.tensor([attention_mask], device=device, dtype=torch.long)
        input_len = input_ids.shape[1]

        def prefix_allowed_tokens_fn(batch_id, sent):
            """
            Callback for constrained beam search that restricts generation to valid ICD prefixes.

            sent contains all tokens so far including the prompt. We slice off the prompt
            (input_len tokens) to get just what the model has generated, then look up
            which token IDs the trie allows to follow that prefix.

            If the trie says the prefix already completes a valid code (is_end=True),
            we add EOS to the allowed set so generation can terminate cleanly.

            If the trie returns an empty allowed set (model went off-vocab), we return
            just [EOS] to end generation immediately rather than looping infinitely.
            """
            gen_prefix = sent[input_len:].tolist()
            allowed, is_end = trie.allowed_next(gen_prefix)

            if is_end:
                allowed = set(allowed)
                allowed.add(tokenizer.eos_token_id)

            if not allowed:
                return [tokenizer.eos_token_id]
            return list(allowed)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,    # deterministic generation
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # num_beams applies to both modes - 1 = greedy, >1 = beam search
            num_beams=max(1, int(args.num_beams)),
        )

        if args.constrain:
            # Beam search with the trie constraint - all beams stay on valid ICD paths
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
            gen_kwargs["early_stopping"] = True

        with torch.no_grad():
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    output_ids = model.generate(**gen_kwargs)
            else:
                output_ids = model.generate(**gen_kwargs)

        # Decode only the tokens the model generated after the prompt
        gen_only_ids = output_ids[0, input_len:]
        gen_only = tokenizer.decode(gen_only_ids, skip_special_tokens=True).strip()

        # Try regex extraction first, then fall back to raw text in constrained mode
        pred_norm = extract_icd_from_text(gen_only)
        if pred_norm == "" and args.constrain:
            # In constrained mode the generated text itself is a valid code
            # (the trie guaranteed it), so we can normalize and use it directly
            pred_norm = normalize_icd(gen_only)

        y_true.append(label2id[gold_norm])
        # Use -1 as the sentinel for invalid/out-of-vocab predictions
        y_pred.append(label2id[pred_norm] if pred_norm in label2id else -1)

        if args.debug_every > 0 and (idx + 1) % args.debug_every == 0:
            full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Example {idx+1}:")
            print(f"GOLD: {gold_norm}")
            print(f"GEN ONLY: {repr(gen_only)}")
            print(f"PRED: {pred_norm if pred_norm else '<EMPTY/INVALID>'}")
            print("FULL(first 200):", repr(full[:200]))
            print()

    # Convert to numpy for sklearn metrics
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    n_total = len(y_true)
    valid_mask = y_pred != -1
    n_valid = int(valid_mask.sum())
    coverage = float(n_valid / max(1, n_total))

    print(f"Coverage (valid preds in vocab): {n_valid} / {n_total} = {coverage:.3f}")

    # Overall metrics - invalid predictions count as wrong automatically
    # since -1 never equals any gold label in [0, num_labels-1]
    overall_acc = float((y_pred == y_true).mean())

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_labels)), average="macro", zero_division=0,
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_labels)), average="micro", zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_labels)), average="weighted", zero_division=0,
    )

    # Build one-hot score matrix from hard predictions for ROC-AUC
    y_score = build_hard_score_matrix(y_pred, num_labels)
    roc_auc_macro_ovr = safe_multiclass_roc_auc(y_true, y_score, average="macro")
    roc_auc_weighted_ovr = safe_multiclass_roc_auc(y_true, y_score, average="weighted")

    print(f"Overall Accuracy (invalid wrong): {overall_acc:.6f}")
    print(f"Precision Macro:                {precision_macro:.6f}")
    print(f"Recall Macro:                   {recall_macro:.6f}")
    print(f"F1 Macro:                       {f1_macro:.6f}")
    print(f"Precision Micro:                {precision_micro:.6f}")
    print(f"Recall Micro:                   {recall_micro:.6f}")
    print(f"F1 Micro:                       {f1_micro:.6f}")
    print(f"Precision Weighted:             {precision_weighted:.6f}")
    print(f"Recall Weighted:                {recall_weighted:.6f}")
    print(f"F1 Weighted:                    {f1_weighted:.6f}")
    if np.isnan(roc_auc_macro_ovr):
        print("ROC-AUC Macro OVR:              NaN")
    else:
        print(f"ROC-AUC Macro OVR:              {roc_auc_macro_ovr:.6f}")
    if np.isnan(roc_auc_weighted_ovr):
        print("ROC-AUC Weighted OVR:           NaN")
    else:
        print(f"ROC-AUC Weighted OVR:           {roc_auc_weighted_ovr:.6f}")
    print()

    # Valid-only metrics are diagnostic - they tell us whether low overall
    # performance is caused by coverage failures or by actual prediction errors
    if n_valid == 0:
        # No valid predictions at all - set everything to zero/nan
        acc_valid = 0.0
        precision_macro_valid = recall_macro_valid = f1_macro_valid = 0.0
        precision_micro_valid = recall_micro_valid = f1_micro_valid = 0.0
        precision_weighted_valid = recall_weighted_valid = f1_weighted_valid = 0.0
        roc_auc_macro_valid = float("nan")
        roc_auc_weighted_valid = float("nan")
        report = "No valid predictions; per-label report unavailable."
        print(report)
    else:
        vt = y_true[valid_mask]
        vp = y_pred[valid_mask]

        acc_valid = accuracy_score(vt, vp)

        precision_macro_valid, recall_macro_valid, f1_macro_valid, _ = precision_recall_fscore_support(
            vt, vp, average="macro", zero_division=0
        )
        precision_micro_valid, recall_micro_valid, f1_micro_valid, _ = precision_recall_fscore_support(
            vt, vp, average="micro", zero_division=0
        )
        precision_weighted_valid, recall_weighted_valid, f1_weighted_valid, _ = precision_recall_fscore_support(
            vt, vp, average="weighted", zero_division=0
        )

        y_score_valid = build_hard_score_matrix(vp, num_labels)
        roc_auc_macro_valid = safe_multiclass_roc_auc(vt, y_score_valid, average="macro")
        roc_auc_weighted_valid = safe_multiclass_roc_auc(vt, y_score_valid, average="weighted")

        print("Metrics on VALID predictions only (diagnostic):")
        print(f"Accuracy(valid):                  {acc_valid:.6f}")
        print(f"Precision Macro(valid):           {precision_macro_valid:.6f}")
        print(f"Recall Macro(valid):              {recall_macro_valid:.6f}")
        print(f"F1 Macro(valid):                  {f1_macro_valid:.6f}")
        print(f"Precision Micro(valid):           {precision_micro_valid:.6f}")
        print(f"Recall Micro(valid):              {recall_micro_valid:.6f}")
        print(f"F1 Micro(valid):                  {f1_micro_valid:.6f}")
        print(f"Precision Weighted(valid):        {precision_weighted_valid:.6f}")
        print(f"Recall Weighted(valid):           {recall_weighted_valid:.6f}")
        print(f"F1 Weighted(valid):               {f1_weighted_valid:.6f}")
        if np.isnan(roc_auc_macro_valid):
            print("ROC-AUC Macro OVR(valid):         NaN")
        else:
            print(f"ROC-AUC Macro OVR(valid):         {roc_auc_macro_valid:.6f}")
        if np.isnan(roc_auc_weighted_valid):
            print("ROC-AUC Weighted OVR(valid):      NaN")
        else:
            print(f"ROC-AUC Weighted OVR(valid):      {roc_auc_weighted_valid:.6f}")
        print()

        report = classification_report(
            vt,
            vp,
            labels=list(range(num_labels)),
            target_names=[id2label[i] for i in range(num_labels)],
            zero_division=0,
        )
        print("Classification Report (valid-only):")
        print(report)

    # Save all metrics to JSON - both overall and valid-only versions
    out_json = os.path.join(args.adapter_dir, "test_results_biomistral_generative.json")
    out_txt = os.path.join(args.adapter_dir, "per_label_report_biomistral_generative.txt")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "coverage": float(coverage),
                "overall_accuracy": float(overall_acc),

                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro),

                "precision_micro": float(precision_micro),
                "recall_micro": float(recall_micro),
                "f1_micro": float(f1_micro),

                "precision_weighted": float(precision_weighted),
                "recall_weighted": float(recall_weighted),
                "f1_weighted": float(f1_weighted),

                "roc_auc_macro_ovr": None if np.isnan(roc_auc_macro_ovr) else float(roc_auc_macro_ovr),
                "roc_auc_weighted_ovr": None if np.isnan(roc_auc_weighted_ovr) else float(roc_auc_weighted_ovr),

                "accuracy_valid_only": float(acc_valid),

                "precision_macro_valid_only": float(precision_macro_valid),
                "recall_macro_valid_only": float(recall_macro_valid),
                "f1_macro_valid_only": float(f1_macro_valid),

                "precision_micro_valid_only": float(precision_micro_valid),
                "recall_micro_valid_only": float(recall_micro_valid),
                "f1_micro_valid_only": float(f1_micro_valid),

                "precision_weighted_valid_only": float(precision_weighted_valid),
                "recall_weighted_valid_only": float(recall_weighted_valid),
                "f1_weighted_valid_only": float(f1_weighted_valid),

                "roc_auc_macro_ovr_valid_only": None if np.isnan(roc_auc_macro_valid) else float(roc_auc_macro_valid),
                "roc_auc_weighted_ovr_valid_only": None if np.isnan(roc_auc_weighted_valid) else float(roc_auc_weighted_valid),

                "n_total": int(n_total),
                "n_valid": int(n_valid),
                "constrain": bool(args.constrain),
                "text_head_frac": float(args.text_head_frac),
                "max_length": int(args.max_length),
                "max_new_tokens": int(args.max_new_tokens),
                "num_beams": int(max(1, int(args.num_beams))),
            },
            f,
            indent=2,
        )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved metrics to {out_json}")
    print(f"Saved per-label report to {out_txt}")


if __name__ == "__main__":
    main()