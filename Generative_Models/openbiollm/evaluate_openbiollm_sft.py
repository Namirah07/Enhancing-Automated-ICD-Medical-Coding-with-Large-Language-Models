# evaluate_openbiollm_sft.py
#
# =============================================================================
# SOURCE ATTRIBUTION - FILE LEVEL
# =============================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
#
# This script is entirely original work mirroring evaluate_meditron_sft.py
# for the OpenBioLLM-8B backbone, with two key differences:
#   1. STRICT MATCHING by default (normalized string vs vocabulary)
#      rather than always trying regex extraction first
#   2. --allow_regex_fallback flag as an optional second attempt
#
# Library component attributions:
#   PeftModel (PEFT library): https://github.com/huggingface/peft
#   BitsAndBytesConfig (QLoRA paper, Dettmers et al., NeurIPS 2023):
#     URL: https://arxiv.org/abs/2305.14314
#   sklearn metrics: https://scikit-learn.org
#
# WHAT IS ENTIRELY ORIGINAL (written by Namirah Imtieaz Shaik):
#   - load_label_vocab(), normalize_icd(): same as other eval scripts
#   - extract_icd_from_text_regex(): regex fallback (off by default for OpenBioLLM)
#   - build_hard_score_matrix(), safe_multiclass_roc_auc(): same as Meditron eval
#   - extract_fields(), safe_label(), crop_head_tail_tokens(): same as other scripts
#   - build_prompt_like_training(): exact training format reproduction
#   - Trie, build_icd_trie(): constrained decoding trie (same as Meditron eval)
#   - main(): inference loop with strict match then optional regex fallback
#   - prefix_allowed_tokens_fn (nested): beam search constraint callback
#   - Coverage metric, valid-only diagnostic metrics section
#   - "strict_match": True field in output JSON to document the evaluation mode
# =============================================================================
#
# Evaluation script for the OpenBioLLM generative ICD-10 model (OpenBioLLM-G).
#
# This script mirrors evaluate_meditron_sft.py and evaluate_biomistral_sft.py
# but has two differences specific to OpenBioLLM:
#
#   1. STRICT MATCHING by default - the generated text is normalized and compared
#      directly against the label vocabulary. If the normalized string is in the
#      vocabulary it is a valid prediction. If not, the example is recorded as
#      invalid (y_pred=-1). This is stricter than the Meditron eval script which
#      always tries regex extraction first.
#
#   2. OPTIONAL REGEX FALLBACK via --allow_regex_fallback - if the strict normalized
#      match fails, we apply the ICD regex as a second attempt. This flag is off by
#      default because OpenBioLLM with constrained decoding almost always produces
#      a valid code directly, making regex extraction unnecessary.
#
# Like all generative eval scripts, three things must match training exactly:
#   1. The same prompt format ([SYSTEM]..[/SYSTEM][USER]..[/USER][ICD]...)
#   2. The same head+tail token-space cropping (text_head_frac, max_length)
#   3. The same comp_reserve budget used in build_text()
#
# The model is loaded in 4-bit NF4 QLoRA for eval just as it was trained.
# Unlike Meditron eval which also uses 4-bit, here we do not use
# prepare_model_for_kbit_training() because we are only doing inference -
# we do not need gradient flow setup, just the quantized weights loaded cleanly.

import os
import re
import json
import argparse
from typing import Dict, Any, List, Tuple, Set

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    roc_auc_score,
)

# Regex pattern for ICD-10 code detection.
# ICD-10 codes start with A-T or V-Z (U is reserved), followed by two digits,
# then an optional dot and up to 4 more alphanumeric characters.
ICD_REGEX = re.compile(r"[A-TV-Z][0-9][0-9A-Z]\.?[0-9A-Z]{0,4}")


# =========================================================================
# SOURCE ATTRIBUTION - load_label_vocab
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Alphabetical sort for consistent index ordering is original design.
# =========================================================================
def load_label_vocab(path: str):
    """
    Read label_vocab.txt and build integer/code mappings.

    Alphabetical sort ensures class index consistency with all other scripts.
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


# =========================================================================
# SOURCE ATTRIBUTION - normalize_icd
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# ICD dot/space/hyphen stripping to handle format differences is original.
# =========================================================================
def normalize_icd(code: str) -> str:
    """
    Strip dots, hyphens, and spaces from an ICD code string and uppercase it.

    OpenBioLLM may generate "I25.10" when our vocabulary contains "I2510",
    or generate with lowercase letters. Normalizing both the generated code
    and the vocabulary entries before comparing prevents false negatives from
    minor formatting differences.
    """
    if code is None:
        return ""
    code = code.strip().upper()
    # Keep only alphanumeric characters
    code = "".join(ch for ch in code if ch.isalnum())
    return code


# =========================================================================
# SOURCE ATTRIBUTION - extract_icd_from_text_regex
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Regex fallback used only when --allow_regex_fallback is passed and strict
# match fails. Off by default for OpenBioLLM because constrained decoding
# makes invalid outputs near-impossible. Entirely original design choice.
# =========================================================================
def extract_icd_from_text_regex(text: str) -> str:
    """
    Search generated text for the first ICD-10 like substring and return it normalized.

    This is the regex fallback used only when --allow_regex_fallback is passed and
    the strict match against the vocabulary failed. The regex finds patterns like
    "I2510" or "I25.10" anywhere in the generated string.

    Returns an empty string if no ICD-like token is found, which the caller then
    treats as an invalid prediction.
    """
    if not text:
        return ""
    m = ICD_REGEX.search(text.upper())
    if not m:
        return ""
    return normalize_icd(m.group(0))


# =========================================================================
# SOURCE ATTRIBUTION - build_hard_score_matrix
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# One-hot score matrix for ROC-AUC from hard generative predictions is
# entirely original. Same as Meditron eval version.
# =========================================================================
def build_hard_score_matrix(y_pred_np: np.ndarray, num_labels: int) -> np.ndarray:
    """
    Convert hard integer predictions to a score matrix for ROC-AUC computation.

    For each example we place a 1.0 at the predicted class index and 0.0 elsewhere.
    Invalid predictions (y_pred=-1) get an all-zero row. This is less informative
    than using real softmax probabilities (which the discriminative models provide)
    but it is the only option for generation-based models that output a single string.
    """
    scores = np.zeros((len(y_pred_np), num_labels), dtype=np.float32)
    valid_mask = (y_pred_np >= 0) & (y_pred_np < num_labels)
    valid_idx = np.where(valid_mask)[0]
    scores[valid_idx, y_pred_np[valid_idx]] = 1.0
    return scores


# =========================================================================
# SOURCE ATTRIBUTION - safe_multiclass_roc_auc
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# NaN-safe wrapper around sklearn roc_auc_score. Same as Meditron eval.
# sklearn: https://scikit-learn.org
# =========================================================================
def safe_multiclass_roc_auc(y_true_np: np.ndarray, y_score: np.ndarray, average: str) -> float:
    """
    Compute multiclass ROC-AUC with one-vs-rest scheme, returning NaN on failure.

    The most common failure is fewer than two unique true classes in the evaluation
    set, which happens when testing on a tiny debug subset. We return NaN rather than
    crashing so the other metrics are still reported normally.
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


# =========================================================================
# SOURCE ATTRIBUTION - extract_fields (evaluate_openbiollm_sft.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Identical to training script version to ensure consistent gold parsing.
# =========================================================================
def extract_fields(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Parse a chat JSONL example and extract system, user, and assistant ICD fields.

    Identical to extract_fields() in the training script. Keeping the implementation
    the same in both files ensures the gold ICD code is parsed consistently.
    """
    system, user, icd = "", "", ""
    msgs = ex.get("messages", [])

    # Handle double-encoded JSON
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
    """Return the ICD code if non-empty, otherwise return Z5111 as a fallback."""
    icd = (icd or "").strip()
    return icd if icd else "Z5111"


# =========================================================================
# SOURCE ATTRIBUTION - crop_head_tail_tokens (evaluate_openbiollm_sft.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Identical to training script version - must match exactly so the note
# is cropped to the same tokens at training and evaluation time.
# =========================================================================
def crop_head_tail_tokens(token_ids: List[int], budget: int, head_frac: float) -> List[int]:
    """
    Trim a token sequence to fit within budget using a head+tail slice.

    Identical to the training version. Using the exact same function in both
    scripts ensures the note is cropped to the same tokens at training and eval time.
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
# SOURCE ATTRIBUTION - build_prompt_like_training (evaluate_openbiollm_sft.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Reproducing the exact training format (same Z5111 dummy, same budget,
# same head+tail crop) at eval time to prevent train/eval mismatch is an
# original design discipline of this thesis. The hardcoded comp_ids[:16]
# matching the training --comp_reserve default is an original alignment choice.
# =========================================================================
def build_prompt_like_training(
    tokenizer,
    ex: Dict[str, Any],
    max_length: int,
    text_head_frac: float,
) -> Tuple[str, str]:
    """
    Build the inference prompt using the same format and token budget as training.

    This function produces the prompt WITHOUT the ICD code at the end - that is
    what the model generates. We return (prompt_text, gold_icd) so the caller can
    compare the generated output against the ground truth.

    The token budget calculation and dummy completion token measurement must exactly
    match build_text() in the training script. We hardcode comp_ids[:16] here for
    the same 16-token reservation used during training. If training was run with
    a different --comp_reserve value this should be updated to match.

    Returns:
        prompt_text - the formatted prompt string ending with "ICD-10 code: "
        gold        - the gold ICD code string from the assistant message
    """
    system, user, gold = extract_fields(ex)

    prefix = "[SYSTEM]\n" + system + "\n[/SYSTEM]\n[USER]\n"
    suffix = "\n[/USER]\n[ICD]\nICD-10 code: "

    # Measure fixed parts to compute how many tokens remain for the note
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]

    # Same dummy completion and reservation as training
    dummy_completion = "Z5111" + tokenizer.eos_token
    comp_ids = tokenizer(dummy_completion, add_special_tokens=False)["input_ids"][:16]

    user_budget = max_length - (len(prefix_ids) + len(suffix_ids) + len(comp_ids))
    user_budget = max(0, user_budget)

    # Crop the note using the same head+tail strategy as training
    user_ids = tokenizer(user, add_special_tokens=False)["input_ids"]
    user_ids = crop_head_tail_tokens(user_ids, user_budget, head_frac=text_head_frac)

    cropped_user = tokenizer.decode(user_ids, skip_special_tokens=True)
    # The prompt ends with the suffix - the model generates what comes next
    prompt_text = prefix + cropped_user + suffix
    return prompt_text, gold


# =========================================================================
# SOURCE ATTRIBUTION - Trie, build_icd_trie
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Same prefix trie for constrained decoding as in Meditron eval.
# Dot-stripped variant insertion is original. Entirely original work.
# =========================================================================
class Trie:
    """
    Prefix trie for constrained token generation.

    Each node maps token IDs to child nodes and marks whether it represents a
    complete valid ICD code. During generation, at each decoding step we look up
    which token IDs are valid continuations given what has been generated so far.
    Only those token IDs are allowed, making it impossible to generate an
    out-of-vocabulary code.
    """

    def __init__(self):
        self.next = {}    # token_id -> child Trie node
        self.end = False  # True if this node completes a valid ICD code

    def insert(self, token_ids: List[int]):
        """Insert a token ID sequence representing one valid ICD code."""
        node = self
        for t in token_ids:
            if t not in node.next:
                node.next[t] = Trie()
            node = node.next[t]
        node.end = True

    def allowed_next(self, prefix_ids: List[int]) -> Tuple[Set[int], bool]:
        """
        Return the set of token IDs that can follow prefix_ids, and whether
        the prefix already completes a valid ICD code.

        If prefix_ids does not match any trie path (model went off-vocab),
        we return an empty set. The generation callback then returns just
        [EOS] to end generation cleanly.
        """
        node = self
        for t in prefix_ids:
            if t not in node.next:
                return set(), False
            node = node.next[t]
        return set(node.next.keys()), node.end


# =========================================================================
# SOURCE ATTRIBUTION - build_icd_trie (evaluate_openbiollm_sft.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Both raw and dot-stripped variants inserted. Identical to Meditron eval.
# =========================================================================
def build_icd_trie(tokenizer, labels: List[str]) -> Trie:
    """
    Build a trie from all valid ICD codes in label_vocab.txt.

    We insert both the raw code and the dot-stripped version because tokenizers
    split ICD codes differently - I25.10 and I2510 may produce different token ID
    sequences. Inserting both variants ensures the trie covers all reasonable
    tokenizations of every valid code.
    """
    trie = Trie()
    for lab in labels:
        variants = {lab.strip(), lab.strip().replace(".", "")}
        for v in variants:
            ids = tokenizer(v, add_special_tokens=False)["input_ids"]
            if len(ids) > 0:
                trie.insert(ids)
    return trie


# =========================================================================
# SOURCE ATTRIBUTION - main (evaluate_openbiollm_sft.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Full inference loop with strict match -> optional regex fallback,
# coverage metric, prefix_allowed_tokens_fn callback, dual overall/valid-only
# metric reporting, and "strict_match": True in output JSON are all
# entirely original. BitsAndBytesConfig usage follows QLoRA paper
# (Dettmers et al., NeurIPS 2023). URL: https://arxiv.org/abs/2305.14314
# PeftModel usage follows PEFT docs. URL: https://github.com/huggingface/peft
# =========================================================================
def main():
    """
    Entry point. Loads the trained OpenBioLLM-G model and evaluates it on the test set.

    OpenBioLLM-G is loaded in 4-bit NF4 quantization for eval, same as training.
    We do not call prepare_model_for_kbit_training() here because we are only doing
    inference - gradient flow setup is not needed.

    Prediction strategy:
      1. Build prompt using build_prompt_like_training()
      2. Generate up to max_new_tokens tokens
      3. Decode only the newly generated tokens (after input_len)
      4. Normalize the generated text and attempt strict vocabulary match
      5. Optionally try regex extraction as a fallback (--allow_regex_fallback)
      6. If nothing matches, record y_pred=-1 (invalid)

    The strict match approach differs from Meditron eval which always runs regex
    extraction first. With constrained decoding the model produces an exact vocabulary
    code so strict match always succeeds; without constraints the regex fallback
    catches cases where the model adds extra text around the code.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_name", type=str, default="aaditya/Llama3-OpenBioLLM-8B")
    # adapter_dir is required - the eval script must know where the LoRA weights are
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--test_path", type=str, default="../shared/test_chat.jsonl")
    parser.add_argument("--label_vocab", type=str, default="../shared/label_vocab.txt")

    parser.add_argument("--max_length", type=int, default=512)
    # max_new_tokens = 8 is enough to cover any ICD code in our vocabulary
    parser.add_argument("--max_new_tokens", type=int, default=8)

    # CRITICAL: text_head_frac must match the value used during training
    parser.add_argument("--text_head_frac", type=float, default=0.40)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # QLoRA eval settings - must match training quantization config
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_double_quant", action="store_true", default=True)

    # --constrain enables trie-based constrained decoding
    parser.add_argument("--constrain", action="store_true",
                        help="restrict outputs to label_vocab via trie")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--debug_every", type=int, default=200)

    # --allow_regex_fallback tries regex extraction when strict vocab match fails.
    # Off by default - not needed when using constrained decoding.
    parser.add_argument(
        "--allow_regex_fallback",
        action="store_true",
        help="If strict match fails, try regex extraction as a fallback.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    labels, id2label, label2id = load_label_vocab(args.label_vocab)
    num_labels = len(labels)
    print(f"Loaded {num_labels} labels from {args.label_vocab}")

    # Load tokenizer from adapter directory if saved there, otherwise from base model
    tok_source = args.adapter_dir if os.path.isdir(args.adapter_dir) else args.base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA config - same settings as training
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        quantization_config=bnb_config if args.load_in_4bit else None,
        # device_map="auto" is required for 4-bit bitsandbytes models
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=(torch.bfloat16 if args.bf16 else torch.float16),
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    # use_cache=True enables the KV cache during generation for faster decoding
    base_model.config.use_cache = True

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
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
    print("Running inference...\n")

    y_true, y_pred = [], []

    for idx, ex in enumerate(test_ds):
        # Build prompt using same format and budget as training
        prompt_text, gold_icd = build_prompt_like_training(
            tokenizer=tokenizer,
            ex=ex,
            max_length=args.max_length,
            text_head_frac=args.text_head_frac,
        )

        gold_norm = normalize_icd(safe_label(gold_icd))
        # Skip examples whose gold label falls outside our 30-code vocabulary
        if gold_norm not in label2id:
            continue

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )

        if device.type == "cuda":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Record prompt length so we can slice out only the generated tokens later
        input_len = inputs["input_ids"].shape[1]

        def prefix_allowed_tokens_fn(batch_id, sent):
            """
            Constrained decoding callback that restricts generation to valid ICD prefixes.

            sent contains all tokens including the prompt. We slice off the first
            input_len tokens to get only what the model has generated so far, then
            look up which token IDs the trie allows to follow.

            If the trie says the prefix already completes a valid code (is_end=True),
            we add EOS so the model can terminate cleanly. If the trie returns an
            empty set (model went off-vocab), we return just [EOS] to stop generation.
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
            max_new_tokens=args.max_new_tokens,
            do_sample=False,         # deterministic greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=max(1, int(args.num_beams)),
        )

        if args.constrain:
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
            gen_kwargs["early_stopping"] = True

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens after the prompt
        gen_only_ids = output_ids[0, input_len:]
        gen_only = tokenizer.decode(gen_only_ids, skip_special_tokens=True).strip()

        # STRICT MATCH: normalize the generated text and check directly against vocabulary
        cand = normalize_icd(gen_only)
        if cand in label2id:
            pred_norm = cand
        else:
            # Strict match failed - optionally try regex as a fallback
            pred_norm = ""
            if args.allow_regex_fallback:
                pred_norm = extract_icd_from_text_regex(gen_only)
                # If regex found something but it is not in our vocabulary, still invalid
                if pred_norm not in label2id:
                    pred_norm = ""

        y_true.append(label2id[gold_norm])
        # -1 is the sentinel for invalid/out-of-vocab predictions
        y_pred.append(label2id[pred_norm] if pred_norm in label2id else -1)

        if args.debug_every > 0 and (idx + 1) % args.debug_every == 0:
            full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Example {idx+1}:")
            print(f"GOLD: {gold_norm}")
            print(f"GEN ONLY: {repr(gen_only)}")
            print(f"PRED: {pred_norm if pred_norm else '<EMPTY/INVALID>'}")
            print("FULL(first 200):", repr(full[:200]))
            print()

    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    n_total = len(y_true)
    if n_total == 0:
        # This should not happen in normal usage - check that label_vocab and test set match
        print("No examples evaluated (checking label vocab / test set).")
        return

    valid_mask = y_pred != -1
    n_valid = int(valid_mask.sum())
    coverage = float(n_valid / n_total)

    # Overall accuracy - invalid predictions (-1) never match any gold label so they
    # count as wrong automatically without needing special handling
    overall_acc = float((y_pred == y_true).mean())

    # Overall metrics including all examples (valid and invalid)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_labels)), average="macro", zero_division=0,
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_labels)), average="micro", zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_labels)), average="weighted", zero_division=0,
    )

    # Build one-hot score matrix for ROC-AUC from hard predictions
    y_score = build_hard_score_matrix(y_pred, num_labels)
    roc_auc_macro_ovr = safe_multiclass_roc_auc(y_true, y_score, average="macro")
    roc_auc_weighted_ovr = safe_multiclass_roc_auc(y_true, y_score, average="weighted")

    print(f"Coverage (valid preds in vocab): {n_valid} / {n_total} = {coverage:.3f}")
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

    # Valid-only metrics are diagnostic - they separate coverage failures from
    # prediction errors. If coverage is high but overall accuracy is low, the model
    # is generating valid codes but picking the wrong ones. If coverage is low, the
    # model is generating unrecognizable strings which strict matching cannot handle.
    if n_valid == 0:
        print("No valid predictions; cannot compute valid-only metrics.")
        acc = None
        precision_macro_valid = recall_macro_valid = f1_macro_valid = None
        precision_micro_valid = recall_micro_valid = f1_micro_valid = None
        precision_weighted_valid = recall_weighted_valid = f1_weighted_valid = None
        roc_auc_macro_valid = roc_auc_weighted_valid = None
        report = ""
    else:
        vt = y_true[valid_mask]
        vp = y_pred[valid_mask]

        acc = accuracy_score(vt, vp)

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
        print(f"Accuracy(valid):                  {acc:.6f}")
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

    # Save all metrics to JSON and the per-label report to text
    out_json = os.path.join(args.adapter_dir, "test_results_openbiollm_generative.json")
    out_txt = os.path.join(args.adapter_dir, "per_label_report_openbiollm_generative.txt")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "coverage": coverage,
                "overall_accuracy": overall_acc,

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

                "accuracy_valid_only": float(acc) if acc is not None else None,

                "precision_macro_valid_only": float(precision_macro_valid) if precision_macro_valid is not None else None,
                "recall_macro_valid_only": float(recall_macro_valid) if recall_macro_valid is not None else None,
                "f1_macro_valid_only": float(f1_macro_valid) if f1_macro_valid is not None else None,

                "precision_micro_valid_only": float(precision_micro_valid) if precision_micro_valid is not None else None,
                "recall_micro_valid_only": float(recall_micro_valid) if recall_micro_valid is not None else None,
                "f1_micro_valid_only": float(f1_micro_valid) if f1_micro_valid is not None else None,

                "precision_weighted_valid_only": float(precision_weighted_valid) if precision_weighted_valid is not None else None,
                "recall_weighted_valid_only": float(recall_weighted_valid) if recall_weighted_valid is not None else None,
                "f1_weighted_valid_only": float(f1_weighted_valid) if f1_weighted_valid is not None else None,

                "roc_auc_macro_ovr_valid_only": None if roc_auc_macro_valid is None or np.isnan(roc_auc_macro_valid) else float(roc_auc_macro_valid),
                "roc_auc_weighted_ovr_valid_only": None if roc_auc_weighted_valid is None or np.isnan(roc_auc_weighted_valid) else float(roc_auc_weighted_valid),

                "n_total": int(n_total),
                "n_valid": int(n_valid),
                "constrain": bool(args.constrain),
                # strict_match=True documents that we used exact vocabulary matching by default
                "strict_match": True,
                "allow_regex_fallback": bool(args.allow_regex_fallback),
                "text_head_frac": float(args.text_head_frac),
                "max_length": int(args.max_length),
                "max_new_tokens": int(args.max_new_tokens),
                "num_beams": int(args.num_beams),
                "load_in_4bit": bool(args.load_in_4bit),
            },
            f,
            indent=2,
        )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report if report else "")

    print(f"\nSaved metrics to {out_json}")
    print(f"Saved per-label report to {out_txt}")


if __name__ == "__main__":
    main()