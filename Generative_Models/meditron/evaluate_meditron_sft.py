# evaluate_meditron_sft.py
#
# =============================================================================
# SOURCE ATTRIBUTION - FILE LEVEL
# =============================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
#
# This script is entirely original work. The following library components
# are used within it:
#
#   PeftModel (LoRA adapter loading):
#     Adapted from: PEFT library documentation
#     URL: https://github.com/huggingface/peft
#
#   BitsAndBytesConfig (4-bit NF4 quantization):
#     Adapted from: QLoRA paper (Dettmers et al., NeurIPS 2023)
#     URL: https://arxiv.org/abs/2305.14314
#
#   sklearn metrics (precision_recall_fscore_support, roc_auc_score):
#     URL: https://scikit-learn.org
#
# WHAT IS ENTIRELY ORIGINAL (written by Namirah Imtieaz Shaik):
#   - load_label_vocab(), normalize_icd(), extract_icd_from_gen_only()
#   - extract_fields(), safe_label(), crop_head_tail_tokens()
#   - build_prompt_like_training(): matching train format exactly at eval time
#   - Trie class and build_icd_trie(): constrained decoding via prefix trie
#   - build_hard_score_matrix(): one-hot score matrix for ROC-AUC from hard preds
#   - safe_multiclass_roc_auc(): NaN-safe ROC-AUC wrapper
#   - prefix_allowed_tokens_fn (nested): beam search constraint callback
#   - Coverage metric tracking (valid preds / total)
#   - Valid-only diagnostic metrics section
#   - Dual JSON + text file output (test_results + per_label_report)
# =============================================================================
#
# Evaluation script for the Meditron generative ICD-10 model (Meditron-G).
#
# This script MUST match train_meditron_sft_chat.py in three things:
#   1. The same prompt format ([SYSTEM]..[/SYSTEM][USER]..[/USER][ICD]...)
#   2. The same head+tail token-space cropping (text_head_frac, max_length)
#   3. The same dummy completion footprint for the budget calculation
#
# Any mismatch between train and eval formatting means the model sees prompts
# at inference time that look different from the ones it trained on, which
# will hurt performance in ways that are hard to diagnose.
#
# The evaluation strategy for generative models differs from discriminative models:
#   - We generate an ICD code string autoregressively (token by token)
#   - If the generated string matches a code in label_vocab.txt, it is a valid prediction
#   - If the model generates something that is not in the vocabulary (e.g. garbage text),
#     we treat it as WRONG but still include it in the overall metrics
#   - We report COVERAGE separately: the fraction of examples where the model
#     generated a valid in-vocabulary code
#
# Two generation modes are supported:
#   - Greedy (default): fastest, picks the highest-probability token at each step
#   - Constrained decoding via trie (--constrain): forces the model to only generate
#     tokens that continue a valid ICD code prefix, making invalid outputs impossible
#
# ROC-AUC note:
#   Because this is a generation-based model, we do not have a probability distribution
#   over all 30 classes. We only get one hard prediction per example. To compute ROC-AUC
#   we convert the predictions to one-hot vectors (1.0 for the predicted class, 0 for all
#   others). This is mathematically valid but less informative than the discriminative
#   ROC-AUC which uses real softmax probabilities.

import os
import re
import json
import argparse
from typing import Dict, Any, List, Tuple, Set, Optional

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
# ICD-10 codes start with a letter (A-Z except U), followed by 2 digits,
# then an optional dot and up to 4 more alphanumeric characters.
# We use this to extract the code from the raw generated text.
ICD_REGEX = re.compile(r"[A-TV-Z][0-9][0-9A-Z]\.?[0-9A-Z]{0,4}")


# =========================================================================
# SOURCE ATTRIBUTION - load_label_vocab
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Alphabetical sort for consistent index ordering matching discriminative
# scripts is an original design choice.
# =========================================================================
def load_label_vocab(path: str):
    """
    Read label_vocab.txt and build the three standard mappings.

    Alphabetical sort ensures consistent integer-to-code ordering
    that matches what the discriminative training scripts produce.
    """
    labels: List[str] = []
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
# ICD code normalization handling I25.10 vs I2510 format differences is
# entirely original.
# =========================================================================
def normalize_icd(code: Optional[str]) -> str:
    """
    Strip dots, spaces, and hyphens from an ICD code and uppercase it.

    This is necessary because the model might generate "I25.10" when the
    vocabulary contains "I2510", or generate with lowercase letters. Normalizing
    both the generated code and the vocabulary entries before comparing ensures
    we do not fail to match a correct prediction due to formatting differences.
    """
    if not code:
        return ""
    code = code.strip().upper()
    # Remove any non-alphanumeric characters (dots, spaces, hyphens)
    code = "".join(ch for ch in code if ch.isalnum())
    return code


# =========================================================================
# SOURCE ATTRIBUTION - extract_icd_from_gen_only
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Searching only the generated tail (not the prompt) for an ICD-like token
# using ICD_REGEX is entirely original.
# =========================================================================
def extract_icd_from_gen_only(gen_only: str) -> str:
    """
    Extract an ICD code from the model's generated output string.

    We only look in the newly generated tokens (not the prompt), so gen_only
    contains just what the model produced after "ICD-10 code: ". We search
    for the first token-like string that matches the ICD_REGEX pattern.

    Returns an empty string if no ICD-like token is found. The caller then
    decides whether to count this as an invalid prediction.
    """
    if not gen_only:
        return ""
    up = gen_only.strip().upper()
    m = ICD_REGEX.search(up)
    return normalize_icd(m.group(0)) if m else ""


# =========================================================================
# SOURCE ATTRIBUTION - extract_fields
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Same function as in the training script - identical implementation kept
# to ensure gold ICD parsing is consistent between train and eval.
# =========================================================================
def extract_fields(ex: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Parse a chat JSONL example and extract the system, user, and assistant fields.

    Handles the same edge cases as the training version: double-encoded JSON,
    single dict instead of list, and missing role keys. Keeping this identical
    to the training version ensures the prompt construction path is the same.
    """
    system, user, icd = "", "", ""
    msgs = ex.get("messages", [])

    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            msgs = []
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
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik. Z5111 fallback is original.
# =========================================================================
def safe_label(icd: str) -> str:
    """Return the ICD code if non-empty, otherwise return the Z5111 fallback."""
    icd = (icd or "").strip()
    return icd if icd else "Z5111"


# =========================================================================
# SOURCE ATTRIBUTION - crop_head_tail_tokens
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Identical to the training script version - must match exactly so the
# note is cropped to the same tokens at train and eval time.
# =========================================================================
def crop_head_tail_tokens(token_ids: List[int], budget: int, head_frac: float) -> List[int]:
    """
    Keep a head slice and a tail slice of token_ids that together fit within budget.

    This function is identical to the one in train_meditron_sft_chat.py.
    Using the exact same function in both scripts guarantees that the note
    the model was trained on and the prompt it receives at eval time are cropped
    in exactly the same way.
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
# SOURCE ATTRIBUTION - build_prompt_like_training
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Reproducing the exact training format at eval time (same Z5111 dummy,
# same budget formula, same head+tail crop) to avoid train/eval mismatch
# is an original design discipline of this thesis.
# =========================================================================
def build_prompt_like_training(
    tokenizer,
    ex: Dict[str, Any],
    max_length: int,
    text_head_frac: float,
    comp_max_tokens: int = 16,
) -> Tuple[str, str]:
    """
    Build the inference prompt using the exact same format and token budget as training.

    This function produces the prompt WITHOUT the ICD code at the end - the model
    generates that part. We return (prompt_text, gold_icd) so the caller can compare
    the generated code against the ground truth.

    The token budget calculation is identical to build_text() in the training script:
      user_budget = max_length - len(prefix_ids) - len(suffix_ids) - len(comp_ids)

    Using Z5111 as the dummy completion is important: if we used a shorter or longer
    dummy the budget would be slightly different, causing the note to be cropped at a
    different point than during training.
    """
    system, user, gold = extract_fields(ex)

    prefix = "[SYSTEM]\n" + system + "\n[/SYSTEM]\n[USER]\n"
    suffix = "\n[/USER]\n[ICD]\nICD-10 code: "

    # Measure the fixed parts to compute how many tokens are available for the note
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]

    # Use the same dummy completion as training to get an identical budget
    dummy_completion = "Z5111" + (tokenizer.eos_token or "")
    comp_ids = tokenizer(dummy_completion, add_special_tokens=False)["input_ids"][:comp_max_tokens]

    user_budget = max_length - (len(prefix_ids) + len(suffix_ids) + len(comp_ids))
    user_budget = max(0, user_budget)

    # Crop the note exactly as training did
    user_ids = tokenizer(user, add_special_tokens=False)["input_ids"]
    user_ids = crop_head_tail_tokens(user_ids, user_budget, head_frac=text_head_frac)
    cropped_user = tokenizer.decode(user_ids, skip_special_tokens=True)

    # The prompt ends at "ICD-10 code: " - the model generates what comes next
    prompt_text = prefix + cropped_user + suffix
    return prompt_text, gold


# =========================================================================
# SOURCE ATTRIBUTION - Trie, build_icd_trie
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# The prefix trie for constrained ICD code generation is entirely original.
# The idea of using a trie for constrained decoding in generation models is
# a known technique in NLP but this specific implementation for ICD-10
# codes, including the dot-stripped variant insertion and the
# prefix_allowed_tokens_fn callback pattern, is original work.
# =========================================================================
class Trie:
    """
    Prefix trie for constrained token generation.

    Each node stores pointers to its children (keyed by token ID) and a flag
    marking whether this node represents a complete valid ICD code.

    During generation, at each step we look up which token IDs continue a valid
    prefix given the tokens generated so far. Only those token IDs are allowed
    by the generation function, making it physically impossible for the model to
    generate an invalid code.
    """

    def __init__(self):
        self.next = {}   # token_id -> Trie node
        self.end = False # True if this node completes a valid ICD code

    def insert(self, token_ids: List[int]):
        """Insert a sequence of token IDs representing one ICD code into the trie."""
        node = self
        for t in token_ids:
            if t not in node.next:
                node.next[t] = Trie()
            node = node.next[t]
        node.end = True

    def allowed_next(self, prefix_ids: List[int]) -> Tuple[Set[int], bool]:
        """
        Given the tokens generated so far (prefix_ids), return the set of token IDs
        that can legally follow, and whether the prefix already completes a valid code.

        If the prefix does not match any path in the trie (the model generated
        something invalid), we return an empty set. The generation function then
        falls back to the EOS token to end generation gracefully.
        """
        node = self
        for t in prefix_ids:
            if t not in node.next:
                return set(), False
            node = node.next[t]
        return set(node.next.keys()), node.end


# =========================================================================
# SOURCE ATTRIBUTION - build_icd_trie
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Inserting both raw and dot-stripped code variants to handle tokenizer
# differences is an original design choice.
# =========================================================================
def build_icd_trie(tokenizer, labels: List[str]) -> Trie:
    """
    Build a trie from all valid ICD codes in the label vocabulary.

    We insert both the raw code and the code with dots removed because
    different tokenizers split ICD codes differently. For example,
    "I2510" might tokenize as ["I", "25", "10"] while "I25.10" might
    tokenize as ["I", "25", ".", "10"]. Inserting both variants ensures
    the trie covers all possible tokenizations.
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
# SOURCE ATTRIBUTION - build_hard_score_matrix
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Converting hard predictions to one-hot score matrices for ROC-AUC
# computation from generation-based models is an original design choice.
# =========================================================================
def build_hard_score_matrix(y_pred_np: np.ndarray, num_labels: int) -> np.ndarray:
    """
    Convert hard integer predictions into a score matrix suitable for ROC-AUC.

    For each example, we create a one-hot vector: 1.0 at the predicted class index,
    0.0 everywhere else. Invalid predictions (y_pred == -1) get an all-zero row,
    which means they contribute nothing to the per-class AUC but do affect the
    aggregated score since they represent missed predictions.

    This is less informative than using real softmax probabilities because it has
    no notion of how confident the model was - but it is the only option when the
    model generates a single string prediction rather than a score for every class.
    """
    scores = np.zeros((len(y_pred_np), num_labels), dtype=np.float32)
    valid_mask = (y_pred_np >= 0) & (y_pred_np < num_labels)
    valid_indices = np.where(valid_mask)[0]
    scores[valid_indices, y_pred_np[valid_indices]] = 1.0
    return scores


# =========================================================================
# SOURCE ATTRIBUTION - safe_multiclass_roc_auc
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# NaN-safe wrapper around sklearn roc_auc_score for edge cases is original.
# sklearn roc_auc_score is from: https://scikit-learn.org
# =========================================================================
def safe_multiclass_roc_auc(y_true_np: np.ndarray, y_score: np.ndarray, average: str) -> float:
    """
    Compute multiclass ROC-AUC with one-vs-rest scheme, returning NaN on failure.

    The most common failure case is when the test split happens to contain fewer
    than two unique classes (e.g. during debugging with a tiny dataset). We return
    NaN rather than crashing so the rest of the metrics still get reported.
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
# SOURCE ATTRIBUTION - main (evaluate_meditron_sft.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# The full inference loop, coverage metric, dual overall/valid-only metric
# reporting, and prefix_allowed_tokens_fn callback are entirely original.
# PeftModel.from_pretrained and BitsAndBytesConfig usage follow PEFT and
# bitsandbytes library documentation patterns.
# PEFT URL: https://github.com/huggingface/peft
# bitsandbytes URL: https://github.com/TimDettmers/bitsandbytes
# =========================================================================
def main():
    """
    Entry point. Loads the trained Meditron-G model and evaluates it on the test set.

    The main inference loop:
      1. Build the prompt using build_prompt_like_training()
      2. Tokenize the prompt and pass it to model.generate()
      3. Decode only the newly generated tokens (after the prompt)
      4. Extract the ICD code using the regex
      5. Compare against the gold label and record the prediction

    We load the model in 4-bit NF4 quantization by default to fit the 7B model
    into a single GPU at inference time. This is fine for eval - quantization
    only affects training convergence meaningfully, not inference quality at this
    level of precision.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_name", type=str, default="epfl-llm/meditron-7b")
    parser.add_argument("--adapter_dir", type=str, default="./output/meditron_sft_chat_sfttrainer")
    parser.add_argument("--test_path", type=str, default="../shared/test_chat.jsonl")
    parser.add_argument("--label_vocab", type=str, default="../shared/label_vocab.txt")

    parser.add_argument("--max_length", type=int, default=512)
    # max_new_tokens limits how many new tokens the model can generate.
    # 8 is enough to cover any ICD code in our vocabulary.
    parser.add_argument("--max_new_tokens", type=int, default=8)
    # comp_max_tokens must match the training script's dummy completion footprint
    parser.add_argument("--comp_max_tokens", type=int, default=16)

    # CRITICAL: text_head_frac must match the value used during training.
    # Using a different value means the model sees different prompt content at eval.
    parser.add_argument("--text_head_frac", type=float, default=0.40)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # 4-bit quantization is the default for eval - fits the 7B model on one GPU
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_double_quant", action="store_true", default=True)

    # --constrain enables trie-based constrained decoding.
    # With this flag the model physically cannot generate an invalid ICD code.
    parser.add_argument("--constrain", action="store_true",
                        help="restrict outputs to label_vocab via trie")
    # num_beams only matters when --constrain is active (beam search + constraints)
    parser.add_argument("--num_beams", type=int, default=8)

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

    # 4-bit quantization config for efficient inference.
    # NF4 (NormalFloat4) preserves the distribution of weights better than
    # standard int4 quantization, which matters for generation quality.
    # double_quant quantizes the quantization constants themselves, saving
    # another ~0.4 bits per parameter.
    compute_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float16)
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
        # device_map="auto" is safe here - we are not training, just generating
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=(torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)),
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    # use_cache=True enables the KV cache during generation, which is the
    # standard behavior and speeds up autoregressive decoding significantly
    base_model.config.use_cache = True

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    # Build the trie if constrained decoding was requested
    trie = None
    if args.constrain:
        trie = build_icd_trie(tokenizer, labels)
        print("Constraint mode ON: generation restricted to label_vocab.")

    print("Loading test dataset...")
    raw = load_dataset("json", data_files={"test": args.test_path})
    test_ds = raw["test"]
    print(f"Test examples: {len(test_ds)}")
    print("Running inference...\n")

    y_true: List[int] = []
    y_pred: List[int] = []

    for idx, ex in enumerate(test_ds):
        # Build the prompt and get the gold ICD code
        prompt_text, gold_icd = build_prompt_like_training(
            tokenizer=tokenizer,
            ex=ex,
            max_length=args.max_length,
            text_head_frac=args.text_head_frac,
            comp_max_tokens=args.comp_max_tokens,
        )

        # Normalize the gold code the same way we will normalize the prediction
        gold_norm = normalize_icd(safe_label(gold_icd))
        if gold_norm not in label2id:
            # Skip examples whose gold label is not in our 30-code vocabulary
            continue

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )

        if device.type == "cuda":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Record the prompt length so we can slice out only the generated tokens later
        input_len = inputs["input_ids"].shape[1]

        def prefix_allowed_tokens_fn(batch_id, sent):
            """
            Callback that tells beam search which token IDs are allowed at each step.

            sent contains all tokens generated so far (prompt + generated). We slice
            off the prompt (input_len tokens) to get only the generated prefix, then
            look up what token IDs the trie allows to follow that prefix.

            If the trie says the current prefix is already a complete code (is_end=True),
            we add EOS to the allowed set so the model can terminate cleanly.

            If the trie returns an empty set (the generated prefix does not match any
            valid code prefix), we return just [EOS] to end generation immediately.
            """
            sent_list = sent.tolist() if hasattr(sent, "tolist") else list(sent)
            gen_prefix = sent_list[input_len:]
            allowed, is_end = trie.allowed_next(gen_prefix)

            if is_end:
                allowed = set(allowed)
                allowed.add(tokenizer.eos_token_id)

            if not allowed:
                return [tokenizer.eos_token_id]
            return list(allowed)

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,          # greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if args.constrain:
            # Beam search with constraints: explores multiple candidate sequences
            # while the trie forces all of them to stay on valid ICD code paths
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
            gen_kwargs["num_beams"] = args.num_beams
            gen_kwargs["early_stopping"] = True
        else:
            gen_kwargs["num_beams"] = 1  # num_beams=1 means greedy decoding

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # Slice out only the tokens the model generated after the prompt
        gen_only_ids = output_ids[0, input_len:]
        gen_only = tokenizer.decode(gen_only_ids, skip_special_tokens=True).strip()

        # Try to extract a valid ICD code from the generated text
        pred_norm = extract_icd_from_gen_only(gen_only)

        # If regex extraction failed and we are in constrained mode,
        # the generated text itself might be a valid code (the trie guaranteed it)
        if pred_norm == "" and args.constrain:
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

    # Convert lists to numpy arrays for metric computation
    y_true_np = np.array(y_true, dtype=np.int64)
    y_pred_np = np.array(y_pred, dtype=np.int64)

    n_total = len(y_true_np)
    valid_mask = (y_pred_np != -1)
    n_valid = int(valid_mask.sum())
    coverage = n_valid / max(1, n_total)

    # Overall accuracy: invalid predictions (-1) never equal any gold label so
    # they automatically count as wrong without needing special handling
    overall_acc = float((y_true_np == y_pred_np).mean())

    # Overall metrics including invalid predictions
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=list(range(num_labels)),
        average="macro",
        zero_division=0,
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=list(range(num_labels)),
        average="micro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=list(range(num_labels)),
        average="weighted",
        zero_division=0,
    )

    # Build one-hot score matrix from hard predictions for ROC-AUC
    y_score = build_hard_score_matrix(y_pred_np, num_labels)
    roc_auc_macro_ovr = safe_multiclass_roc_auc(y_true_np, y_score, average="macro")
    roc_auc_weighted_ovr = safe_multiclass_roc_auc(y_true_np, y_score, average="weighted")

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

    # Valid-only metrics are useful for diagnosing coverage failures.
    # If coverage is low but valid-only accuracy is high, the model knows
    # the right answer but is producing malformed code strings.
    if n_valid > 0:
        vt = y_true_np[valid_mask]
        vp = y_pred_np[valid_mask]

        acc_valid = accuracy_score(vt, vp)

        prec_macro_valid, rec_macro_valid, f1_macro_valid, _ = precision_recall_fscore_support(
            vt, vp, average="macro", zero_division=0
        )
        prec_micro_valid, rec_micro_valid, f1_micro_valid, _ = precision_recall_fscore_support(
            vt, vp, average="micro", zero_division=0
        )
        prec_weighted_valid, rec_weighted_valid, f1_weighted_valid, _ = precision_recall_fscore_support(
            vt, vp, average="weighted", zero_division=0
        )

        y_score_valid = build_hard_score_matrix(vp, num_labels)
        roc_auc_macro_valid = safe_multiclass_roc_auc(vt, y_score_valid, average="macro")
        roc_auc_weighted_valid = safe_multiclass_roc_auc(vt, y_score_valid, average="weighted")

        print("Metrics on VALID predictions only (diagnostic):")
        print(f"Accuracy(valid):                  {acc_valid:.6f}")
        print(f"Precision Macro(valid):           {prec_macro_valid:.6f}")
        print(f"Recall Macro(valid):              {rec_macro_valid:.6f}")
        print(f"F1 Macro(valid):                  {f1_macro_valid:.6f}")
        print(f"Precision Micro(valid):           {prec_micro_valid:.6f}")
        print(f"Recall Micro(valid):              {rec_micro_valid:.6f}")
        print(f"F1 Micro(valid):                  {f1_micro_valid:.6f}")
        print(f"Precision Weighted(valid):        {prec_weighted_valid:.6f}")
        print(f"Recall Weighted(valid):           {rec_weighted_valid:.6f}")
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
    else:
        # No valid predictions at all - set all diagnostic metrics to zero/nan
        acc_valid = 0.0
        prec_macro_valid = rec_macro_valid = f1_macro_valid = 0.0
        prec_micro_valid = rec_micro_valid = f1_micro_valid = 0.0
        prec_weighted_valid = rec_weighted_valid = f1_weighted_valid = 0.0
        roc_auc_macro_valid = float("nan")
        roc_auc_weighted_valid = float("nan")
        report = "No valid predictions; per-label report unavailable."
        print(report)

    # Save all metrics to JSON - save both overall and valid-only versions
    out_json = os.path.join(args.adapter_dir, "test_results_meditron_generative.json")
    out_txt = os.path.join(args.adapter_dir, "per_label_report_meditron_generative.txt")

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

                "precision_macro_valid_only": float(prec_macro_valid),
                "recall_macro_valid_only": float(rec_macro_valid),
                "f1_macro_valid_only": float(f1_macro_valid),

                "precision_micro_valid_only": float(prec_micro_valid),
                "recall_micro_valid_only": float(rec_micro_valid),
                "f1_micro_valid_only": float(f1_micro_valid),

                "precision_weighted_valid_only": float(prec_weighted_valid),
                "recall_weighted_valid_only": float(rec_weighted_valid),
                "f1_weighted_valid_only": float(f1_weighted_valid),

                "roc_auc_macro_ovr_valid_only": None if np.isnan(roc_auc_macro_valid) else float(roc_auc_macro_valid),
                "roc_auc_weighted_ovr_valid_only": None if np.isnan(roc_auc_weighted_valid) else float(roc_auc_weighted_valid),

                "n_total": int(n_total),
                "n_valid": int(n_valid),
                "constrain": bool(args.constrain),
                "text_head_frac": float(args.text_head_frac),
                "max_length": int(args.max_length),
                "max_new_tokens": int(args.max_new_tokens),
                "load_in_4bit": bool(args.load_in_4bit),
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