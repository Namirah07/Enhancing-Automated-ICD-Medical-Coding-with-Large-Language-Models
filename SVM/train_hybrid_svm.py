"""
train_hybrid_svm.py - Hybrid SVM Baseline for ICD-10 Classification

# =============================================================================
# SOURCE ATTRIBUTION - FILE LEVEL
# =============================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
#
# This script is entirely original work. The following library components
# are used within it:
#
#   Bio_ClinicalBERT encoder (feature extractor):
#     Adapted from: Alsentzer et al. (2019) "Publicly Available Clinical BERT
#     Embeddings" (NAACL Clinical NLP Workshop 2019)
#     URL: https://github.com/EmilyAlsentzer/clinicalBERT
#     HuggingFace model: emilyalsentzer/Bio_ClinicalBERT
#     Used as a FROZEN feature extractor only - no fine-tuning.
#
#   LinearSVC:
#     From: scikit-learn library
#     URL: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
#     Reference: Cortes & Vapnik (1995) "Support-Vector Networks"
#
#   CalibratedClassifierCV (Platt scaling):
#     From: scikit-learn library
#     URL: https://scikit-learn.org/stable/modules/calibration.html
#     Reference: Platt (1999) "Probabilistic outputs for support vector machines"
#
#   Mean pooling pattern:
#     Standard pattern from HuggingFace community for sentence embeddings.
#     Reference: https://huggingface.co/blog/how-to-train-sentence-transformers
#
# WHAT IS ENTIRELY ORIGINAL (written by Namirah Imtieaz Shaik):
#   - The "hybrid" design combining frozen Bio_ClinicalBERT + LinearSVC
#   - max_chars=2000 character pre-filter before tokenization
#   - maybe_cache_embeddings() caching strategy for development speed
#   - encode_texts() batched encoding with tqdm progress bar
#   - The choice to use Platt scaling (sigmoid method) for probability calibration
#   - evaluate() computing both macro and weighted ROC-AUC OVR alongside F1
#   - Saving embeddings as .npy files for reuse across SVM hyperparameter runs
# =============================================================================

This script implements the classical machine learning baseline for the thesis
"Enhancing Automated ICD-10 Medical Coding with Large Language Models."

The model is called "hybrid" because it combines two separate components:
  1. A frozen Bio_ClinicalBERT encoder that converts discharge notes into
     fixed 768-dimensional embedding vectors (deep learning component)
  2. A LinearSVC classifier trained on those embeddings (classical ML component)

The encoder is never updated during training - it is used purely as a
feature extractor. Only the SVM parameters are learned.

Why this design?
  The SVM baseline answers: "how well can a linear classifier do if we just
  use pretrained BERT embeddings as features, with no task-specific fine-tuning?"
  This sets a reference point for comparing against the PLM-ICD and LLM models
  which actually update the backbone during training.

Why Bio_ClinicalBERT?
  Bio_ClinicalBERT was pretrained on MIMIC clinical notes, making its
  representations more relevant to our discharge summary task than
  general-purpose BERT.

Probability calibration:
  LinearSVC does not produce probabilities natively - it outputs raw margin
  scores. We need probabilities for ROC-AUC. Wrapping in CalibratedClassifierCV
  with sigmoid method (Platt scaling) learns to convert the SVM margin scores
  into calibrated class probabilities using 3-fold cross validation.

Usage:
  python train_hybrid_svm.py \
    --data_dir ./shared \
    --out_dir ./output/svm_hybrid \
    --use_gpu \
    --cache_embeddings
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import torch

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# =========================================================================
# SOURCE ATTRIBUTION - load_label_vocab
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Reads label_vocab.txt preserving file order (unlike the discriminative
# scripts which sort alphabetically). File-order indexing is a deliberate
# design decision for this script — it does not need to match other scripts
# since the SVM is evaluated standalone.
# =========================================================================
def load_label_vocab(label_vocab_path: str):
    """
    Read label_vocab.txt and build two-way mappings between ICD codes and integers.

    The file has one ICD-10 code per line, and we preserve that file order as
    the class index ordering. This is important — all models in the thesis must
    use the same index for the same ICD code so results are comparable.

    Returns:
        label2id: dict mapping ICD code string → integer index  (e.g. "I2510" → 7)
        id2label: dict mapping integer index → ICD code string  (e.g. 7 → "I2510")
    """
    labels = []
    with open(label_vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            code = line.strip()
            if code:
                labels.append(code)

    id2label = {i: code for i, code in enumerate(labels)}
    label2id = {code: i for i, code in id2label.items()}
    return label2id, id2label


# =========================================================================
# SOURCE ATTRIBUTION - load_split
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# max_chars=2000 character pre-filter design and the coarse approximation
# (2000 chars ≈ 512 tokens for clinical text) are original choices.
# =========================================================================
def load_split(csv_path: str, text_col="TEXT", label_col="LABELS", max_chars: int | None = None):
    """
    Load a train/dev/test CSV file and optionally truncate the note text.

    max_chars controls character-level truncation before tokenization.
    We use 2000 characters because Bio_ClinicalBERT handles 512 tokens, and
    roughly 4-5 characters map to one token, so 2000 chars ≈ 512 tokens.
    This is a fast coarse pre-filter — the tokenizer will do the final
    precise truncation to exactly 512 tokens afterward.

    Note: this means we only see about 19% of the average MIMIC discharge
    note (mean length ~10,550 chars). The PLM-ICD and LLM models handle
    longer inputs via chunking and head-tail cropping respectively.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str).str.strip()

    if max_chars is not None and max_chars > 0:
        # s[:max_chars] keeps only the first max_chars characters.
        # Everything after that position is discarded entirely.
        df[text_col] = df[text_col].apply(lambda s: s[:max_chars])

    return df


# =========================================================================
# SOURCE ATTRIBUTION - encode_labels
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# The upfront unknown-label sanity check that raises immediately rather
# than failing silently during training is original defensive design.
# =========================================================================
def encode_labels(label_series: pd.Series, label2id: dict):
    """
    Convert a pandas Series of ICD code strings into a numpy array of integers.

    Also does a sanity check upfront — if any label in the CSV is not in
    label_vocab.txt, we raise an error immediately rather than silently
    producing wrong results later during training or evaluation.
    """
    unknown = sorted(set(label_series.unique()) - set(label2id.keys()))
    if unknown:
        raise ValueError(
            f"Found {len(unknown)} labels in CSV that are NOT in label_vocab.txt. "
            f"First few: {unknown[:10]}"
        )
    return label_series.map(label2id).astype(int).to_numpy()


# =========================================================================
# SOURCE ATTRIBUTION - mean_pool
# =========================================================================
# ADAPTED FROM: Standard masked mean pooling pattern from HuggingFace
#   community for sentence embedding tasks.
#   Reference: https://huggingface.co/blog/how-to-train-sentence-transformers
# The specific implementation (mask unsqueeze, expand, masked sum, clamp)
# is written by Namirah Imtieaz Shaik following this standard pattern.
# =========================================================================
def mean_pool(last_hidden_state, attention_mask):
    """
    Average the token hidden states, ignoring padding tokens.

    Bio_ClinicalBERT produces one 768-dim vector for every token in the input.
    We need a single vector per document to feed into the SVM. Mean pooling
    computes the average across all real (non-padding) token positions.

    The attention_mask has 1 for real tokens and 0 for padding. Expanding it
    to match the hidden state shape and multiplying zeros out the padding
    positions before summing, so padding does not affect the average.

    clamp(min=1e-9) prevents a division by zero if somehow a sequence has
    no valid tokens — shouldn't happen in practice but safer to guard against.
    """
    # Expand mask from [B, T] to [B, T, H] to match hidden state dimensions
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    # Zero out padding token vectors, then sum across the token dimension
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)          # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)  # [B, 1] — number of real tokens
    return summed / counts              # [B, H] — the masked mean


# =========================================================================
# SOURCE ATTRIBUTION - encode_texts
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# The batched encoding loop, tqdm progress bar, batch_size parameter,
# and the "mean" vs "cls" pooling switch are all original design choices.
# The AutoModel.from_pretrained call and tokenizer usage follow HuggingFace
# standard documentation patterns.
# URL: https://huggingface.co/docs/transformers
# =========================================================================
def encode_texts(
    texts,
    tokenizer,
    encoder,
    device,
    batch_size=16,
    max_length=512,
    pooling="mean",
):
    """
    Run all texts through the frozen Bio_ClinicalBERT encoder to get embeddings.

    The encoder is put into eval() mode before encoding, which disables dropout
    so the embeddings are deterministic. torch.no_grad() skips gradient
    computation entirely since we are not training the encoder — this saves
    significant memory and makes encoding faster.

    We process in batches of 16 rather than all at once to avoid running out
    of GPU memory on large datasets.

    pooling options:
      "mean" — average all token vectors (default, generally works better for
               longer texts because it considers all token representations)
      "cls"  — take only the [CLS] token's hidden state (position 0), which
               BERT is specifically trained to use as a sequence summary

    Returns a 2D numpy array of shape (num_texts, 768) — one embedding per note.
    """
    all_embeddings = []

    encoder.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size]

            # Tokenize the batch — padding pads shorter sequences to the longest
            # one in this batch, truncation cuts anything beyond max_length tokens
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Move token tensors to GPU if available
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = encoder(**enc)

            if pooling == "cls":
                # Take the [CLS] token hidden state at position 0
                emb = outputs.last_hidden_state[:, 0, :]
            else:
                # Masked mean pooling over all real token positions
                emb = mean_pool(outputs.last_hidden_state, enc["attention_mask"])

            # Move back to CPU and collect as numpy
            all_embeddings.append(emb.cpu().numpy())

    # Stack all batch embeddings into a single (N, 768) matrix
    return np.vstack(all_embeddings)


# =========================================================================
# SOURCE ATTRIBUTION - maybe_cache_embeddings
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# The lazy-load caching strategy using a build_fn lambda and .npy files
# is entirely original. No external reference was used.
# =========================================================================
def maybe_cache_embeddings(cache_path, build_fn, use_cache=True):
    """
    Load embeddings from disk if they exist, otherwise compute and save them.

    Encoding 16,540 training notes through Bio_ClinicalBERT takes several
    minutes even on a GPU. During development you often want to retrain
    or tune the SVM without re-running the encoder each time. This function
    saves the embedding matrix as a .npy file after the first run and
    loads it directly on subsequent runs.

    build_fn is a lambda that, when called, performs the actual encoding.
    We use a lambda so the encoding only runs if we actually need it.
    """
    if use_cache and cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from: {cache_path}")
        return np.load(cache_path)

    # Cache does not exist or caching is disabled - encode from scratch
    arr = build_fn()

    if use_cache and cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, arr)
        print(f"Saved embeddings to: {cache_path}")

    return arr


# =========================================================================
# SOURCE ATTRIBUTION - evaluate
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# The combination of macro and weighted ROC-AUC OVR alongside F1 metrics,
# and the use of predict_proba() for ROC-AUC (rather than decision_function)
# are original design choices. sklearn metric functions are from:
# https://scikit-learn.org
# =========================================================================
def evaluate(name: str, model, X, y, num_classes: int):
    """
    Run the calibrated SVM on a split and compute all evaluation metrics.

    model.predict_proba() returns the sigmoid-calibrated probability matrix
    [N, num_classes]. We use argmax to get the predicted class (equivalent
    to picking the highest-probability class) and the full probability matrix
    for ROC-AUC computation.

    Metrics reported:
      Accuracy       — same as Micro F1 in single-label multiclass (ΣFP = ΣFN)
      Macro P/R/F1   — equal weight to every class including rare ones
      Micro F1       — weighted by class frequency, our primary metric
      ROC-AUC OVR   — one-vs-rest scheme, macro and weighted versions

    zero_division=0 means if a class has no predictions at all in this split,
    its precision/recall are set to 0 rather than raising a warning.
    """
    probs = model.predict_proba(X)   # [N, num_classes] — calibrated probabilities
    pred = probs.argmax(axis=1)      # [N] — predicted class index

    acc = accuracy_score(y, pred)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0
    )
    f1_micro = precision_recall_fscore_support(
        y, pred, average="micro", zero_division=0
    )[2]

    # ROC-AUC needs the full probability matrix, not just the predicted class.
    # multi_class="ovr" treats each of the 30 classes as a binary problem
    # (this class vs all others) and averages the per-class AUC scores.
    auc_macro_ovr = None
    auc_weighted_ovr = None
    try:
        auc_macro_ovr = roc_auc_score(
            y,
            probs,
            labels=list(range(num_classes)),
            multi_class="ovr",
            average="macro",      # Equal weight to each class
        )
        auc_weighted_ovr = roc_auc_score(
            y,
            probs,
            labels=list(range(num_classes)),
            multi_class="ovr",
            average="weighted",   # Weight by class frequency - our reported metric
        )
    except Exception as e:
        print(f"[WARN] ROC-AUC could not be computed for {name}: {repr(e)}")

    print(f"\n===== {name} =====")
    print(f"Accuracy              : {acc:.6f}")
    print(f"Macro Precision       : {prec_macro:.6f}")
    print(f"Macro Recall          : {rec_macro:.6f}")
    print(f"Macro F1              : {f1_macro:.6f}")
    print(f"Micro F1              : {f1_micro:.6f}")
    if auc_macro_ovr is not None:
        print(f"ROC AUC (Macro OVR)   : {auc_macro_ovr:.6f}")
    if auc_weighted_ovr is not None:
        print(f"ROC AUC (Weighted OVR): {auc_weighted_ovr:.6f}")

    out = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
    }
    if auc_macro_ovr is not None:
        out["roc_auc_macro_ovr"] = float(auc_macro_ovr)
    if auc_weighted_ovr is not None:
        out["roc_auc_weighted_ovr"] = float(auc_weighted_ovr)

    return out


# =========================================================================
# SOURCE ATTRIBUTION - main (train_hybrid_svm.py)
# =========================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# The full pipeline design (frozen encoder -> cached embeddings -> LinearSVC
# -> Platt scaling -> Pipeline) is entirely original. LinearSVC and
# CalibratedClassifierCV are standard scikit-learn components used in an
# original hybrid architecture design for this thesis.
# sklearn URL: https://scikit-learn.org
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder containing train_full.csv, dev_full.csv, test_full.csv, label_vocab.txt")
    ap.add_argument("--out_dir", default="svm_hybrid_out", help="Where to save model + metrics + cached embeddings")

    ap.add_argument("--encoder_name", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--pooling", choices=["mean", "cls"], default="mean")

    # 2000 chars ≈ 512 tokens for clinical text - coarse pre-filter before tokenization
    ap.add_argument("--max_chars", type=int, default=2000, help="Character truncation before tokenization")
    ap.add_argument("--max_length", type=int, default=512, help="Transformer token length")
    ap.add_argument("--batch_size", type=int, default=16)

    # Number of folds for Platt scaling cross-validation
    ap.add_argument("--calib_cv", type=int, default=3)
    ap.add_argument("--text_col", default="TEXT")
    ap.add_argument("--label_col", default="LABELS")

    ap.add_argument("--use_gpu", action="store_true")
    # --cache_embeddings saves encoded embeddings to .npy files after the first run
    ap.add_argument("--cache_embeddings", action="store_true")

    args = ap.parse_args()

    # Build paths to the three data splits and the label vocabulary file
    train_csv = os.path.join(args.data_dir, "train_full.csv")
    dev_csv = os.path.join(args.data_dir, "dev_full.csv")
    test_csv = os.path.join(args.data_dir, "test_full.csv")
    vocab_path = os.path.join(args.data_dir, "label_vocab.txt")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load the label mappings from label_vocab.txt
    # Note: unlike run_icd.py which sorts labels alphabetically, this script
    # preserves the file order from label_vocab.txt as the class index ordering.
    label2id, id2label = load_label_vocab(vocab_path)
    num_classes = len(id2label)

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loaded {num_classes} ICD codes from label_vocab.txt")

    # Load all three splits with character-level truncation applied
    train_df = load_split(train_csv, args.text_col, args.label_col, max_chars=args.max_chars)
    dev_df = load_split(dev_csv, args.text_col, args.label_col, max_chars=args.max_chars)
    test_df = load_split(test_csv, args.text_col, args.label_col, max_chars=args.max_chars)

    # Separate text and integer label arrays for each split
    X_train_text = train_df[args.text_col].tolist()
    y_train = encode_labels(train_df[args.label_col], label2id)

    X_dev_text = dev_df[args.text_col].tolist()
    y_dev = encode_labels(dev_df[args.label_col], label2id)

    X_test_text = test_df[args.text_col].tolist()
    y_test = encode_labels(test_df[args.label_col], label2id)

    # Load the Bio_ClinicalBERT encoder - weights are frozen, never updated
    print(f"Loading encoder: {args.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    encoder = AutoModel.from_pretrained(args.encoder_name).to(device)

    # Cache paths for storing pre-computed embeddings to disk
    train_cache = os.path.join(args.out_dir, "train_embeddings.npy")
    dev_cache = os.path.join(args.out_dir, "dev_embeddings.npy")
    test_cache = os.path.join(args.out_dir, "test_embeddings.npy")

    # Encode all three splits through the frozen Bio_ClinicalBERT encoder.
    # After this step X_train/X_dev/X_test are numpy arrays of shape (N, 768).
    # The encoder is no longer used after this - the SVM trains on these fixed vectors.
    X_train = maybe_cache_embeddings(
        train_cache,
        lambda: encode_texts(
            X_train_text, tokenizer, encoder, device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
        ),
        use_cache=args.cache_embeddings,
    )

    X_dev = maybe_cache_embeddings(
        dev_cache,
        lambda: encode_texts(
            X_dev_text, tokenizer, encoder, device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
        ),
        use_cache=args.cache_embeddings,
    )

    X_test = maybe_cache_embeddings(
        test_cache,
        lambda: encode_texts(
            X_test_text, tokenizer, encoder, device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
        ),
        use_cache=args.cache_embeddings,
    )

    print("\nTraining hybrid baseline (Transformer embeddings + LinearSVC + calibration)...")

    # LinearSVC is the core classifier. We use a linear kernel because the 768-dim
    # BERT embedding space is already rich enough that a linear decision boundary
    # works well — and LinearSVC scales much better to high-dimensional data than
    # kernel SVMs would.
    base_svm = LinearSVC()

    # Wrap the SVM in CalibratedClassifierCV to get class probabilities.
    # method="sigmoid" means Platt scaling: fits a sigmoid function A*score+B
    # on the SVM's raw margin scores using cv=3 cross-validation folds.
    # This is necessary because LinearSVC has no predict_proba() on its own,
    # and we need probabilities for ROC-AUC computation.
    calibrated = CalibratedClassifierCV(base_svm, cv=args.calib_cv, method="sigmoid")

    # Wrap in a sklearn Pipeline so the whole thing behaves as a single estimator
    # with fit() / predict() / predict_proba() methods.
    # There is only one step here, but using Pipeline keeps the code consistent
    # with sklearn conventions and makes it easy to add preprocessing later.
    model = Pipeline([
        ("clf", calibrated),
    ])

    # Train the SVM on the pre-computed Bio_ClinicalBERT embeddings.
    # This is the only part of the pipeline that actually learns - the encoder
    # was already frozen and did not update during the encode_texts() calls above.
    model.fit(X_train, y_train)

    # Save the trained model using joblib, which handles the numpy arrays inside
    # sklearn models efficiently. Also save the id2label mapping for later inference.
    joblib.dump(model, os.path.join(args.out_dir, "hybrid_svm_model.joblib"))
    json.dump(id2label, open(os.path.join(args.out_dir, "id2label.json"), "w", encoding="utf-8"), indent=2)

    # Save the experiment settings alongside the results so we always know
    # exactly what configuration produced these numbers
    metrics = {
        "settings": {
            "encoder_name": args.encoder_name,
            "pooling": args.pooling,
            "max_chars": int(args.max_chars),
            "max_length": int(args.max_length),
            "batch_size": int(args.batch_size),
            "calib_cv": int(args.calib_cv),
            "use_gpu": bool(args.use_gpu),
            "cache_embeddings": bool(args.cache_embeddings),
        }
    }

    # Evaluate on dev and test sets —-dev is for monitoring, test is the final number
    metrics["dev"] = evaluate("DEV", model, X_dev, y_dev, num_classes)
    metrics["test"] = evaluate("TEST", model, X_test, y_test, num_classes)

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model + metrics to: {args.out_dir}")


if __name__ == "__main__":
    main()