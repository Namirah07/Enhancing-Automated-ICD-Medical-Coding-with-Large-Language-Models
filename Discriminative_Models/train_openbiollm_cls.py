# train_openbiollm_cls.py
#
# =============================================================================
# SOURCE ATTRIBUTION - FILE LEVEL
# =============================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
#
# This script is entirely original work. It mirrors the structure of
# train_biomistral_cls.py but targets OpenBioLLM-8B (Llama-3 based) and
# adds --quant_4bit (4-bit NF4 bitsandbytes quantization) to handle the
# larger 8B parameter count on limited VRAM.
#
# Library components used within this original code:
#
#   PEFT / LoRA adapter usage:
#     Adapted from: PEFT library documentation and QLoRA paper
#     (Dettmers et al., NeurIPS 2023 - "QLoRA: Efficient Finetuning of
#      Quantized LLMs")
#     URL: https://github.com/huggingface/peft
#     URL: https://arxiv.org/abs/2305.14314
#     Specifically: LoraConfig, get_peft_model, PeftModel, BitsAndBytesConfig
#
#   MeanPooler class:
#     Standard masked mean pooling pattern from the HuggingFace community.
#     Reference: https://huggingface.co/blog/how-to-train-sentence-transformers
#
#   Manual training loop (AdamW, cosine LR scheduler, GradScaler):
#     Adapted from: HuggingFace Transformers training documentation
#     URL: https://huggingface.co/docs/transformers/training
#
# WHAT IS ENTIRELY ORIGINAL (written by Namirah Imtieaz Shaik):
#   - CustomClassifierHead: two-layer MLP head for 30-class ICD-10 prediction
#   - CustomSeqClassifier: AutoModel backbone + pooling + MLP head wrapper
#     with SimpleNamespace output
#   - The decision to use AutoModel (not AutoModelForCausalLM) for hidden states
#   - --quant_4bit flag and BitsAndBytesConfig NF4 setup for 8B model
#   - encode_disc_batch_headtail() / DataCollatorWithPadding pipeline
#   - load_best_model_for_eval(): rebuild logic from saved LoRA adapter + head
#   - Early stopping on macro F1 with patience and min_delta
#   - evaluate_loader() with ROC-AUC weighted OVR alongside F1 metrics
#   - Checkpoint saving strategy (LoRA adapter and custom head separately)
#   - GPU cleanup gc.collect() + empty_cache() before test evaluation
# =============================================================================
#
# Single-label ICD text classification with LoRA + custom dense head on
# Llama3-OpenBioLLM-8B. Mirrors BioMistral script structure and adds
# --eval_only, --quant_4bit, and ROC-AUC reporting.

import os, json, argparse, warnings, csv, gc
from typing import Dict, List, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import numpy as np

from datasets import load_dataset, Features, Value
from torch.utils.data import DataLoader

from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    roc_auc_score,
)

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    get_scheduler,
    set_seed,
)

from peft import LoraConfig, get_peft_model, PeftModel

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision("high")


# =============================================================================
# SOURCE ATTRIBUTION - build_label_maps
# =============================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Reads label_vocab.txt and builds sorted integer-to-ICD-code mappings.
# Alphabetical sort ensures class indices are deterministic and consistent
# across all three discriminative model training scripts.
# No external reference was used for this function.
# =============================================================================
def build_label_maps(path: str) -> Tuple[List[str], Dict[int, str], Dict[str, int]]:
    """
    Read label_vocab.txt and build the two-way mappings between ICD codes and integers.

    One ICD code per line. Labels are sorted alphabetically so the class index
    ordering is deterministic and consistent across all discriminative scripts.
    This means class index 0 always maps to the same ICD code regardless of
    which model or machine runs the script.

    Returns:
        labels   - sorted list of ICD code strings
        id2label - integer to ICD code mapping
        label2id - ICD code to integer mapping
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


# =============================================================================
# SOURCE ATTRIBUTION - MeanPooler
# =============================================================================
# ADAPTED FROM: Standard masked mean pooling pattern widely used in the
#   HuggingFace community for sentence embedding tasks.
#   Reference: https://huggingface.co/blog/how-to-train-sentence-transformers
# The specific implementation here (mask expand, masked sum, clamp denom)
# is written by Namirah Imtieaz Shaik following this standard pattern.
# =============================================================================
class MeanPooler(nn.Module):
    """
    Compute the masked mean of token hidden states over the sequence dimension.

    Multiplying by the attention mask zeros out padding positions before summing,
    so padding tokens do not contribute to the document representation. The clamp
    on the denominator prevents division by zero for zero-length sequences, which
    should not occur in practice but is a safe guard.
    """

    def forward(self, last_hidden_state, attention_mask):
        """
        Pool [B, T, H] token hidden states to a [B, H] document vector.

        Args:
            last_hidden_state: token-level hidden states, shape [B, T, H]
            attention_mask:    1 for real tokens, 0 for padding, shape [B, T]

        Returns:
            Masked mean pooled vector of shape [B, H].
        """
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, T, 1]
        summed = (last_hidden_state * mask).sum(dim=1)                   # [B, H]
        denom = mask.sum(dim=1).clamp(min=1e-6)                          # [B, 1]
        return summed / denom                                             # [B, H]


# =============================================================================
# SOURCE ATTRIBUTION - CustomClassifierHead
# =============================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# Two-layer MLP classification head designed specifically for this thesis.
# Architecture was validated through the hyperparameter tuning experiments
# (EXP1 and EXP2) in train_openbiollm_HPT1.py. No external reference was
# used for this class.
# =============================================================================
class CustomClassifierHead(nn.Module):
    """
    Two-layer MLP that maps the pooled document vector to 30 class logits.

    Architecture: Dropout -> Linear(in_dim -> hidden_dim) -> GELU
                           -> Dropout -> Linear(hidden_dim -> num_labels)

    GELU activation is chosen for consistency with transformer-style architectures.
    Dropout is applied before each linear layer for regularization.
    """

    def __init__(self, in_dim, hidden_dim, num_labels, dropout=0.1):
        """
        Build the two-layer MLP head.

        Args:
            in_dim:     input dimension, must match backbone hidden_size (4096 for OpenBioLLM-8B)
            hidden_dim: width of the intermediate hidden layer (default 1536)
            num_labels: number of output classes (30 for our MIMIC-IV benchmark)
            dropout:    dropout probability applied before each linear layer (default 0.1)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x):
        """
        Project pooled document vector to class logits.

        Args:
            x: pooled document vector of shape [batch, in_dim]

        Returns:
            Raw logits of shape [batch, num_labels]. No softmax applied here -
            CrossEntropyLoss applies softmax internally during training.
        """
        return self.net(x)


# =============================================================================
# SOURCE ATTRIBUTION - CustomSeqClassifier
# =============================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
# This wrapper class combining a causal LM backbone (AutoModel) with a custom
# MLP classification head is entirely original. Key design decisions that are
# original contributions of this thesis:
#   - Using AutoModel (not AutoModelForCausalLM) to get hidden states without
#     the language modelling head attached
#   - SimpleNamespace return object for .logits and .loss access
#   - The pooling="mean"/"last" switch for different model families
# =============================================================================
class CustomSeqClassifier(nn.Module):
    """
    Full OpenBioLLM-D model: LoRA-adapted AutoModel backbone + pooling + MLP head.

    Combines three components into a single nn.Module:
      1. base: the OpenBioLLM-8B backbone loaded via AutoModel (hidden states only,
               no LM head attached), with LoRA adapter weights injected
      2. pool: MeanPooler that collapses the token sequence to one document vector
      3. head: CustomClassifierHead that projects to 30 ICD logits

    Returns a SimpleNamespace with .logits and .loss so the training loop can
    use outputs.logits and outputs.loss just like a HuggingFace model output,
    without requiring CustomSeqClassifier to subclass PreTrainedModel.
    """

    def __init__(self, base_model, num_labels, hidden_dim, dropout=0.1, pooling="mean", dtype=None):
        """
        Wire together the backbone, pooling layer, and classification head.

        Args:
            base_model: the LoRA-wrapped AutoModel backbone
            num_labels: number of ICD classes (30)
            hidden_dim: intermediate MLP width (default 1536)
            dropout:    dropout probability for the head
            pooling:    "mean" for masked mean pooling, "last" for last real token
            dtype:      weight dtype for the head (matches backbone dtype)
        """
        super().__init__()
        self.base = base_model
        self.num_labels = num_labels
        self.hidden_size = base_model.config.hidden_size
        self.pooling = pooling
        self.pool = MeanPooler() if pooling == "mean" else None
        self.head = CustomClassifierHead(self.hidden_size, hidden_dim, num_labels, dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.dtype = dtype

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """
        Full forward pass: backbone -> pool -> head -> optional loss.

        Args:
            input_ids:      token IDs, shape [B, T]
            attention_mask: 1 for real tokens, 0 for padding, shape [B, T]
            labels:         integer class indices, shape [B], optional.
                            If provided, CrossEntropyLoss is computed and
                            returned in the output namespace.

        Returns:
            SimpleNamespace with:
              .logits: raw class scores, shape [B, num_labels]
              .loss:   CrossEntropyLoss scalar if labels provided, else None
        """
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        last_hidden = out.last_hidden_state  # [B, T, H]

        if self.pooling == "mean":
            pooled = self.pool(last_hidden, attention_mask)  # [B, H]
        else:
            # "Last" token pooling: find the last real (non-padding) token position.
            # attention_mask.sum - 1 gives the index of the last 1 in each row.
            lengths = attention_mask.sum(dim=1) - 1
            pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), lengths]

        logits = self.head(pooled)  # [B, num_labels]

        loss = None
        if labels is not None:
            if labels.dtype != torch.long:
                labels = labels.long()
            loss = self.criterion(logits, labels)

        return SimpleNamespace(logits=logits, loss=loss)


def main():
    """
    Entry point. Parses arguments, builds the model, and either trains or evaluates.

    The training pipeline:
      1. Load label vocab and build id2label / label2id maps
      2. Load and preprocess CSV datasets via HuggingFace datasets
      3. Tokenize with dynamic padding via DataCollatorWithPadding
      4. Load OpenBioLLM-8B via AutoModel (optionally in 4-bit NF4)
      5. Inject LoRA adapters via PEFT
      6. Wrap in CustomSeqClassifier
      7. Train with AdamW + cosine LR + fp16/bf16 + early stopping on macro F1
      8. Save LoRA adapter + custom head separately at the best checkpoint
      9. Reload best checkpoint and evaluate on the test set
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="aaditya/Llama3-OpenBioLLM-8B")

    parser.add_argument("--train_csv", type=str, default=r".\shared\train_full.csv")
    parser.add_argument("--dev_csv",   type=str, default=r".\shared\dev_full.csv")
    parser.add_argument("--test_csv",  type=str, default=r".\shared\test_full.csv")
    parser.add_argument("--label_vocab", type=str, default=r".\shared\label_vocab.txt")

    parser.add_argument("--out_dir", type=str, default=r".\output\openbiollm8b_cls_lora_custom")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true",
                        help="Prefer bf16 on A100/L40 GPUs")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # --quant_4bit loads the backbone in 4-bit NF4 quantization to fit OpenBioLLM-8B
    # into a single GPU. Not available in the 7B Meditron/BioMistral scripts because
    # those models fit in fp16 without quantization.
    parser.add_argument("--quant_4bit", action="store_true",
                        help="Use 4-bit NF4 bitsandbytes quantization. Recommended for low VRAM.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training and only evaluate using existing best_checkpoint.")

    parser.add_argument("--cls_hidden_dim", type=int, default=1536)
    parser.add_argument("--cls_dropout", type=float, default=0.10)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"])

    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Dataset loading and preprocessing
    # -------------------------------------------------------------------------
    labels, id2label, label2id = build_label_maps(args.label_vocab)
    num_labels = len(labels)
    if num_labels < 2:
        raise ValueError("label_vocab must contain at least 2 labels.")

    # Force HuggingFace datasets to treat TEXT and LABELS as string columns.
    # Without this, pandas may infer numeric types for some ICD codes.
    features = Features({"TEXT": Value("string"), "LABELS": Value("string")})
    ds = load_dataset(
        "csv",
        data_files={"train": args.train_csv, "validation": args.dev_csv, "test": args.test_csv},
        features=features,
    )

    # Drop any extra columns that are not TEXT or LABELS
    keep = {"TEXT", "LABELS"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    def clean_and_map(example):
        """
        Normalize one CSV row and map the ICD code string to an integer label index.

        Rows whose LABELS value is not in label_vocab are marked with None so
        the filter step below can remove them. This keeps the task well-defined:
        only the 30 ICD codes in our vocabulary are valid predictions.

        Returns a dict with TEXT (str), LABELS (str or None), label (int or None).
        """
        txt = example.get("TEXT") or ""
        lab = example.get("LABELS")
        if lab is None:
            return {"TEXT": txt, "LABELS": None, "label": None}
        lab = str(lab).strip()
        if lab not in label2id:
            return {"TEXT": txt, "LABELS": None, "label": None}
        return {"TEXT": txt, "LABELS": lab, "label": label2id[lab]}

    ds = ds.map(clean_and_map)

    def _row_has_valid_label(ex):
        """
        Filter predicate for HuggingFace datasets.filter().

        Keeps only rows that successfully mapped to a valid integer label.
        Rows with unknown ICD codes or missing LABELS were marked None by
        clean_and_map() and are removed here.
        """
        return ex["LABELS"] is not None and ex["label"] is not None

    for split in ["train", "validation", "test"]:
        ds[split] = ds[split].filter(_row_has_valid_label)

    # -------------------------------------------------------------------------
    # Tokenizer setup
    # -------------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        # OpenBioLLM (Llama-3 based) has no pad token by default.
        # Setting pad_token = eos_token is the standard workaround for causal LMs.
        tok.pad_token = tok.eos_token

    def tokenize_fn(batch):
        """
        Tokenize a HuggingFace batch and attach integer labels.

        padding=False here means no padding is applied at tokenization time.
        Padding is applied later by DataCollatorWithPadding on a per-batch basis,
        which pads only to the longest sequence in each batch rather than globally
        to max_length. This reduces unnecessary padding tokens and speeds up training.

        Args:
            batch: dict with TEXT (list of strings) and label (list of ints)

        Returns:
            dict with input_ids, attention_mask, and labels fields
        """
        enc = tok(
            batch["TEXT"],
            truncation=True,
            max_length=args.max_length,
            padding=False,  # DataCollatorWithPadding handles padding per batch
        )
        enc["labels"] = batch["label"]
        return enc

    keep_after = {"input_ids", "attention_mask", "labels", "token_type_ids"}
    cols_to_drop = [c for c in ds["train"].column_names if c not in keep_after]
    ds = ds.map(tokenize_fn, batched=True, remove_columns=cols_to_drop)

    # pad_to_multiple_of=8 aligns sequence lengths to multiples of 8 for
    # efficient GPU tensor core utilization on modern CUDA hardware
    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(ds["validation"], batch_size=args.batch_size, shuffle=False,
                              collate_fn=collator, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(ds["test"], batch_size=args.batch_size, shuffle=False,
                              collate_fn=collator, num_workers=2, pin_memory=True)

    # -------------------------------------------------------------------------
    # Dtype and device configuration
    # -------------------------------------------------------------------------
    # bf16 and fp16 are mutually exclusive - bf16 takes priority if both are set
    use_bf16 = bool(args.bf16)
    use_fp16 = bool(args.fp16) and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = use_fp16 or use_bf16
    autocast_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)

    # -------------------------------------------------------------------------
    # 4-bit quantization config (OpenBioLLM-8B specific)
    # =========================================================================
    # SOURCE ATTRIBUTION - BitsAndBytesConfig (QLoRA)
    # =========================================================================
    # ADAPTED FROM: QLoRA paper (Dettmers et al., NeurIPS 2023) and
    #   bitsandbytes library documentation
    #   URL: https://arxiv.org/abs/2305.14314
    #   URL: https://github.com/TimDettmers/bitsandbytes
    # NF4 quantization type, double quantization, and compute dtype pattern
    # follow the QLoRA paper and bitsandbytes documentation exactly.
    # The --quant_4bit flag and the try/except RuntimeError wrapper are
    # original code by Namirah Imtieaz Shaik.
    # =========================================================================
    quant_config = None
    device_map = None
    if args.quant_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError(
                "You requested --quant_4bit but BitsAndBytesConfig is unavailable.\n"
                "Install/upgrade: pip install -U bitsandbytes accelerate transformers"
            ) from e

        # NF4 (NormalFloat4) preserves the weight distribution better than
        # standard int4, which matters for generation quality.
        # double_quant quantizes the quantization constants themselves,
        # saving another ~0.4 bits per parameter.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
        # device_map="auto" is required when using bitsandbytes quantization -
        # it lets bitsandbytes manage GPU placement of quantized layers
        device_map = "auto"

    base_kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if quant_config is not None:
        base_kwargs["quantization_config"] = quant_config
        base_kwargs["device_map"] = device_map

    # -------------------------------------------------------------------------
    # Evaluation helper
    # =========================================================================
    # SOURCE ATTRIBUTION - evaluate_loader
    # =========================================================================
    # ORIGINAL CODE: Written by Namirah Imtieaz Shaik
    # The evaluation loop structure (no_grad, argmax, softmax for ROC-AUC,
    # autocast context) is original. ROC-AUC weighted OVR metric alongside
    # F1 metrics is an original contribution of the thesis evaluation design.
    # sklearn metric functions are from: https://scikit-learn.org
    # =========================================================================
    def evaluate_loader(model, loader):
        """
        Run the model on all batches in loader and compute full evaluation metrics.

        Collects predictions, gold labels, and class probability distributions
        across all batches. Computes:
          - Macro and micro precision, recall, F1
          - Overall accuracy
          - ROC-AUC macro OVR and weighted OVR (from softmax probabilities)

        ROC-AUC is computed from real softmax class probabilities (not hard
        predictions), making it more informative than for the generative models
        which only produce one hard prediction per example.

        Note: float32 softmax is used for ROC-AUC computation to avoid
        half-precision noise that can cause numerical issues in sklearn.

        Args:
            model:  the CustomSeqClassifier to evaluate
            loader: DataLoader over the split to evaluate

        Returns:
            dict with loss, accuracy, precision_macro, recall_macro, f1_macro,
            f1_micro, preds, labels, and optionally roc_auc_macro_ovr and
            roc_auc_weighted_ovr.
        """
        model.eval()
        all_preds, all_gold, all_probs = [], [], []
        total_loss, n_batches = 0.0, 0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                if use_autocast and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        outputs = model(**batch)
                else:
                    outputs = model(**batch)

                loss = outputs.loss
                logits = outputs.logits

                if loss is not None:
                    total_loss += float(loss.item())
                n_batches += 1

                preds = logits.argmax(dim=-1).detach().cpu().numpy()
                gold = batch["labels"].detach().cpu().numpy()
                # Cast to float32 before softmax to avoid half-precision noise
                probs = torch.softmax(logits.to(torch.float32), dim=-1).detach().cpu().numpy()

                all_preds.append(preds)
                all_gold.append(gold)
                all_probs.append(probs)

        all_preds = np.concatenate(all_preds)
        all_gold = np.concatenate(all_gold)
        all_probs = np.concatenate(all_probs, axis=0)

        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            all_gold, all_preds, average="macro", zero_division=0
        )
        f1_micro = precision_recall_fscore_support(
            all_gold, all_preds, average="micro", zero_division=0
        )[2]
        acc = (all_preds == all_gold).mean()

        auc_macro_ovr = None
        auc_weighted_ovr = None
        try:
            auc_macro_ovr = roc_auc_score(
                all_gold,
                all_probs,
                labels=list(range(num_labels)),
                multi_class="ovr",
                average="macro",
            )
            auc_weighted_ovr = roc_auc_score(
                all_gold,
                all_probs,
                labels=list(range(num_labels)),
                multi_class="ovr",
                average="weighted",
            )
        except Exception as e:
            # Most commonly happens when a split is missing a class entirely,
            # which can occur with small debug subsets
            print("ROC-AUC could not be computed:", repr(e))
            print("Unique classes in this loader:", np.unique(all_gold))
            print("all_probs shape:", all_probs.shape)

        out = {
            "loss": total_loss / max(1, n_batches),
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
            "preds": all_preds,
            "labels": all_gold,
        }
        if auc_macro_ovr is not None:
            out["roc_auc_macro_ovr"] = float(auc_macro_ovr)
        if auc_weighted_ovr is not None:
            out["roc_auc_weighted_ovr"] = float(auc_weighted_ovr)

        return out

    # =========================================================================
    # SOURCE ATTRIBUTION - load_best_model_for_eval
    # =========================================================================
    # ORIGINAL CODE: Written by Namirah Imtieaz Shaik
    # This function rebuilds the full CustomSeqClassifier from the saved
    # LoRA adapter and custom head weights at inference time. The two-step
    # load (base model from HuggingFace + LoRA adapter from checkpoint) is
    # an original design choice of this thesis - other approaches save the
    # full merged model which uses far more disk space.
    # =========================================================================
    def load_best_model_for_eval():
        """
        Rebuild the full CustomSeqClassifier from the best checkpoint for evaluation.

        The checkpoint contains only the LoRA adapter weights and custom head weights,
        not the full backbone. We reload the backbone from HuggingFace and then
        attach the saved LoRA adapter via PeftModel.from_pretrained().

        This two-step rebuild is necessary because model.base.save_pretrained()
        only saves the adapter deltas, not the full 8B parameter backbone.

        Returns:
            CustomSeqClassifier loaded with best checkpoint weights, in eval mode.

        Raises:
            FileNotFoundError: if the LoRA adapter or head file are missing.
        """
        best_path = os.path.join(args.out_dir, "best_checkpoint")
        adapter_dir = os.path.join(best_path, "lora_adapter")
        head_path = os.path.join(best_path, "custom_head.pt")

        if not os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
            raise FileNotFoundError(f"Missing LoRA adapter in {adapter_dir}. Did you train first?")
        if not os.path.isfile(head_path):
            raise FileNotFoundError(f"Missing head weights: {head_path}. Did you train first?")

        # Load fresh backbone from HuggingFace - this is the full 8B model
        base_eval = AutoModel.from_pretrained(args.model_name, **base_kwargs)
        base_eval.config.pad_token_id = tok.pad_token_id
        base_eval.config.use_cache = False
        if args.gradient_checkpointing:
            base_eval.gradient_checkpointing_enable()

        # Attach the saved LoRA adapter weights on top of the frozen backbone
        base_eval = PeftModel.from_pretrained(base_eval, adapter_dir)

        m = CustomSeqClassifier(
            base_model=base_eval,
            num_labels=num_labels,
            hidden_dim=args.cls_hidden_dim,
            dropout=args.cls_dropout,
            pooling=args.pooling,
            dtype=dtype,
        )

        # Load the custom MLP head weights separately (not saved by PEFT)
        state = torch.load(head_path, map_location="cpu")
        m.head.load_state_dict(state)

        m.to(device)
        m.eval()
        return m

    # -------------------------------------------------------------------------
    # Eval-only path
    # -------------------------------------------------------------------------
    if args.eval_only:
        # Clear any GPU memory from dataset loading before loading the 8B model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = load_best_model_for_eval()
        test_metrics = evaluate_loader(model, test_loader)
        print("Test:", {k: v for k, v in test_metrics.items() if k not in ("preds", "labels")})

        with open(os.path.join(args.out_dir, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in test_metrics.items() if k not in ("preds", "labels")}, f, indent=2)
        return

    # -------------------------------------------------------------------------
    # Training path - model loading and LoRA injection
    # =========================================================================
    # SOURCE ATTRIBUTION - LoRA configuration and application
    # =========================================================================
    # ADAPTED FROM: PEFT library documentation and QLoRA paper
    #   (Dettmers et al., NeurIPS 2023)
    #   URL: https://github.com/huggingface/peft
    #   URL: https://arxiv.org/abs/2305.14314
    # LoraConfig parameters (r=16, lora_alpha=32, target_modules, task_type=SEQ_CLS)
    # follow the PEFT library documentation pattern.
    # The specific target_modules list and r/alpha values were validated through
    # the HPT experiments in train_openbiollm_HPT1.py (original work).
    # enable_input_require_grads() pattern is from PEFT documentation.
    # =========================================================================
    _ = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )

    base_model = AutoModel.from_pretrained(args.model_name, **base_kwargs)
    base_model.config.pad_token_id = tok.pad_token_id

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
    # use_cache must be False when using gradient checkpointing - they are incompatible
    base_model.config.use_cache = False

    # LoRA on all seven projection matrix types - same target_modules as all other
    # discriminative models. task_type=SEQ_CLS is PEFT bookkeeping for the adapter.
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    # enable_input_require_grads is needed so gradients flow from the LoRA adapter
    # matrices back through the frozen embedding layer during backpropagation
    base_model.enable_input_require_grads()
    base_model = get_peft_model(base_model, lora_cfg)
    base_model.print_trainable_parameters()  # confirms ~0.1% of params are trainable

    model = CustomSeqClassifier(
        base_model=base_model,
        num_labels=num_labels,
        hidden_dim=args.cls_hidden_dim,
        dropout=args.cls_dropout,
        pooling=args.pooling,
        dtype=dtype,
    )
    model.to(device)

    # =========================================================================
    # SOURCE ATTRIBUTION - Training loop, optimizer, scheduler, GradScaler
    # =========================================================================
    # ADAPTED FROM: HuggingFace Transformers training documentation
    #   URL: https://huggingface.co/docs/transformers/training
    # AdamW optimizer, cosine LR scheduler with 5% warmup, GradScaler for fp16,
    # and gradient accumulation follow standard HuggingFace training conventions.
    #
    # ORIGINAL CODE (written by Namirah Imtieaz Shaik):
    # - Early stopping monitoring macro F1 with patience and min_delta
    # - Checkpoint saving: LoRA adapter and MLP head saved separately
    # - GPU cleanup (gc.collect + empty_cache) before test evaluation
    # - The manual loop design chosen over HuggingFace Trainer because
    #   CustomSeqClassifier is not a standard HuggingFace PreTrainedModel
    # =========================================================================
    # Only pass trainable parameters to the optimizer - frozen backbone weights
    # have requires_grad=False so no optimizer state is wasted on them
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=0.01)

    total_steps = (len(train_loader) // max(1, args.grad_accum)) * args.epochs
    # 5% warmup: ramp LR from 0 to args.lr over the first 5% of total steps
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # GradScaler for fp16 only - bf16 does not need gradient scaling
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # -------------------------------------------------------------------------
    # Training loop with early stopping
    # -------------------------------------------------------------------------
    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_path = os.path.join(args.out_dir, "best_checkpoint")
    os.makedirs(best_path, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_autocast and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    outputs = model(**batch)
                    # Divide loss by grad_accum to get the average over the accumulation window
                    loss = outputs.loss / args.grad_accum
                if use_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum
                loss.backward()

            running_loss += float(loss.item())

            # Only step optimizer every grad_accum micro-batches
            if step % args.grad_accum == 0:
                if use_fp16 and device.type == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if global_step % 50 == 0:
                print(f"[epoch {epoch}] step {global_step}/{total_steps} | loss {running_loss:.4f}")
                running_loss = 0.0

        # Validation at end of each epoch
        val_metrics = evaluate_loader(model, val_loader)
        auc_str = ""
        if "roc_auc_macro_ovr" in val_metrics:
            auc_str = f" roc_auc_macro_ovr={val_metrics['roc_auc_macro_ovr']:.4f}"

        print(
            f"[epoch {epoch}] val: "
            f"loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"f1_macro={val_metrics['f1_macro']:.4f} "
            f"f1_micro={val_metrics['f1_micro']:.4f}"
            f"{auc_str}"
        )

        # Early stopping: monitor macro F1 improvement
        improved = val_metrics["f1_macro"] > (best_val_f1 + args.early_stop_min_delta)
        if improved:
            best_val_f1 = val_metrics["f1_macro"]
            epochs_no_improve = 0

            # Save LoRA adapter separately from the backbone (saves ~150 MB vs ~16 GB)
            adapter_dir = os.path.join(best_path, "lora_adapter")
            os.makedirs(adapter_dir, exist_ok=True)
            model.base.save_pretrained(adapter_dir)
            # Save the custom MLP head separately (PEFT does not include it)
            torch.save(model.head.state_dict(), os.path.join(best_path, "custom_head.pt"))
            tok.save_pretrained(best_path)
            with open(os.path.join(best_path, "label_map.json"), "w", encoding="utf-8") as f:
                json.dump({"id2label": {int(i): lab for i, lab in id2label.items()},
                           "label2id": label2id}, f, indent=2)

            print(f" saved new best checkpoint to {best_path} (f1_macro={best_val_f1:.6f})")
        else:
            epochs_no_improve += 1
            print(f" no improvement (+{epochs_no_improve}/{args.early_stop_patience})")
            if epochs_no_improve >= args.early_stop_patience:
                print(" Early stopping triggered.")
                break

    # -------------------------------------------------------------------------
    # GPU cleanup before test evaluation
    # -------------------------------------------------------------------------
    # The training model, optimizer, scheduler, and scaler all hold GPU memory.
    # Moving to CPU and deleting them before reloading the best checkpoint
    # prevents OOM errors on GPUs with limited VRAM. This is especially
    # important for the 8B OpenBioLLM model.
    try:
        model.to("cpu")
    except Exception:
        pass
    try:
        del model, optimizer, scheduler, scaler
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Reload best checkpoint and evaluate on test set
    # -------------------------------------------------------------------------
    best_adapter = os.path.join(best_path, "lora_adapter")
    head_path = os.path.join(best_path, "custom_head.pt")

    if os.path.isdir(best_adapter):
        print(f"Reloading best adapter from: {best_adapter}")

        base_model = AutoModel.from_pretrained(args.model_name, **base_kwargs)
        base_model.config.pad_token_id = tok.pad_token_id
        base_model.config.use_cache = False
        if args.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()

        base_model = PeftModel.from_pretrained(base_model, best_adapter)

        model = CustomSeqClassifier(
            base_model=base_model,
            num_labels=num_labels,
            hidden_dim=args.cls_hidden_dim,
            dropout=args.cls_dropout,
            pooling=args.pooling,
            dtype=dtype,
        )

        if os.path.isfile(head_path):
            state = torch.load(head_path, map_location="cpu")
            model.head.load_state_dict(state)

        model.to(device)
        model.eval()
    else:
        raise FileNotFoundError(f"No best adapter found at {best_adapter}")

    test_metrics = evaluate_loader(model, test_loader)
    print("Test:", {k: v for k, v in test_metrics.items() if k not in ("preds", "labels")})

    with open(os.path.join(args.out_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in test_metrics.items() if k not in ("preds", "labels")}, f, indent=2)

    # Per-label metrics CSV and full classification report
    y_true = test_metrics["labels"]
    y_pred = test_metrics["preds"]

    labels_idx = list(range(len(labels)))
    prec, rec, f1s, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average=None, zero_division=0
    )

    per_label_csv = os.path.join(args.out_dir, "per_label_f1_test.csv")
    with open(per_label_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label_index", "icd_code", "precision", "recall", "f1", "support"])
        for i in labels_idx:
            writer.writerow([i, id2label[i], f"{prec[i]:.6f}", f"{rec[i]:.6f}", f"{f1s[i]:.6f}", int(supp[i])])

    report_txt = classification_report(
        y_true, y_pred, labels=labels_idx, target_names=[id2label[i] for i in labels_idx], zero_division=0
    )
    with open(os.path.join(args.out_dir, "per_label_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(f" Saved per-label metrics to {per_label_csv}")


if __name__ == "__main__":
    main()