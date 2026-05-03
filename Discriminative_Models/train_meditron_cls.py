# train_meditron_cls.py
#
# =============================================================================
# SOURCE ATTRIBUTION - FILE LEVEL
# =============================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
#
# This entire script is original work. No existing training script was used
# as a direct template. The following standard library components were used
# within the original code:
#
#   PEFT / LoRA adapter usage:
#     Adapted from: PEFT library documentation and QLoRA paper
#     (Dettmers et al., NeurIPS 2023 - "QLoRA: Efficient Finetuning of
#      Quantized LLMs")
#     URL: https://github.com/huggingface/peft
#     Specifically: LoraConfig, get_peft_model, PeftModel pattern
#
#   MeanPooler class:
#     Standard masked mean pooling pattern widely used in the HuggingFace
#     community for sentence embedding tasks.
#     Reference: https://huggingface.co/blog/how-to-train-sentence-transformers
#
#   Manual training loop structure (gradient accumulation, GradScaler,
#   cosine LR scheduler with warmup):
#     Adapted from: HuggingFace Transformers training documentation
#     URL: https://huggingface.co/docs/transformers/training
#
#   Early stopping pattern:
#     Standard practice; no specific source.
#
# WHAT IS ENTIRELY ORIGINAL (written by Namirah Imtieaz Shaik):
#   - CustomClassifierHead: two-layer MLP head for ICD-10 classification
#   - CustomSeqClassifier: combining AutoModel backbone + pooling + MLP head
#     as a single nn.Module with SimpleNamespace output
#   - The decision to use AutoModel (not AutoModelForCausalLM) for hidden states
#   - Head+tail text cropping via crop_head_tail_tokens()
#   - encode_disc_batch_headtail() encoding pipeline
#   - Checkpoint saving strategy (LoRA adapter + head separately)
#   - load_best_model_for_eval() rebuild logic
#   - Early stopping on macro F1 (not loss or micro F1)
#   - All evaluation metric choices (Micro F1, ROC-AUC Weighted OVR)
# =============================================================================
#
# This is the main training script for the Meditron discriminative classifier.
# It fine-tunes Meditron-7B (a Llama-2 based biomedical LLM) for 30-class
# ICD-10 prediction using LoRA adapters + a custom two-layer MLP head.
#
# The backbone is loaded via AutoModel (not AutoModelForCausalLM) so we get
# hidden states directly without a language modelling head attached.
# LoRA is applied to q/k/v/o/gate/up/down projection matrices inside the
# transformer blocks — only the adapter weights and the classification head
# actually get trained. Everything else in the 7B backbone stays frozen.
#
# Why a manual training loop instead of HuggingFace Trainer?
#   The custom MLP head is not a standard HuggingFace module so Trainer
#   cannot manage it natively. The manual loop gives full control over
#   gradient accumulation, mixed precision, early stopping, and checkpoint
#   saving logic without fighting against Trainer's assumptions.

import os, json, argparse, warnings
import pandas as pd
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader  # wraps dataset into mini-batches for training/validation
from transformers import get_scheduler
from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_auc_score

import csv
from peft import PeftModel
import torch
import numpy as np
from datasets import load_dataset, Features, Value
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from peft import LoraConfig
from peft import get_peft_model
warnings.filterwarnings("ignore", category=UserWarning)


def build_label_maps(path: str) -> Tuple[List[str], Dict[int, str], Dict[str, int]]:
    """
    Read label_vocab.txt and build the two-way mappings between ICD codes and integers.

    Labels are sorted alphabetically so the class index ordering is deterministic
    and consistent across all three discriminative model scripts. This means
    class index 0 always maps to the same ICD code regardless of which model
    or which machine runs the script.
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


def main():
    # -------------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------------
    # All hyperparameters are exposed as command-line flags so experiments can
    # be launched without touching the code. Defaults reflect the configuration
    # used for the final reported results in the thesis.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="epfl-llm/meditron-7b")
    parser.add_argument("--train_csv", type=str, default=r".\shared\train_full.csv")
    parser.add_argument("--dev_csv",   type=str, default=r".\shared\dev_full.csv")
    parser.add_argument("--test_csv",  type=str, default=r".\shared\test_full.csv")
    parser.add_argument("--label_vocab", type=str, default=r".\shared\label_vocab.txt")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and only evaluate using an existing best_checkpoint in --out_dir",
    )
    parser.add_argument("--out_dir", type=str, default=r".\output\meditron_cls_lora")
    # 1024 tokens gives the full note context Meditron can handle without quantization
    parser.add_argument("--max_length", type=int, default=1024)
    # batch_size=1 with grad_accum=8 gives effective batch size of 8
    # without exceeding GPU memory for a 7B model
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    # cls_hidden_dim is the intermediate size of the two-layer MLP head
    # 1536 was selected through the hyperparameter tuning experiments
    parser.add_argument("--cls_hidden_dim", type=int, default=1536,
                        help="Hidden size of the custom classifier layer (e.g., 1024-2048)")
    parser.add_argument("--cls_dropout", type=float, default=0.1,
                        help="Dropout in the custom classifier head")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "last"],
                        help="How to pool token embeddings into a sentence embedding")
    # Early stopping monitors macro F1 on the validation set — sensitive to rare
    # classes which makes it a better training signal than micro F1 alone
    parser.add_argument("--early_stop_patience", type=int, default=2,
                        help="Stop if val F1-macro doesn't improve for this many validations")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                        help="Minimum F1-macro improvement to reset patience")
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    import torch.nn as nn
    from types import SimpleNamespace

    # -------------------------------------------------------------------------
    # Custom model components
    # -------------------------------------------------------------------------

    # =========================================================================
    # SOURCE ATTRIBUTION - MeanPooler
    # =========================================================================
    # ADAPTED FROM: Standard masked mean pooling pattern in the HuggingFace
    #   community for deriving sentence embeddings from token hidden states.
    #   Reference: https://huggingface.co/blog/how-to-train-sentence-transformers
    # The specific implementation here (mask expand, sum, clamp) is written
    # by Namirah Imtieaz Shaik following this standard pattern.
    # =========================================================================
    class MeanPooler(nn.Module):
        """
        Compute the average of all non-padding token hidden states.

        Meditron produces one hidden state per token. We need a single
        fixed-size vector per document to pass into the classifier head.
        Mean pooling averages across all real token positions, ignoring padding.

        The attention_mask has 1 for real tokens and 0 for padding, so
        multiplying by the mask zeros out padding positions before averaging.
        """
        def forward(self, last_hidden_state, attention_mask):
            # masked mean: sum(h * mask) / sum(mask)
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, T, 1]
            summed = (last_hidden_state * mask).sum(dim=1)  # [B, H]
            denom = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1] — clamp avoids divide-by-zero
            return summed / denom  # [B, H]

    # =========================================================================
    # SOURCE ATTRIBUTION - CustomClassifierHead
    # =========================================================================
    # ORIGINAL CODE: Written by Namirah Imtieaz Shaik
    # This two-layer MLP head (Dropout -> Linear -> GELU -> Dropout -> Linear)
    # is original work. The architecture was designed specifically for this
    # thesis and validated through the hyperparameter tuning experiments
    # (EXP1 and EXP2 in train_meditron_HPT1.py).
    # GELU activation is chosen for consistency with transformer architectures.
    # =========================================================================
    class CustomClassifierHead(nn.Module):
        """
        Two-layer MLP that maps the pooled document vector to 30 class logits.

        Architecture: Dropout → Linear(4096→1536) → GELU → Dropout → Linear(1536→30)

        Why two layers instead of one?
          The hyperparameter tuning experiments showed L1 and L2 perform
          comparably, with marginal gains from L2. We went with L2 as a
          conservative choice to keep a non-linear transformation between
          the LLM representation space and the ICD label space.

        Why GELU?
          GELU is used throughout transformer architectures and is smoother
          than ReLU. Since we are fine-tuning on top of a transformer backbone
          it makes sense to keep the same activation function for consistency.
        """
        def __init__(self, in_dim, hidden_dim, num_labels, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )

        def forward(self, x):
            return self.net(x)

    # =========================================================================
    # SOURCE ATTRIBUTION - CustomSeqClassifier
    # =========================================================================
    # ORIGINAL CODE: Written by Namirah Imtieaz Shaik
    # This wrapper class combining a causal LM backbone with a custom MLP
    # classification head is entirely original. The design decision to use
    # AutoModel (not AutoModelForCausalLM) to obtain raw hidden states,
    # the mean pooling strategy, and the SimpleNamespace return object are
    # all original contributions of this thesis.
    # =========================================================================
    class CustomSeqClassifier(nn.Module):
        """
        Full classification model: LoRA-adapted Meditron backbone + MLP head.

        Wraps the base LM and head into one nn.Module so training and evaluation
        can call model(**batch) and get back outputs.logits and outputs.loss
        in the same style as HuggingFace model outputs.

        SimpleNamespace is used as a lightweight return object — it gives us
        attribute access (outputs.logits) without needing a full dataclass.
        """

        def __init__(self, base_model, num_labels, hidden_dim, dropout=0.1, pooling="mean", dtype=None):
            super().__init__()
            self.base = base_model          # LoRA-wrapped Meditron backbone
            self.num_labels = num_labels
            self.hidden_size = base_model.config.hidden_size  # 4096 for Meditron-7B
            self.pooling = pooling
            self.pool = MeanPooler() if pooling == "mean" else None  # "last" token handled inline
            self.head = CustomClassifierHead(self.hidden_size, hidden_dim, num_labels, dropout)
            self.criterion = nn.CrossEntropyLoss()
            self.dtype = dtype

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            # Run the backbone — we only need the last hidden state, not intermediate layers
            out = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
            last_hidden = out.last_hidden_state  # [B, T, H]

            if self.pooling == "mean":
                pooled = self.pool(last_hidden, attention_mask)  # [B, H]
            else:
                # Last-token pooling: use the final real token (not a padding token)
                # attention_mask.sum()-1 gives the index of the last real token
                lengths = attention_mask.sum(dim=1) - 1  # [B]
                pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), lengths]

            logits = self.head(pooled)  # [B, num_labels]

            loss = None
            if labels is not None:
                if labels.dtype != torch.long:
                    labels = labels.long()
                loss = self.criterion(logits, labels)

            # Return a SimpleNamespace so callers can do outputs.logits and outputs.loss
            return SimpleNamespace(logits=logits, loss=loss)

    # -------------------------------------------------------------------------
    # Label vocabulary
    # -------------------------------------------------------------------------
    labels, id2label, label2id = build_label_maps(args.label_vocab)
    num_labels = len(labels)  # should be 30 for our MIMIC-IV benchmark
    if num_labels < 2:
        raise ValueError("label_vocab must contain at least 2 labels.")

    # -------------------------------------------------------------------------
    # Dataset loading and cleaning
    # -------------------------------------------------------------------------
    # Force the schema to TEXT/LABELS strings so the datasets library does not
    # try to infer types and accidentally cast ICD codes to numbers.
    features = Features({"TEXT": Value("string"), "LABELS": Value("string")})
    ds = load_dataset(
        "csv",
        data_files={"train": args.train_csv, "validation": args.dev_csv, "test": args.test_csv},
        features=features,
    )
    # Remove any extra columns the CSV might have beyond TEXT and LABELS
    keep = {"TEXT", "LABELS"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    def clean_and_map(example):
        """
        Map each ICD code string to its integer class index.

        Any example with a None label or a label not in label_vocab.txt is
        marked for removal. We return None values rather than raising errors
        here so the filter step below can remove them cleanly.
        """
        txt = example.get("TEXT")
        lab = example.get("LABELS")
        if txt is None:
            txt = ""
        if lab is None:
            return {"TEXT": txt, "LABELS": None, "label": None}
        lab = str(lab).strip()
        if lab not in label2id:
            # Unknown label — mark as None so it gets filtered out
            return {"TEXT": txt, "LABELS": None, "label": None}
        return {"TEXT": txt, "LABELS": lab, "label": label2id[lab]}

    # Apply to all three splits at once
    ds = ds.map(clean_and_map)
    for split in ["train", "validation", "test"]:
        ds[split] = ds[split].filter(lambda ex: ex["LABELS"] is not None and ex["label"] is not None)

    # -------------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------------
    # Meditron uses the Llama-2 SentencePiece tokenizer which has no pad token
    # by default. Setting pad_token = eos_token is the standard workaround —
    # the DataCollatorWithPadding will then pad with the EOS token ID.
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize_fn(batch):
        """
        Convert raw TEXT strings to token ID sequences.

        padding=False here because we pad dynamically at batch time in the
        DataCollatorWithPadding — this is more memory efficient than padding
        all sequences to a global maximum length upfront.
        """
        enc = tok(
            batch["TEXT"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        # Attach the integer class label alongside the tokenized inputs
        enc["labels"] = batch["label"]
        return enc

    # Only keep columns the model actually needs — drop TEXT, LABELS, label strings
    keep_after = {"input_ids", "attention_mask", "labels", "token_type_ids"}
    cols_to_drop = [c for c in ds["train"].column_names if c not in keep_after]
    ds = ds.map(tokenize_fn, batched=True, remove_columns=cols_to_drop)

    # Dynamic padding: pad each batch to the longest sequence in that batch.
    # pad_to_multiple_of=8 ensures tensor dimensions are multiples of 8 for
    # efficient GPU memory alignment.
    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------
    # num_workers=2 loads the next batch in background while the GPU trains
    # on the current one. pin_memory=True speeds up CPU→GPU transfer.
    train_loader = DataLoader(
        ds["train"],
        batch_size=args.batch_size,
        shuffle=True,         # shuffle training data each epoch
        collate_fn=collator,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        ds["validation"],
        batch_size=args.batch_size,
        shuffle=False,        # no shuffle for eval — want deterministic results
        collate_fn=collator,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        ds["test"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True
    )

    # -------------------------------------------------------------------------
    # Base model and LoRA setup
    # -------------------------------------------------------------------------
    # Load the config with our task settings attached — num_labels, id2label etc.
    cfg = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )
    # Choose compute precision: bf16 on A100/H100, fp16 on V100, fp32 otherwise
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    from transformers import AutoModel

    # Load the backbone WITHOUT a classification head — we attach our own custom head.
    # AutoModel gives raw hidden states, AutoModelForCausalLM would give LM logits.
    base_model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    )
    base_model.config.pad_token_id = tok.pad_token_id

    # Print architecture info so we know what we are working with
    print("hidden_size:", base_model.config.hidden_size)        # 4096 for Meditron-7B
    print("num_attention_heads:", base_model.config.num_attention_heads)
    print("num_hidden_layers:", base_model.config.num_hidden_layers)

    # LoRA configuration:
    # r=16: rank of the low-rank matrices (higher = more capacity, more memory)
    # lora_alpha=32: scaling factor, alpha/r = 2.0 controls the effective learning rate for adapters
    # target_modules: the projection matrices inside each transformer block where LoRA is injected
    # task_type=SEQ_CLS: tells PEFT this is a classification task
    # =========================================================================
    # SOURCE ATTRIBUTION - LoRA configuration and application
    # =========================================================================
    # ADAPTED FROM: PEFT library documentation and QLoRA paper
    #   (Dettmers et al., NeurIPS 2023)
    #   URL: https://github.com/huggingface/peft
    #   URL: https://arxiv.org/abs/2305.14314
    # The LoraConfig parameters (r=16, lora_alpha=32, target_modules,
    # task_type=SEQ_CLS) follow standard PEFT usage from the library docs.
    # The specific target_modules list and the choice of r=16, alpha=32
    # were validated through EXP3 in train_meditron_HPT1.py (original work).
    # enable_input_require_grads() pattern is from PEFT documentation.
    # =========================================================================
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )

    # enable_input_require_grads() is needed before applying LoRA because by
    # default the input embeddings do not require gradients, and LoRA needs
    # gradients to flow through the entire computation graph.
    base_model.enable_input_require_grads()
    base_model = get_peft_model(base_model, lora_cfg)
    base_model.print_trainable_parameters()  # shows how few params LoRA actually trains

    # Wrap backbone + head into one model
    model = CustomSeqClassifier(
        base_model=base_model,
        num_labels=num_labels,
        hidden_dim=args.cls_hidden_dim,
        dropout=args.cls_dropout,
        pooling=args.pooling,
        dtype=dtype,
    )

    # -------------------------------------------------------------------------
    # Device and precision settings
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = args.fp16 or args.bf16
    autocast_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None)

    # -------------------------------------------------------------------------
    # Evaluation helper
    # -------------------------------------------------------------------------
    def evaluate_loader(loader):
        """
        Run inference on a dataloader and compute all evaluation metrics.

        torch.no_grad() disables gradient computation during eval — we do not
        need gradients for inference and skipping them saves memory and time.

        Softmax converts raw logits to probabilities for ROC-AUC.
        Argmax on logits (not probabilities) gives the predicted class —
        argmax is invariant to softmax so we can do it directly on logits.
        """
        model.eval()
        all_preds, all_gold = [], []
        all_probs = []
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
                logits = outputs.logits  # [B, num_labels]

                total_loss += float(loss.item())
                n_batches += 1

                preds = logits.argmax(dim=-1).detach().cpu().numpy()
                gold = batch["labels"].detach().cpu().numpy()
                # Cast to float32 before softmax — avoids precision issues with fp16 logits
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

        # ROC-AUC — one-vs-rest across all 30 classes using full softmax probabilities
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

    def load_best_model_for_eval():
        """
        Rebuild the model from saved checkpoint files for test evaluation.

        We save the LoRA adapter and custom head separately:
          - lora_adapter/: contains the LoRA weight matrices (tiny, ~100MB)
          - custom_head.pt: contains the MLP head state dict

        Loading on CPU first (low_cpu_mem_usage=True) avoids GPU memory spikes
        from having both the old training model and new eval model on GPU at once.
        We move to GPU only once everything is assembled.
        """
        best_path = os.path.join(args.out_dir, "best_checkpoint")
        best_adapter = os.path.join(best_path, "lora_adapter")
        head_path = os.path.join(best_path, "custom_head.pt")

        if not os.path.isdir(best_adapter):
            raise FileNotFoundError(f"Missing LoRA adapter dir: {best_adapter}")
        if not os.path.isfile(head_path):
            raise FileNotFoundError(f"Missing classifier head file: {head_path}")

        from transformers import AutoModel
        base_eval = AutoModel.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        base_eval.config.pad_token_id = tok.pad_token_id

        # Attach the LoRA adapter weights on top of the fresh backbone
        base_eval = PeftModel.from_pretrained(base_eval, best_adapter)

        m = CustomSeqClassifier(
            base_model=base_eval,
            num_labels=num_labels,
            hidden_dim=args.cls_hidden_dim,
            dropout=args.cls_dropout,
            pooling=args.pooling,
            dtype=dtype,
        )

        # Restore the trained head weights
        state = torch.load(head_path, map_location="cpu")
        m.head.load_state_dict(state)

        m.to(device)
        m.eval()
        return m

    # -------------------------------------------------------------------------
    # Eval-only path
    # -------------------------------------------------------------------------
    if args.eval_only:
        print("EVAL ONLY: loading best checkpoint and evaluating test set...")

        # Delete the training model that was built above before loading the
        # eval model — keeps peak GPU memory under control
        try:
            del model
        except Exception:
            pass
        torch.cuda.empty_cache()

        model = load_best_model_for_eval()
        test_metrics = evaluate_loader(test_loader)

        print("Test:", {k: v for k, v in test_metrics.items() if k not in ("preds", "labels")})

        with open(os.path.join(args.out_dir, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in test_metrics.items() if k not in ("preds", "labels")}, f, indent=2)

        return

    # -------------------------------------------------------------------------
    # Training path
    # -------------------------------------------------------------------------
    # =========================================================================
    # SOURCE ATTRIBUTION - Training loop, optimizer, scheduler, GradScaler
    # =========================================================================
    # ADAPTED FROM: HuggingFace Transformers training documentation
    #   URL: https://huggingface.co/docs/transformers/training
    # AdamW optimizer, cosine LR scheduler with warmup, GradScaler for fp16,
    # and gradient accumulation follow standard HuggingFace training conventions.
    #
    # ORIGINAL CODE (written by Namirah Imtieaz Shaik):
    # - Early stopping monitoring macro F1 with patience and min_delta
    # - Checkpoint saving (LoRA adapter and head saved separately)
    # - GPU memory cleanup before test evaluation
    # - Manual loop design chosen over HuggingFace Trainer because
    #   CustomSeqClassifier is not a standard HuggingFace model
    # =========================================================================
    model.to(device)

    # AdamW with weight decay 0.01 — standard choice for transformer fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Total optimizer steps = (batches per epoch / grad_accum) * epochs
    total_steps = (len(train_loader) // max(1, args.grad_accum)) * args.epochs
    # 5% warmup: ramp LR from 0 to args.lr over the first 5% of steps
    # to avoid large unstable gradients at the start when the head is random
    warmup_steps = int(0.05 * total_steps)

    # Cosine schedule: smoothly decreases LR following a cosine curve to 0
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # GradScaler for fp16 mixed precision — scales loss to prevent underflow
    # in float16 gradients. Only active when --fp16 is passed.
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_path = os.path.join(args.out_dir, "best_checkpoint")
    os.makedirs(best_path, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()  # enable dropout
        running_loss = 0.0
        # set_to_none=True frees gradient memory entirely instead of zeroing —
        # slightly more efficient than optimizer.zero_grad()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_autocast and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    outputs = model(**batch)
                    # Divide loss by grad_accum so the gradient magnitude stays
                    # correct regardless of how many steps we accumulate over
                    loss = outputs.loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum
                loss.backward()

            running_loss += loss.item()

            # Only update weights after accumulating grad_accum gradient steps.
            # This simulates a batch_size * grad_accum effective batch size.
            if step % args.grad_accum == 0:
                if use_autocast and device.type == "cuda":
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

        # ---- Validation at the end of each epoch ----
        val_metrics = evaluate_loader(val_loader)
        auc_str = ""
        if "roc_auc_macro_ovr" in val_metrics:
            auc_str = f" roc_auc_macro_ovr={val_metrics['roc_auc_macro_ovr']:.4f}"
        print(f"[epoch {epoch}] val: "
              f"loss={val_metrics['loss']:.4f} "
              f"acc={val_metrics['accuracy']:.4f} "
              f"f1_macro={val_metrics['f1_macro']:.4f} "
              f"f1_micro={val_metrics['f1_micro']:.4f}"
              f"{auc_str}")

        # ---- Early stopping check ----
        # We monitor macro F1 rather than micro F1 because macro is more sensitive
        # to rare class performance, which makes it a better training signal.
        # The model with the best validation macro F1 is saved as the checkpoint.
        improved = val_metrics["f1_macro"] > (best_val_f1 + args.early_stop_min_delta)
        if improved:
            best_val_f1 = val_metrics["f1_macro"]
            epochs_no_improve = 0

            # Save LoRA adapter and head separately:
            # - save_pretrained on the PEFT model only saves the adapter weights (small)
            # - torch.save on the head saves just the MLP state dict
            # This way we do not need to save the full 7B backbone weights
            adapter_dir = os.path.join(best_path, "lora_adapter")
            os.makedirs(adapter_dir, exist_ok=True)
            model.base.save_pretrained(adapter_dir)
            torch.save(model.head.state_dict(), os.path.join(best_path, "custom_head.pt"))
            tok.save_pretrained(best_path)
            with open(os.path.join(best_path, "label_map.json"), "w", encoding="utf-8") as f:
                json.dump({"id2label": {int(i): lab for i, lab in id2label.items()},
                           "label2id": label2id}, f, indent=2)
            print(f"  ✓ saved new best checkpoint to {best_path} (f1_macro={best_val_f1:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  ↳ no improvement (+{epochs_no_improve}/{args.early_stop_patience})")
            if epochs_no_improve >= args.early_stop_patience:
                print(" Early stopping triggered.")
                break

    # -------------------------------------------------------------------------
    # Final test evaluation
    # -------------------------------------------------------------------------
    # Free all training state from GPU before loading the best checkpoint.
    # The training model, optimizer, scheduler, and scaler all hold GPU memory.
    # Releasing them first ensures the eval model can load cleanly.
    try:
        model.to("cpu")
    except Exception:
        pass
    for obj in ("optimizer", "scheduler", "scaler"):
        if obj in locals():
            del locals()[obj]
    del model
    torch.cuda.empty_cache()

    best_adapter = os.path.join(best_path, "lora_adapter")
    if os.path.isdir(best_adapter):
        print(f"Reloading best adapter from: {best_adapter}")

        # Load fresh backbone on CPU to avoid GPU peak memory from having
        # both old and new models in GPU memory simultaneously
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            # do NOT use device_map="auto" here unless using Accelerate —
            # we want CPU load first, then a single .to(device) call
        )
        base_model.config.pad_token_id = tok.pad_token_id

        # Re-attach the saved LoRA adapter weights
        base_model = PeftModel.from_pretrained(base_model, best_adapter)

        # Rebuild the full classifier with the same head config as training
        model = CustomSeqClassifier(
            base_model=base_model,
            num_labels=num_labels,
            hidden_dim=args.cls_hidden_dim,
            dropout=args.cls_dropout,
            pooling=args.pooling,
            dtype=dtype,
        )

        # Restore the trained head weights
        head_path = os.path.join(best_path, "custom_head.pt")
        state = torch.load(head_path, map_location="cpu")
        model.head.load_state_dict(state)

        # Move the assembled model to GPU all at once
        model.to(device)
        model.eval()
    else:
        print(" No best adapter found, using in-memory model.")
        model.eval()

    test_metrics = evaluate_loader(test_loader)
    print("Test:", {k: v for k, v in test_metrics.items() if k not in ("preds", "labels")})

    # Save scalar metrics to JSON (exclude numpy arrays preds/labels)
    with open(os.path.join(args.out_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in test_metrics.items() if k not in ("preds", "labels")}, f, indent=2)

    # ---- Per-label breakdown ----
    # average=None gives per-class arrays instead of one aggregated number
    y_true = test_metrics["labels"]
    y_pred = test_metrics["preds"]

    prec, rec, f1s, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), average=None, zero_division=0
    )

    # Write a CSV with one row per ICD code so it is easy to inspect in Excel
    per_label_csv = os.path.join(args.out_dir, "per_label_f1_test.csv")
    with open(per_label_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label_index", "icd_code", "precision", "recall", "f1", "support"])
        for i in range(len(labels)):
            writer.writerow([i, id2label[i], f"{prec[i]:.6f}", f"{rec[i]:.6f}", f"{f1s[i]:.6f}", int(supp[i])])

    # Also save the full sklearn classification_report as a text file
    report_txt = classification_report(
        y_true, y_pred,
        labels=list(range(len(labels))),
        target_names=[id2label[i] for i in range(len(labels))],
        zero_division=0
    )
    with open(os.path.join(args.out_dir, "per_label_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(f"✓ Saved per-label metrics to {per_label_csv}")


if __name__ == "__main__":
    main()