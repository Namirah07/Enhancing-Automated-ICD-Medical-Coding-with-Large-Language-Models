# train_biomistral_cls.py
#
# =============================================================================
# SOURCE ATTRIBUTION - FILE LEVEL
# =============================================================================
# ORIGINAL CODE: Written by Namirah Imtieaz Shaik
#
# This script is original work, structurally identical to train_meditron_cls.py
# with BioMistral-7B as the backbone instead of Meditron-7B.
# The same library component attributions apply as in train_meditron_cls.py:
#
#   PEFT / LoRA adapter usage:
#     Adapted from: PEFT library documentation and QLoRA paper
#     (Dettmers et al., NeurIPS 2023)
#     URL: https://github.com/huggingface/peft
#
#   MeanPooler class:
#     Standard masked mean pooling pattern from HuggingFace community.
#     Reference: https://huggingface.co/blog/how-to-train-sentence-transformers
#
#   Manual training loop structure:
#     Adapted from: HuggingFace Transformers training documentation
#     URL: https://huggingface.co/docs/transformers/training
#
# WHAT IS ENTIRELY ORIGINAL (written by Namirah Imtieaz Shaik):
#   - CustomClassifierHead: two-layer MLP head for ICD-10 classification
#   - CustomSeqClassifier: AutoModel backbone + pooling + MLP head wrapper
#   - Explicit gc.collect() cleanup steps (more aggressive than Meditron
#     version due to BioMistral's larger residual GPU allocations)
#   - load_best_model_for_eval() with device_map=None CPU-first loading
#   - Early stopping on macro F1
#   - Checkpoint saving strategy (LoRA adapter + head separately)
# =============================================================================
#
# Main training script for the BioMistral discriminative classifier (BioMistral-D).
#
# This is structurally identical to train_meditron_cls.py — same LoRA setup,
# same custom MLP head, same manual training loop. The only meaningful differences
# are the backbone (BioMistral-7B instead of Meditron-7B) and slightly more
# aggressive GPU memory management via explicit gc.collect() calls throughout.
#
# BioMistral-7B was created by fine-tuning Mistral-7B on PubMed Central and
# other biomedical text corpora. It uses a Mistral architecture rather than
# Llama-2, which is why it tends to use a bit more memory at 1024 tokens
# and why we added the extra cleanup steps.
#
# Architecture summary:
#   BioMistral backbone (frozen) + LoRA adapters (trainable)
#   → mean pooling over all token hidden states
#   → Dropout → Linear(4096→1536) → GELU → Dropout → Linear(1536→30)
#   → CrossEntropyLoss

import os, json, argparse, warnings
from typing import Dict, List, Tuple
import csv
import gc

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_auc_score

from datasets import load_dataset, Features, Value
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
    set_seed,
)

from peft import LoraConfig, get_peft_model, PeftModel

warnings.filterwarnings("ignore", category=UserWarning)


def build_label_maps(path: str) -> Tuple[List[str], Dict[int, str], Dict[str, int]]:
    """
    Read label_vocab.txt and build integer <-> ICD code mappings.

    Labels are sorted alphabetically so the class index ordering is identical
    across all three discriminative model scripts. Class 0 always maps to the
    same ICD code regardless of which script is running.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="BioMistral/BioMistral-7B")
    parser.add_argument("--train_csv", type=str, default=r".\shared\train_full.csv")
    parser.add_argument("--dev_csv",   type=str, default=r".\shared\dev_full.csv")
    parser.add_argument("--test_csv",  type=str, default=r".\shared\test_full.csv")
    parser.add_argument("--label_vocab", type=str, default=r".\shared\label_vocab.txt")

    parser.add_argument("--out_dir", type=str, default=r".\output\biomistral_cls_lora_custom")
    # 1024 tokens is the max BioMistral handles without quantization
    parser.add_argument("--max_length", type=int, default=1024)
    # batch_size=1 + grad_accum=8 gives effective batch size of 8
    # without running out of memory for a 7B model at 1024 tokens
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    # cls_hidden_dim is the intermediate layer size in the two-layer MLP head
    parser.add_argument("--cls_hidden_dim", type=int, default=1536,
                        help="Hidden size in the custom MLP head (e.g., 1024-2048)")
    parser.add_argument("--cls_dropout", type=float, default=0.10,
                        help="Dropout used in the custom head")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "last"],
                        help="Pooling strategy to get a single document vector from token states")

    # Early stopping monitors macro F1 on the validation set.
    # Macro is more sensitive to rare class performance than micro,
    # making it a better training signal across all 30 ICD codes.
    parser.add_argument("--early_stop_patience", type=int, default=2,
                        help="Stop if val F1-macro does not improve for this many epochs")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                        help="Minimum improvement in F1-macro to count as progress")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training and only evaluate using existing best_checkpoint in --out_dir")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    import torch.nn as nn
    from types import SimpleNamespace
    from transformers import AutoModel

    # -------------------------------------------------------------------------
    # Custom model components
    # -------------------------------------------------------------------------

    # =========================================================================
    # SOURCE ATTRIBUTION - MeanPooler
    # =========================================================================
    # ADAPTED FROM: Standard masked mean pooling pattern in the HuggingFace
    #   community. Reference:
    #   https://huggingface.co/blog/how-to-train-sentence-transformers
    # Implementation written by Namirah Imtieaz Shaik following this pattern.
    # =========================================================================
    class MeanPooler(nn.Module):
        """
        Average all non-padding token hidden states into one document vector.

        Multiplying by the attention mask zeros out padding positions before
        summing, so padding tokens do not affect the result. The clamp on the
        denominator is just a safety guard against a zero-length sequence,
        which should not happen in practice.
        """
        def forward(self, last_hidden_state, attention_mask):
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, T, 1]
            summed = (last_hidden_state * mask).sum(dim=1)                   # [B, H]
            denom = mask.sum(dim=1).clamp(min=1e-6)                          # [B, 1]
            return summed / denom                                             # [B, H]

    # =========================================================================
    # SOURCE ATTRIBUTION - CustomClassifierHead
    # =========================================================================
    # ORIGINAL CODE: Written by Namirah Imtieaz Shaik
    # Two-layer MLP head designed and validated through hyperparameter tuning
    # experiments (EXP1/EXP2) in train_biomistral_HPT1.py.
    # =========================================================================
    class CustomClassifierHead(nn.Module):
        """
        Two-layer MLP that maps the pooled document vector to 30 class logits.

        Architecture: Dropout -> Linear(H->hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim->30)

        The hyperparameter tuning experiments showed L2 (two layers) and L1 (one layer)
        perform comparably, so we keep two layers as a mild non-linear transformation
        between the backbone representation space and the ICD label space.
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
    # Wrapper combining AutoModel backbone + pooling + MLP head is entirely
    # original. The use of AutoModel instead of AutoModelForCausalLM and the
    # SimpleNamespace return object are original design decisions.
    # =========================================================================
    class CustomSeqClassifier(nn.Module):
        """
        Full BioMistral-D model: LoRA-adapted backbone + mean pooling + MLP head.

        Returns a SimpleNamespace with .logits and .loss so the training loop can
        use outputs.logits and outputs.loss just like a HuggingFace model output.
        """
        def __init__(self, base_model, num_labels, hidden_dim, dropout=0.1, pooling="mean", dtype=None):
            super().__init__()
            self.base = base_model          # LoRA-wrapped BioMistral backbone
            self.num_labels = num_labels
            self.hidden_size = base_model.config.hidden_size  # 4096 for BioMistral-7B
            self.pooling = pooling
            self.pool = MeanPooler() if pooling == "mean" else None
            self.head = CustomClassifierHead(self.hidden_size, hidden_dim, num_labels, dropout)
            self.criterion = nn.CrossEntropyLoss()
            self.dtype = dtype

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            # Get token-level hidden states from the backbone
            out = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
            last_hidden = out.last_hidden_state  # [B, T, H]

            if self.pooling == "mean":
                pooled = self.pool(last_hidden, attention_mask)       # [B, H]
            else:
                # Last-token pooling: find the last real token (not padding)
                lengths = attention_mask.sum(dim=1) - 1               # [B]
                pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), lengths]

            logits = self.head(pooled)                                 # [B, 30]
            loss = None
            if labels is not None:
                if labels.dtype != torch.long:
                    labels = labels.long()
                loss = self.criterion(logits, labels)
            return SimpleNamespace(logits=logits, loss=loss)

    # -------------------------------------------------------------------------
    # Labels and dataset
    # -------------------------------------------------------------------------
    labels, id2label, label2id = build_label_maps(args.label_vocab)
    num_labels = len(labels)  # 30 for our MIMIC-IV benchmark
    if num_labels < 2:
        raise ValueError("label_vocab must contain at least 2 labels.")

    # Force TEXT and LABELS to strings so the CSV loader does not try to
    # infer types and accidentally cast an ICD code like "E11" to float
    features = Features({"TEXT": Value("string"), "LABELS": Value("string")})
    ds = load_dataset(
        "csv",
        data_files={"train": args.train_csv, "validation": args.dev_csv, "test": args.test_csv},
        features=features,
    )
    # Drop any stray columns that might be in the CSV beyond TEXT and LABELS
    keep = {"TEXT", "LABELS"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    def clean_and_map(example):
        """Map ICD code strings to integer indices, mark unknown codes as None for filtering."""
        txt = example.get("TEXT") or ""
        lab = example.get("LABELS")
        if lab is None:
            return {"TEXT": txt, "LABELS": None, "label": None}
        lab = str(lab).strip()
        if lab not in label2id:
            return {"TEXT": txt, "LABELS": None, "label": None}
        return {"TEXT": txt, "LABELS": lab, "label": label2id[lab]}

    ds = ds.map(clean_and_map)
    for split in ["train", "validation", "test"]:
        ds[split] = ds[split].filter(lambda ex: ex["LABELS"] is not None and ex["label"] is not None)

    # -------------------------------------------------------------------------
    # Tokenizer and dataloaders
    # -------------------------------------------------------------------------
    # BioMistral uses the Mistral tokenizer, which like Llama has no pad token.
    # Setting pad_token = eos_token is the standard fix.
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize_fn(batch):
        # padding=False here because we pad dynamically per batch in the collator
        enc = tok(
            batch["TEXT"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        enc["labels"] = batch["label"]
        return enc

    keep_after = {"input_ids", "attention_mask", "labels", "token_type_ids"}
    cols_to_drop = [c for c in ds["train"].column_names if c not in keep_after]
    ds = ds.map(tokenize_fn, batched=True, remove_columns=cols_to_drop)

    # Dynamic padding per batch + alignment to multiples of 8 for GPU efficiency
    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(ds["validation"], batch_size=args.batch_size, shuffle=False,
                              collate_fn=collator, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(ds["test"], batch_size=args.batch_size, shuffle=False,
                              collate_fn=collator, num_workers=2, pin_memory=True)

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_autocast = args.fp16 or args.bf16
    autocast_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None)

    # -------------------------------------------------------------------------
    # Evaluation helper
    # -------------------------------------------------------------------------
    def evaluate_loader(model, loader):
        """
        Run inference on a dataloader and compute all evaluation metrics.

        model is passed as an explicit argument so this function works correctly
        when called from eval-only mode with a freshly loaded checkpoint.
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
                # Cast to float32 before softmax to avoid precision issues with fp16 logits
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

        # ROC-AUC uses the full softmax probability matrix rather than hard predictions
        auc_macro_ovr = None
        auc_weighted_ovr = None
        try:
            auc_macro_ovr = roc_auc_score(
                all_gold, all_probs,
                labels=list(range(num_labels)),
                multi_class="ovr",
                average="macro",
            )
            auc_weighted_ovr = roc_auc_score(
                all_gold, all_probs,
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

    # -------------------------------------------------------------------------
    # Checkpoint loading for eval-only mode
    # -------------------------------------------------------------------------
    def load_best_model_for_eval():
        """
        Reconstruct the model from saved checkpoint files.

        We load the backbone on CPU first (device_map=None, low_cpu_mem_usage=True)
        to avoid having both the training model and the eval model on GPU at once.
        Everything is moved to GPU in one shot once fully assembled.

        gradient_checkpointing and use_cache=False are re-applied here too for
        consistency with training — they do not affect inference results but
        suppress warnings from BioMistral's architecture.
        """
        best_path = os.path.join(args.out_dir, "best_checkpoint")
        adapter_dir = os.path.join(best_path, "lora_adapter")
        head_path = os.path.join(best_path, "custom_head.pt")

        if not os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
            raise FileNotFoundError(f"Missing LoRA adapter in {adapter_dir}. Did you train first?")
        if not os.path.isfile(head_path):
            raise FileNotFoundError(f"Missing head weights: {head_path}. Did you train first?")

        base_eval = AutoModel.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None,  # load on CPU first, move to GPU once assembled
        )
        base_eval.config.pad_token_id = tok.pad_token_id
        base_eval.config.use_cache = False
        base_eval.gradient_checkpointing_enable()

        # Attach the saved LoRA adapter weights onto the fresh backbone
        base_eval = PeftModel.from_pretrained(base_eval, adapter_dir)

        m = CustomSeqClassifier(
            base_model=base_eval,
            num_labels=num_labels,
            hidden_dim=args.cls_hidden_dim,
            dropout=args.cls_dropout,
            pooling=args.pooling,
            dtype=dtype,
        )

        # Restore the trained MLP head weights
        state = torch.load(head_path, map_location="cpu")
        m.head.load_state_dict(state)

        m.to(device)
        m.eval()
        return m

    # -------------------------------------------------------------------------
    # Eval-only path
    # -------------------------------------------------------------------------
    if args.eval_only:
        # Clean up any residual memory from the setup code above before loading
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
    # Training path
    # -------------------------------------------------------------------------
    _ = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )

    base_model = AutoModel.from_pretrained(args.model_name, torch_dtype=dtype)
    base_model.config.pad_token_id = tok.pad_token_id

    # Gradient checkpointing: recomputes activations during backward instead of
    # storing them, saving significant GPU memory. use_cache=False must accompany it.
    base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False

    print("hidden_size:", base_model.config.hidden_size)
    print("num_attention_heads:", base_model.config.num_attention_heads)
    print("num_hidden_layers:", base_model.config.num_hidden_layers)

    # =========================================================================
    # SOURCE ATTRIBUTION - LoRA configuration and application
    # =========================================================================
    # ADAPTED FROM: PEFT library documentation and QLoRA paper
    #   (Dettmers et al., NeurIPS 2023)
    #   URL: https://github.com/huggingface/peft
    #   URL: https://arxiv.org/abs/2305.14314
    # LoraConfig parameters and get_peft_model usage follow PEFT library docs.
    # Specific target_modules and r/alpha values validated through EXP3 in
    # train_biomistral_HPT1.py (original work by Namirah Imtieaz Shaik).
    # =========================================================================
    # Inject LoRA adapters into all seven projection matrix types.
    # Only these adapter weights (~0.1% of total params) are trained.
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,  # effective scale = alpha/r = 2.0
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    # Required so gradients can flow back through the frozen embedding layer
    # to reach the LoRA adapter weight matrices
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
    # AdamW optimizer, cosine LR scheduler with warmup, GradScaler for fp16,
    # and gradient accumulation follow standard HuggingFace training conventions.
    #
    # ORIGINAL CODE (written by Namirah Imtieaz Shaik):
    # - Early stopping logic monitoring macro F1 with patience and min_delta
    # - Explicit gc.collect() cleanup (more aggressive than Meditron version
    #   due to BioMistral leaving larger residual GPU allocations)
    # - Checkpoint saving: LoRA adapter and MLP head saved separately
    # - load_best_model_for_eval() with CPU-first loading (device_map=None)
    # =========================================================================
    # Only include parameters with requires_grad=True — frozen backbone weights
    # have requires_grad=False so no optimizer state is wasted on them
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )
    total_steps = (len(train_loader) // max(1, args.grad_accum)) * args.epochs
    warmup_steps = int(0.05 * total_steps)  # 5% warmup before cosine decay
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

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
                    loss = outputs.loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum
                loss.backward()

            running_loss += float(loss.item())

            # Update weights only after accumulating grad_accum gradient steps
            if step % args.grad_accum == 0:
                if args.fp16 and device.type == "cuda":
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

        # Save checkpoint whenever macro F1 genuinely improves
        improved = val_metrics["f1_macro"] > (best_val_f1 + args.early_stop_min_delta)
        if improved:
            best_val_f1 = val_metrics["f1_macro"]
            epochs_no_improve = 0

            # Save only the adapter (~100 MB) and the head, not the full backbone
            adapter_dir = os.path.join(best_path, "lora_adapter")
            os.makedirs(adapter_dir, exist_ok=True)
            model.base.save_pretrained(adapter_dir)
            torch.save(model.head.state_dict(), os.path.join(best_path, "custom_head.pt"))
            tok.save_pretrained(best_path)
            with open(os.path.join(best_path, "label_map.json"), "w", encoding="utf-8") as f:
                json.dump({"id2label": {int(i): lab for i, lab in id2label.items()},
                           "label2id": label2id}, f, indent=2)

            print(f"  saved new best checkpoint to {best_path} (f1_macro={best_val_f1:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  no improvement (+{epochs_no_improve}/{args.early_stop_patience})")
            if epochs_no_improve >= args.early_stop_patience:
                print(" Early stopping triggered.")
                break

    # -------------------------------------------------------------------------
    # Cleanup training state before test evaluation
    # -------------------------------------------------------------------------
    # BioMistral leaves larger residual GPU allocations than Meditron, so
    # explicit gc.collect() + empty_cache() here matters more than in the
    # Meditron script. Without this the checkpoint reload below can hit OOM.
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
    # Reload best checkpoint and run final test evaluation
    # -------------------------------------------------------------------------
    best_adapter = os.path.join(best_path, "lora_adapter")
    head_path = os.path.join(best_path, "custom_head.pt")

    if os.path.isdir(best_adapter):
        print(f"Reloading best adapter from: {best_adapter}")

        # Load backbone on CPU first to avoid GPU memory spike from having
        # two large models on GPU simultaneously
        base_model = AutoModel.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        base_model.config.pad_token_id = tok.pad_token_id
        base_model.config.use_cache = False
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

        # Move the fully assembled model to GPU in one shot
        model.to(device)
        model.eval()
    else:
        raise FileNotFoundError(f"No best adapter found at {best_adapter}")

    test_metrics = evaluate_loader(model, test_loader)
    print("Test:", {k: v for k, v in test_metrics.items() if k not in ("preds", "labels")})

    with open(os.path.join(args.out_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in test_metrics.items() if k not in ("preds", "labels")}, f, indent=2)

    # Per-label breakdown - one row per ICD code
    y_true = test_metrics["labels"]
    y_pred = test_metrics["preds"]

    prec, rec, f1s, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), average=None, zero_division=0
    )

    per_label_csv = os.path.join(args.out_dir, "per_label_f1_test.csv")
    with open(per_label_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label_index", "icd_code", "precision", "recall", "f1", "support"])
        for i in range(len(labels)):
            writer.writerow([i, id2label[i], f"{prec[i]:.6f}", f"{rec[i]:.6f}", f"{f1s[i]:.6f}", int(supp[i])])

    report_txt = classification_report(
        y_true, y_pred,
        labels=list(range(len(labels))),
        target_names=[id2label[i] for i in range(len(labels))],
        zero_division=0
    )
    with open(os.path.join(args.out_dir, "per_label_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(f"Saved per-label metrics to {per_label_csv}")


if __name__ == "__main__":
    main()