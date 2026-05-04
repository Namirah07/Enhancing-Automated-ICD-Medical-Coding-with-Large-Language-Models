# app.py
#
# Gradio web application for interactive ICD-10 prediction and SHAP explanation.
#
# This is the UI demo shown during the thesis defense. It loads any of the six
# trained models (three PLM-ICD deep learning models and three discriminative LLMs),
# runs prediction on five preloaded clinical notes, and generates token-level SHAP
# explanations for any predicted ICD code the user selects.
#
# The application has two main actions:
#
#   1. Predict top-30: loads the selected model, runs it on the selected note,
#      and displays the top 30 ICD codes ranked by probability in a table and
#      a horizontal bar chart.
#
#   2. Explain selected ICD (SHAP): runs SHAP partition explainer on the selected
#      note with the selected ICD code as the output dimension, and renders the
#      resulting token attribution as a force plot inside an iframe.
#
# Memory management is a central concern in this application. The six models
# range from 110M (BERT) to 8B (OpenBioLLM) parameters and cannot all live in
# GPU memory simultaneously. The application enforces a one-model-at-a-time policy:
# when the user switches to a different model, all cached models and explainers
# are evicted from GPU memory before the new model is loaded.
#
# File layout expected by this application:
#   ../PLM-ICD/output/         - PLM-ICD model checkpoints
#   ../PLM-ICD/src/            - PLM-ICD custom modeling files (modeling_bert.py etc.)
#   ../PLM-ICD/data/           - label_vocab.txt for PLM-ICD models
#   ../Discriminative_Models/  - discriminative LLM checkpoints
#   ./notes.json               - five preloaded clinical notes with title, text, label
#   ./icd_short_titles.json    - mapping from ICD codes to short human-readable titles

import os
import sys
import gc
import json
import html
import base64
import logging
import matplotlib

# Use non-interactive backend - required for servers without a display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gradio as gr
import shap
import importlib.util

from typing import Dict, List, Optional, Tuple
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# Disable tokenizer parallelism warnings in the Gradio worker threads
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Suppress safetensors conversion messages for older checkpoint formats
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

# ---- Compatibility patch for old PLM-ICD models vs new transformers ----
# PLM-ICD checkpoints were saved with an older version of HuggingFace transformers.
# Newer versions of transformers call mark_tied_weights_as_initialized() during
# from_pretrained() and expect the model to have an all_tied_weights_keys attribute.
# PLM-ICD custom models do not have this attribute, so we patch the method to add
# a safe fallback before it tries to access the attribute.
try:
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "_patched_mark_tied_weights_as_initialized"):
        _orig_mark = PreTrainedModel.mark_tied_weights_as_initialized

        def _patched_mark_tied_weights_as_initialized(self, loading_info=None):
            if not hasattr(self, "all_tied_weights_keys"):
                old = getattr(self, "_tied_weights_keys", None)
                if old is None:
                    old = []
                # Build the expected dict format from the old list format
                self.all_tied_weights_keys = {k: None for k in old}
            return _orig_mark(self, loading_info)

        PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark_tied_weights_as_initialized
        PreTrainedModel._patched_mark_tied_weights_as_initialized = True
except Exception as e:
    print("[WARN] Could not apply transformers tied-weights patch:", repr(e))


# ---- Path configuration ----
# All paths are resolved relative to this file so the app works regardless
# of the working directory it is launched from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PLM_DIR = os.path.join(PROJECT_ROOT, "PLM-ICD", "output")
PLM_DATA_DIR = os.path.join(PROJECT_ROOT, "PLM-ICD", "data")
PLM_SRC_DIR = os.path.join(PROJECT_ROOT, "PLM-ICD", "src")
PLM_LABEL_VOCAB = os.path.join(PLM_DATA_DIR, "label_vocab.txt")

# PLM-ICD model checkpoint directories - one per backbone
PLM_MODELS = {
    "BERT (PLM-ICD)": os.path.join(PLM_DIR, "bert_single_label"),
    "Longformer (PLM-ICD)": os.path.join(PLM_DIR, "longformer_single_label"),
    "RoBERTa (PLM-ICD)": os.path.join(PLM_DIR, "roberta_single_label"),
}

DISC_DIR = os.path.join(PROJECT_ROOT, "Discriminative_Models")

# Discriminative LLM model configs - each entry specifies the checkpoint directory,
# the HuggingFace base model name for loading weights, and the pooling strategy.
# The pooling strategy must match what was used during training.
DISC_MODELS = {
    "Meditron (discriminative)": {
        "checkpoint_dir": os.path.join(DISC_DIR, "meditron", "final_best_seed42", "best_checkpoint"),
        "base_model_name": "epfl-llm/meditron-7b",
        "pooling": "mean",
    },
    "BioMistral (discriminative)": {
        "checkpoint_dir": os.path.join(DISC_DIR, "biomistral", "final_biomistral_seed42", "best_checkpoint"),
        "base_model_name": "BioMistral/BioMistral-7B",
        "pooling": "last",
    },
    "OpenBioLLM (discriminative)": {
        "checkpoint_dir": os.path.join(DISC_DIR, "openbiollm", "final_openbiollm_seed42", "best_checkpoint"),
        "base_model_name": "aaditya/Llama3-OpenBioLLM-8B",
        "pooling": "mean",
    },
}

NOTES_JSON = os.path.join(os.path.dirname(__file__), "notes.json")
ICD_SHORT_TITLES_JSON = os.path.join(os.path.dirname(__file__), "icd_short_titles.json")

# ---- Sequence length and chunking constants ----
# PLM-ICD models process notes in chunks of 256 tokens, with 2 chunks = 512 total.
# Discriminative LLMs use a flat 512-token sequence with head+tail cropping.
MAX_LEN_PLM = 512
MAX_LEN_LLM = 512

CHUNK_SIZE = 256
assert MAX_LEN_PLM % CHUNK_SIZE == 0
NUM_CHUNKS = MAX_LEN_PLM // CHUNK_SIZE

TOPK = 30                  # number of top ICD predictions to display
DISC_TEXT_HEAD_FRAC = 0.40 # fraction of token budget taken from the note beginning

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fp16 on GPU for memory efficiency; fp32 on CPU as fp16 is not well supported there
DISC_DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

# Max SHAP evaluations - controls the number of perturbations per token.
# Higher = more accurate but slower. 300 gives a good quality/speed tradeoff.
SHAP_MAX_EVALS_PLM = 300
SHAP_MAX_EVALS_LLM = 300


# ---- GPU memory utilities ----

def free_gpu_memory():
    """
    Run Python garbage collection and clear the CUDA memory caches.

    We call both empty_cache() and ipc_collect() - empty_cache releases cached
    but currently unused memory back to the OS, while ipc_collect() handles
    inter-process CUDA memory that may have been allocated by the tokenizer
    workers. Both are wrapped in try/except because they are not available on
    all platforms and CUDA configurations.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def move_module_to_cpu_if_possible(obj):
    """
    Move a model or module to CPU if it has a .cpu() method.

    Called before deleting a model object to ensure the GPU memory is freed
    promptly rather than waiting for Python's garbage collector, which does
    not always release CUDA tensors in time for the next model to load.
    """
    try:
        if hasattr(obj, "cpu"):
            obj.cpu()
    except Exception:
        pass


def clear_all_cached_models():
    """
    Evict all cached model wrappers and SHAP explainers from GPU memory.

    Called when the user switches to a different model. We move each model's
    sub-modules to CPU before clearing the cache dict so the GPU memory is
    released before we try to load the new model.

    The DiscWrapper stores its model as self.model which contains self.base
    (the LLM backbone) and self.head (the MLP classifier). The PLMWrapper
    stores its model directly as self.model. We try to move all of these
    to CPU regardless of which type of wrapper it is.
    """
    global MODEL_CACHE, EXPLAINER_CACHE, ACTIVE_MODEL_CHOICE

    for _, w in list(MODEL_CACHE.items()):
        try:
            if hasattr(w, "model"):
                move_module_to_cpu_if_possible(w.model)
            if hasattr(w, "base"):
                move_module_to_cpu_if_possible(w.base)
            if hasattr(w, "head"):
                move_module_to_cpu_if_possible(w.head)
        except Exception:
            pass

    MODEL_CACHE.clear()
    EXPLAINER_CACHE.clear()
    ACTIVE_MODEL_CHOICE = None
    free_gpu_memory()


def ensure_only_one_active_model(model_choice: str):
    """
    Enforce the one-model-at-a-time memory policy.

    If the requested model is the same as the currently active one, do nothing.
    If it is different, evict all cached models before the new one loads.
    This prevents two large LLMs from being in GPU memory simultaneously.
    """
    global ACTIVE_MODEL_CHOICE
    if ACTIVE_MODEL_CHOICE is None:
        ACTIVE_MODEL_CHOICE = model_choice
        return
    if ACTIVE_MODEL_CHOICE != model_choice:
        clear_all_cached_models()
        ACTIVE_MODEL_CHOICE = model_choice


def model_from_pretrained_compat(model_name: str, dtype: torch.dtype, **kwargs):
    """
    Load a HuggingFace model with the correct dtype keyword for the installed version.

    Older versions of transformers accept dtype=... while newer versions require
    torch_dtype=... . We try the newer form first and fall back to the older one
    if it raises a TypeError, so the app works with a range of transformers versions.
    """
    try:
        return AutoModel.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModel.from_pretrained(model_name, torch_dtype=dtype, **kwargs)


# Meditron-7B was pretrained with a vocabulary of exactly 32,000 tokens.
# If the checkpoint was saved with a different vocab size we need to reset it
# to this value before loading to avoid an embedding size mismatch.
MEDITRON_BASE_VOCAB = 32000


def load_notes() -> List[dict]:
    """
    Load the five preloaded clinical notes from notes.json.

    Each note is a dict with at least "title", "text", and "label" keys.
    The title is shown in the dropdown, the text is fed to the model,
    and the label is displayed alongside predictions as the ground truth.
    """
    with open(NOTES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label_vocab(path: str) -> Dict[int, str]:
    """
    Read label_vocab.txt and build an integer-to-ICD-code mapping.

    Labels are read in file order (not alphabetical) for PLM-ICD models
    because those models were trained with file-order class indices.
    The discriminative models use their own label_map.json from the checkpoint.
    """
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            code = line.strip()
            if code:
                labels.append(code)
    return {i: code for i, code in enumerate(labels)}


_ICD_SHORT_TITLE_CACHE: Optional[Dict[str, str]] = None


def load_icd_short_titles() -> Dict[str, str]:
    """
    Load the ICD code to short description mapping from icd_short_titles.json.

    The result is cached in a module-level variable after the first load so
    subsequent calls do not re-read the file. The cache is never invalidated
    because the ICD description file does not change during a session.
    """
    global _ICD_SHORT_TITLE_CACHE
    if _ICD_SHORT_TITLE_CACHE is not None:
        return _ICD_SHORT_TITLE_CACHE
    out: Dict[str, str] = {}
    if os.path.isfile(ICD_SHORT_TITLES_JSON):
        try:
            with open(ICD_SHORT_TITLES_JSON, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                out = {str(k).strip(): str(v).strip() for k, v in raw.items() if str(k).strip()}
        except Exception as e:
            print("[WARN] Could not load ICD short titles:", repr(e))
    _ICD_SHORT_TITLE_CACHE = out
    return out


def short_title_for_icd(code: str) -> str:
    """
    Look up the short human-readable title for an ICD code.

    Returns a dash if the code is not found in the short titles file,
    which lets the UI display something sensible even for codes that were
    not included in the title mapping.
    """
    t = load_icd_short_titles().get(code.strip())
    return t if t else "-"


def build_topk_probability_figure(codes: List[str], probs: List[float], short_titles: List[str]):
    """
    Build a horizontal bar chart showing the top-30 ICD code probabilities.

    Bar color encodes probability magnitude: dark blue for p >= 0.1 (high
    confidence), medium blue for p >= 0.01 (moderate), and light blue below
    that (low confidence). This gives the user an immediate visual sense of
    how confident the model is without needing to read the numeric values.

    We dynamically scale the figure height based on the number of codes so
    the bars are always readable regardless of how many predictions are shown.
    """
    if not codes or not probs:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No predictions", ha="center", va="center")
        ax.axis("off")
        return fig
    n = len(codes)

    # Truncate long labels to keep the chart readable
    labels: List[str] = []
    for c, t in zip(codes, short_titles):
        if t and t != "-":
            line = f"{c}  {t}"
        else:
            line = c
        if len(line) > 60:
            line = line[:57] + "..."
        labels.append(line)

    fig_h = max(4.0, min(24.0, 0.38 * n + 1.5))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8f9fa")

    y = np.arange(n)
    # Three-tier color coding based on probability thresholds
    colors = ["#1a6eb5" if p >= 0.1 else "#5ba3d9" if p >= 0.01 else "#a8cde8" for p in probs]
    bars = ax.barh(y, probs, color=colors, height=0.6, edgecolor="white", linewidth=0.5)

    # Print numeric probability next to each bar
    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{p:.3f}", va="center", ha="left", fontsize=7.5, color="#444"
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()  # highest probability at the top
    ax.set_xlabel("Probability", fontsize=10, color="#555")
    ax.set_title("Top-30 predicted ICD codes", fontsize=12, fontweight="bold", pad=12, color="#222")
    pmax = max(probs) if probs else 1.0
    ax.set_xlim(0, min(pmax * 1.18, 1.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", linestyle="--", alpha=0.4, color="#ccc")
    ax.axvline(x=0, color="#aaa", linewidth=0.8)
    fig.tight_layout(pad=1.5)
    return fig


def load_disc_label_map(checkpoint_dir: str) -> Dict[int, str]:
    """
    Load the id2label mapping from the checkpoint's label_map.json file.

    The discriminative models save their label mapping at training time so
    the class indices in the saved head weights correspond to the correct
    ICD codes at inference time, regardless of the order in label_vocab.txt.
    """
    p = os.path.join(checkpoint_dir, "label_map.json")
    with open(p, "r", encoding="utf-8") as f:
        lm = json.load(f)
    return {int(k): v for k, v in lm["id2label"].items()}


def topk_indices(probs: np.ndarray, k: int) -> List[int]:
    """
    Return the indices of the top k probabilities in descending order.

    np.argsort returns ascending order so we take the last k elements
    and reverse them with [::-1] to get descending order.
    """
    return np.argsort(probs)[-k:][::-1].tolist()


def ensure_list_of_str(texts) -> List[str]:
    """
    Normalize any text input into a list of strings.

    SHAP and Gradio can pass text inputs in many forms: a single string,
    a list, a tuple, or a numpy array. This function converts all of them
    to a plain Python list of strings so the model wrappers can always
    iterate over a consistent input type.
    """
    if texts is None:
        return [""]
    if isinstance(texts, str):
        return [texts]
    if isinstance(texts, (list, tuple)):
        return [t if isinstance(t, str) else str(t) for t in texts]
    if hasattr(texts, "tolist"):
        flat = texts.tolist()
        if isinstance(flat, list):
            return [str(x) for x in flat]
        return [str(flat)]
    return [str(texts)]


def dynamic_import(module_name: str, file_path: str):
    """
    Import a Python file as a module at runtime using importlib.

    The PLM-ICD custom modeling files (modeling_bert.py, modeling_longformer.py,
    modeling_roberta.py) live in the PLM-ICD/src directory and are not installed
    as a Python package. We use importlib to load them by file path at runtime
    so we can access the custom BertForSingleLabelClassification class etc.
    without modifying sys.path or installing the PLM-ICD package.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def find_plm_class(module, keyword: str):
    """
    Find the classification class in a dynamically imported PLM-ICD module.

    The class names in the modeling files are things like BertForSingleLabelClassification,
    LongformerForSingleLabelClassification, etc. We look for a class whose name
    contains both the backbone keyword (bert, roberta, longformer) and "classif".
    We sort by name length descending to prefer more specific names in case
    multiple matching classes exist in the module.
    """
    candidates = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type):
            lname = name.lower()
            if keyword.lower() in lname and "classif" in lname:
                candidates.append(obj)
    if not candidates:
        raise ImportError(f"Could not find a classification class in {module.__file__} for keyword={keyword}")
    candidates.sort(key=lambda c: len(c.__name__), reverse=True)
    return candidates[0]


def chunk_batch(enc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Reshape a flat tokenized batch into chunks for PLM-ICD models.

    PLM-ICD models expect input of shape (batch_size, num_chunks, chunk_size)
    rather than the standard (batch_size, seq_len). This function takes the
    flat (batch_size, MAX_LEN_PLM) tensors from the tokenizer and reshapes
    them to (batch_size, NUM_CHUNKS, CHUNK_SIZE).

    If the sequence is shorter than MAX_LEN_PLM we pad with zeros; if longer
    we truncate. This ensures the tensor dimensions are always divisible by
    CHUNK_SIZE before the view() call.
    """
    out = {}
    for k, v in enc.items():
        if v is None or v.ndim != 2:
            out[k] = v
            continue
        b, l = v.shape
        if l != MAX_LEN_PLM:
            if l > MAX_LEN_PLM:
                v = v[:, :MAX_LEN_PLM]
            else:
                pad = torch.zeros((b, MAX_LEN_PLM - l), dtype=v.dtype, device=v.device)
                v = torch.cat([v, pad], dim=1)
        out[k] = v.view(b, NUM_CHUNKS, CHUNK_SIZE)
    return out


def is_llm_model_choice(model_choice: str) -> bool:
    """Return True if the selected model is a discriminative LLM (not PLM-ICD)."""
    return model_choice in DISC_MODELS


def shrink_text_for_shap(model_choice: str, text: str, tokenizer) -> str:
    """
    Truncate a clinical note to the maximum length the model can process.

    SHAP perturbs the input text many times, and each perturbation must fit
    within the model's context window. We truncate the text before passing it
    to the SHAP explainer so that every perturbed version also fits.

    We first try to use the tokenizer's offset_mapping to find the exact
    character boundary corresponding to the last real token. This gives us
    a character-accurate truncation that does not split words. If offset
    mapping fails (some tokenizers do not support it), we fall back to
    decoding the truncated token IDs back to a string. As a last resort
    we truncate by character count.
    """
    max_len = MAX_LEN_LLM if is_llm_model_choice(model_choice) else MAX_LEN_PLM
    safe_len = max_len - 4  # leave a small margin for special tokens

    try:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=safe_len,
            return_offsets_mapping=True,
        )
        offsets = enc.get("offset_mapping", None)
        if offsets:
            # Find the character position of the last real (non-special) token
            real_offsets = [(s, e) for s, e in offsets if not (s == 0 and e == 0)]
            if real_offsets:
                last_char = max(int(e) for s, e in real_offsets)
                if last_char > 0:
                    return text[:last_char]
    except Exception as ex:
        print("[WARN] shrink_text_for_shap offset mapping failed:", repr(ex))

    # Fallback: decode truncated token IDs back to text
    try:
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=safe_len,
        )["input_ids"]
        return tokenizer.decode(ids, skip_special_tokens=True)
    except Exception:
        # Last resort: character-level truncation
        return text[:4000]


def shap_budget(model_choice: str) -> int:
    """
    Return the SHAP max_evals budget for the chosen model family.

    The budget controls how many perturbation evaluations SHAP runs per token.
    We use the same value for PLM and LLM models currently, but keeping them
    separate makes it easy to tune them independently if needed.
    """
    return SHAP_MAX_EVALS_LLM if is_llm_model_choice(model_choice) else SHAP_MAX_EVALS_PLM


def crop_head_tail_tokens(token_ids: List[int], budget: int, head_frac: float) -> List[int]:
    """
    Keep a head slice and a tail slice of token_ids that together fit within budget.

    Same function used in all discriminative training scripts - keeping it identical
    ensures the SHAP explainer sees the same text representation the model was trained on.
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


def _encode_disc_token_ids_manual(tokenizer, ids: List[int], max_length: int) -> Tuple[List[int], List[int]]:
    """
    Manually add BOS token, pad, and build attention mask for a token ID sequence.

    This is the fallback path when tokenizer.prepare_for_model() is not available.
    We detect the padding side from the tokenizer's padding_side attribute and pad
    accordingly - some LLM tokenizers pad on the left rather than the right.

    We add a BOS token if the tokenizer expects one and the sequence does not already
    start with it. This is important for Llama-based models (Meditron, OpenBioLLM)
    which use the BOS token as a generation sentinel.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    full_ids = list(ids)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    add_bos = getattr(tokenizer, "add_bos_token", True)
    if add_bos and bos_id is not None and (not full_ids or full_ids[0] != bos_id):
        full_ids = [bos_id] + full_ids

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    attention = [1] * len(full_ids)
    pad_len = max_length - len(full_ids)
    if pad_len > 0:
        side = getattr(tokenizer, "padding_side", "right")
        if side == "left":
            full_ids = [pad_id] * pad_len + full_ids
            attention = [0] * pad_len + attention
        else:
            full_ids = full_ids + [pad_id] * pad_len
            attention = attention + [0] * pad_len

    return full_ids, attention


def encode_disc_batch_headtail(tokenizer, texts: List[str], max_length: int, head_frac: float):
    """
    Tokenize a batch of texts with head+tail cropping for discriminative LLM models.

    The encoding pipeline:
      1. Tokenize without truncation to get all token IDs
      2. Apply head+tail cropping to fit within the token budget
      3. Add special tokens (BOS, EOS, padding) via prepare_for_model()
         or the manual fallback if prepare_for_model() is not available
      4. Stack into tensors and move to the inference device

    We compute special_budget by asking the tokenizer how many special tokens
    it adds (BOS, EOS etc.) so we can compute the remaining budget for text
    content precisely: text_budget = max_length - num_special_tokens.
    """
    input_ids_batch = []
    attention_mask_batch = []

    # Reserve space for special tokens (BOS, EOS, etc.)
    special_budget = tokenizer.num_special_tokens_to_add(pair=False)
    text_budget = max(1, max_length - special_budget)

    use_prepare = callable(getattr(tokenizer, "prepare_for_model", None))

    for text in texts:
        # Tokenize without any truncation or special tokens so we can crop manually
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        ids = crop_head_tail_tokens(ids, text_budget, head_frac)

        if use_prepare:
            # prepare_for_model handles BOS/EOS addition, padding, and attention mask
            prepared = tokenizer.prepare_for_model(
                ids,
                add_special_tokens=True,
                truncation=False,
                padding="max_length",
                max_length=max_length,
                return_attention_mask=True,
            )
            full_ids = prepared["input_ids"]
            attention_mask = prepared["attention_mask"]
        else:
            # Manual fallback for tokenizers that do not implement prepare_for_model
            full_ids, attention_mask = _encode_disc_token_ids_manual(tokenizer, ids, max_length)

        # Safety truncation in case prepare_for_model returned a longer sequence
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        input_ids_batch.append(full_ids)
        attention_mask_batch.append(attention_mask)

    input_ids = torch.tensor(input_ids_batch, dtype=torch.long, device=DEVICE)
    attention_mask = torch.tensor(attention_mask_batch, dtype=torch.long, device=DEVICE)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ---- Note HTML rendering ----

def build_note_html_with_spans(text: str, tokenizer, max_len: int) -> str:
    """
    Render the clinical note as HTML with each token wrapped in an individual span.

    Each span has a data-idx attribute matching the token's position in the
    tokenized sequence. This allows the SHAP callback to highlight specific tokens
    by updating span backgrounds via JavaScript without re-rendering the whole note.

    We use offset_mapping from the tokenizer to find the exact character boundaries
    of each token in the original text, then insert HTML span tags around each token
    while preserving all inter-token whitespace and punctuation as plain text nodes.

    If offset_mapping is not available (some slow tokenizers do not support it),
    we fall back to rendering the entire note as a single escaped text block.
    """
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_len,
    )
    offsets = enc.get("offset_mapping", None)
    if offsets is None:
        safe = html.escape(text)
        return f"<div id='note_box' style='white-space:normal; line-height:1.7;'>{safe}</div>"

    out = []
    last = 0
    for i, (s, e) in enumerate(offsets):
        s = int(s)
        e = int(e)
        if s < last:
            s = last
        if s > last:
            # Text between tokens (spaces, punctuation) - render as plain text
            out.append(html.escape(text[last:s]))
        piece = html.escape(text[s:e])
        out.append(
            f"<span class='note_tok' data-idx='{i}' "
            f"style='display:inline; border-radius:4px; padding:1px 1px; background:transparent;'>"
            f"{piece}</span>"
        )
        last = e

    if last < len(text):
        out.append(html.escape(text[last:]))

    return (
        "<div id='note_box' style='white-space:normal; line-height:1.7; background:#fff; color:#111; "
        "padding:12px; border:1px solid #ddd; border-radius:10px; font-size:16px;'>"
        + "".join(out)
        + "</div>"
    )


def _leaf_shap_tokens_values(exp: shap.Explanation) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
    """
    Extract the token list and SHAP value array from a SHAP Explanation object.

    SHAP's internal data structure can be nested differently depending on the
    version of SHAP and the explainer used. This function tries to use SHAP's
    own internal process_shap_values() and unpack_shap_explanation_contents()
    helpers to navigate the structure correctly.

    If values are 2D (one column per output class) we take either the first
    column or squeeze if there is only one column, since SHAP text plots
    expect 1D values.

    Returns (None, None) if extraction fails, in which case the caller falls
    back to a simpler rendering path.
    """
    try:
        from shap.plots._text import process_shap_values, unpack_shap_explanation_contents

        row = exp[0] if hasattr(exp, "__len__") and len(exp) > 0 else exp
        values, clustering = unpack_shap_explanation_contents(row)
        values = np.asarray(values)
        if values.ndim == 2 and values.shape[1] == 1:
            values = np.squeeze(values, axis=-1)
        elif values.ndim == 2 and values.shape[1] > 1:
            values = values[:, 0]
        tokens, vals, _gs = process_shap_values(
            row.data, values, 0.01, "", clustering
        )
        return list(tokens), np.asarray(vals).squeeze()
    except Exception as ex:
        print("[WARN] _leaf_shap_tokens_values:", repr(ex))
        return None, None


def build_note_html_with_shap_colors(text: str, tokenizer, max_len: int, exp: shap.Explanation) -> str:
    """
    Render the clinical note with token backgrounds colored by SHAP attribution.

    Uses SHAP's own red_transparent_blue colormap so the colors match the
    force plot - red tokens pushed the prediction up, blue pushed it down,
    and transparency encodes the magnitude.

    We extract token-level SHAP values via _leaf_shap_tokens_values() and
    then map each value to an RGBA background color using SHAP's colormap.
    The scaled value is 0.5 + 0.5 * (v / cmax), which maps the range
    [-cmax, +cmax] to [0, 1] for the colormap input.

    If extraction fails we fall back to the plain span version without colors.
    """
    from shap.plots import colors as shap_colors

    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    tokens, vals = _leaf_shap_tokens_values(exp)
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_len,
    )
    offsets = enc.get("offset_mapping", None)
    if offsets is None or tokens is None or vals is None:
        return build_note_html_with_spans(text, tokenizer, max_len=max_len)

    vals = np.asarray(vals).ravel()
    n_tok = min(len(offsets), len(tokens), len(vals))
    if n_tok <= 0:
        return build_note_html_with_spans(text, tokenizer, max_len=max_len)

    cmax = float(np.max(np.abs(vals[:n_tok]))) if n_tok else 1.0
    cmax = max(cmax, 1e-8)  # avoid division by zero for all-zero attributions

    out = []
    last = 0
    for i in range(len(offsets)):
        s, e = int(offsets[i][0]), int(offsets[i][1])
        if i < n_tok:
            v = float(vals[i])
            # Map [-cmax, cmax] to [0, 1] for the red-transparent-blue colormap
            scaled = 0.5 + 0.5 * v / (cmax + 1e-8)
            c = shap_colors.red_transparent_blue(scaled)
            rgba = (c[0] * 255, c[1] * 255, c[2] * 255, c[3])
            bg = f"rgba({rgba[0]:.0f},{rgba[1]:.0f},{rgba[2]:.0f},{rgba[3]:.3f})"
        else:
            bg = "transparent"

        if s < last:
            s = last
        if s > last:
            out.append(html.escape(text[last:s]))
        piece = html.escape(text[s:e])
        out.append(
            f"<span class='note_tok' data-idx='{i}' "
            f"style='display:inline; border-radius:4px; padding:1px 1px; background:{bg};'>"
            f"{piece}</span>"
        )
        last = e

    if last < len(text):
        out.append(html.escape(text[last:]))

    return (
        "<div id='note_box' style='white-space:normal; line-height:1.7; background:#fff; color:#111; "
        "padding:12px; border:1px solid #ddd; border-radius:10px; font-size:16px;'>"
        + "".join(out)
        + "</div>"
    )


# ---- SHAP HTML rendering ----

def _fix_shap_html(fragment: str) -> str:
    """
    Fix SHAP-generated HTML to render correctly in an iframe.

    SHAP's text plot HTML can contain np.float64(...) and numpy.float64(...)
    strings that are left over from converting numpy scalars to Python floats.
    These are not valid JavaScript or HTML so we strip them with regex, replacing
    each occurrence with just the numeric value inside the parentheses.

    We also append a style block to force white background and dark text, because
    Gradio's default theme would otherwise cause the iframe content to inherit
    dark theme colors that make the SHAP force plot unreadable.
    """
    import re
    fragment = re.sub(
        r'np\.float64\(([^)]+)\)',
        lambda m: str(float(m.group(1))),
        fragment
    )
    fragment = re.sub(
        r'numpy\.float64\(([^)]+)\)',
        lambda m: str(float(m.group(1))),
        fragment
    )
    fragment += (
        "<style>"
        "html,body{background:#fff!important;color:#111!important;}"
        "svg{background:#fff!important;}"
        "</style>"
    )
    return fragment


def embed_shap_html_in_iframe(html_fragment: str, min_height: int = 900) -> str:
    """
    Wrap SHAP HTML in a sandboxed iframe using a base64-encoded data URI.

    We embed the SHAP plot in an iframe rather than rendering it directly in
    the Gradio HTML component because SHAP's text plot uses JavaScript for
    interactive highlighting, and Gradio sanitizes JavaScript out of gr.HTML().

    The sandbox attribute with allow-scripts permits JavaScript inside the iframe
    while blocking access to the parent page's cookies, localStorage, and DOM.
    The base64 data URI avoids cross-origin issues that would arise from serving
    the HTML as a separate file.
    """
    cleaned = _fix_shap_html(html_fragment)
    doc = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/>"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>"
        "</head>"
        "<body style='margin:0;padding:10px;font-family:system-ui,sans-serif;"
        "background:#fff;color:#111;'>"
        + cleaned
        + "</body></html>"
    )
    b64 = base64.b64encode(doc.encode("utf-8")).decode("ascii")
    return (
        f"<iframe title=\"SHAP text plot\" sandbox=\"allow-scripts\" "
        f"style=\"width:100%;min-height:{min_height}px;border:1px solid #ccc;"
        f"border-radius:8px;background:#fff;\" "
        f"src=\"data:text/html;charset=utf-8;base64,{b64}\"></iframe>"
    )


def coerce_explanation_for_shap_text_plot(
    exp: shap.Explanation, output_label: str, label_idx: int
) -> shap.Explanation:
    """
    Reshape and clean a SHAP Explanation object so shap.plots.text() accepts it.

    shap.plots.text() has strict requirements on the Explanation structure:
      - values must be 1D (one value per token)
      - base_values must be a scalar float
      - data must not be wrapped in a single-element tuple
      - optional attributes (clustering, hierarchical_values) must match data length

    Different versions of SHAP and different explainer algorithms produce
    Explanation objects with slightly different shapes. This function normalizes
    the structure to what shap.plots.text() expects, trying several fallback
    strategies if the primary reshape fails.

    We reconstruct the Explanation from scratch using keyword arguments so we
    have precise control over which attributes are included and their shapes.
    """
    try:
        shp = getattr(exp, "shape", ())
        if len(shp) >= 2 and int(shp[0]) == 1:
            e = exp[0]
        else:
            e = exp
    except Exception:
        e = exp

    try:
        values = np.asarray(e.values)

        # Ensure values is 1D - take the right column if multi-output
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        elif values.ndim == 2 and values.shape[1] > 1:
            col = min(label_idx, values.shape[1] - 1)
            values = values[:, col]

        # base_values must be a scalar
        base_values = np.asarray(e.base_values)
        if base_values.ndim >= 1:
            base_values = float(base_values.ravel()[0])

        # Unwrap single-element tuples from data
        data = e.data
        if isinstance(data, tuple) and len(data) == 1:
            data = data[0]

        kw: Dict = {
            "values": values,
            "base_values": base_values,
            "data": data,
            "output_names": output_label,
        }

        def unwrap(v):
            # Recursively unwrap single-element tuples
            while isinstance(v, tuple) and len(v) == 1:
                v = v[0]
            return v

        # Include optional attributes only if they match the data length
        for attr in ("display_data", "clustering", "hierarchical_values"):
            v = getattr(e, attr, None)
            if v is not None:
                v = unwrap(v)
                try:
                    arr = np.asarray(v)
                    if arr.shape[0] == len(data):
                        kw[attr] = v
                except Exception:
                    pass

        return shap.Explanation(**kw)

    except Exception:
        return exp


def shap_native_text_plot_html(
    exp: shap.Explanation, output_label: str, label_idx: int
) -> Optional[str]:
    """
    Generate SHAP's native text plot HTML for a token-level explanation.

    Tries several candidate Explanation objects in sequence, applying the coercion
    from coerce_explanation_for_shap_text_plot() first. For each candidate we call
    shap.plots.text() and try to extract the HTML string from whatever it returns -
    different SHAP versions return a string, an object with _repr_html_(), or an
    object with a .data attribute.

    Returns the HTML string if any candidate succeeds, or None if all fail.
    The caller then falls back to the manual red/blue token plot.
    """
    def candidates() -> List[shap.Explanation]:
        out: List[shap.Explanation] = []
        seen: set = set()

        def add(x: shap.Explanation) -> None:
            if id(x) not in seen:
                seen.add(id(x))
                out.append(x)

        coerced = coerce_explanation_for_shap_text_plot(exp, output_label, label_idx)
        add(coerced)
        try:
            shp = getattr(coerced, "shape", ())
            if len(shp) >= 2 and int(shp[0]) == 1:
                add(coerced[0])
        except Exception:
            pass
        if id(coerced) != id(exp):
            add(exp)
        return out

    last_err: Optional[Exception] = None
    for cand in candidates():
        try:
            raw = shap.plots.text(cand, display=False, num_starting_labels=0)
            if raw is None:
                continue
            # Extract the HTML string from whatever shap.plots.text() returned
            if hasattr(raw, "_repr_html_") and not isinstance(raw, str):
                try:
                    raw = raw._repr_html_()
                except Exception:
                    raw = str(raw)
            if hasattr(raw, "data") and not isinstance(raw, str):
                raw = getattr(raw, "data", "")
            if isinstance(raw, str) and raw.strip():
                return raw
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        print("[WARN] shap.plots.text failed:", repr(last_err))
    return None


def build_doc_style_outputs_row(code: str, label_idx: int, p: float) -> str:
    """
    Build the HTML header row shown above the SHAP force plot.

    Displays the predicted ICD code, a confidence label (High/Moderate/Low)
    with color-coded badge, and a brief explanation of how to read the SHAP plot.
    The thresholds p >= 0.5 for High and p >= 0.1 for Moderate were chosen to
    roughly match the practical meaning of those terms in clinical coding context.
    """
    safe_code = html.escape(code)
    confidence = "High" if p >= 0.5 else "Moderate" if p >= 0.1 else "Low"
    conf_color = "#1a7a3a" if p >= 0.5 else "#b45d00" if p >= 0.1 else "#a32d2d"
    conf_bg = "#eaf5ee" if p >= 0.5 else "#fef3e2" if p >= 0.1 else "#fcebeb"
    return (
        "<div style='margin-bottom:16px;padding-bottom:16px;border-bottom:1px solid #eee;'>"
        "<div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;'>"
        f"<div style='font-size:22px;font-weight:600;color:#111;'>{safe_code}</div>"
        f"<div style='font-size:13px;color:#555;'>ICD-10 code</div>"
        f"<div style='margin-left:auto;background:{conf_bg};color:{conf_color};"
        f"font-size:13px;font-weight:500;padding:4px 12px;border-radius:99px;'>"
        f"{confidence} confidence - {p:.1%}</div>"
        "</div>"
        "<div style='font-size:13px;color:#666;margin-top:8px;'>"
        "The SHAP force bar below shows which tokens pushed the model's prediction "
        "above (red) or below (blue) its baseline probability. "
        "Longer arrows indicate stronger influence."
        "</div>"
        "</div>"
    )


def wrap_shap_notebook_layout(outputs_block: str, inner: str) -> str:
    """
    Wrap the SHAP output header and force plot in a styled container div.

    Provides a consistent white card layout with a light border and rounded
    corners that matches the Gradio Soft theme while ensuring the SHAP plot
    always renders on a white background regardless of the Gradio theme.
    """
    return (
        "<div style='background:#fff;color:#111;padding:16px 18px;"
        "border:1px solid #ddd;border-radius:10px;overflow:auto;max-width:100%;'>"
        f"{outputs_block}"
        f"{inner}"
        "</div>"
    )


def shap_redblue_token_plot(exp: shap.Explanation) -> str:
    """
    Render a simple red/blue token coloring as a fallback when shap.plots.text() fails.

    This is a pure HTML/CSS implementation that does not rely on SHAP's JavaScript
    force plot. Each token is colored with a red or blue background whose alpha
    encodes the magnitude of its SHAP value: stronger attributions get more opaque
    color, weaker ones are more transparent.

    We normalize all SHAP values by the maximum absolute value so the colors are
    always in a useful range - without normalization small SHAP values would all
    look nearly transparent even when they are the strongest attributions relative
    to each other.
    """
    row = exp[0] if hasattr(exp, "__len__") and len(exp) > 0 else exp

    tokens = getattr(row, "data", None)
    vals = getattr(row, "values", None)

    if tokens is None or vals is None:
        return "<pre>Could not extract SHAP tokens/values.</pre>"

    vals = np.array(vals).squeeze()
    if vals.ndim != 1 or len(tokens) != len(vals):
        return "<pre>Unexpected SHAP shapes.</pre>"

    max_abs = float(np.max(np.abs(vals))) if len(vals) else 1.0
    max_abs = max(max_abs, 1e-8)

    parts = []
    for i, (tok, v) in enumerate(zip(tokens, vals)):
        t = "" if tok is None else str(tok)
        safe = html.escape(t).replace("\r", " ").replace("\n", " ").replace("\t", " ")
        if t.strip() == "":
            parts.append(safe)
            continue

        # Map |v| / max_abs to an alpha in [0.12, 0.67] so even the weakest
        # attributions have a visible tint and the strongest are clearly colored
        strength = min(abs(float(v)) / max_abs, 1.0)
        alpha = 0.12 + 0.55 * strength
        bg = f"rgba(255,0,0,{alpha:.3f})" if v >= 0 else f"rgba(0,102,255,{alpha:.3f})"

        parts.append(
            f"<span class='shap_tok' data-idx='{i}' data-bg='{bg}' "
            f"style='background:{bg}; border-radius:4px; padding:1px 2px; cursor:pointer;'>"
            f"{safe}</span>"
        )

    return (
        "<div style='background:#fff; color:#111; padding:12px; border:1px solid #ddd; border-radius:10px;'>"
        "<div style='font-size:0.95em; margin-bottom:8px; color:#444;'>"
        "<b>Red</b> increases the selected ICD probability, <b>Blue</b> decreases it."
        "</div>"
        "<div style='white-space:normal; line-height:1.8; font-family: system-ui, -apple-system, Segoe UI, Roboto;'>"
        + "".join(parts)
        + "</div>"
        "</div>"
    )


# ---- Discriminative head loading ----

def _strip_net_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Strip the "net." prefix from state dict keys if present.

    When the custom MLP head was saved inside an nn.Sequential named "net",
    the saved keys look like "net.1.weight" instead of "1.weight". We strip
    the prefix so the keys match the architecture we reconstruct below.
    """
    return {(k[len("net."):] if k.startswith("net.") else k): v for k, v in state.items()}


def build_head_from_state_dict_strict(state: Dict[str, torch.Tensor]) -> Tuple[nn.Module, int]:
    """
    Reconstruct the MLP classification head from its saved state dict.

    The head architecture (number of layers and dimensions) is inferred directly
    from the weight tensor shapes in the state dict, so we do not need to pass
    the architecture as a separate argument. This makes the loader resilient to
    models trained with different --head_layers settings.

    We support three head depths:
      1-layer: Dropout -> Linear(in -> num_labels)
      2-layer: Dropout -> Linear(in -> hidden1) -> GELU -> Dropout -> Linear(hidden1 -> num_labels)
      3-layer: same as 2-layer but with an additional hidden layer

    The in_dim is read from the first linear weight's shape[1], and num_labels
    from the last linear weight's shape[0]. We only consider linear weight keys
    that have a numeric prefix (1, 3, 5 etc.) which corresponds to the position
    in the nn.Sequential.
    """
    state2 = _strip_net_prefix(state)

    # Find all linear weight tensors and sort by their sequential index
    linear_items = []
    for k, v in state2.items():
        if k.endswith(".weight") and getattr(v, "ndim", 0) == 2:
            try:
                idx = int(k.split(".")[0])
                linear_items.append((idx, k))
            except Exception:
                pass
    linear_items.sort(key=lambda x: x[0])
    linear_weights = [k for _, k in linear_items][:3]  # at most 3 linear layers
    if not linear_weights:
        raise ValueError("custom_head.pt missing expected keys like '1.weight'")

    first_w = state2[linear_weights[0]]
    in_dim = int(first_w.shape[1])

    if len(linear_weights) == 1:
        # 1-layer head: direct linear projection
        num_labels = int(first_w.shape[0])
        head = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_dim, num_labels))
    elif len(linear_weights) == 2:
        # 2-layer head: standard configuration used in the thesis
        hidden1 = int(first_w.shape[0])
        num_labels = int(state2[linear_weights[1]].shape[0])
        head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_dim, hidden1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, num_labels),
        )
    else:
        # 3-layer head: tested in EXP1 but not selected as the final config
        hidden1 = int(first_w.shape[0])
        hidden2 = int(state2[linear_weights[1]].shape[0])
        num_labels = int(state2[linear_weights[2]].shape[0])
        head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(in_dim, hidden1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden2, num_labels),
        )

    head.load_state_dict(state2, strict=True)
    return head, num_labels


class MeanPooler(nn.Module):
    """
    Compute the masked mean of token hidden states.

    Identical to the MeanPooler used in the discriminative training scripts.
    The attention_mask zeros out padding positions before summing so padding
    tokens do not contribute to the document representation.
    """
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom


class DiscClassifier(nn.Module):
    """
    Inference-only wrapper combining the LLM backbone, pooling, and MLP head.

    This mirrors the CustomSeqClassifier from the training scripts but is
    simplified for inference: no criterion, no labels argument, just forward().

    The pooling argument controls how token hidden states are collapsed to a
    single document vector:
      "mean" - average all non-padding token states (standard for Meditron, OpenBioLLM)
      "last" - take the hidden state of the final real token (used for BioMistral)

    For "last" pooling with left-padded tokenizers we take position -1, which is
    always the last real token regardless of sequence length. For right-padded
    tokenizers we find the last real token by summing the attention mask.

    We cast the pooled vector to the head's dtype before passing it through the
    head, because the backbone may run in fp16 while the head parameters are in
    a different dtype due to how they were saved and loaded.
    """
    def __init__(self, base_model, head: nn.Module, pooling: str = "mean", padding_side: str = "right"):
        super().__init__()
        self.base = base_model
        self.head = head
        self.pooling = pooling
        self.padding_side = padding_side
        self.pool = MeanPooler() if pooling == "mean" else None

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        last_hidden = out.last_hidden_state

        if self.pooling == "mean":
            pooled = self.pool(last_hidden, attention_mask)
        else:
            # "last" pooling - find the last real token
            if self.padding_side == "left":
                pooled = last_hidden[:, -1, :]  # always position -1 for left-padded
            else:
                lengths = attention_mask.sum(dim=1) - 1
                lengths = lengths.clamp(min=0)
                pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), lengths]

        # Cast to the head's dtype to avoid mixed-precision matmul errors
        head_dtype = next(self.head.parameters()).dtype
        if pooled.dtype != head_dtype:
            pooled = pooled.to(head_dtype)

        return self.head(pooled)


# ---- Model wrappers ----

class BaseWrapper:
    """Abstract base class for model wrappers. Subclasses implement predict_probs()."""
    def predict_probs(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class PLMWrapper(BaseWrapper):
    """
    Wrapper for PLM-ICD models (BERT, Longformer, RoBERTa).

    Loads the appropriate custom model class from the PLM-ICD src directory
    using dynamic_import() and find_plm_class(). The model is identified by
    the backbone name in the checkpoint directory (bert_single_label,
    longformer_single_label, roberta_single_label).

    predict_probs() tokenizes the input, reshapes to chunks, runs the model,
    and returns softmax probabilities over the 30 ICD classes.
    """
    def __init__(self, name: str, model_dir: str, id2label: Dict[int, str]):
        self.name = name
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

        base = os.path.basename(model_dir).lower()
        if "longformer" in base:
            mod = dynamic_import("plm_modeling_longformer", os.path.join(PLM_SRC_DIR, "modeling_longformer.py"))
            cls = find_plm_class(mod, "longformer")
            self.model = cls.from_pretrained(model_dir)
        elif "roberta" in base:
            mod = dynamic_import("plm_modeling_roberta", os.path.join(PLM_SRC_DIR, "modeling_roberta.py"))
            cls = find_plm_class(mod, "roberta")
            self.model = cls.from_pretrained(model_dir)
        elif "bert" in base:
            mod = dynamic_import("plm_modeling_bert", os.path.join(PLM_SRC_DIR, "modeling_bert.py"))
            cls = find_plm_class(mod, "bert")
            self.model = cls.from_pretrained(model_dir)
        else:
            # Fallback for any other architecture - try standard AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        self.model.to(DEVICE).eval()
        self.id2label = id2label

    @torch.inference_mode()
    def predict_probs(self, texts: List[str]) -> np.ndarray:
        """
        Run PLM-ICD forward pass and return softmax probabilities.

        Tokenizes with padding to MAX_LEN_PLM, then reshapes to the chunked
        format (batch, num_chunks, chunk_size) expected by the custom PLM-ICD
        model forward methods.
        """
        texts = ensure_list_of_str(texts)
        enc = self.tokenizer(texts, truncation=True, max_length=MAX_LEN_PLM, padding="max_length", return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        enc = chunk_batch(enc)
        out = self.model(**enc)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()


class DiscWrapper(BaseWrapper):
    """
    Wrapper for discriminative LLM models (Meditron, BioMistral, OpenBioLLM).

    Loading a 7-8B LLM involves several steps:
      1. Load the base model weights from HuggingFace in the target dtype
      2. Resize token embeddings if the checkpoint tokenizer has a different
         vocabulary size than the original base model
      3. Attach the saved LoRA adapter weights via PeftModel.from_pretrained()
      4. Load and reconstruct the MLP classification head from custom_head.pt
      5. Wrap everything in DiscClassifier and freeze all parameters

    Meditron requires special handling because its base vocabulary of 32,000
    tokens may differ from what the tokenizer reports after training, and the
    loading report from transformers can be noisy due to the vocab resize.
    """
    def __init__(self, name: str, checkpoint_dir: str, base_model_name: str, pooling: str):
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.base_model_name = base_model_name
        self.pooling = pooling

        self.id2label = load_disc_label_map(checkpoint_dir)
        self.num_labels = len(self.id2label)

        is_meditron = "meditron" in name.lower() or "meditron" in base_model_name.lower()

        # Meditron may have saved a tokenizer_config.json in the checkpoint dir
        # that differs from the base model tokenizer. Load from checkpoint if present.
        if is_meditron:
            tcfg = os.path.join(checkpoint_dir, "tokenizer_config.json")
            tok_source = checkpoint_dir if os.path.isfile(tcfg) else base_model_name
        else:
            tok_source = checkpoint_dir

        self.tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[INFO] Loading {name} in dtype={DISC_DTYPE} on {DEVICE} ...")
        print(f"[INFO] {name}: tokenizer from {tok_source}")

        tok_vocab = len(self.tokenizer)

        load_kw = dict(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_safetensors=False,  # some checkpoints are not in safetensors format
            ignore_mismatched_sizes=is_meditron,
        )

        if is_meditron:
            # Force the config vocab_size to the base Meditron value to prevent
            # a mismatch error when the checkpoint has a different saved vocab size
            config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
            if getattr(config, "vocab_size", None) is not None and config.vocab_size != MEDITRON_BASE_VOCAB:
                config.vocab_size = MEDITRON_BASE_VOCAB
            # Suppress the verbose loading report that appears during vocab resize
            _lr = logging.getLogger("transformers.utils.loading_report")
            _prev_lr = _lr.level
            _lr.setLevel(logging.ERROR)
            try:
                base = model_from_pretrained_compat(
                    base_model_name,
                    DISC_DTYPE,
                    config=config,
                    **load_kw,
                )
            finally:
                _lr.setLevel(_prev_lr)
        else:
            base = model_from_pretrained_compat(base_model_name, DISC_DTYPE, **load_kw)

        base.config.pad_token_id = self.tokenizer.pad_token_id

        # Resize embeddings if the tokenizer vocabulary is larger than the base model's
        base_vocab = base.get_input_embeddings().num_embeddings
        if tok_vocab != base_vocab:
            print(f"[INFO] Resizing token embeddings for {name}: {base_vocab} -> {tok_vocab}")
            base.resize_token_embeddings(tok_vocab)

        # Attach the saved LoRA adapter weights to the base model
        adapter_dir = os.path.join(checkpoint_dir, "lora_adapter")
        base = PeftModel.from_pretrained(base, adapter_dir)

        # Load and reconstruct the MLP classification head from the saved state dict
        head_state = torch.load(os.path.join(checkpoint_dir, "custom_head.pt"), map_location="cpu")
        head, inferred = build_head_from_state_dict_strict(head_state)
        if inferred != self.num_labels:
            print(f"[WARN] {name}: head num_labels={inferred}, label_map num_labels={self.num_labels}")

        head = head.to(dtype=DISC_DTYPE)

        pad_side = getattr(self.tokenizer, "padding_side", None) or "right"
        self.model = DiscClassifier(
            base, head=head, pooling=pooling, padding_side=pad_side
        ).to(device=DEVICE, dtype=DISC_DTYPE).eval()

        # Freeze all parameters - inference only, no gradient computation needed
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def predict_probs(self, texts: List[str]) -> np.ndarray:
        """
        Run the discriminative LLM forward pass and return softmax probabilities.

        Uses autocast for mixed precision on GPU for faster inference and lower
        memory usage. The logits are cast to float32 before softmax to avoid
        numerical issues from fp16 precision.
        """
        texts = ensure_list_of_str(texts)
        enc = encode_disc_batch_headtail(
            tokenizer=self.tokenizer,
            texts=texts,
            max_length=MAX_LEN_LLM,
            head_frac=DISC_TEXT_HEAD_FRAC,
        )

        if DEVICE.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=DISC_DTYPE):
                logits = self.model(enc["input_ids"], enc["attention_mask"])
        else:
            logits = self.model(enc["input_ids"], enc["attention_mask"])

        return torch.softmax(logits.float(), dim=-1).cpu().numpy()


# ---- Global model and explainer cache ----
# Only one model can be active at a time to stay within GPU memory limits.
# Both caches are dicts keyed by the model choice string from the UI dropdown.
MODEL_CACHE: Dict[str, BaseWrapper] = {}
EXPLAINER_CACHE: Dict[str, shap.Explainer] = {}
ACTIVE_MODEL_CHOICE = None

# Load PLM-ICD label vocab once at startup - shared across all three PLM models
PLM_ID2LABEL = load_label_vocab(PLM_LABEL_VOCAB)


def get_wrapper(model_choice: str) -> BaseWrapper:
    """
    Return the model wrapper for the chosen model, loading it if not cached.

    Calls ensure_only_one_active_model() first to evict any cached models
    from a different model family before loading the new one.
    """
    ensure_only_one_active_model(model_choice)

    if model_choice in MODEL_CACHE:
        return MODEL_CACHE[model_choice]

    if model_choice in PLM_MODELS:
        w = PLMWrapper(model_choice, PLM_MODELS[model_choice], PLM_ID2LABEL)
    else:
        cfg = DISC_MODELS[model_choice]
        w = DiscWrapper(model_choice, cfg["checkpoint_dir"], cfg["base_model_name"], cfg["pooling"])

    MODEL_CACHE[model_choice] = w
    return w


def get_explainer(model_choice: str) -> shap.Explainer:
    """
    Return the SHAP partition explainer for the chosen model, building it if not cached.

    We use shap.maskers.Text with the model's tokenizer so SHAP knows how to
    split the text into tokens for perturbation. The partition algorithm works by
    hierarchically masking groups of tokens and measuring the change in model output,
    which gives more stable attributions than the simpler token-by-token approach.

    The predict_fn wrapper normalizes the input through ensure_list_of_str() to
    handle whatever format SHAP passes at perturbation time.
    """
    ensure_only_one_active_model(model_choice)

    if model_choice in EXPLAINER_CACHE:
        return EXPLAINER_CACHE[model_choice]

    w = get_wrapper(model_choice)

    def predict_fn(texts):
        return w.predict_probs(ensure_list_of_str(texts))

    masker = shap.maskers.Text(w.tokenizer)
    explainer = shap.Explainer(predict_fn, masker, algorithm="partition")
    EXPLAINER_CACHE[model_choice] = explainer
    return explainer


# ---- UI callbacks ----

NOTES = load_notes()
NOTE_TITLE_TO_OBJ = {n["title"]: n for n in NOTES}

# LAST stores the prediction state so explain_selected() can reuse it without
# re-running inference when the user just changes the selected ICD code.
LAST = {"probs": None, "text_for_shap": "", "model_choice": "", "note_title": ""}


def run_topk_only(model_choice: str, note_title: str):
    """
    Run prediction for the selected model and note, update all UI outputs.

    This callback is triggered when the user clicks "Predict top-30". It:
      1. Loads the model wrapper (from cache or freshly)
      2. Runs predict_probs() on the full note text
      3. Selects the top 30 predictions by probability
      4. Builds the results table, bar chart, dropdown options, and note HTML
      5. Stores the predictions in LAST so explain_selected() can reuse them

    The text_for_shap stored in LAST is the truncated version of the note -
    this is what SHAP will actually perturb, and it is also what we use to
    render the note HTML so the token spans align with what SHAP expects.
    """
    LAST.update({"probs": None, "text_for_shap": "", "model_choice": "", "note_title": ""})

    note = NOTE_TITLE_TO_OBJ[note_title]
    text = note["text"]
    true_label = note.get("label", "")

    w = get_wrapper(model_choice)
    probs = w.predict_probs([text])[0]
    top_idx = topk_indices(probs, TOPK)

    rows = []
    dd = []
    plot_codes: List[str] = []
    plot_probs: List[float] = []
    plot_titles: List[str] = []
    for i in top_idx:
        code = w.id2label.get(int(i), f"LABEL_{i}")
        st = short_title_for_icd(code)
        p = float(probs[i])
        rows.append([code, st, int(i), p])
        # Dropdown option format encodes both the display text and the index
        # so explain_selected() can parse the label_idx without a separate lookup
        dd.append(f"{code} - {st} | idx={int(i)} | p={p:.6f}")
        plot_codes.append(code)
        plot_probs.append(p)
        plot_titles.append(st)

    bar_fig = build_topk_probability_figure(plot_codes, plot_probs, plot_titles)

    text_for_shap = shrink_text_for_shap(model_choice, text, w.tokenizer)
    max_len = MAX_LEN_LLM if is_llm_model_choice(model_choice) else MAX_LEN_PLM
    note_html_val = build_note_html_with_spans(text_for_shap, w.tokenizer, max_len=max_len)

    LAST.update({
        "probs": probs,
        "text_for_shap": text_for_shap,
        "model_choice": model_choice,
        "note_title": note_title
    })

    header = (
        f"<b>True label in file:</b> <code>{html.escape(true_label)}</code>"
        if true_label else "<b>True label in file:</b> (not provided)"
    )
    return (
        rows,
        header,
        gr.update(value=note_html_val),
        gr.update(choices=dd, value=(dd[0] if dd else None)),
        gr.update(value="<i>Click Explain to see the SHAP attribution plot for the selected ICD code.</i>"),
        bar_fig,
    )


def explain_selected(model_choice: str, note_title: str, explain_choice: str):
    """
    Run SHAP explanation for the selected ICD code and update the SHAP plot.

    This callback is triggered when the user clicks "Explain selected ICD (SHAP)".
    It:
      1. Re-runs prediction if the model or note has changed since last predict
      2. Parses the label index from the dropdown choice string
      3. Runs the SHAP partition explainer with outputs=[label_idx] to get
         attributions for only the selected class
      4. Tries shap.plots.text() first, falls back to the manual red/blue plot
      5. Wraps the result in the styled container and returns HTML

    We pass outputs=[label_idx] to the explainer so SHAP only perturbs the
    output dimension corresponding to the selected ICD code. Without this,
    SHAP would compute attributions for all 30 classes simultaneously, which
    would be 30 times slower and return a 2D attribution matrix.
    """
    if not explain_choice:
        return "<b>Select an ICD first.</b>", gr.update()

    # Re-run prediction if the model or note changed since last predict call
    if LAST["model_choice"] != model_choice or LAST["note_title"] != note_title:
        run_topk_only(model_choice, note_title)

    probs = LAST["probs"]
    text_for_shap = LAST["text_for_shap"]
    if probs is None:
        return "<b>Click Predict top-30 first.</b>", gr.update()

    # Parse the label index from the dropdown option string (format: "... | idx=7 | ...")
    try:
        label_idx = int(explain_choice.split("idx=")[1].split("|")[0].strip())
    except Exception:
        return f"<b>Could not parse idx from:</b> {html.escape(explain_choice)}", gr.update()

    w = get_wrapper(model_choice)
    explainer = get_explainer(model_choice)

    p = float(probs[label_idx])
    code = w.id2label.get(int(label_idx), f"LABEL_{label_idx}")

    # Run SHAP with outputs=[label_idx] to get attributions for only the selected class.
    # max_evals controls the number of perturbations - higher is more accurate but slower.
    sv = explainer([text_for_shap], outputs=[int(label_idx)], max_evals=shap_budget(model_choice))

    # Try native SHAP force plot first, fall back to the manual red/blue plot
    native = shap_native_text_plot_html(sv, code, label_idx)
    outputs_row = build_doc_style_outputs_row(code, label_idx, p)
    if native:
        inner = embed_shap_html_in_iframe(native)
    else:
        inner = shap_redblue_token_plot(sv)
    plot = wrap_shap_notebook_layout(outputs_row, inner)

    return plot, gr.update()


# ---- Gradio UI layout ----

CSS = """
#note_area, #shap_area { background: #fff !important; color: #111 !important; }
#note_box { background: #fff !important; color: #111 !important; }
#note_box span.note_tok { display:inline; }
.shap-notebook-wrap, .shap-doc-outputs { background: #fff !important; color: #111 !important; }
"""

model_choices = list(PLM_MODELS.keys()) + list(DISC_MODELS.keys())
note_titles = [n["title"] for n in NOTES]

with gr.Blocks(title="ICD Coding Explainability (Top-30 + SHAP)") as demo:
    gr.Markdown("""
# ICD Coding Explainability
**Automated ICD-10 code prediction from clinical notes using medical LLMs + SHAP interpretability**

**How to use:** Select a model and a clinical note, click **Predict top-30**, select an ICD code from the dropdown, then click **Explain selected ICD (SHAP)**
""")

    with gr.Row():
        model_dd = gr.Dropdown(choices=model_choices, value=model_choices[0], label="Model")
        note_dd = gr.Dropdown(choices=note_titles, value=note_titles[0], label="Clinical note")

    predict_btn = gr.Button("Predict top-30", variant="primary")

    gr.Markdown("### Predictions")
    top_df = gr.Dataframe(
        headers=["icd_code", "short_description", "label_index", "probability"],
        datatype=["str", "str", "number", "number"],
        label="Top-30 ICD predictions",
        interactive=False,
        wrap=True,
    )

    topk_prob_plot = gr.Plot(
        label="Probability distribution",
        format="png",
    )

    true_label_html = gr.HTML(label="True label")

    gr.Markdown("### Clinical note (model input)")
    note_html = gr.HTML(label="", elem_id="note_area")

    gr.Markdown("### SHAP explanation")
    with gr.Row():
        explain_dd = gr.Dropdown(choices=[], value=None, label="Select ICD code to explain")
        explain_btn = gr.Button("Explain selected ICD (SHAP)", variant="primary")

    shap_html = gr.HTML(
        label="",
        elem_id="shap_area",
        max_height=1200,
    )

    # Wire predict button to run_topk_only, which updates all main outputs
    predict_btn.click(
        fn=run_topk_only,
        inputs=[model_dd, note_dd],
        outputs=[top_df, true_label_html, note_html, explain_dd, shap_html, topk_prob_plot],
    )

    # Wire explain button to explain_selected, which updates the SHAP plot
    explain_btn.click(
        fn=explain_selected,
        inputs=[model_dd, note_dd, explain_dd],
        outputs=[shap_html, note_html],
    )

# Launch on localhost only (not exposed externally) on the standard Gradio port
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    css=CSS,
    theme=gr.themes.Soft(),
)