# Enhancing Automated ICD-10 Medical Coding with Large Language Models

> Master's Project — California State University, Sacramento (CSUS)
> Department of Computer Science
> Author: Namirah Imtieaz Shaik
> Committee Chair: Dr. Haiquan Chen | Second Reader: Dr. Jin Ying

\---

## Overview

This repository contains the full implementation of my Master's project, which investigates the use of biomedical large language models (LLMs) for automated ICD-10 medical coding on clinical discharge summaries from the MIMIC-IV dataset.

We frame ICD-10 coding as a **30-class single-label classification problem** over the most frequent diagnostic codes in MIMIC-IV and evaluate three families of models:

* **Classical ML Baseline** — Hybrid SVM with frozen Bio\_ClinicalBERT features
* **Deep Learning (PLM-ICD)** — BERT, Longformer, and RoBERTa with chunk-wise processing
* **Discriminative LLMs** — Meditron-7B, BioMistral-7B, and OpenBioLLM-8B with LoRA fine-tuning and a custom MLP classification head
* **Generative LLMs** — Meditron-7B, BioMistral-7B, and OpenBioLLM-8B fine-tuned with SFT to autoregressively generate ICD codes

An interactive Gradio demo with SHAP explainability is included for the six discriminative and PLM-ICD models.

\---

## Results

|Model|Micro F1|ROC-AUC (Weighted OVR)|
|-|-|-|
|SVM-ML (Hybrid)|0.6576|0.9598|
|BERT-PLM-ICD|0.7466|0.9794|
|Longformer-PLM-ICD|0.7316|0.9791|
|RoBERTa-PLM-ICD|0.7282|0.9765|
|Meditron-D|0.7668 ± 0.0023|0.9838 ± 0.0002|
|BioMistral-D|0.7776 ± 0.0051|0.9857 ± 0.0011|
|**OpenBioLLM-D**|**0.7802 ± 0.0045**|**0.9857 ± 0.0004**|
|Meditron-G|0.7591|0.8729|
|BioMistral-G|0.7093|0.8484|
|**OpenBioLLM-G**|**0.7896**|**0.8897**|

Discriminative LLM results are reported as mean ± std across 5 random seeds.

\---

## Repository Structure

```
Enhancing-Automated-ICD-Medical-Coding-with-Large-Language-Models/
│
├── PLM\_ICD/src/
│   ├── modeling\_bert.py              # BERT single-label classifier (adapted from PLM-ICD)
│   ├── modeling\_longformer.py        # Longformer single-label classifier (adapted from PLM-ICD)
│   ├── modeling\_roberta.py           # RoBERTa single-label classifier (adapted from PLM-ICD)
│   └── run\_icd.py                    # Main training script for all PLM-ICD models

│

├── Baseline/
│   └── train\_hybrid\_svm.py           # Hybrid SVM baseline (Bio\_ClinicalBERT + LinearSVC)

│

├── Generative\_Models/
│   ├── train\_meditron\_sft\_chat.py    # Meditron-G SFT training
│   ├── evaluate\_meditron\_sft.py      # Meditron-G evaluation with constrained decoding
│   ├── train\_biomistral\_sft\_chat.py  # BioMistral-G SFT training with label mixing
│   ├── evaluate\_biomistral\_sft.py    # BioMistral-G evaluation
│   ├── train\_openbiollm\_sft\_chat.py  # OpenBioLLM-G SFT training with QLoRA
│   └── evaluate\_openbiollm\_sft.py    # OpenBioLLM-G evaluation with strict matching
│
├── Discriminative\_Models/
│   ├── train\_meditron\_cls.py         # Meditron-D main training script
│   ├── train\_biomistral\_cls.py       # BioMistral-D main training script
│   ├── train\_openbiollm\_cls.py       # OpenBioLLM-D main training script
│  

├── Demo/
│   ├── app.py                        # Gradio UI with SHAP explainability

│
└── README.md
|
|__notebook.ipynb
|
|__requirements.txt
```

\---

## Dataset

All experiments use the **MIMIC-IV** clinical dataset (Johnson et al., 2023).

* **Source:** PhysioNet — https://physionet.org/content/mimiciv/
* **Access:** Requires credentialed access and signing the data use agreement
* **Split:** 20,676 discharge summaries, stratified 80/10/10 train/val/test split
* **Task:** 30-class single-label ICD-10 classification (top 30 most frequent codes)

> The processed dataset cannot be shared publicly due to the MIMIC-IV data use agreement. Please follow the PhysioNet access instructions to download the raw data and use `make\_chat\_data.py` to reproduce the training splits.

\---

## Model Architecture

### PLM-ICD (Deep Learning)

* Backbone: Bio\_ClinicalBERT / Longformer / RoBERTa (fully fine-tuned)
* Input: Discharge note split into 256-token chunks, stacked to shape (B, num\_chunks, chunk\_size)
* Aggregation: CLS-sum or LAAT (Label-Aware Attention Pooling)
* Loss: CrossEntropyLoss (single-label adaptation from original multi-label PLM-ICD)

### Discriminative LLMs (Meditron-D / BioMistral-D / OpenBioLLM-D)

* Backbone: AutoModel (hidden states only, no LM head)
* LoRA: r=16, alpha=32, targeting q/k/v/o/gate/up/down projections
* Pooling: Mean pooling over non-padding tokens
* Head: Two-layer MLP (4096 → 1536 → 30)
* Loss: CrossEntropyLoss
* Training: AdamW + cosine LR + early stopping on macro F1

### Generative LLMs (Meditron-G / BioMistral-G / OpenBioLLM-G)

* Prompt format:

```
  \[SYSTEM]
  You are a medical coding assistant...
  \[/SYSTEM]
  \[USER]
  <discharge note (head+tail cropped)>
  \[/USER]
  \[ICD]
  ICD-10 code: <ICD code>
  ```

* Training: SFT with tail-only supervision (loss only on ICD completion tokens)
* Evaluation: Greedy decoding + optional constrained decoding via prefix trie

\---

## Installation

```bash
git clone https://github.com/Namirah07/Enhancing-Automated-ICD-Medical-Coding-with-Large-Language-Models.git
cd Enhancing-Automated-ICD-Medical-Coding-with-Large-Language-Models

pip install torch transformers peft trl datasets scikit-learn
pip install gradio shap matplotlib numpy pandas
pip install ray\[tune] bitsandbytes accelerate
```

\---

## Usage

### Step 1 — Prepare the Data

```bash
# Convert CSV splits to chat JSONL for generative models
python Generative\_Models/make\_chat\_data.py
```

### Step 2 — Train PLM-ICD Models

```bash
python PLM\_ICD/run\_icd.py \\
  --model\_name\_or\_path emilyalsentzer/Bio\_ClinicalBERT \\
  --train\_file shared/train\_full.csv \\
  --validation\_file shared/dev\_full.csv \\
  --output\_dir output/bert\_single\_label \\
  --model\_mode cls-sum \\
  --num\_train\_epochs 3
```

### Step 3 — Train Discriminative LLMs

```bash
# Meditron-D
python Discriminative\_Models/train\_meditron\_cls.py \\
  --model\_name epfl-llm/meditron-7b \\
  --train\_csv shared/train\_full.csv \\
  --dev\_csv shared/dev\_full.csv \\
  --out\_dir output/meditron\_cls \\
  --bf16

# BioMistral-D
python Discriminative\_Models/train\_biomistral\_cls.py \\
  --model\_name BioMistral/BioMistral-7B \\
  --train\_csv shared/train\_full.csv \\
  --dev\_csv shared/dev\_full.csv \\
  --out\_dir output/biomistral\_cls \\
  --bf16

# OpenBioLLM-D
python Discriminative\_Models/train\_openbiollm\_cls.py \\
  --model\_name aaditya/Llama3-OpenBioLLM-8B \\
  --train\_csv shared/train\_full.csv \\
  --dev\_csv shared/dev\_full.csv \\
  --out\_dir output/openbiollm\_cls \\
  --bf16 --quant\_4bit
```

### Step 4 — Train Generative LLMs

```bash
# Meditron-G
python Generative\_Models/train\_meditron\_sft\_chat.py \\
  --model\_name epfl-llm/meditron-7b \\
  --train\_path shared/train\_chat.jsonl \\
  --dev\_path shared/dev\_chat.jsonl \\
  --out\_dir output/meditron\_sft \\
  --bf16

# Evaluate Meditron-G
python Generative\_Models/evaluate\_meditron\_sft.py \\
  --base\_model\_name epfl-llm/meditron-7b \\
  --adapter\_dir output/meditron\_sft \\
  --test\_path shared/test\_chat.jsonl \\
  --constrain
```

### Step 5 — Run the Gradio Demo

```bash
python Demo/app.py
# Open http://127.0.0.1:7860 in your browser
```

\---

## Key Design Contributions

1. **Tail-only supervision** for generative SFT - loss computed only on ICD completion tokens, not the full prompt
2. **Head+tail note cropping** - keep 40% from note start and 60% from note end to fit within context window while preserving clinically relevant sections
3. **Label mixing** for BioMistral-G - 40% head codes / 60% tail codes to correct head-code over-prediction bias
4. **Prefix trie constrained decoding** - forces generative models to only produce valid ICD codes from the label vocabulary
5. **QLoRA training** for OpenBioLLM-8B - 4-bit NF4 quantization enabling 8B model training on a single GPU
6. **SHAP explainability** in the Gradio demo - token-level attribution for any predicted ICD code

\---

## References

* Alsentzer et al. (2019). Publicly Available Clinical BERT Embeddings. NAACL Clinical NLP.
* Beltagy et al. (2020). Longformer: The Long-Document Transformer. arXiv.
* Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS.
* EPFL LLM Group (2023). Meditron-7B. HuggingFace.
* Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
* Huang et al. (2022). PLM-ICD: Automatic ICD Coding with Pretrained Language Models. ClinicalNLP.
* Johnson et al. (2023). MIMIC-IV. PhysioNet.
* Lundberg \& Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
* Mullenbach et al. (2018). Explainable Prediction of Medical Codes. NAACL.
* OpenBioLLM Team (2024). Llama3-OpenBioLLM-8B. HuggingFace.
* BioMistral Team (2023). BioMistral-7B. HuggingFace.
* von Werra et al. (2020). TRL: Transformer Reinforcement Learning. GitHub.

\---

## Citation

If you use this code or refer to this work please cite:

```bibtex
@mastersproject{shaik2025icd10,
  author    = {Namirah Imtieaz Shaik},
  title     = {Enhancing Automated ICD-10 Medical Coding with Large Language Models},
  school    = {California State University, Sacramento},
  year      = {2025},
  advisor   = {Dr. Haiquan Chen}
}
```

\---

## License

This project is licensed under the MIT License.

The MIMIC-IV dataset is subject to its own data use agreement available at PhysioNet. The pre-trained model weights (Meditron, BioMistral, OpenBioLLM, Bio\_ClinicalBERT) are subject to their respective licenses on HuggingFace.

