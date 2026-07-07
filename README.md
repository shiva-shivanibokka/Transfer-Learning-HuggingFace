---
title: Transfer Learning Project
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Transfer Learning & HuggingFace Model Hub Showcase

[![CI](https://github.com/shiva-shivanibokka/Transfer-Learning-HuggingFace/actions/workflows/ci.yml/badge.svg)](https://github.com/shiva-shivanibokka/Transfer-Learning-HuggingFace/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-ee4c2c)

An empirical study of transfer learning efficiency across **4 vision architectures** (ResNet-50, EfficientNet-B0, ViT-Base, DINOv2-Base) and **2 text encoders** (RoBERTa, ModernBERT) on a niche satellite domain. Answers three questions no other project answers:

1. Does DINOv2's self-supervised pretraining transfer better than supervised ViT to satellite imagery?
2. At what labeled-data crossover point does CLIP zero-shot beat fine-tuned CNN?
3. How sensitive is CLIP zero-shot accuracy to prompt wording — and does ensembling recover it?

---

## Key Results

**Three findings, from the experiments below:**

1. **Self-supervised features transfer far better *frozen*.** DINOv2's **linear probe hits 95.4%** — its frozen features nearly match everyone else's *fully fine-tuned* models, while the CNNs' linear probes languish at ~78%. A **17-point gap** with zero backbone training.
2. **…but full fine-tuning DINOv2 on scarce data backfires.** At 1% data, full fine-tune collapses to **29%** (overfitting 86M params on 162 images), whereas **ViT-Base stays at 90.5%**. Lesson: freeze DINOv2, fine-tune ViT.
3. **CLIP zero-shot is weak *and* prompt-fragile on satellite imagery.** Accuracy swings **42%→52%** across five prompt templates; a **5-template ensemble recovers to 53.1%** — but that is still far below a fine-tuned CNN's linear probe (78%), so *any* labeled data makes fine-tuning the better choice.

### Vision: Strategy Comparison (EuroSAT, 100% data)

Test accuracy by fine-tuning strategy. Latency is single-image PyTorch CPU inference (ONNX in parentheses).

| Model | Family | Year | Linear Probe | Partial Unfreeze | Full Fine-tune | CPU Latency (ms) |
|---|---|---|---|---|---|---|
| ResNet-50 | CNN | 2015 | 77.8% | 98.0% | 98.5% | 589 (23 ONNX) |
| EfficientNet-B0 | CNN | 2019 | 79.5% | 92.9% | 98.0% | **25 (6 ONNX)** |
| ViT-Base | Transformer | 2020 | 88.9% | 96.2% | **99.0%** | 1040 (590 ONNX) |
| DINOv2-Base | Self-supervised | 2023 | **95.4%** | 97.8% | 96.9% | 131 (163 ONNX) |

*EfficientNet-B0 is the deployment sweet spot: 98% accuracy at 6 ms/image (ONNX). ViT-Base is most accurate but ~100× slower.*

### Vision: Data Efficiency (full fine-tune, test accuracy)

| Model | 1% data | 5% data | 10% data | 100% data |
|---|---|---|---|---|
| ResNet-50 | 47.3% | 87.4% | 95.5% | 98.5% |
| EfficientNet-B0 | 62.4% | 88.2% | 95.0% | 98.0% |
| ViT-Base | **90.5%** | **94.2%** | **97.2%** | **99.0%** |
| DINOv2-Base | 29.1% | 66.5% | 91.1% | 96.9% |

*ViT-Base is remarkably data-efficient — 90.5% from just 162 labeled images. DINOv2 full fine-tune is the opposite: it needs data (or, better, a frozen probe — see 95.4% above).*

### Text: RoBERTa vs ModernBERT + Calibration (dair-ai/emotion)

DistilBERT included as an efficiency reference. ECE = Expected Calibration Error (lower is better); temperature scaling is fit on the validation set.

| Model | Test Acc | F1 Macro | ECE Before | ECE After | Temperature T |
|---|---|---|---|---|---|
| RoBERTa | 92.7% | 87.9% | 0.0288 | **0.0230** | 1.169 |
| ModernBERT | 92.7% | **88.9%** | 0.0386 | **0.0305** | 1.293 |
| DistilBERT (ref) | 92.9% | 88.5% | 0.0308 | 0.0273 | 1.157 |

*All three land within 0.2% accuracy; ModernBERT edges F1. Temperature scaling reduces calibration error in every case (T > 1 ⇒ the raw models were mildly overconfident).*

### CLIP: Prompt Sensitivity (EuroSAT zero-shot)

| Prompt Template | Accuracy |
|---|---|
| "a photo of {cls}" | 42.1% |
| "a satellite image of {cls} land use" | 49.8% |
| "an aerial photograph showing {cls}" | 43.2% |
| "a remote sensing image of {cls}" | 45.6% |
| "{cls} viewed from above" | 51.5% |
| **Ensemble (all 5)** | **53.1%** |

*Domain-aware prompts beat the generic "a photo of" by ~10 points; averaging text embeddings across all five templates beats the best single template.*

> Reproduce: `python scripts/run_grid_resumable.py` (vision + text + CLIP), then `python scripts/build_clip_index.py`. Trained on an RTX 4060.

---

## Architecture

```
Transfer-Learning-HuggingFace/
├── configs/
│   ├── vision_config.py      # Model registry, training strategies, data fractions
│   ├── text_config.py        # Text model registry, calibration config
│   └── clip_config.py        # CLIP model, prompt templates, class descriptions
│
├── src/
│   ├── vision/
│   │   ├── model.py          # Model factory: load + freeze strategy application
│   │   └── trainer.py        # HF Trainer wrapper + ONNX export + latency benchmark
│   ├── text/
│   │   └── trainer.py        # Text fine-tuning + temperature scaling calibration
│   ├── clip/
│   │   └── pipeline.py       # Zero-shot, few-shot, retrieval, prompt ensembling
│   └── utils/
│       ├── data.py            # EuroSAT loading, stratified fraction sampling
│       ├── metrics.py         # ECE, temperature scaler, latency benchmark, ONNX
│       ├── mlflow_utils.py    # MLflow logging helpers
│       └── visualization.py   # Confusion matrix, reliability diagram, attention rollout
│
├── scripts/
│   ├── train_vision.py        # CLI: run vision experiments
│   ├── train_text.py          # CLI: run text experiments
│   └── train_clip.py          # CLI: run CLIP pipeline
│
├── notebooks/
│   ├── 01_vision_cnn_vit_dinov2.ipynb          # Notebook 1: Vision study
│   ├── 02_text_roberta_modernbert_calibration.ipynb  # Notebook 2: Text + calibration
│   └── 03_clip_zeroshot_prompt_engineering.ipynb    # Notebook 3: CLIP study
│
├── app/
│   └── gradio_app.py          # 4-tab Gradio app (deploy to HF Spaces)
│
└── results/                   # Saved per-run JSON results + MLflow logs
```

---

## Setup

```bash
git clone https://github.com/shiva-shivanibokka/Transfer-Learning-HuggingFace
cd Transfer-Learning-HuggingFace
pip install -r requirements.txt        # full training stack
# or, for serving the demo only:
pip install -r requirements-app.txt
cp .env.example .env
# Add HF_TOKEN if you want to push models to the Hub
```

Run the tests and linter:

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

## Running experiments

```bash
# Vision: single quick run (EfficientNet, full fine-tune, 10% data)
python scripts/train_vision.py --model efficientnet_b0 --strategy full_finetune --fraction 0.1

# Vision: full strategy comparison study
python scripts/train_vision.py --study strategy_comparison

# Vision: data efficiency study
python scripts/train_vision.py --study data_efficiency

# Text: train all models + calibration
python scripts/train_text.py

# CLIP: zero-shot + few-shot + retrieval
python scripts/train_clip.py

# Launch MLflow UI
mlflow ui --port 5000
```

## Notebooks (run in order)

```bash
jupyter notebook
# Open notebooks/01_vision_cnn_vit_dinov2.ipynb
# Open notebooks/02_text_roberta_modernbert_calibration.ipynb
# Open notebooks/03_clip_zeroshot_prompt_engineering.ipynb
```

## Gradio demo

```bash
python app/gradio_app.py
# Opens at http://localhost:7860
```

---

## Deployment — Hugging Face Spaces (free CPU tier)

The Gradio app deploys to HF Spaces using the included `Dockerfile`, which
installs only `requirements-app.txt` (slim inference set) — not the full
training stack — so the Space build stays fast on the free tier.

1. Create a new Space → **SDK: Docker** → Hardware: **CPU basic (free)**.
2. Add this YAML front matter to the top of the Space's `README.md` so HF
   builds it as a Docker app on port 7860:
   ```yaml
   ---
   title: Transfer Learning Demo
   emoji: 🛰️
   sdk: docker
   app_port: 7860
   ---
   ```
3. Push this repo to the Space remote (or connect the GitHub repo).
4. Commit your trained artifacts under `results/` so the app serves real
   weights (see **Populating results** below). Large `.pt` files use Git LFS.

Free CPU Spaces sleep after 48h idle and cold-start on the next visit —
expected on the free tier. The app fails loudly at startup if required env
vars are missing rather than dying mid-request.

### Populating results
The app loads checkpoints produced by the training scripts (paths are defined
once in `src/utils/paths.py`, shared by trainers and the app):
- `results/vision/<model>/full_finetune/frac1.00/best_model.pt`
- `results/text/<model>/best_model.pt` and `temperature.json`
- `results/clip/retrieval_index.pt` (built in notebook 03)

Train locally on a GPU, then commit these files for the Space to serve. Until
they exist the app logs a clear warning and serves randomly-initialised
weights, so an empty Space never crashes — but its predictions are meaningless
until real checkpoints are committed.

---

## What's technically new in this project

| Feature | Why it's not in other repos |
|---|---|
| **DINOv2** (Meta AI 2023) | No other repo uses self-supervised ViT pretraining |
| **ViT attention rollout** | Correct ViT visualization (Abnar & Zuidema 2020) — Grad-CAM is invalid for ViTs |
| **Temperature scaling / ECE** | Calibration on a classification model — not done in any other repo |
| **ModernBERT** (2024) | 2024 BERT successor with RoPE + Flash Attention — current research |
| **CLIP prompt sensitivity study** | Quantifies accuracy variance across 5 prompt templates on a niche domain |
| **Prompt ensembling** | Averages text embeddings across templates — mirrors Radford et al. (2021) technique |
| **ONNX export + latency benchmark** | Production deployment answer: latency vs accuracy tradeoff |
| **HuggingFace Hub push** | Actually publishes trained models — makes work visible on HF profile |

---

## References

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)
- [Learning Transferable Visual Models From Natural Language (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [On Calibration of Modern Neural Networks (Guo et al., 2017)](https://arxiv.org/abs/1706.04599)
- [Quantifying Attention Flow in Transformers (Abnar & Zuidema, 2020)](https://arxiv.org/abs/2005.00928)
- [ModernBERT (Warner et al., 2024)](https://arxiv.org/abs/2412.13663)
