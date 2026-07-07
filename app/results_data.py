"""Static experiment results + model registry served by the JSON API.

These are the real numbers from the training grid (also in the README tables).
Bundling them here means the deployed Space needs no local results/ files to
power the frontend's Results dashboard and model info cards.
"""

from __future__ import annotations

import os

# Single source of truth for the HF Hub account. gradio_app.py reads the same
# HF_HUB_USER env var, so a fork with a different account gets consistent
# hub_ids in the /models payload and in the live model loaders.
HUB_USER = os.getenv("HF_HUB_USER", "shiva-1993")

# Vision strategy comparison @ 100% data. latency_ms = single-image PyTorch CPU.
VISION_STRATEGY = [
    {"model": "ResNet-50", "family": "CNN", "year": 2015, "linear_probe": 77.8, "partial_unfreeze": 98.0, "full_finetune": 98.5, "latency_ms": 589, "onnx_ms": 23},
    {"model": "EfficientNet-B0", "family": "CNN", "year": 2019, "linear_probe": 79.5, "partial_unfreeze": 92.9, "full_finetune": 98.0, "latency_ms": 25, "onnx_ms": 6},
    {"model": "ViT-Base", "family": "Transformer", "year": 2020, "linear_probe": 88.9, "partial_unfreeze": 96.2, "full_finetune": 99.0, "latency_ms": 1040, "onnx_ms": 590},
    {"model": "DINOv2-Base", "family": "Self-supervised", "year": 2023, "linear_probe": 95.4, "partial_unfreeze": 97.8, "full_finetune": 96.9, "latency_ms": 131, "onnx_ms": 163},
]

# Vision data efficiency (full fine-tune, test accuracy).
VISION_DATA_EFFICIENCY = [
    {"model": "ResNet-50", "p1": 47.3, "p5": 87.4, "p10": 95.5, "p100": 98.5},
    {"model": "EfficientNet-B0", "p1": 62.4, "p5": 88.2, "p10": 95.0, "p100": 98.0},
    {"model": "ViT-Base", "p1": 90.5, "p5": 94.2, "p10": 97.2, "p100": 99.0},
    {"model": "DINOv2-Base", "p1": 29.1, "p5": 66.5, "p10": 91.1, "p100": 96.9},
]

# Text: RoBERTa vs ModernBERT (+ DistilBERT reference) + calibration.
TEXT_RESULTS = [
    {"model": "RoBERTa", "accuracy": 92.7, "f1_macro": 87.9, "ece_before": 0.0288, "ece_after": 0.0230, "temperature": 1.169},
    {"model": "ModernBERT", "accuracy": 92.7, "f1_macro": 88.9, "ece_before": 0.0386, "ece_after": 0.0305, "temperature": 1.293},
    {"model": "DistilBERT", "accuracy": 92.9, "f1_macro": 88.5, "ece_before": 0.0308, "ece_after": 0.0273, "temperature": 1.157},
]

# CLIP zero-shot prompt sensitivity (EuroSAT).
CLIP_PROMPTS = [
    {"template": "a photo of {cls}", "accuracy": 42.1},
    {"template": "a satellite image of {cls} land use", "accuracy": 49.8},
    {"template": "an aerial photograph showing {cls}", "accuracy": 43.2},
    {"template": "a remote sensing image of {cls}", "accuracy": 45.6},
    {"template": "{cls} viewed from above", "accuracy": 51.5},
    {"template": "Ensemble (all 5)", "accuracy": 53.1, "ensemble": True},
]

# Model registry powering the frontend dropdowns + info cards.
VISION_MODELS = [
    {"key": "ResNet-50", "hub_id": f"{HUB_USER}/eurosat-resnet50", "family": "CNN", "year": 2015, "params_m": 25.6, "accuracy": 98.5, "latency_ms": 589, "onnx_ms": 23, "has_attention": False},
    {"key": "EfficientNet-B0", "hub_id": f"{HUB_USER}/eurosat-efficientnet-b0", "family": "CNN", "year": 2019, "params_m": 5.3, "accuracy": 98.0, "latency_ms": 25, "onnx_ms": 6, "has_attention": False},
    {"key": "ViT-Base", "hub_id": f"{HUB_USER}/eurosat-vit-base", "family": "Transformer", "year": 2020, "params_m": 86.0, "accuracy": 99.0, "latency_ms": 1040, "onnx_ms": 590, "has_attention": True},
    {"key": "DINOv2-Base", "hub_id": f"{HUB_USER}/eurosat-dinov2-base", "family": "Self-supervised", "year": 2023, "params_m": 86.0, "accuracy": 96.9, "latency_ms": 131, "onnx_ms": 163, "has_attention": True},
]
TEXT_MODELS = [
    {"key": "RoBERTa", "hub_id": f"{HUB_USER}/emotion-roberta", "params_m": 125, "year": 2019, "accuracy": 92.7, "temperature": 1.169},
    {"key": "ModernBERT", "hub_id": f"{HUB_USER}/emotion-modernbert", "params_m": 149, "year": 2024, "accuracy": 92.7, "temperature": 1.293},
]

# Source of truth for class label ORDER (maps to prediction indices).
# gradio_app.py imports these; do not reorder.
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
EMOTION_CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def results_payload() -> dict:
    return {
        "vision_strategy": VISION_STRATEGY,
        "vision_data_efficiency": VISION_DATA_EFFICIENCY,
        "text": TEXT_RESULTS,
        "clip_prompts": CLIP_PROMPTS,
        "findings": [
            "DINOv2 frozen features transfer far better: 95.4% linear probe vs ~78% for CNNs.",
            "Full fine-tuning DINOv2 on 1% data collapses to 29%; ViT-Base stays at 90.5%.",
            "CLIP zero-shot is weak and prompt-fragile (42-52%); ensembling recovers to 53.1%.",
        ],
    }


def models_payload() -> dict:
    return {"vision": VISION_MODELS, "text": TEXT_MODELS,
            "eurosat_classes": EUROSAT_CLASSES, "emotion_classes": EMOTION_CLASSES}
