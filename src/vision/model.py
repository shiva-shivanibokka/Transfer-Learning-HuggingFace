"""
Vision model factory: loads pretrained models from HuggingFace Hub / timm,
replaces the classification head, and applies the freezing strategy.

Supports: ResNet-50, EfficientNet-B0, ViT-Base, DINOv2-Base.
Each model is returned as a standard nn.Module with a .forward(x) → logits interface.
"""

from __future__ import annotations

import torch.nn as nn
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForImageClassification,
)

# ── Model builders ─────────────────────────────────────────────────────────────


def build_model(
    model_key: str,
    num_classes: int,
    id2label: dict,
    label2id: dict,
    strategy: str = "full_finetune",
    dropout: float = 0.1,
    pretrained: bool = True,
) -> tuple[nn.Module, AutoImageProcessor]:
    """
    Build and return a (model, processor) pair ready for training.

    Args:
        model_key:   One of "resnet50", "efficientnet_b0", "vit_base", "dinov2_base".
        num_classes: Number of output classes.
        id2label:    Mapping from class index to label name.
        label2id:    Mapping from label name to class index.
        strategy:    "linear_probe" | "partial_unfreeze" | "full_finetune"
        dropout:     Dropout rate for the classification head.
        pretrained:  Whether to load pretrained weights.

    Returns:
        (model, processor) where processor converts PIL images to tensors.
    """
    from configs.vision_config import STRATEGIES, VISION_MODELS

    hf_id = VISION_MODELS[model_key]["hf_id"]
    strategy_cfg = STRATEGIES[strategy]

    # Load processor
    try:
        processor = AutoImageProcessor.from_pretrained(hf_id)
    except Exception:
        processor = AutoFeatureExtractor.from_pretrained(hf_id)

    # Load model with a fresh classification head
    model = AutoModelForImageClassification.from_pretrained(
        hf_id,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # allows replacing the head
    )

    # Apply freezing strategy
    _apply_strategy(model, model_key, strategy_cfg)

    return model, processor


def _apply_strategy(model: nn.Module, model_key: str, strategy_cfg: dict) -> None:
    """
    Freeze/unfreeze parameters according to the chosen strategy.
    The classification head is always trainable.
    """
    if not strategy_cfg["freeze_backbone"]:
        # Full fine-tune: all parameters trainable
        for p in model.parameters():
            p.requires_grad = True
        return

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Always unfreeze the classification head
    _unfreeze_head(model, model_key)

    n_blocks = strategy_cfg.get("unfreeze_last_n_blocks", 0)
    if n_blocks > 0:
        _unfreeze_last_n_blocks(model, model_key, n_blocks)


def _unfreeze_head(model: nn.Module, model_key: str) -> None:
    """Unfreeze the classification head for the given model family."""
    if model_key in ("resnet50", "efficientnet_b0"):
        # HuggingFace wraps these as timm-style models;
        # the head is model.classifier
        for name, p in model.named_parameters():
            if "classifier" in name or "head" in name:
                p.requires_grad = True
    elif model_key in ("vit_base", "dinov2_base"):
        # ViT/DINOv2: head is model.classifier
        for name, p in model.named_parameters():
            if "classifier" in name:
                p.requires_grad = True


def _unfreeze_last_n_blocks(model: nn.Module, model_key: str, n: int) -> None:
    """Unfreeze the last n transformer blocks / residual stages."""
    if model_key == "resnet50":
        # ResNet has 4 layers (layer1–layer4); unfreeze last n
        stages = [model.resnet.encoder.stages[-i] for i in range(1, n + 1) if i <= 4]
        for stage in stages:
            for p in stage.parameters():
                p.requires_grad = True

    elif model_key == "efficientnet_b0":
        # EfficientNet blocks
        blocks = list(model.efficientnet.encoder.blocks)
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

    elif model_key in ("vit_base", "dinov2_base"):
        # ViT/DINOv2 have 12 transformer layers
        encoder = getattr(model, "vit", None) or getattr(model, "dinov2", None)
        if encoder and hasattr(encoder, "encoder"):
            layers = encoder.encoder.layer
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True


def count_trainable_params(model: nn.Module) -> dict:
    """Return trainable and total parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 2) if total > 0 else 0,
    }
