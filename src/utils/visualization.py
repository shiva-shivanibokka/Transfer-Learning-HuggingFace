"""
Shared plotting utilities: confusion matrix, reliability diagram (calibration),
learning curves, latency/accuracy scatter, attention rollout for ViT/DINOv2.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ── Colour palette ────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "resnet50": "#E15759",
    "efficientnet_b0": "#F28E2B",
    "vit_base": "#4E79A7",
    "dinov2_base": "#59A14F",
    "distilbert": "#B07AA1",
    "roberta": "#76B7B2",
    "modernbert": "#FF9DA7",
}


# ── Confusion matrix ──────────────────────────────────────────────────────────


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(
                j,
                i,
                val,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Reliability diagram (calibration) ─────────────────────────────────────────


def plot_reliability_diagram(
    bin_data_before: dict,
    bin_data_after: Optional[dict] = None,
    ece_before: float = 0.0,
    ece_after: float = 0.0,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot reliability diagram showing confidence vs accuracy.
    Optionally overlays pre- and post-temperature-scaling curves.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Perfect calibration line
    ax.plot(
        [0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration", alpha=0.7
    )

    lowers = bin_data_before["bin_lowers"]
    uppers = bin_data_before["bin_uppers"]
    centers = [(lo + up) / 2 for lo, up in zip(lowers, uppers)]

    # Before calibration
    ax.bar(
        centers,
        bin_data_before["bin_accuracy"],
        width=[(up - lo) * 0.9 for lo, up in zip(lowers, uppers)],
        alpha=0.5,
        color="#4E79A7",
        label=f"Before scaling (ECE={ece_before:.3f})",
    )
    ax.plot(
        centers,
        bin_data_before["bin_accuracy"],
        "o-",
        color="#4E79A7",
        linewidth=1.5,
        markersize=4,
    )

    # After calibration
    if bin_data_after:
        ax.bar(
            centers,
            bin_data_after["bin_accuracy"],
            width=[(up - lo) * 0.9 for lo, up in zip(lowers, uppers)],
            alpha=0.4,
            color="#F28E2B",
            label=f"After scaling (ECE={ece_after:.3f})",
        )
        ax.plot(
            centers,
            bin_data_after["bin_accuracy"],
            "s--",
            color="#F28E2B",
            linewidth=1.5,
            markersize=4,
        )

    ax.set_xlabel("Mean predicted confidence", fontsize=12)
    ax.set_ylabel("Fraction of correct predictions", fontsize=12)
    ax.set_title(f"Reliability Diagram — {model_name}", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Learning curves ───────────────────────────────────────────────────────────


def plot_learning_curves(
    results: dict[str, list[float]],
    x_label: str = "Epoch",
    y_label: str = "Accuracy",
    title: str = "Learning Curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot accuracy vs epoch for multiple models."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_key, values in results.items():
        color = MODEL_COLORS.get(model_key, None)
        ax.plot(
            range(1, len(values) + 1),
            values,
            marker="o",
            markersize=4,
            label=model_key,
            color=color,
            linewidth=2,
        )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_data_efficiency_curves(
    results: dict[str, dict[float, float]],
    title: str = "Data Efficiency: Accuracy vs Training Set Fraction",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    results: {model_key: {data_fraction: accuracy}}
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for model_key, frac_acc in results.items():
        fracs = sorted(frac_acc.keys())
        accs = [frac_acc[f] for f in fracs]
        color = MODEL_COLORS.get(model_key, None)
        ax.plot(
            [f * 100 for f in fracs],
            accs,
            marker="o",
            label=model_key,
            color=color,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Training set size (%)", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 100])
    ax.set_xticklabels(["1%", "5%", "10%", "100%"])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_latency_accuracy_scatter(
    model_results: dict[str, dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of accuracy vs latency (ms/image) for all models.
    model_results: {model_key: {"accuracy": float, "latency_ms": float, "params_M": float}}
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for model_key, info in model_results.items():
        color = MODEL_COLORS.get(model_key, "#888")
        size = info.get("params_M", 10) * 3  # bubble size proportional to params
        ax.scatter(
            info["latency_ms"],
            info["accuracy"],
            s=size,
            color=color,
            alpha=0.8,
            zorder=5,
        )
        ax.annotate(
            model_key,
            (info["latency_ms"], info["accuracy"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )

    ax.set_xlabel("CPU Inference Latency (ms/image)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Accuracy vs. Latency (bubble size = #parameters)", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Attention Rollout for ViT/DINOv2 ──────────────────────────────────────────


def compute_attention_rollout(
    attentions: list[torch.Tensor],
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """
    Abnar & Zuidema (2020) attention rollout for Vision Transformers.

    Propagates attention weights through all transformer layers to produce
    a single (num_patches,) attention map from the [CLS] token.

    Args:
        attentions: List of attention tensors, one per layer.
                    Each tensor has shape (batch, heads, seq_len, seq_len).
        discard_ratio: Fraction of lowest attention values to zero out per layer
                       (helps suppress noise from uniform attention heads).

    Returns:
        A (num_patches,) numpy array of attention scores (excluding CLS).
    """
    result = torch.eye(attentions[0].shape[-1])

    for attention in attentions:
        # Average over heads: (batch, seq_len, seq_len) → (seq_len, seq_len)
        attn = attention[0].mean(dim=0)

        # Discard lowest attention weights (set to 0 before normalising)
        flat = attn.view(-1)
        threshold = flat.kthvalue(int(discard_ratio * flat.numel())).values
        attn = attn * (attn >= threshold).float()

        # Add residual and re-normalise rows
        attn = attn + torch.eye(attn.size(0))
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        result = attn @ result

    # CLS token attends to all patches — return its row (excluding itself)
    mask = result[0, 1:]
    return mask.numpy()


def plot_attention_rollout(
    image: Image.Image,
    attention_map: np.ndarray,
    num_patches_side: int,
    title: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlay the attention rollout heatmap on the original image.

    Args:
        image:            Original PIL image.
        attention_map:    (num_patches,) attention scores.
        num_patches_side: sqrt(num_patches), e.g. 14 for ViT-B/16 on 224×224.
        title:            Plot title.
        save_path:        If given, save figure to this path.
    """
    img = np.array(image.resize((224, 224)))
    attn = attention_map.reshape(num_patches_side, num_patches_side)

    # Resize attention map to image size
    attn_img = Image.fromarray(attn.astype(np.float32))
    attn_img = np.array(attn_img.resize((224, 224), Image.BILINEAR))
    attn_img = (attn_img - attn_img.min()) / (attn_img.max() - attn_img.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(attn_img, cmap="hot")
    axes[1].set_title("Attention Rollout", fontsize=11)
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(attn_img, cmap="hot", alpha=0.6)
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
