"""
Shared evaluation metrics used across all three notebooks:
accuracy, per-class F1, confusion matrix, ECE, reliability diagram data,
inference latency benchmarking, and ONNX export helpers.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

# ── Classification metrics ────────────────────────────────────────────────────


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Returns accuracy, macro F1, weighted F1, per-class F1, and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    result = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_per_class": f1_per_class.tolist(),
        "confusion_matrix": cm.tolist(),
    }

    if class_names:
        result["per_class"] = {
            cls: float(f1_per_class[i])
            for i, cls in enumerate(class_names)
            if i < len(f1_per_class)
        }

    return result


# ── Calibration (ECE) ─────────────────────────────────────────────────────────


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, dict]:
    """
    Expected Calibration Error (ECE).

    Bins predictions by confidence, computes average accuracy in each bin,
    and returns the weighted mean absolute deviation.

    Args:
        confidences: Max softmax probability per sample, shape (N,)
        predictions: Predicted class index per sample, shape (N,)
        labels:      Ground-truth class index per sample, shape (N,)
        n_bins:      Number of bins for reliability diagram

    Returns:
        (ece_score, bin_data) where bin_data has 'bin_confidence', 'bin_accuracy',
        'bin_counts' for plotting the reliability diagram.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_confidence = []
    bin_accuracy = []
    bin_counts = []

    ece = 0.0
    n = len(confidences)

    for lower, upper in zip(bin_lowers, bin_uppers):
        mask = (confidences > lower) & (confidences <= upper)
        count = mask.sum()
        bin_counts.append(int(count))

        if count > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = (predictions[mask] == labels[mask]).mean()
            ece += (count / n) * abs(avg_conf - avg_acc)
            bin_confidence.append(float(avg_conf))
            bin_accuracy.append(float(avg_acc))
        else:
            bin_confidence.append(float((lower + upper) / 2))
            bin_accuracy.append(0.0)

    bin_data = {
        "bin_confidence": bin_confidence,
        "bin_accuracy": bin_accuracy,
        "bin_counts": bin_counts,
        "bin_lowers": bin_lowers.tolist(),
        "bin_uppers": bin_uppers.tolist(),
    }

    return float(ece), bin_data


# ── Temperature scaling ───────────────────────────────────────────────────────


class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration via temperature scaling (Guo et al., 2017).
    Learns a single scalar temperature T and rescales logits: p = softmax(logits / T).

    This is a logits-based API: callers extract logits themselves (HF models
    return a ``ModelOutput``, not a bare tensor) and pass the tensor in. If a
    ``ModelOutput`` is passed anyway, its ``.logits`` are used automatically.
    """

    def __init__(self, model: nn.Module = None):
        super().__init__()
        # ``model`` is retained only for backward compatibility; scaling itself
        # operates purely on logits tensors.
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    @staticmethod
    def _as_logits(x) -> torch.Tensor:
        """Accept either a raw logits tensor or an HF ModelOutput."""
        if isinstance(x, torch.Tensor):
            return x
        logits = getattr(x, "logits", None)
        if logits is not None:
            return logits
        raise TypeError(
            "TemperatureScaler expected a logits Tensor or a ModelOutput with "
            f".logits, got {type(x)!r}"
        )

    def forward(self, logits):
        return self._scale(self._as_logits(logits))

    def _scale(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)

    def fit_from_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> list[float]:
        """
        Fit the temperature directly on precomputed validation logits + labels
        using LBFGS (Guo et al.). Only the temperature parameter is optimised.

        Returns the list of NLL losses recorded during optimisation.
        """
        logits = self._as_logits(logits).detach().to(torch.float32)
        labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        losses: list[float] = []

        def closure():
            optimizer.zero_grad()
            scaled = self._scale(logits)
            loss = nll_criterion(scaled, labels)
            loss.backward()
            losses.append(loss.item())
            return loss

        optimizer.step(closure)
        return losses

    @property
    def T(self) -> float:
        return self.temperature.item()


# ── Inference latency benchmark ────────────────────────────────────────────────


@torch.no_grad()
def benchmark_latency(
    model: nn.Module,
    image_size: int = 224,
    n_warmup: int = 20,
    n_runs: int = 100,
    batch_size: int = 1,
    device: str = "cpu",
) -> dict:
    """
    Measure inference latency in ms/image for a PyTorch model.

    Args:
        model:      The model to benchmark.
        image_size: Input spatial size (assumes square, 3-channel).
        n_warmup:   Warmup runs (not timed).
        n_runs:     Timed runs.
        batch_size: Batch size per forward pass.
        device:     "cpu" or "cuda".

    Returns:
        Dict with mean_ms, std_ms, p50_ms, p95_ms, throughput_imgs_per_sec.
    """
    model = model.to(device).eval()
    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device)

    # Warmup
    for _ in range(n_warmup):
        _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000 / batch_size)

    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "throughput_imgs_per_sec": float(1000 / arr.mean()),
        "device": device,
        "batch_size": batch_size,
    }


# ── ONNX export ────────────────────────────────────────────────────────────────


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    image_size: int = 224,
    opset_version: int = 17,
    device: str = "cpu",
) -> str:
    """
    Export a PyTorch classification model to ONNX.

    Returns the path to the exported file.
    """
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    return output_path


def benchmark_onnx_latency(
    onnx_path: str,
    image_size: int = 224,
    n_warmup: int = 20,
    n_runs: int = 100,
) -> dict:
    """Benchmark ONNX Runtime inference latency."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    for _ in range(n_warmup):
        sess.run(None, {"pixel_values": dummy})

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"pixel_values": dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "mean_ms": float(arr.mean()),
        "p95_ms": float(np.percentile(arr, 95)),
        "throughput_imgs_per_sec": float(1000 / arr.mean()),
        "runtime": "onnxruntime_cpu",
    }
