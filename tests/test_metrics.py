import numpy as np
import torch.nn as nn

from src.utils.metrics import benchmark_latency, compute_ece


def test_ece_zero_when_perfectly_calibrated():
    # All predictions correct with confidence 1.0 -> ECE 0.
    conf = np.ones(100)
    preds = np.zeros(100, dtype=int)
    labels = np.zeros(100, dtype=int)
    ece, bins = compute_ece(conf, preds, labels, n_bins=15)
    assert ece == 0.0
    assert len(bins["bin_counts"]) == 15


def test_ece_high_when_overconfident_and_wrong():
    conf = np.ones(100)  # confidence 1.0
    preds = np.zeros(100, dtype=int)
    labels = np.ones(100, dtype=int)  # always wrong -> accuracy 0
    ece, _ = compute_ece(conf, preds, labels, n_bins=15)
    assert ece > 0.9


def test_benchmark_latency_shape():
    class Flat(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3 * 8 * 8, 10)

        def forward(self, x):
            return self.lin(x.flatten(1))

    out = benchmark_latency(Flat(), image_size=8, n_warmup=2, n_runs=5, device="cpu")
    assert {"mean_ms", "p95_ms", "throughput_imgs_per_sec"} <= set(out)
