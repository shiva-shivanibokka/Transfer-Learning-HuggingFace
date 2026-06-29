import numpy as np
import torch

from src.utils.visualization import compute_attention_rollout


def test_rollout_returns_one_value_per_patch():
    # 2 layers, batch 1, 4 heads, seq=5 (1 CLS + 4 patches)
    attentions = [torch.rand(1, 4, 5, 5) for _ in range(2)]
    out = compute_attention_rollout(attentions)
    assert out.shape == (4,)  # patches only, CLS removed
    assert np.all(out >= 0)
