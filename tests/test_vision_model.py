"""Unit tests for the vision freeze-strategy / trainable-param logic in
src/vision/model.py.

Network-free: we drive ``_apply_strategy`` + ``count_trainable_params`` against a
tiny fake nn.Module that mimics the attribute names the freeze code keys off
(``.vit`` backbone + ``.classifier`` head), so no HuggingFace download is needed.
"""

import pytest

pytest.importorskip("transformers")  # src.vision.model imports transformers at top

import torch.nn as nn  # noqa: E402

from src.vision.model import (  # noqa: E402
    _apply_strategy,
    count_trainable_params,
)


def _fake_vit() -> nn.Module:
    class FakeViT(nn.Module):
        def __init__(self):
            super().__init__()
            # Backbone under the ``.vit`` attribute (matches ViT/DINOv2 naming).
            self.vit = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
            # Head under ``.classifier`` — the freeze code always keeps this trainable.
            self.classifier = nn.Linear(8, 3)

        def forward(self, x):
            return self.classifier(self.vit(x))

    return FakeViT()


def test_linear_probe_freezes_backbone_keeps_head():
    model = _fake_vit()
    _apply_strategy(
        model,
        "vit_base",
        {"freeze_backbone": True, "unfreeze_last_n_blocks": 0},
    )

    counts = count_trainable_params(model)
    head_params = sum(p.numel() for p in model.classifier.parameters())
    backbone_params = sum(p.numel() for p in model.vit.parameters())

    # Only the classification head trains.
    assert counts["trainable_params"] == head_params
    assert counts["trainable_params"] < backbone_params
    assert counts["trainable_params"] < counts["total_params"]
    assert all(not p.requires_grad for p in model.vit.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_full_finetune_trains_everything():
    model = _fake_vit()
    _apply_strategy(model, "vit_base", {"freeze_backbone": False})

    counts = count_trainable_params(model)
    assert counts["trainable_params"] == counts["total_params"]
    assert counts["trainable_pct"] == 100
    assert all(p.requires_grad for p in model.parameters())


def test_linear_probe_trains_far_fewer_params_than_full():
    lp = _fake_vit()
    _apply_strategy(lp, "vit_base", {"freeze_backbone": True, "unfreeze_last_n_blocks": 0})
    ft = _fake_vit()
    _apply_strategy(ft, "vit_base", {"freeze_backbone": False})

    assert (
        count_trainable_params(lp)["trainable_params"]
        < count_trainable_params(ft)["trainable_params"]
    )
