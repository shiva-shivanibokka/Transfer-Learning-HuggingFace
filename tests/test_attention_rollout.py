import numpy as np
import pytest
import torch

from src.utils.visualization import compute_attention_rollout

# ── Pure helper (network-free; runs everywhere, incl. lean CI) ─────────────────


def test_rollout_returns_one_value_per_patch():
    # 2 layers, batch 1, 4 heads, seq=5 (1 CLS + 4 patches)
    attentions = [torch.rand(1, 4, 5, 5) for _ in range(2)]
    out = compute_attention_rollout(attentions)
    assert out.shape == (4,)  # patches only, CLS removed
    assert np.all(out >= 0)


def test_rollout_peaked_attention_picks_that_patch():
    """Discriminating check: if every token attends almost entirely to one patch
    column, that patch must come out as the argmax of the CLS rollout.

    The old ``assert out >= 0`` was near-tautological — attention weights and
    their products are always non-negative — so it could not catch a rollout
    that pointed at the wrong patch. This asserts *where* the map peaks.
    """
    seq = 5  # 1 CLS + 4 patches
    heads = 4
    target_col = 3  # a patch column (seq index 3 → patch index 2 after CLS drop)
    attentions = []
    for _ in range(2):
        a = torch.full((1, heads, seq, seq), 0.02)
        a[:, :, :, target_col] = 1.0
        a = a / a.sum(dim=-1, keepdim=True)  # row-normalise like a softmax
        attentions.append(a)

    out = compute_attention_rollout(attentions)
    assert out.shape == (4,)
    assert int(out.argmax()) == target_col - 1


# ── App path: app.gradio_app._compute_attention_rollout_app ────────────────────
# This is the function where the real bug lived (it must call
# model(tensor, output_attentions=True) and read getattr(out, "attentions")).
# Importing app.gradio_app pulls in gradio + transformers, so these skip cleanly
# on a lean CI runner while running locally / in full verification.


def _load_app_gradio():
    pytest.importorskip("gradio")
    pytest.importorskip("transformers")
    from app import gradio_app

    return gradio_app


def test_app_rollout_returns_map_for_transformer_output():
    """A fake ViT-style output (an object with a non-empty ``.attentions`` tuple)
    must yield a normalised 2-D 224x224 attention map."""
    gradio_app = _load_app_gradio()

    heads, seq = 3, 17  # 16 patches → a clean 4x4 grid (side*side == num_patches)
    attentions = tuple(
        torch.softmax(torch.rand(1, heads, seq, seq), dim=-1) for _ in range(2)
    )

    class _Out:
        def __init__(self, att):
            self.attentions = att

    class _FakeViT:
        def __call__(self, tensor, output_attentions=False):
            # The function under test MUST request attentions explicitly.
            assert output_attentions is True
            return _Out(attentions)

    dummy = torch.rand(1, 3, 224, 224)
    arr = gradio_app._compute_attention_rollout_app(_FakeViT(), dummy)

    assert arr is not None
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape == (224, 224)
    assert np.isfinite(arr).all()
    # Normalised to [0, 1] by the app helper.
    assert arr.min() >= 0.0 and arr.max() <= 1.0 + 1e-5


def test_app_rollout_returns_none_for_cnn_output():
    """A CNN-style output (``attentions is None``) must return None, not crash."""
    gradio_app = _load_app_gradio()

    class _Out:
        attentions = None

    class _FakeCNN:
        def __call__(self, tensor, output_attentions=False):
            return _Out()

    dummy = torch.rand(1, 3, 224, 224)
    assert gradio_app._compute_attention_rollout_app(_FakeCNN(), dummy) is None
