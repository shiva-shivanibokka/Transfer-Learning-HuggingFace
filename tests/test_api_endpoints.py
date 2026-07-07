"""Contract tests for the JSON inference API (app/api.py) via FastAPI TestClient.

These exercise the *validation* layer, which by contract runs BEFORE any model
is loaded — so every assertion here is network-free and needs no weight
download. Guarded with importorskip so a lean CI runner (no fastapi/gradio)
skips the whole module cleanly.

We set the HF offline env vars defensively: if a handler ever fails to
short-circuit and reaches the Hub, the call fails fast instead of hanging the
suite on a network timeout.
"""

import base64
import io
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

pytest.importorskip("fastapi")
pytest.importorskip("gradio")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Building the app mounts Gradio + wires the /api router. Local example data
    # only — no network at import time. Skip gracefully if the import is too
    # heavy / unavailable in this environment.
    try:
        from app.serve import app
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"could not import app.serve ({exc})")
    return TestClient(app)


def _tiny_png_b64() -> str:
    """A small, genuinely decodable PNG so the 502 test gets PAST image decode
    and into the model-load path we monkeypatch to fail."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (120, 60, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── GET endpoints (no model download) ──────────────────────────────────────────


def test_get_models_ok(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    for key in ("vision", "text", "eurosat_classes", "emotion_classes"):
        assert key in body


def test_get_results_ok(client):
    r = client.get("/api/results")
    assert r.status_code == 200
    body = r.json()
    for key in ("vision_strategy", "text", "clip_prompts"):
        assert key in body


# ── POST /api/vision validation (before model load) ────────────────────────────


def test_vision_unknown_model_400(client):
    r = client.post("/api/vision", json={"image": "abc", "model": "NoSuchModel"})
    assert r.status_code == 400


def test_vision_malformed_base64_400(client):
    r = client.post(
        "/api/vision", json={"image": "!!!not-valid-base64!!!", "model": "DINOv2-Base"}
    )
    assert r.status_code == 400


def test_vision_empty_image_400(client):
    r = client.post("/api/vision", json={"image": "", "model": "DINOv2-Base"})
    assert r.status_code == 400


def test_vision_oversized_payload_413(client):
    # A raw base64 string longer than 4,000,000 chars must be rejected as too
    # large BEFORE any decode/model work.
    huge = "A" * 4_000_001
    r = client.post("/api/vision", json={"image": huge, "model": "DINOv2-Base"})
    assert r.status_code == 413


# ── POST /api/text validation ──────────────────────────────────────────────────


def test_text_empty_400(client):
    r = client.post("/api/text", json={"text": "   ", "model": "RoBERTa"})
    assert r.status_code == 400


def test_text_too_long_400(client):
    r = client.post("/api/text", json={"text": "a" * 2001, "model": "RoBERTa"})
    assert r.status_code == 400


# ── POST /api/clip-search validation ───────────────────────────────────────────


def test_clip_empty_query_400(client):
    r = client.post("/api/clip-search", json={"query": "   ", "k": 5})
    assert r.status_code == 400


# ── 502 path must not leak internal error text ─────────────────────────────────


def test_vision_model_load_failure_is_generic_502(client, monkeypatch):
    """When the model loader raises, the API must return 502 with a GENERIC
    detail — the raw exception text must never reach the client body."""
    from app import gradio_app as G

    secret = "SUPER_SECRET_INTERNAL_TRACEBACK_abc123"

    def _boom(_name):
        raise RuntimeError(secret)

    monkeypatch.setattr(G, "_load_vision_model", _boom)

    r = client.post(
        "/api/vision", json={"image": _tiny_png_b64(), "model": "DINOv2-Base"}
    )
    assert r.status_code == 502
    assert secret not in r.text
