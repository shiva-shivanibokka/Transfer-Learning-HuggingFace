"""JSON inference API for the custom (Next.js) frontend.

Reuses the model-loading + inference helpers from gradio_app so there is ONE
inference path; this layer just returns raw JSON (probability arrays, base64
images) instead of Gradio components. Mounted at /api/* by build_app().
"""

from __future__ import annotations

import base64
import io
import time

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Request
from PIL import Image
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from app import gradio_app as G
from app import results_data as R
from src.utils.logging_utils import get_logger

log = get_logger("api")
router = APIRouter(prefix="/api")

# Per-client-IP rate limiter; registered on the FastAPI app in build_app().
limiter = Limiter(key_func=get_remote_address)

# ── validation limits ─────────────────────────────────────────────────────────
MAX_B64_LEN = 4_000_000        # raw base64 string length cap → 413 if exceeded
MAX_IMAGE_DIM = 4096           # px on either side → 413 if exceeded
MAX_TEXT_LEN = 2000            # chars → 400 if exceeded
# Cap decoded pixel count to defuse decompression bombs before .convert() runs.
Image.MAX_IMAGE_PIXELS = 8_000_000


# ── helpers ──────────────────────────────────────────────────────────────────
def _decode_image(b64: str) -> Image.Image:
    if not b64 or not b64.strip():
        raise HTTPException(400, "empty image")
    if len(b64) > MAX_B64_LEN:
        raise HTTPException(413, "image too large")
    if "," in b64[:64]:  # strip data URL prefix if present
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        log.exception("base64 decode failed")
        raise HTTPException(400, "malformed image")
    if not raw:
        raise HTTPException(400, "empty image")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        log.exception("image decode failed")
        raise HTTPException(400, "invalid image")
    if img.width > MAX_IMAGE_DIM or img.height > MAX_IMAGE_DIM:
        raise HTTPException(413, "image dimensions too large")
    return img


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _probs_list(probs: np.ndarray, classes: list[str]) -> list[dict]:
    return [{"label": c, "prob": float(p)} for c, p in zip(classes, probs)]


# ── request models ───────────────────────────────────────────────────────────
class VisionReq(BaseModel):
    image: str  # base64 (optionally a data URL)
    model: str = "DINOv2-Base"


class TextReq(BaseModel):
    text: str
    model: str = "RoBERTa"


class ClipReq(BaseModel):
    query: str
    k: int = 5


# ── endpoints ────────────────────────────────────────────────────────────────
@router.get("/models")
def models():
    return R.models_payload()


@router.get("/results")
def results():
    return R.results_payload()


@router.post("/vision")
@limiter.limit("20/minute")
def vision(request: Request, req: VisionReq):
    # Validate everything BEFORE loading any model.
    if req.model not in G.VISION_MODEL_IDS:
        raise HTTPException(400, "unknown model")
    pil = _decode_image(req.image)
    try:
        model = G._load_vision_model(req.model)
    except Exception:
        log.exception("vision model load failed for %s", req.model)
        raise HTTPException(502, "model load failed")
    tensor = G._vision_transform(pil)

    # Single timed forward pass for latency (static benchmark latency lives in
    # results_data.py for display).
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(tensor)
        latency_ms = (time.perf_counter() - t0) * 1000
    logits = (out.logits if hasattr(out, "logits") else out).squeeze(0).cpu().numpy()
    probs = G._softmax(logits)
    idx = int(probs.argmax())

    attention_png = None
    if req.model in ("ViT-Base", "DINOv2-Base"):
        rollout = G._compute_attention_rollout_app(model, tensor)
        if rollout is not None:
            attention_png = _png_b64(G._overlay_attention(pil, rollout))

    return {
        "label": G.EUROSAT_CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": _probs_list(probs, G.EUROSAT_CLASSES),
        "attention_png": attention_png,
        "latency_ms": float(latency_ms),
        "device": G.DEVICE,
    }


@router.post("/text")
@limiter.limit("20/minute")
def text(request: Request, req: TextReq):
    # Validate everything BEFORE loading any model.
    if req.model not in G.TEXT_MODEL_IDS:
        raise HTTPException(400, "unknown model")
    if not req.text or not req.text.strip():
        raise HTTPException(400, "empty text")
    if len(req.text) > MAX_TEXT_LEN:
        raise HTTPException(400, "text too long")
    try:
        tokenizer, model, temperature = G._load_text_model(req.model)
    except Exception:
        log.exception("text model load failed for %s", req.model)
        raise HTTPException(502, "model load failed")

    enc = tokenizer(req.text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    enc = {k: v.to(G.DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits.squeeze(0).cpu().numpy()
    raw = G._softmax(logits)
    cal = G._softmax(logits / temperature)
    idx = int(cal.argmax())
    return {
        "label": G.EMOTION_CLASSES[idx],
        "confidence": float(cal[idx]),
        "raw": _probs_list(raw, G.EMOTION_CLASSES),
        "calibrated": _probs_list(cal, G.EMOTION_CLASSES),
        "temperature": float(temperature),
    }


@router.post("/clip-search")
@limiter.limit("20/minute")
def clip_search(request: Request, req: ClipReq):
    # Validate everything BEFORE loading any model.
    if not req.query or not req.query.strip():
        raise HTTPException(400, "empty query")
    k = max(1, min(int(req.k), G.MAX_CLIP_K))
    try:
        clip_model, clip_processor = G._load_clip()
        index = G._load_clip_index()
    except Exception:
        log.exception("CLIP load failed")
        raise HTTPException(502, "model load failed")

    feats = index["features"].float()
    labels, images = index["labels"], index["images"]
    inputs = clip_processor(text=[req.query], return_tensors="pt", padding=True,
                            truncation=True, max_length=77)
    inputs = {kk: vv.to(G.DEVICE) for kk, vv in inputs.items()}
    with torch.no_grad():
        tf = clip_model.get_text_features(**inputs)
        tf = (tf / tf.norm(dim=-1, keepdim=True)).cpu()
    sims = (feats @ tf.T).squeeze(1)
    top = sims.topk(k).indices.tolist()

    results = []
    for i in top:
        arr = images[i]
        arr = arr.numpy() if isinstance(arr, torch.Tensor) else arr
        pil = Image.fromarray(np.asarray(arr).astype(np.uint8)) if not isinstance(arr, Image.Image) else arr
        lbl = labels[i]
        lbl = int(lbl.item()) if hasattr(lbl, "item") else int(lbl)
        results.append({
            "image_png": _png_b64(pil),
            "label": G.EUROSAT_CLASSES[lbl],
            "similarity": float(sims[i].item()),
        })
    return {"results": results, "query": req.query}
