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
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

from app import gradio_app as G
from app import results_data as R
from src.utils.logging_utils import get_logger

log = get_logger("api")
router = APIRouter(prefix="/api")


# ── helpers ──────────────────────────────────────────────────────────────────
def _decode_image(b64: str) -> Image.Image:
    if "," in b64[:64]:  # strip data URL prefix if present
        b64 = b64.split(",", 1)[1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"invalid image: {exc}")


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
def vision(req: VisionReq):
    if req.model not in G.VISION_MODEL_IDS:
        raise HTTPException(400, f"unknown model {req.model}")
    pil = _decode_image(req.image)
    try:
        model = G._load_vision_model(req.model)
    except Exception as exc:
        raise HTTPException(502, f"model load failed: {exc}")
    tensor = G._vision_transform(pil)

    with torch.no_grad():
        for _ in range(2):  # warm-up
            model(tensor)
    times = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter()
            out = model(tensor)
            times.append((time.perf_counter() - t0) * 1000)
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
        "latency_ms": float(np.median(times)),
        "device": G.DEVICE,
    }


@router.post("/text")
def text(req: TextReq):
    if req.model not in G.TEXT_MODEL_IDS:
        raise HTTPException(400, f"unknown model {req.model}")
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    try:
        tokenizer, model, temperature = G._load_text_model(req.model)
    except Exception as exc:
        raise HTTPException(502, f"model load failed: {exc}")

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
def clip_search(req: ClipReq):
    if not req.query.strip():
        raise HTTPException(400, "empty query")
    k = max(1, min(int(req.k), 12))
    try:
        clip_model, clip_processor = G._load_clip()
        index = G._load_clip_index()
    except Exception as exc:
        raise HTTPException(502, f"CLIP load failed: {exc}")

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
