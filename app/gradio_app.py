"""
Transfer Learning HuggingFace — Gradio Demo App
================================================
4 tabs:
  1. Vision Classifier      — EfficientNet / ResNet / ViT / DINOv2 on EuroSAT
  2. Text Emotion Detector  — RoBERTa / ModernBERT with calibration
  3. CLIP Image Search      — text-to-image retrieval over EuroSAT index
  4. Experiment Results     — loads saved CSV/JSON result files
"""

import json
import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
from pathlib import Path

import gradio as gr
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logging_utils import get_logger  # noqa: E402

log = get_logger("app")

# ── lazy model cache ──────────────────────────────────────────────────────────
_MODEL_CACHE: dict = {}

# ── constants ─────────────────────────────────────────────────────────────────
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

EMOTION_CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

VISION_MODEL_IDS = {
    "EfficientNet-B0": "efficientnet_b0",
    "ResNet-50": "resnet50",
    "ViT-Base": "vit_base",
    "DINOv2-Base": "dinov2_base",
}

TEXT_MODEL_IDS = {
    "RoBERTa": "roberta",
    "ModernBERT": "modernbert",
}

# Fine-tuned models are served from the Hugging Face Hub (published by
# scripts/push_models_to_hub.py), so the Space needs no local weight files.
# Override the account with the HF_HUB_USER env var if you fork this.
HUB_USER = os.getenv("HF_HUB_USER", "shiva-1993")
VISION_HUB_IDS = {
    "resnet50": f"{HUB_USER}/eurosat-resnet50",
    "efficientnet_b0": f"{HUB_USER}/eurosat-efficientnet-b0",
    "vit_base": f"{HUB_USER}/eurosat-vit-base",
    "dinov2_base": f"{HUB_USER}/eurosat-dinov2-base",
}
TEXT_HUB_IDS = {
    "roberta": f"{HUB_USER}/emotion-roberta",
    "modernbert": f"{HUB_USER}/emotion-modernbert",
}
# CLIP retrieval index (1000 EuroSAT embeddings) hosted as a Hub dataset repo.
CLIP_INDEX_REPO = f"{HUB_USER}/eurosat-clip-index"

RESULTS_DIR = ROOT / "results"
VISION_DIR = RESULTS_DIR / "vision"
TEXT_DIR = RESULTS_DIR / "text"
CLIP_DIR = RESULTS_DIR / "clip"
FIGURES_DIR = RESULTS_DIR / "figures"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def _bar_chart(
    labels: list, values: list, title: str, color: str = "#3498db"
) -> plt.Figure:
    """Return a matplotlib Figure for a horizontal confidence bar chart."""
    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.45)))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=color, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Confidence")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    return fig


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — VISION CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════


def _load_vision_model(model_display_name: str):
    """Lazy-load a vision model; cache after first call."""
    cache_key = f"vision_{model_display_name}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_key = VISION_MODEL_IDS[model_display_name]
    hub_id = VISION_HUB_IDS[model_key]

    try:
        from transformers import AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(hub_id)
        model.to(DEVICE).eval()
        log.info("Loaded vision model from Hub: %s", hub_id)
        _MODEL_CACHE[cache_key] = model
        return model

    except Exception as exc:
        raise gr.Error(f"Could not load vision model '{model_display_name}' from {hub_id}: {exc}")


def _vision_transform(pil_image: Image.Image) -> torch.Tensor:
    import torchvision.transforms as T

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)


def _compute_attention_rollout_app(model, tensor: torch.Tensor) -> np.ndarray | None:
    """Run the model with output_attentions and roll attention up to a 224x224 map.

    compute_attention_rollout expects a LIST of per-layer attention tensors,
    not (model, tensor) — passing the wrong args was why the overlay silently
    never appeared. Returns None for CNNs (no attentions) or on any failure.
    """
    try:
        from src.utils.visualization import compute_attention_rollout

        with torch.no_grad():
            out = model(tensor, output_attentions=True)
        attentions = getattr(out, "attentions", None)
        if not attentions:
            return None
        rollout = compute_attention_rollout([a.cpu() for a in attentions])  # (P,)
        side = int(round(float(np.sqrt(rollout.shape[0]))))
        if side * side != rollout.shape[0]:
            return None
        attn_map = rollout.reshape(side, side).astype(np.float32)
        attn_img = Image.fromarray(attn_map).resize((224, 224), Image.BILINEAR)
        arr = np.asarray(attn_img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr
    except Exception as exc:
        log.warning("attention rollout failed: %s", exc)
        return None


def _overlay_attention(pil_img: Image.Image, rollout: np.ndarray) -> Image.Image:
    """Overlay attention heatmap on the original image."""
    img_np = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    heatmap = cm.jet(rollout)[:, :, :3]
    overlay = 0.55 * img_np + 0.45 * heatmap
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def vision_predict(pil_image: Image.Image, model_name: str):
    """
    Returns:
      - label_str   : str   — predicted class + confidence
      - conf_chart  : Figure
      - attn_image  : PIL Image or None
      - latency_str : str
    """
    if pil_image is None:
        return "Upload an image first.", None, None, ""

    model = _load_vision_model(model_name)
    tensor = _vision_transform(pil_image)

    # Latency benchmark (3 warm-up + 20 timed runs)
    with torch.no_grad():
        for _ in range(3):
            _ = model(tensor)
    times = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            output = model(tensor)
            times.append((time.perf_counter() - t0) * 1000)
    latency_ms = float(np.median(times))

    # Logits → probabilities
    logits = output.logits if hasattr(output, "logits") else output
    logits_np = logits.squeeze(0).cpu().numpy()
    probs = _softmax(logits_np)
    pred_idx = int(probs.argmax())
    pred_cls = EUROSAT_CLASSES[pred_idx]
    conf = probs[pred_idx]

    label_str = f"Predicted: **{pred_cls}** ({conf * 100:.1f}% confidence)"
    latency_str = f"Inference latency: {latency_ms:.1f} ms (median over 20 runs, {DEVICE.upper()})"

    conf_chart = _bar_chart(
        labels=EUROSAT_CLASSES,
        values=probs.tolist(),
        title=f"{model_name} — Class Probabilities",
        color="#3498db",
    )

    # Attention rollout (only for ViT / DINOv2)
    attn_image = None
    if model_name in ("ViT-Base", "DINOv2-Base"):
        rollout = _compute_attention_rollout_app(model, tensor)
        if rollout is not None:
            attn_image = _overlay_attention(pil_image, rollout)

    log.info(
        "vision_predict model=%s pred=%s conf=%.3f latency_ms=%.1f",
        model_name, pred_cls, conf, latency_ms,
    )
    return label_str, conf_chart, attn_image, latency_str


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — TEXT EMOTION DETECTOR
# ══════════════════════════════════════════════════════════════════════════════


def _load_text_model(model_display_name: str):
    cache_key = f"text_{model_display_name}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_key = TEXT_MODEL_IDS[model_display_name]
    hub_id = TEXT_HUB_IDS[model_key]

    try:
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hub_id)
        model = AutoModelForSequenceClassification.from_pretrained(hub_id)
        model.to(DEVICE).eval()
        log.info("Loaded text model from Hub: %s", hub_id)

        # Calibration temperature ships in the model repo (temperature.json).
        temperature = 1.0
        try:
            temp_file = hf_hub_download(hub_id, "temperature.json")
            with open(temp_file) as f:
                temperature = json.load(f).get("temperature", 1.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("no temperature.json for %s (%s); using T=1.0", hub_id, exc)

        _MODEL_CACHE[cache_key] = (tokenizer, model, temperature)
        return _MODEL_CACHE[cache_key]

    except Exception as exc:
        raise gr.Error(f"Could not load text model '{model_display_name}' from {hub_id}: {exc}")


def text_predict(text_input: str, model_name: str):
    """
    Returns:
      - label_str      : str
      - conf_chart_raw : Figure (uncalibrated)
      - conf_chart_cal : Figure (calibrated)
    """
    if not text_input or not text_input.strip():
        return "Enter some text first.", None, None

    tokenizer, model, temperature = _load_text_model(model_name)

    max_len = 128
    enc = tokenizer(
        text_input,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        output = model(**enc)
        logits = output.logits.squeeze(0).cpu().numpy()

    probs_raw = _softmax(logits)
    probs_cal = _softmax(logits / temperature)

    pred_idx_cal = int(probs_cal.argmax())
    pred_cls = EMOTION_CLASSES[pred_idx_cal]
    conf = probs_cal[pred_idx_cal]

    label_str = (
        f"Predicted emotion: **{pred_cls.upper()}** "
        f"({conf * 100:.1f}% calibrated confidence)"
    )

    conf_chart_raw = _bar_chart(
        labels=EMOTION_CLASSES,
        values=probs_raw.tolist(),
        title=f"{model_name} — Uncalibrated Confidence",
        color="#e74c3c",
    )

    conf_chart_cal = _bar_chart(
        labels=EMOTION_CLASSES,
        values=probs_cal.tolist(),
        title=f"{model_name} — Calibrated (T={temperature:.3f})",
        color="#2ecc71",
    )

    log.info(
        "text_predict model=%s pred=%s conf=%.3f chars=%d",
        model_name, pred_cls, conf, len(text_input),
    )
    return label_str, conf_chart_raw, conf_chart_cal


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — CLIP IMAGE SEARCH
# ══════════════════════════════════════════════════════════════════════════════


def _load_clip():
    if "clip" in _MODEL_CACHE:
        return _MODEL_CACHE["clip"]

    try:
        from transformers import CLIPModel, CLIPProcessor

        from configs.clip_config import CLIP_MODEL_ID

        log.info("Loading CLIP model %s", CLIP_MODEL_ID)
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        clip_model.eval()
        _MODEL_CACHE["clip"] = (clip_model, clip_processor)
        return _MODEL_CACHE["clip"]
    except Exception as exc:
        raise gr.Error(f"Could not load CLIP: {exc}")


def _load_clip_index():
    """Load the cached EuroSAT retrieval index (1000 image embeddings).

    Prefers a local file (dev); on the deployed Space it downloads the index
    from the Hub dataset repo. weights_only=False because the index holds numpy
    image arrays, not just tensors (torch>=2.6 defaults to weights_only=True).
    """
    if "clip_index" in _MODEL_CACHE:
        return _MODEL_CACHE["clip_index"]

    from src.utils.paths import clip_index_path

    local = clip_index_path()
    if local.exists():
        path = str(local)
    else:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(
                CLIP_INDEX_REPO, "retrieval_index.pt", repo_type="dataset"
            )
        except Exception as exc:
            raise gr.Error(
                f"CLIP retrieval index not found locally or on the Hub "
                f"({CLIP_INDEX_REPO}): {exc}"
            )

    data = torch.load(path, map_location="cpu", weights_only=False)
    _MODEL_CACHE["clip_index"] = data
    return data


def clip_search(query: str, k: int):
    """
    Returns:
      - gallery : list of (PIL Image, caption) tuples
    """
    if not query or not query.strip():
        return []

    t0 = time.perf_counter()
    clip_model, clip_processor = _load_clip()
    index_data = _load_clip_index()

    index_features = index_data["features"].float()
    index_labels = index_data["labels"]
    index_images = index_data["images"]  # list of np.ndarray or PIL Images

    # Encode query
    inputs = clip_processor(
        text=[query], return_tensors="pt", padding=True, truncation=True, max_length=77
    )
    inputs = {kk: vv.to(DEVICE) for kk, vv in inputs.items()}

    with torch.no_grad():
        text_feat = clip_model.get_text_features(**inputs)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.cpu()

    sims = (index_features @ text_feat.T).squeeze(1)
    top_k_idx = sims.topk(int(k)).indices.tolist()

    gallery = []
    for idx in top_k_idx:
        img_arr = index_images[idx]
        if isinstance(img_arr, torch.Tensor):
            img_arr = img_arr.numpy()
        if isinstance(img_arr, np.ndarray):
            pil_img = Image.fromarray(img_arr.astype(np.uint8))
        else:
            pil_img = img_arr

        lbl = index_labels[idx]
        lbl = int(lbl.item()) if hasattr(lbl, "item") else int(lbl)
        cls_name = EUROSAT_CLASSES[lbl]
        sim_score = sims[idx].item()
        caption = f"{cls_name} (sim={sim_score:.3f})"
        gallery.append((pil_img, caption))

    log.info(
        "clip_search query=%r k=%d results=%d elapsed_ms=%.1f",
        query, int(k), len(gallery), (time.perf_counter() - t0) * 1000,
    )
    return gallery


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — EXPERIMENT RESULTS
# ══════════════════════════════════════════════════════════════════════════════


def _load_vision_summary() -> pd.DataFrame | None:
    csv_path = VISION_DIR / "summary.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    json_path = VISION_DIR / "strategy_results.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        rows = []
        for model, strategies in data.items():
            for strategy, metrics in strategies.items():
                rows.append({"model": model, "strategy": strategy, **metrics})
        return pd.DataFrame(rows)
    return None


def _load_text_summary() -> pd.DataFrame | None:
    json_path = TEXT_DIR / "training_results.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        rows = []
        for model, metrics in data.items():
            rows.append(
                {
                    "model": model,
                    "test_acc": metrics.get("test_acc", ""),
                    "f1_macro": metrics.get("f1_macro", ""),
                    "ece_before": metrics.get("ece_before", ""),
                    "ece_after": metrics.get("ece_after", ""),
                    "temperature": metrics.get("temperature", ""),
                }
            )
        return pd.DataFrame(rows)
    return None


def _load_clip_summary() -> pd.DataFrame | None:
    json_path = CLIP_DIR / "clip_results.json"
    if not json_path.exists():
        # Try prompt sensitivity file
        json_path = CLIP_DIR / "prompt_sensitivity.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            rows = []
            for template, res in data.items():
                rows.append(
                    {"template": template, "accuracy": res.get("overall_acc", "")}
                )
            return pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    return None


def _df_to_str(df: pd.DataFrame | None, title: str) -> str:
    if df is None:
        return (
            f"**{title}**: No results file found. Run the training notebooks first.\n"
        )
    # Round floats
    df = df.copy()
    for col in df.select_dtypes("float").columns:
        df[col] = df[col].round(4)
    return f"**{title}**\n\n{df.to_markdown(index=False)}\n"


def load_results():
    """Load and format all experiment results."""
    vision_df = _load_vision_summary()
    text_df = _load_text_summary()
    clip_df = _load_clip_summary()

    output = ""
    output += _df_to_str(vision_df, "Vision — Strategy Comparison (EuroSAT)")
    output += "\n---\n\n"
    output += _df_to_str(text_df, "Text — Calibration Results (dair-ai/emotion)")
    output += "\n---\n\n"
    output += _df_to_str(clip_df, "CLIP — Prompt Sensitivity (EuroSAT Zero-Shot)")

    # Data efficiency if available
    de_path = VISION_DIR / "data_efficiency_results.json"
    if de_path.exists():
        with open(de_path) as f:
            de_data = json.load(f)
        rows = []
        for model, fracs in de_data.items():
            for frac, metrics in fracs.items():
                rows.append(
                    {
                        "model": model,
                        "data_fraction": frac,
                        "test_acc": round(metrics.get("test_acc", 0), 4),
                    }
                )
        de_df = pd.DataFrame(rows)
        output += "\n---\n\n"
        output += _df_to_str(de_df, "Vision — Data Efficiency (Full Fine-Tune)")

    return output


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

HEADER_MD = """
# Transfer Learning with HuggingFace
**CNN vs ViT vs DINOv2 | RoBERTa vs ModernBERT | CLIP Zero-Shot**

Explore vision classification, emotion detection, and CLIP image search — all powered by pretrained HuggingFace models.
"""

def build_demo() -> gr.Blocks:
    """Build and return the Gradio app.

    Wrapped in a function so importing this module (e.g. in tests) does NOT
    construct the UI or fetch remote example assets at import time.
    """
    with gr.Blocks(title="Transfer Learning Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(HEADER_MD)

        # ── TAB 1: Vision Classifier ──────────────────────────────────────────
        with gr.Tab("Vision Classifier"):
            gr.Markdown(
                "## Land-Use Classification on EuroSAT\nUpload a satellite image patch and classify it using one of four pretrained vision models."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    vision_img_input = gr.Image(
                        type="pil",
                        label="Input Image (64×64 or larger satellite patch)",
                    )
                    vision_model_dd = gr.Dropdown(
                        choices=list(VISION_MODEL_IDS.keys()),
                        value="DINOv2-Base",
                        label="Model",
                    )
                    vision_submit_btn = gr.Button("Classify", variant="primary")

                with gr.Column(scale=2):
                    vision_label_out = gr.Markdown(label="Prediction")
                    vision_latency_out = gr.Markdown(label="Latency")
                    vision_conf_chart = gr.Plot(label="Class Probabilities")
                    vision_attn_img = gr.Image(
                        type="pil", label="Attention Rollout (ViT / DINOv2 only)"
                    )

            vision_submit_btn.click(
                fn=vision_predict,
                inputs=[vision_img_input, vision_model_dd],
                outputs=[
                    vision_label_out,
                    vision_conf_chart,
                    vision_attn_img,
                    vision_latency_out,
                ],
            )
            # NOTE: a remote-URL example was removed — fetching an external image
            # at build time made the Space build fragile. Upload a local patch
            # or add a committed sample image under results/ to restore examples.

        # ── TAB 2: Text Emotion Detector ──────────────────────────────────────
        with gr.Tab("Text Emotion Detector"):
            gr.Markdown(
                "## Emotion Detection from Text\nEnter a sentence or tweet and detect its emotion. Compares raw vs temperature-scaled confidence."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Input Text",
                        placeholder="I feel absolutely amazing today!",
                        lines=4,
                    )
                    text_model_dd = gr.Dropdown(
                        choices=list(TEXT_MODEL_IDS.keys()),
                        value="ModernBERT",
                        label="Model",
                    )
                    text_submit_btn = gr.Button("Detect Emotion", variant="primary")

                with gr.Column(scale=2):
                    text_label_out = gr.Markdown(label="Prediction")
                    with gr.Row():
                        text_conf_raw = gr.Plot(label="Uncalibrated Confidence")
                        text_conf_cal = gr.Plot(label="Calibrated Confidence")

            text_submit_btn.click(
                fn=text_predict,
                inputs=[text_input, text_model_dd],
                outputs=[text_label_out, text_conf_raw, text_conf_cal],
            )

            gr.Examples(
                examples=[
                    ["I can't stop crying, everything feels so hopeless.", "RoBERTa"],
                    ["Best day ever! Just got promoted!", "ModernBERT"],
                    ["omg I'm so scared I have a job interview tomorrow", "ModernBERT"],
                    ["I love spending time with my family :)", "RoBERTa"],
                ],
                inputs=[text_input, text_model_dd],
                label="Example Inputs",
            )

        # ── TAB 3: CLIP Image Search ──────────────────────────────────────────
        with gr.Tab("CLIP Image Search"):
            gr.Markdown(
                "## Text-to-Image Retrieval with CLIP\n"
                "Search through a cached index of **1,000 EuroSAT images** (100 per class) "
                "using natural language queries. Powered by CLIP's shared embedding space."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    clip_query_input = gr.Textbox(
                        label="Text Query",
                        placeholder="a satellite image of a river",
                        lines=2,
                    )
                    clip_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Number of results (k)",
                    )
                    clip_search_btn = gr.Button("Search", variant="primary")

                with gr.Column(scale=3):
                    clip_gallery = gr.Gallery(
                        label="Retrieved Images",
                        columns=5,
                        height="auto",
                        object_fit="cover",
                    )

            clip_search_btn.click(
                fn=clip_search,
                inputs=[clip_query_input, clip_k_slider],
                outputs=[clip_gallery],
            )

            gr.Examples(
                examples=[
                    ["a satellite image of a river", 5],
                    ["urban areas with buildings", 5],
                    ["green forested areas", 5],
                    ["large bodies of water", 5],
                    ["industrial land", 5],
                    ["crop fields", 8],
                    ["residential neighborhoods", 8],
                ],
                inputs=[clip_query_input, clip_k_slider],
                label="Example Queries",
            )

        # ── TAB 4: Experiment Results ─────────────────────────────────────────
        with gr.Tab("Experiment Results"):
            gr.Markdown(
                "## Experiment Results Summary\n"
                "Loads pre-computed results from `results/vision/`, `results/text/`, "
                "and `results/clip/`. Run the training notebooks first to populate these files."
            )

            results_display = gr.Markdown(value="Click **Refresh** to load results.")
            results_refresh_btn = gr.Button("Refresh Results", variant="secondary")

            results_refresh_btn.click(
                fn=load_results,
                inputs=[],
                outputs=[results_display],
            )

            # Auto-load on tab render
            demo.load(fn=load_results, inputs=[], outputs=[results_display])

    return demo


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def build_app():
    """Build the FastAPI app: JSON /api/* (for the Next.js frontend), the Gradio
    demo at /, and a /health probe.

    - ``/api/*``  — JSON inference API (app.api.router), consumed by the custom frontend
    - ``/``       — the Gradio demo (kept as a fallback / direct demo)
    - ``/health`` — liveness probe (no model load)

    CORS is open so the Vercel-hosted frontend can call the API from the browser.
    Run via ``uvicorn app.serve:app`` — NOT ``python app/gradio_app.py`` (that
    would import this module twice and split the model cache).
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from app.api import router as api_router

    fastapi_app = FastAPI(title="Transfer Learning API + Demo")
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # public read-only inference API
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    fastapi_app.include_router(api_router)

    demo = build_demo()
    return gr.mount_gradio_app(fastapi_app, demo.queue(), path="/")
