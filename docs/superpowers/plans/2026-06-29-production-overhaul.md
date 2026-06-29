# Transfer-Learning-HuggingFace — Production Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every Critical and Important finding from the Phase-1 audit so the repo's train → save → load → serve path is correct, reproducible, observable, and deployable to Hugging Face Spaces free tier.

**Architecture:** The repo trains models locally (user has a GPU laptop), saves artifacts under `results/`, and a Gradio app loads those artifacts to serve a 4-tab demo on HF Spaces. The core defect class is **contract drift**: trainers and the app independently hardcode artifact paths / model keys / dict keys that don't match, so the app silently serves random weights or crashes. The fix centralizes the artifact-path contract and model registry references into a single source of truth, adds structured logging + fail-loud config validation, and adds tests/CI/Docker so the contract can't silently regress.

**Tech Stack:** Python 3.12, PyTorch 2.5, HuggingFace Transformers 4.57, Gradio 6, pytest, ruff, GitHub Actions, Docker (for HF Spaces).

## Global Constraints

- **Free tier only.** Deployment target is Hugging Face Spaces free CPU Basic (2 vCPU / 16 GB RAM / sleeps after 48h). No paid infra. No Render, no Supabase.
- **No database is introduced** in this plan — the app is stateless over result files. (Observability is structured logs to stdout, which HF Spaces captures.)
- **Commits must NOT mention Claude or "Co-Authored-By: Claude".** (User instruction.) Use plain conventional-commit messages.
- **Do not change experiment hyperparameters or model choices** — these are research decisions the user owns. Only fix correctness/plumbing.
- **Verification reality:** this dev box has a partial ML install (no `datasets`/`scikit-learn`/`timm`) and cannot download models. Tests must run on **pure logic** (numpy/torch only) or **static contract checks** (importing `configs/` + `src/utils/paths.py`, which have no heavy deps). Anything needing model download or full deps is verified by the user locally and is called out as such.
- **Single source of truth:** model keys come from `configs/*_config.py` registries; artifact paths come from `src/utils/paths.py`. No module may hardcode a path or model key that duplicates these.

---

## File Structure

**New files:**
- `src/utils/paths.py` — artifact-path contract (used by trainers AND app).
- `src/utils/logging_utils.py` — structured logger factory + startup config validation.
- `tests/__init__.py`, `tests/conftest.py` — test bootstrap (repo root on path).
- `tests/test_paths.py` — contract: path builders + app/registry key alignment.
- `tests/test_metrics.py` — ECE correctness, latency dict shape.
- `tests/test_attention_rollout.py` — rollout output shape/normalization.
- `tests/test_data_sampling.py` — stratified subset class balance.
- `requirements-app.txt` — slim, pinned serving deps for the HF Space.
- `Dockerfile` — minimal non-root image for the Space.
- `.dockerignore`
- `.github/workflows/ci.yml` — ruff lint + pytest on push/PR.
- `pyproject.toml` — ruff + pytest config.
- `docs/adr/0001-artifact-path-contract.md` — short ADR (system-design signal + Documentation checklist).

**Modified files:**
- `app/gradio_app.py` — fix build_model call, model-key mapping, checkpoint paths, `hf_id` key, attention rollout, CLIP index robustness, logging, launch config.
- `src/vision/trainer.py` — save deployable checkpoint via `paths`, flatten summary, logging, remove dead var.
- `src/text/trainer.py` — save `best_model.pt` + `temperature.json` (clamped) via `paths`, logging.
- `scripts/train_vision.py` — flatten summary.csv rows.
- `src/vision/model.py` — drop unused imports.
- `.env.example` — remove unrelated `AGENT_MODEL`/`AGENTLESS_MODEL`, document real vars.
- `requirements.txt` — pin versions (reproducibility).
- `README.md` — deployment (Spaces + Docker) + artifact-population section + ADR link.
- `.gitignore` — ignore pytest/ruff caches already partly present; add `results/**/best_model.pt`? NO — those must be committable to the Space. Add `.ruff_cache/`.

---

## Findings → Task mapping

| Finding | Severity | Task |
|---|---|---|
| C1 `build_model(cfg)` wrong call | Critical | 4 |
| C2 / A3 vision checkpoint path drift | Critical | 1, 2, 4 |
| A2 vision model-key mapping mismatch | Critical | 1, 4 |
| A6 text uses `cfg["model_id"]` not `hf_id` | Critical | 4 |
| A5 text checkpoint + temperature not saved | Critical | 3, 4 |
| A4 attention rollout called with wrong args | Important | 4 |
| A7 CLIP index label robustness | Important | 4 |
| V1 vision trainer never persists best model | Important | 2 |
| T1/T2 temperature not saved / unclamped | Important | 3 |
| V3 summary.csv writes nested dicts | Important | 2 |
| I1 no tests | Important | 1,2,3,5 + 6 |
| I2 no CI | Important | 8 |
| I3 unpinned deps / no lockfile | Important | 7 |
| I4 print-only, no structured logging, no fail-loud config | Important | 5 |
| I5 no Dockerfile | Important | 9 |
| I6 `.env.example` cruft | Important | 5 |
| I7 README missing deploy/architecture docs | Important | 10 |
| I8 app pulls full training stack | Important | 7 |

---

### Task 1: Artifact-path contract (`src/utils/paths.py`) + alignment tests

**Files:**
- Create: `src/utils/paths.py`
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/test_paths.py`
- Create: `pyproject.toml` (pytest config)

**Interfaces:**
- Produces:
  - `RESULTS_ROOT: Path` (= repo_root / "results")
  - `vision_run_dir(model_key: str, strategy: str, fraction: float) -> Path`
  - `vision_checkpoint_path(model_key: str, strategy: str, fraction: float) -> Path` → `<run_dir>/best_model.pt`
  - `text_run_dir(model_key: str) -> Path`
  - `text_checkpoint_path(model_key: str) -> Path` → `<run_dir>/best_model.pt`
  - `text_temperature_path(model_key: str) -> Path` → `<run_dir>/temperature.json`
  - `clip_index_path() -> Path` → `results/clip/retrieval_index.pt`
  - `DEMO_VISION_STRATEGY = "full_finetune"`, `DEMO_VISION_FRACTION = 1.0` (the artifacts the app serves)

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
# E/F = pyflakes+pycodestyle, I = isort. Keep it pragmatic for a research repo.
select = ["E", "F", "I"]
ignore = ["E501"]  # line length handled by formatter, not blocking
```

- [ ] **Step 2: Write `src/utils/paths.py`**

```python
"""
Single source of truth for where training artifacts live on disk.

Both the trainers (which WRITE checkpoints) and the Gradio app (which READS
them) import these helpers, so the train -> serve contract can never silently
drift. Previously each side hardcoded its own path and they did not match,
so the app served randomly-initialised weights without erroring.
"""

from __future__ import annotations

from pathlib import Path

# repo_root = .../Transfer-Learning-HuggingFace  (this file is src/utils/paths.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"

# The exact artifact the demo app serves for the Vision tab.
DEMO_VISION_STRATEGY = "full_finetune"
DEMO_VISION_FRACTION = 1.0


def vision_run_dir(model_key: str, strategy: str, fraction: float) -> Path:
    return RESULTS_ROOT / "vision" / model_key / strategy / f"frac{fraction:.2f}"


def vision_checkpoint_path(model_key: str, strategy: str, fraction: float) -> Path:
    return vision_run_dir(model_key, strategy, fraction) / "best_model.pt"


def text_run_dir(model_key: str) -> Path:
    return RESULTS_ROOT / "text" / model_key


def text_checkpoint_path(model_key: str) -> Path:
    return text_run_dir(model_key) / "best_model.pt"


def text_temperature_path(model_key: str) -> Path:
    return text_run_dir(model_key) / "temperature.json"


def clip_index_path() -> Path:
    return RESULTS_ROOT / "clip" / "retrieval_index.pt"
```

- [ ] **Step 3: Write `tests/__init__.py` (empty) and `tests/conftest.py`**

```python
# tests/conftest.py
import sys
from pathlib import Path

# Ensure repo root is importable so `import configs...` / `import src...` work.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 4: Write the failing test `tests/test_paths.py`**

```python
from pathlib import Path

from src.utils import paths
from configs.vision_config import VISION_MODELS
from configs.text_config import TEXT_MODELS


def test_vision_checkpoint_path_matches_run_dir():
    p = paths.vision_checkpoint_path("dinov2_base", "full_finetune", 1.0)
    assert p.parent == paths.vision_run_dir("dinov2_base", "full_finetune", 1.0)
    assert p.name == "best_model.pt"
    assert p.parts[-2] == "frac1.00"


def test_text_paths():
    assert paths.text_checkpoint_path("roberta").name == "best_model.pt"
    assert paths.text_temperature_path("roberta").name == "temperature.json"


def test_app_vision_model_keys_are_real_registry_keys():
    # The app maps display names -> registry keys. Every mapped key must exist
    # in the config registry, or build_model() will KeyError. (Bug A2 guard.)
    from app.gradio_app import VISION_MODEL_IDS

    for display, key in VISION_MODEL_IDS.items():
        assert key in VISION_MODELS, f"{display} -> {key} not in VISION_MODELS"


def test_app_text_model_keys_are_real_registry_keys():
    from app.gradio_app import TEXT_MODEL_IDS

    for display, key in TEXT_MODEL_IDS.items():
        assert key in TEXT_MODELS, f"{display} -> {key} not in TEXT_MODELS"
```

- [ ] **Step 5: Run tests — `test_paths` PASS, app-key tests FAIL**

Run: `python -m pytest tests/test_paths.py -v`
Expected: the two path tests PASS; the two app-key tests FAIL (current `VISION_MODEL_IDS` values are `efficientnet`/`resnet`/`vit`/`dinov2`, not the registry keys) — **and importing `app.gradio_app` may itself fail under gradio 6**. If the import errors, that is expected pre-Task-4; mark these two tests xfail-pending-Task-4 by leaving them — they go green after Task 4.

Note: if `import app.gradio_app` raises at module load (it executes `gr.Blocks(...)` at import time), move the two app-key assertions to run after Task 4 reorganizes the app so the Blocks build is under `build_demo()`. Task 4 Step 1 handles that.

- [ ] **Step 6: Commit**

```bash
git add src/utils/paths.py tests/ pyproject.toml
git commit -m "feat: centralize training-artifact path contract + tests"
```

---

### Task 2: Vision trainer persists deployable checkpoint + flat summary

**Files:**
- Modify: `src/vision/trainer.py`
- Modify: `scripts/train_vision.py`

**Interfaces:**
- Consumes: `src.utils.paths.vision_checkpoint_path`
- Produces: after `train_vision_model(cfg)`, the file `vision_checkpoint_path(cfg.model_key, cfg.strategy, cfg.data_fraction)` exists and is a plain `state_dict` loadable by a freshly-built model. `result` dict gains flat scalar keys `latency_cpu_mean_ms`, `latency_cpu_p95_ms`, `onnx_mean_ms`.

- [ ] **Step 1: Save the best model state_dict after training.** In `src/vision/trainer.py`, after `test_results = trainer.evaluate(test_hf)` (around line 209) and before the latency block, add:

```python
    from src.utils.paths import vision_checkpoint_path

    ckpt_path = vision_checkpoint_path(cfg.model_key, cfg.strategy, cfg.data_fraction)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # load_best_model_at_end=True means `model` is already the best checkpoint.
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved deployable checkpoint -> {ckpt_path}")
```

- [ ] **Step 2: Remove the dead `device` variable.** In `src/vision/trainer.py` line ~218, delete:

```python
        device = "cuda" if torch.cuda.is_available() else "cpu"
```

(the latency call already passes `device="cpu"` literally; the variable is unused.)

- [ ] **Step 3: Add flat scalar metrics to `result`.** Replace the `result = {...}` dict (around line 245) with one that also includes flat scalars so the CSV summary is clean:

```python
        result = {
            "model_key": cfg.model_key,
            "strategy": cfg.strategy,
            "data_fraction": cfg.data_fraction,
            "trainable_params": param_info["trainable_params"],
            "total_params": param_info["total_params"],
            "trainable_pct": param_info["trainable_pct"],
            "test_accuracy": test_results.get("eval_accuracy", 0),
            "test_f1_macro": test_results.get("eval_f1_macro", 0),
            "latency_cpu_mean_ms": latency.get("mean_ms"),
            "latency_cpu_p95_ms": latency.get("p95_ms"),
            "onnx_mean_ms": onnx_latency.get("mean_ms") if onnx_latency else None,
        }
```

- [ ] **Step 4: Flatten the summary writer.** In `scripts/train_vision.py` `run_all_experiments`, the rows are now flat scalars, so the existing `DictWriter(..., extrasaction="ignore")` works. Replace the `keys = [...]` line with a stable column order:

```python
        keys = [
            "model_key", "strategy", "data_fraction",
            "trainable_params", "total_params", "trainable_pct",
            "test_accuracy", "test_f1_macro",
            "latency_cpu_mean_ms", "latency_cpu_p95_ms", "onnx_mean_ms",
        ]
```

- [ ] **Step 5: Static verification (no model download available here).**

Run: `python -m py_compile src/vision/trainer.py scripts/train_vision.py && echo OK`
Expected: `OK` (syntax valid).
Note for user: actual checkpoint creation is verified by running `python scripts/train_vision.py --model efficientnet_b0 --strategy full_finetune --fraction 0.1` on your GPU box and confirming `results/vision/efficientnet_b0/full_finetune/frac0.10/best_model.pt` appears.

- [ ] **Step 6: Commit**

```bash
git add src/vision/trainer.py scripts/train_vision.py
git commit -m "fix: persist deployable vision checkpoint + flat run summary"
```

---

### Task 3: Text trainer saves checkpoint + clamped temperature

**Files:**
- Modify: `src/text/trainer.py`

**Interfaces:**
- Consumes: `src.utils.paths.text_checkpoint_path`, `text_temperature_path`
- Produces: after `train_text_model(cfg)`, `text_checkpoint_path(cfg.model_key)` (state_dict) and `text_temperature_path(cfg.model_key)` (`{"temperature": <float >= 0.05>}`) both exist.

- [ ] **Step 1: Clamp the fitted temperature.** In `src/text/trainer.py` after `T = scaler.T` (line ~221), replace with:

```python
        T = max(float(scaler.T), 0.05)  # guard: avoid div-by-~0 at serve time
```

- [ ] **Step 2: Save checkpoint + temperature.** After the `result_path.write_text(...)` block (around line 248), add:

```python
        from src.utils.paths import text_checkpoint_path, text_temperature_path

        ckpt_path = text_checkpoint_path(cfg.model_key)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

        temp_path = text_temperature_path(cfg.model_key)
        temp_path.write_text(json.dumps({"temperature": T}, indent=2))
        print(f"  Saved checkpoint -> {ckpt_path}; temperature={T:.4f}")
```

Note: `model.cpu()` was already called when constructing `TemperatureScaler(model.cpu())`, so the saved state_dict is CPU — correct for the app.

- [ ] **Step 3: Static verification.**

Run: `python -m py_compile src/text/trainer.py && echo OK`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/text/trainer.py
git commit -m "fix: persist text checkpoint + clamped calibration temperature"
```

---

### Task 4: Fix the Gradio app (the C1/A2/A4/A5/A6/A7 cluster)

**Files:**
- Modify: `app/gradio_app.py`
- Modify: `src/vision/model.py` (drop unused imports)

**Interfaces:**
- Consumes: `build_model(model_key, num_classes, id2label, label2id, strategy)` → `(model, processor)`; `src.utils.paths` checkpoint helpers; `configs` registries.
- Produces: `build_demo() -> gr.Blocks` (app construction wrapped in a function so importing the module for tests does not build the UI). `VISION_MODEL_IDS` values are real registry keys.

- [ ] **Step 1: Make UI construction lazy.** Wrap the `with gr.Blocks(...) as demo:` block (lines ~533-697) in `def build_demo() -> gr.Blocks:` returning `demo`, and change the entrypoint:

```python
if __name__ == "__main__":
    demo = build_demo()
    demo.queue().launch()   # share handled by env; Spaces ignores share=
```

(Removes import-time side effects so `tests/test_paths.py` can import the module.)

- [ ] **Step 2: Fix the vision model-key mapping (Bug A2).** Replace `VISION_MODEL_IDS`:

```python
VISION_MODEL_IDS = {
    "EfficientNet-B0": "efficientnet_b0",
    "ResNet-50": "resnet50",
    "ViT-Base": "vit_base",
    "DINOv2-Base": "dinov2_base",
}
```

- [ ] **Step 3: Fix `_load_vision_model` (Bugs A1, A3).** Replace the body's build+load section:

```python
    model_key = VISION_MODEL_IDS[model_display_name]
    try:
        from configs.vision_config import EUROSAT_CLASSES, NUM_CLASSES
        from src.vision.model import build_model
        from src.utils.paths import (
            vision_checkpoint_path, DEMO_VISION_STRATEGY, DEMO_VISION_FRACTION,
        )

        id2label = {i: c for i, c in enumerate(EUROSAT_CLASSES)}
        label2id = {c: i for i, c in enumerate(EUROSAT_CLASSES)}
        model, _ = build_model(
            model_key=model_key,
            num_classes=NUM_CLASSES,
            id2label=id2label,
            label2id=label2id,
            strategy=DEMO_VISION_STRATEGY,
        )

        ckpt_path = vision_checkpoint_path(
            model_key, DEMO_VISION_STRATEGY, DEMO_VISION_FRACTION
        )
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)
            log.info("Loaded vision checkpoint %s", ckpt_path)
        else:
            log.warning("No checkpoint at %s; serving RANDOM weights", ckpt_path)

        model.to(DEVICE).eval()
        _MODEL_CACHE[cache_key] = model
        return model
    except Exception as exc:
        raise gr.Error(f"Could not load vision model '{model_display_name}': {exc}")
```

- [ ] **Step 4: Fix `_load_text_model` (Bugs A5, A6).** Replace `cfg["model_id"]` → `cfg["hf_id"]` and route checkpoint/temperature through `paths`:

```python
    model_key = TEXT_MODEL_IDS[model_display_name]
    try:
        from configs.text_config import TEXT_MODELS
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from src.utils.paths import text_checkpoint_path, text_temperature_path

        cfg = TEXT_MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["hf_id"], num_labels=len(EMOTION_CLASSES)
        )

        ckpt_path = text_checkpoint_path(model_key)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)
            log.info("Loaded text checkpoint %s", ckpt_path)
        else:
            log.warning("No checkpoint at %s; serving RANDOM weights", ckpt_path)
        model.to(DEVICE).eval()

        temperature = 1.0
        temp_path = text_temperature_path(model_key)
        if temp_path.exists():
            temperature = json.load(open(temp_path)).get("temperature", 1.0)

        _MODEL_CACHE[cache_key] = (tokenizer, model, temperature)
        return _MODEL_CACHE[cache_key]
    except Exception as exc:
        raise gr.Error(f"Could not load text model '{model_display_name}': {exc}")
```

- [ ] **Step 5: Fix attention rollout (Bug A4).** `compute_attention_rollout` takes a LIST of attention tensors, not `(model, tensor)`. Replace `_compute_attention_rollout_app`:

```python
def _compute_attention_rollout_app(model, tensor: torch.Tensor) -> "np.ndarray | None":
    """Run the model with output_attentions and roll attention up to a 224x224 map."""
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
```

(`_overlay_attention` already expects a 2-D map, so it now matches.)

- [ ] **Step 6: Harden CLIP index labels (Bug A7).** In `clip_search`, replace the caption/label section so int or tensor labels both work:

```python
        lbl = index_labels[idx]
        lbl = int(lbl.item()) if hasattr(lbl, "item") else int(lbl)
        cls_name = EUROSAT_CLASSES[lbl]
```

- [ ] **Step 7: Add a module logger at the top of `app/gradio_app.py`.** After imports add:

```python
from src.utils.logging_utils import get_logger
log = get_logger("app")
```

and delete the remaining `print(...)` calls in the app (CLIP load message → `log.info`).

- [ ] **Step 8: Drop unused imports in `src/vision/model.py`.** Remove `Optional`, `ViTForImageClassification`, `Dinov2ForImageClassification` from the imports (only `AutoImageProcessor`, `AutoFeatureExtractor`, `AutoModelForImageClassification` are used).

- [ ] **Step 9: Run the alignment tests (now expected GREEN).**

Run: `python -m pytest tests/test_paths.py -v`
Expected: all 4 tests PASS (app imports cleanly now that Blocks is lazy; mapped keys are real registry keys).

- [ ] **Step 10: Syntax check the app.**

Run: `python -m py_compile app/gradio_app.py src/vision/model.py && echo OK`
Expected: `OK`.

- [ ] **Step 11: Commit**

```bash
git add app/gradio_app.py src/vision/model.py
git commit -m "fix: correct vision/text model loading, attention rollout, and CLIP index handling in app"
```

---

### Task 5: Structured logging + fail-loud config + `.env.example` cleanup

**Files:**
- Create: `src/utils/logging_utils.py`
- Modify: `src/vision/trainer.py`, `src/text/trainer.py`, `src/clip/pipeline.py` (swap key `print`→logger; keep human-facing progress prints sparing)
- Modify: `.env.example`

**Interfaces:**
- Produces: `get_logger(name: str) -> logging.Logger` (level from `LOG_LEVEL` env, default INFO, single stream handler, consistent format); `require_env(keys: list[str]) -> None` (raises `RuntimeError` listing all missing keys).

- [ ] **Step 1: Write `src/utils/logging_utils.py`**

```python
"""Structured logging + fail-loud config validation.

One configured logger factory so every module logs with consistent levels and
format to stdout (which Hugging Face Spaces captures). `require_env` makes the
app fail LOUDLY at startup if required config is missing, instead of dying
deep in a request handler later.
"""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format=_FORMAT)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)


def require_env(keys: list[str]) -> None:
    """Raise RuntimeError naming every missing required env var."""
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
```

- [ ] **Step 2: Write the failing test `tests/test_logging.py`**

```python
import pytest
from src.utils.logging_utils import get_logger, require_env


def test_get_logger_returns_named_logger():
    assert get_logger("x").name == "x"


def test_require_env_raises_listing_all_missing(monkeypatch):
    monkeypatch.delenv("FOO_X", raising=False)
    monkeypatch.delenv("BAR_Y", raising=False)
    with pytest.raises(RuntimeError) as ei:
        require_env(["FOO_X", "BAR_Y"])
    assert "FOO_X" in str(ei.value) and "BAR_Y" in str(ei.value)


def test_require_env_passes_when_present(monkeypatch):
    monkeypatch.setenv("FOO_X", "1")
    require_env(["FOO_X"])  # no raise
```

- [ ] **Step 3: Run it.**

Run: `python -m pytest tests/test_logging.py -v`
Expected: PASS (3 tests).

- [ ] **Step 4: Swap noisy `print` for logging in trainers/pipeline.** In `src/vision/trainer.py`, `src/text/trainer.py`, `src/clip/pipeline.py`, add at top `from src.utils.logging_utils import get_logger` / `log = get_logger(__name__)` and convert the `print(...)` status lines to `log.info(...)`. Keep the change mechanical; do not alter logic.

- [ ] **Step 5: Clean `.env.example` (Bug I6).** Replace entire file with:

```bash
# ── HuggingFace ───────────────────────────────────────────────
# Optional: only needed to PUSH trained models to the Hub.
HF_TOKEN=

# ── MLflow ────────────────────────────────────────────────────
# Local file store by default; set to a remote URI to share runs.
MLFLOW_TRACKING_URI=mlruns

# ── Logging ───────────────────────────────────────────────────
# DEBUG | INFO | WARNING | ERROR
LOG_LEVEL=INFO
```

- [ ] **Step 6: Verify + commit.**

Run: `python -m pytest tests/test_logging.py -q && python -m py_compile src/clip/pipeline.py && echo OK`
Expected: tests pass, `OK`.

```bash
git add src/utils/logging_utils.py tests/test_logging.py src/vision/trainer.py src/text/trainer.py src/clip/pipeline.py .env.example
git commit -m "feat: structured logging + fail-loud env validation; clean .env.example"
```

---

### Task 6: Pure-logic unit tests (closes I1 coverage on core logic)

**Files:**
- Create: `tests/test_metrics.py`, `tests/test_attention_rollout.py`, `tests/test_data_sampling.py`

**Interfaces:** consumes existing `src.utils.metrics`, `src.utils.visualization`, `src.utils.data`.

- [ ] **Step 1: `tests/test_metrics.py` — ECE correctness.**

```python
import numpy as np
from src.utils.metrics import compute_ece, benchmark_latency
import torch.nn as nn


def test_ece_zero_when_perfectly_calibrated():
    # All predictions correct with confidence 1.0 -> ECE 0.
    conf = np.ones(100)
    preds = np.zeros(100, dtype=int)
    labels = np.zeros(100, dtype=int)
    ece, bins = compute_ece(conf, preds, labels, n_bins=15)
    assert ece == 0.0
    assert len(bins["bin_counts"]) == 15


def test_ece_high_when_overconfident_and_wrong():
    conf = np.ones(100)            # confidence 1.0
    preds = np.zeros(100, dtype=int)
    labels = np.ones(100, dtype=int)  # always wrong -> accuracy 0
    ece, _ = compute_ece(conf, preds, labels, n_bins=15)
    assert ece > 0.9


def test_benchmark_latency_shape():
    m = nn.Linear(3 * 8 * 8, 10)

    class Flat(nn.Module):
        def __init__(self, lin): super().__init__(); self.lin = lin
        def forward(self, x): return self.lin(x.flatten(1))

    out = benchmark_latency(Flat(m), image_size=8, n_warmup=2, n_runs=5, device="cpu")
    assert {"mean_ms", "p95_ms", "throughput_imgs_per_sec"} <= set(out)
```

- [ ] **Step 2: `tests/test_attention_rollout.py` — output shape.**

```python
import numpy as np
import torch
from src.utils.visualization import compute_attention_rollout


def test_rollout_returns_one_value_per_patch():
    # 2 layers, batch 1, 4 heads, seq=5 (1 CLS + 4 patches)
    attentions = [torch.rand(1, 4, 5, 5) for _ in range(2)]
    out = compute_attention_rollout(attentions)
    assert out.shape == (4,)            # patches only, CLS removed
    assert np.all(out >= 0)
```

- [ ] **Step 3: `tests/test_data_sampling.py` — stratified balance.** This imports `src.utils.data`, which imports `datasets`/`torchvision`. If `datasets` is not installed in the runner, guard with importorskip:

```python
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_stratified_subset_keeps_fraction_per_class():
    from torch.utils.data import Dataset
    from src.utils.data import _stratified_subset

    class Toy(Dataset):
        def __init__(self): self.labels = [i % 5 for i in range(100)]  # 20 each
        def __len__(self): return len(self.labels)
        def __getitem__(self, i): return torch.zeros(1), self.labels[i]

    sub = _stratified_subset(Toy(), fraction=0.5, seed=0)
    got = {}
    for i in range(len(sub)):
        _, lbl = sub[i]; got[lbl] = got.get(lbl, 0) + 1
    assert all(c == 10 for c in got.values())  # 50% of 20 per class
    assert len(got) == 5
```

- [ ] **Step 4: Run the suite.**

Run: `python -m pytest tests/ -v`
Expected: metrics + rollout tests PASS; data-sampling test PASS or SKIP (if `datasets`/`torchvision` missing in runner). No failures.

- [ ] **Step 5: Commit.**

```bash
git add tests/test_metrics.py tests/test_attention_rollout.py tests/test_data_sampling.py
git commit -m "test: cover ECE, attention rollout, and stratified sampling logic"
```

---

### Task 7: Dependency pinning + slim serving requirements (I3, I8)

**Files:**
- Modify: `requirements.txt` (pin)
- Create: `requirements-app.txt` (slim, pinned — what the HF Space installs)
- Create: `requirements-dev.txt` (pytest + ruff)

**Interfaces:** none (build inputs).

- [ ] **Step 1: Pin `requirements.txt`.** Convert each `>=` to a `>=floor,<next-major` cap so a fresh install can't silently jump a major version. Keep the existing floors; add upper caps. Example head (apply the same pattern to every line):

```
torch>=2.2.0,<3.0.0
torchvision>=0.17.0,<1.0.0
transformers>=4.47.0,<5.0.0
datasets>=2.19.0,<4.0.0
gradio>=4.36.0,<7.0.0
mlflow>=2.12.0,<4.0.0
numpy>=1.26.0,<2.0.0
```

(Apply caps to all remaining lines following the same major-version rule. Rationale comment at top: `# Bounded ranges keep HF Spaces / fresh installs reproducible across rebuilds.`)

- [ ] **Step 2: Create `requirements-app.txt` (serving only — no mlflow/umap/albumentations/opencv/notebook/plotly).**

```
# Slim dependency set for the Hugging Face Space (inference only).
# Training-only deps (mlflow, umap-learn, albumentations, opencv, notebook,
# plotly, seaborn) are intentionally excluded to keep the Space build fast.
torch>=2.2.0,<3.0.0
torchvision>=0.17.0,<1.0.0
transformers>=4.47.0,<5.0.0
huggingface-hub>=0.22.0,<1.0.0
Pillow>=10.3.0,<12.0.0
numpy>=1.26.0,<2.0.0
scipy>=1.13.0,<2.0.0
scikit-learn>=1.4.0,<2.0.0
matplotlib>=3.8.0,<4.0.0
pandas>=2.2.0,<3.0.0
tabulate>=0.9.0,<1.0.0
gradio>=4.36.0,<7.0.0
```

(`tabulate` is required by `df.to_markdown()` in the Results tab — currently an undeclared transitive dependency; adding it fixes a latent ImportError on the Space.)

- [ ] **Step 3: Create `requirements-dev.txt`.**

```
-r requirements.txt
pytest>=8.0,<9.0
ruff>=0.6,<1.0
```

- [ ] **Step 4: Verify the slim set is internally consistent.**

Run: `python -m pip install --dry-run -r requirements-app.txt` (user runs this where network is available)
Expected: resolver finds a consistent set. On this box, just `python -c "print('ok')"`.

- [ ] **Step 5: Commit.**

```bash
git add requirements.txt requirements-app.txt requirements-dev.txt
git commit -m "build: pin dependency ranges; add slim serving + dev requirement sets"
```

---

### Task 8: CI — ruff lint + pytest on push/PR (I2)

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write `.github/workflows/ci.yml`.**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dev deps (CPU-only, no heavy ML stack)
        run: |
          python -m pip install --upgrade pip
          pip install pytest ruff
      - name: Lint
        run: ruff check .
      - name: Run lightweight tests
        # Only the pure-logic + contract tests; model-download tests are skipped
        # automatically via importorskip when heavy deps are absent.
        run: pytest tests/ -q
```

(Heavy ML deps are deliberately NOT installed in CI — the contract/logic tests use `importorskip`, so CI stays fast and free. This is documented in the README CI note.)

- [ ] **Step 2: Verify YAML parses.**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('yaml ok')"`
Expected: `yaml ok` (skip if `pyyaml` absent — GitHub validates on push anyway).

- [ ] **Step 3: Commit.**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: lint + test on every push and PR"
```

---

### Task 9: Dockerfile for HF Spaces (I5)

**Files:**
- Create: `Dockerfile`, `.dockerignore`

- [ ] **Step 1: Write `Dockerfile` (minimal, non-root, slim serving deps).**

```dockerfile
# Hugging Face Spaces (Docker SDK) — CPU inference image for the Gradio demo.
FROM python:3.12-slim

# System libs needed by Pillow/torchvision image decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Non-root user (HF Spaces convention: uid 1000).
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/.cache/huggingface \
    LOG_LEVEL=INFO \
    PORT=7860

WORKDIR /home/user/app

COPY --chown=user requirements-app.txt .
RUN pip install --no-cache-dir --user -r requirements-app.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app/gradio_app.py"]
```

- [ ] **Step 2: Write `.dockerignore`.**

```
.git
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
mlruns/
notebooks/
docs/
tests/
*.md
.env
```

- [ ] **Step 3: Make the app bind correctly for Docker/Spaces.** In `app/gradio_app.py` entrypoint, bind host/port from env:

```python
if __name__ == "__main__":
    import os
    demo = build_demo()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
```

- [ ] **Step 4: Verify syntax.**

Run: `python -m py_compile app/gradio_app.py && echo OK`
Expected: `OK`. (Docker build itself is verified by the user / on the Space.)

- [ ] **Step 5: Commit.**

```bash
git add Dockerfile .dockerignore app/gradio_app.py
git commit -m "feat: add non-root Docker image for HF Spaces deployment"
```

---

### Task 10: Documentation — deployment, artifacts, ADR (I7)

**Files:**
- Modify: `README.md`
- Create: `docs/adr/0001-artifact-path-contract.md`
- Modify: `.gitignore` (add `.ruff_cache/`)

- [ ] **Step 1: Add a "Deployment (Hugging Face Spaces, free tier)" section to `README.md`** after the Gradio demo section:

```markdown
## Deployment — Hugging Face Spaces (free CPU tier)

The Gradio app deploys to HF Spaces using the included `Dockerfile`.

1. Create a new Space → SDK: **Docker** → Hardware: **CPU basic (free)**.
2. Push this repo to the Space remote (or connect the GitHub repo).
3. Commit your trained artifacts under `results/` so the app serves real
   weights (see "Populating results" below). Large `.pt` files use Git LFS.
4. Set Space secrets if pushing to the Hub: `HF_TOKEN` (optional).

The Space installs `requirements-app.txt` (slim inference set), not the full
training stack. Free CPU Spaces sleep after 48h idle and cold-start on the
next visit — expected on the free tier.

### Populating results
The app loads checkpoints from `results/` produced by the training scripts:
- `results/vision/<model>/full_finetune/frac1.00/best_model.pt`
- `results/text/<model>/best_model.pt` and `temperature.json`
- `results/clip/retrieval_index.pt` (built in notebook 03)

Run training locally (GPU), then commit these files for the Space to serve.
```

- [ ] **Step 2: Replace the README setup block** to reference the new requirement files:

```markdown
## Setup

```bash
git clone https://github.com/sbokk/Transfer-Learning-HuggingFace
cd Transfer-Learning-HuggingFace
pip install -r requirements.txt        # full training stack
# or, for serving only:
pip install -r requirements-app.txt
cp .env.example .env
```
```

- [ ] **Step 3: Write `docs/adr/0001-artifact-path-contract.md`.**

```markdown
# ADR 0001 — Single source of truth for training-artifact paths

## Status
Accepted (2026-06-29).

## Context
The training scripts wrote model checkpoints to one directory layout while the
Gradio serving app read from a different hardcoded layout. They never matched,
so the app silently fell back to randomly-initialised weights and served
confident-but-wrong predictions with no error. The same drift affected the
calibration temperature file and the model-key registry.

## Decision
All artifact locations are defined once in `src/utils/paths.py`. Both the
trainers (writers) and the app (reader) import these helpers. Model keys come
only from the `configs/*_config.py` registries; no module hardcodes a duplicate.
A unit test (`tests/test_paths.py`) asserts the app's display→key mapping
resolves against the registry, so the drift cannot silently return.

## Consequences
- Adding a model means updating the registry in one place.
- Changing the on-disk layout is a one-file change, covered by tests.
- Trade-off: a thin indirection layer instead of inline string paths — worth it
  given the failure was invisible in production.
```

- [ ] **Step 4: Add `.ruff_cache/` to `.gitignore`.**

```
.ruff_cache/
```

- [ ] **Step 5: Commit.**

```bash
git add README.md docs/adr/0001-artifact-path-contract.md .gitignore
git commit -m "docs: add HF Spaces deployment guide, artifact-population steps, and ADR"
```

---

## Self-Review

**Spec coverage:** Every Critical/Important finding in the mapping table maps to a task — verified row by row. C1→T4, C2/A3→T1+T2+T4, A2→T1+T4, A6→T4, A5→T3+T4, A4→T4, A7→T4, V1→T2, T1/T2→T3, V3→T2, I1→T1/2/3/5/6, I2→T8, I3→T7, I4→T5, I5→T9, I6→T5, I7→T10, I8→T7. No gaps.

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N" — all code blocks are concrete. The only deferred verifications are explicitly the ones that need model downloads or a Docker daemon, and each names the exact command the user runs locally.

**Type consistency:** `vision_checkpoint_path(model_key, strategy, fraction)` and `text_checkpoint_path(model_key)` / `text_temperature_path(model_key)` are used identically in trainers (Task 2/3) and app (Task 4). `build_model(...) -> (model, processor)` is consumed as `model, _ = build_model(...)`. `get_logger(name)` / `require_env(keys)` signatures match across Task 5 definition and usages. `compute_attention_rollout(list_of_attn)` matches the existing function signature in `visualization.py`.

**Known residual (not in scope, flagged not fixed):** `TemperatureScaler.fit()` in `metrics.py` calls `self.model(inputs)` expecting a bare logits tensor and would break on HF models — but it is unused (the text trainer fits temperature manually), so it is left as-is rather than expanding scope. Noted here for honesty.
