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
