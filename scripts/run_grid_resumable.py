"""Resumable full-grid runner.

Iterates every target run (vision strategy-comparison + data-efficiency, then
text, then CLIP) and SKIPS any run whose deployable checkpoint already exists,
so an interrupted grid resumes without repeating completed work. Writes the
vision summary.csv from all per-run result.json files at the end.

Run via the __main__ guard (required for Windows DataLoader worker spawning):
    python scripts/run_grid_resumable.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _vision_targets():
    from configs.vision_config import STRATEGIES, VISION_MODELS

    targets = []
    # Strategy comparison: every model x every strategy at 100% data.
    for model_key in VISION_MODELS:
        for strategy in STRATEGIES:
            targets.append((model_key, strategy, 1.0))
    # Data efficiency: every model, full fine-tune, at 1/5/10% data.
    for model_key in VISION_MODELS:
        for fraction in (0.01, 0.05, 0.10):
            targets.append((model_key, "full_finetune", fraction))
    return targets


def run_vision():
    from configs.vision_config import VisionTrainingConfig
    from src.utils.paths import RESULTS_ROOT, vision_checkpoint_path
    from src.vision.trainer import train_vision_model

    targets = _vision_targets()
    done = sum(vision_checkpoint_path(*t).exists() for t in targets)
    print(f"[resume] vision: {done}/{len(targets)} already complete", flush=True)

    for model_key, strategy, fraction in targets:
        ckpt = vision_checkpoint_path(model_key, strategy, fraction)
        if ckpt.exists():
            print(f"[skip] {model_key} | {strategy} | frac{fraction:.2f}", flush=True)
            continue
        print(f"\n{'=' * 60}\n[run] {model_key} | {strategy} | frac{fraction:.2f}\n{'=' * 60}", flush=True)
        cfg = VisionTrainingConfig(model_key=model_key, strategy=strategy, data_fraction=fraction)
        train_vision_model(cfg)

    # Rebuild summary.csv from every per-run result.json (done + new).
    rows = []
    for rj in sorted((RESULTS_ROOT / "vision").rglob("result.json")):
        try:
            rows.append(json.loads(rj.read_text()))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not read {rj}: {exc}", flush=True)
    if rows:
        keys = [
            "model_key", "strategy", "data_fraction",
            "trainable_params", "total_params", "trainable_pct",
            "test_accuracy", "test_f1_macro",
            "latency_cpu_mean_ms", "latency_cpu_p95_ms", "onnx_mean_ms",
        ]
        summary = RESULTS_ROOT / "vision" / "summary.csv"
        with open(summary, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"[resume] wrote {summary} ({len(rows)} rows)", flush=True)


def run_text():
    from configs.text_config import TEXT_MODELS, TextTrainingConfig
    from src.text.trainer import train_text_model
    from src.utils.paths import text_checkpoint_path

    for model_key in TEXT_MODELS:
        if text_checkpoint_path(model_key).exists():
            print(f"[skip] text {model_key}", flush=True)
            continue
        print(f"\n[run] text {model_key}", flush=True)
        train_text_model(TextTrainingConfig(model_key=model_key))


def run_clip():
    from configs.clip_config import CLIPConfig
    from src.clip.pipeline import run_clip_pipeline
    from src.utils.paths import clip_index_path

    # The zero-shot / prompt-sensitivity results are cheap to recompute; the
    # retrieval index is the expensive artifact. Run the pipeline regardless.
    print(f"\n[run] clip (index target: {clip_index_path()})", flush=True)
    run_clip_pipeline(CLIPConfig())


def main():
    print("[resume] === VISION ===", flush=True)
    run_vision()
    print("[resume] === TEXT ===", flush=True)
    run_text()
    print("[resume] === CLIP ===", flush=True)
    run_clip()
    print("[resume] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
