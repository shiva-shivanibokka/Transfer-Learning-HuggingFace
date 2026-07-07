"""Resumable full-grid runner.

Iterates every target run (vision strategy-comparison + data-efficiency, then
text, then CLIP) and SKIPS any run whose FINAL artifact (result.json) already
exists, so an interrupted grid resumes without repeating completed work — and
crucially RE-RUNS any run interrupted mid-way (checkpoint written but no
result.json yet). Writes the vision summary.csv from all per-run result.json
files at the end.

A per-target lockfile prevents two concurrent launches from double-training the
same target (this repo has a history of orphaned/duplicated training).

Run via the __main__ guard (required for Windows DataLoader worker spawning):
    python scripts/run_grid_resumable.py
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@contextlib.contextmanager
def target_lock(lock_dir: Path, name: str):
    """Claim an exclusive per-target lock via O_CREAT|O_EXCL (robust on Windows).

    Yields True if the lock was acquired (caller should proceed) or False if
    another process already holds it (caller should skip). The lock file is
    always removed in the finally block when we own it.
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{name}.lock"
    fd = None
    try:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            print(f"[lock] {name} is already locked ({lock_path}); skipping", flush=True)
            yield False
            return
        os.write(fd, str(os.getpid()).encode())
        yield True
    finally:
        if fd is not None:
            os.close(fd)
            with contextlib.suppress(FileNotFoundError):
                os.remove(lock_path)


def _vision_targets():
    from configs.vision_config import DATA_FRACTIONS, STRATEGIES, VISION_MODELS

    targets = []
    # Strategy comparison: every model x every strategy at 100% data.
    for model_key in VISION_MODELS:
        for strategy in STRATEGIES:
            targets.append((model_key, strategy, 1.0))
    # Data efficiency: every model, full fine-tune, at the sub-100% fractions.
    # Single source of truth: configs.vision_config.DATA_FRACTIONS. 1.0 is
    # excluded here because the strategy sweep above already covers full data.
    for model_key in VISION_MODELS:
        for fraction in DATA_FRACTIONS:
            if fraction == 1.0:
                continue
            targets.append((model_key, "full_finetune", fraction))
    return targets


def run_vision():
    from configs.vision_config import VisionTrainingConfig
    from src.utils.paths import RESULTS_ROOT, vision_run_dir
    from src.vision.trainer import train_vision_model

    def result_json(model_key, strategy, fraction):
        # result.json is the LAST artifact the vision trainer writes (after the
        # checkpoint AND the latency benchmark), so keying resume off it — not
        # best_model.pt — means a run interrupted mid-benchmark re-runs instead
        # of being silently skipped and dropped from summary.csv.
        return vision_run_dir(model_key, strategy, fraction) / "result.json"

    targets = _vision_targets()
    done = sum(result_json(*t).exists() for t in targets)
    print(f"[resume] vision: {done}/{len(targets)} already complete", flush=True)

    for model_key, strategy, fraction in targets:
        if result_json(model_key, strategy, fraction).exists():
            print(f"[skip] {model_key} | {strategy} | frac{fraction:.2f}", flush=True)
            continue
        lock_dir = vision_run_dir(model_key, strategy, fraction)
        with target_lock(lock_dir, "train") as acquired:
            if not acquired:
                continue
            # Re-check under the lock in case another process finished it.
            if result_json(model_key, strategy, fraction).exists():
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
    from src.utils.paths import text_checkpoint_path, text_run_dir

    for model_key in TEXT_MODELS:
        if text_checkpoint_path(model_key).exists():
            print(f"[skip] text {model_key}", flush=True)
            continue
        with target_lock(text_run_dir(model_key), "train") as acquired:
            if not acquired:
                continue
            if text_checkpoint_path(model_key).exists():
                print(f"[skip] text {model_key}", flush=True)
                continue
            print(f"\n[run] text {model_key}", flush=True)
            train_text_model(TextTrainingConfig(model_key=model_key))


def run_clip():
    from configs.clip_config import CLIPConfig
    from src.clip.pipeline import run_clip_pipeline
    from src.utils.paths import RESULTS_ROOT

    clip_results = RESULTS_ROOT / "clip" / "clip_results.json"
    if clip_results.exists():
        print(f"[skip] clip (results exist: {clip_results})", flush=True)
        return

    with target_lock(RESULTS_ROOT / "clip", "clip") as acquired:
        if not acquired:
            return
        if clip_results.exists():
            print(f"[skip] clip (results exist: {clip_results})", flush=True)
            return
        print("\n[run] clip pipeline (zero-shot / prompt sensitivity / few-shot)", flush=True)
        run_clip_pipeline(CLIPConfig())

        # The pipeline computes metrics but does NOT build the retrieval index
        # the app serves — wire the index build here so the grid produces it.
        print("\n[run] clip retrieval index", flush=True)
        try:
            from scripts.build_clip_index import build_index

            out = build_index()
            print(f"[resume] built clip retrieval index -> {out}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] failed to build CLIP retrieval index: {exc}\n"
                f"       run `python scripts/build_clip_index.py` to produce it.",
                flush=True,
            )


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
