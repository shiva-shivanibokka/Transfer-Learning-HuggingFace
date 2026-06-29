"""
Script entry point for Notebook 1 vision experiments.

Runs all combinations of models × strategies × data fractions,
logs every run to MLflow, and saves a summary CSV.

Usage:
    # All experiments (takes hours on GPU, ~30min on CPU for subset)
    python scripts/train_vision.py

    # Quick single run for testing
    python scripts/train_vision.py --model efficientnet_b0 --strategy full_finetune --fraction 0.1

    # Data efficiency study only (full fine-tune, all fractions)
    python scripts/train_vision.py --study data_efficiency

    # Strategy comparison only (all models, 100% data)
    python scripts/train_vision.py --study strategy_comparison
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.vision_config import (
    DATA_FRACTIONS,
    STRATEGIES,
    VISION_MODELS,
    VisionTrainingConfig,
)
from src.vision.trainer import train_vision_model


def run_single(
    model_key: str, strategy: str, fraction: float, push_to_hub: bool = False
) -> dict:
    cfg = VisionTrainingConfig(
        model_key=model_key,
        strategy=strategy,
        data_fraction=fraction,
        push_to_hub=push_to_hub,
    )
    return train_vision_model(cfg)


def run_all_experiments(study: str = "all", push_to_hub: bool = False) -> list[dict]:
    """
    Run the full experiment grid.

    study="all":                 All models × all strategies × all fractions
    study="strategy_comparison": All models × all strategies × 100% data only
    study="data_efficiency":     All models × full_finetune × all fractions
    """
    results = []

    if study in ("all", "strategy_comparison"):
        for model_key in VISION_MODELS:
            for strategy in STRATEGIES:
                fraction = 1.0
                print(f"\n{'=' * 60}")
                print(f"STRATEGY COMPARISON: {model_key} | {strategy} | 100% data")
                print(f"{'=' * 60}")
                r = run_single(model_key, strategy, fraction, push_to_hub)
                results.append(r)

    if study in ("all", "data_efficiency"):
        for model_key in VISION_MODELS:
            for fraction in DATA_FRACTIONS:
                if fraction == 1.0 and study == "all":
                    continue  # already done above
                print(f"\n{'=' * 60}")
                print(
                    f"DATA EFFICIENCY: {model_key} | full_finetune | {fraction * 100:.0f}% data"
                )
                print(f"{'=' * 60}")
                r = run_single(model_key, "full_finetune", fraction, push_to_hub)
                results.append(r)

    # Save summary CSV
    summary_path = Path("results/vision/summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        keys = [
            "model_key", "strategy", "data_fraction",
            "trainable_params", "total_params", "trainable_pct",
            "test_accuracy", "test_f1_macro",
            "latency_cpu_mean_ms", "latency_cpu_p95_ms", "onnx_mean_ms",
        ]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSummary saved to {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train vision transfer learning models on EuroSAT"
    )
    parser.add_argument("--model", choices=list(VISION_MODELS.keys()), default=None)
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument(
        "--study",
        choices=["all", "strategy_comparison", "data_efficiency"],
        default="all",
        help="Which experiment suite to run",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    if args.model and args.strategy and args.fraction is not None:
        result = run_single(args.model, args.strategy, args.fraction, args.push_to_hub)
        print(
            f"\nResult: accuracy={result['test_accuracy']:.4f}, "
            f"latency={result['latency_cpu_mean_ms']:.1f}ms"
        )
    else:
        results = run_all_experiments(args.study, args.push_to_hub)
        print(f"\nCompleted {len(results)} experiments.")


if __name__ == "__main__":
    main()
