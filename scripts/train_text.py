"""
Script entry point for Notebook 2 text experiments.

Usage:
    python scripts/train_text.py                        # all models
    python scripts/train_text.py --model roberta        # single model
    python scripts/train_text.py --push-to-hub          # train + push best model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.text_config import TEXT_MODELS, TextTrainingConfig
from src.text.trainer import train_text_model


def main():
    parser = argparse.ArgumentParser(
        description="Train text classification models on dair-ai/emotion"
    )
    parser.add_argument(
        "--model",
        choices=list(TEXT_MODELS.keys()),
        default=None,
        help="Single model to train. Default: train all.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the best model to HuggingFace Hub after training.",
    )
    parser.add_argument(
        "--hub-id", default="", help="HuggingFace Hub model ID for pushing."
    )
    args = parser.parse_args()

    models_to_train = [args.model] if args.model else list(TEXT_MODELS.keys())
    all_results = []

    for model_key in models_to_train:
        cfg = TextTrainingConfig(
            model_key=model_key,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_id or f"eurosat-emotion-{model_key}",
        )
        result = train_text_model(cfg)
        all_results.append(result)
        print(
            f"\n{model_key}: acc={result['test_accuracy']:.4f}, "
            f"ECE {result['ece_before']:.4f} → {result['ece_after']:.4f}, T={result['temperature']:.3f}"
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(
            f"{r['model_key']:15s}  acc={r['test_accuracy']:.4f}  "
            f"f1={r['test_f1_macro']:.4f}  "
            f"ECE_before={r['ece_before']:.4f}  ECE_after={r['ece_after']:.4f}"
        )


if __name__ == "__main__":
    main()
