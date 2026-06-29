"""
Script entry point for Notebook 3 CLIP experiments.

Usage:
    python scripts/train_clip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.clip_config import CLIPConfig
from src.clip.pipeline import run_clip_pipeline


def main():
    cfg = CLIPConfig()
    results = run_clip_pipeline(cfg)

    print("\n" + "=" * 60)
    print("CLIP RESULTS")
    print("=" * 60)
    zs = results["zero_shot"]
    print(f"Zero-shot (best single template): {zs['best_template_accuracy']:.4f}")
    print(f"Zero-shot (ensemble 5 templates): {zs['ensemble_accuracy']:.4f}")
    print(f"Accuracy std across templates:    {zs['std_template_accuracy']:.4f}")
    print("\nPer-template accuracies:")
    from configs.clip_config import PROMPT_TEMPLATES

    for tmpl, acc in zip(PROMPT_TEMPLATES, zs["per_template_accuracy"]):
        print(f"  {acc:.4f}  {tmpl}")

    print("\nFew-shot linear probe:")
    for k, acc in results["few_shot"].items():
        print(f"  k={k:3s}: {float(acc):.4f}")


if __name__ == "__main__":
    main()
