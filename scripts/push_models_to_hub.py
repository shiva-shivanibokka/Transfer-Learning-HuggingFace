"""Publish the deployable fine-tuned models to the Hugging Face Hub.

Loads each best_model.pt state_dict into its architecture and pushes it in
proper HF format (config.json + safetensors) so the serving app can load it with
`from_pretrained("<user>/<repo>")`. Text repos also get their calibration
temperature.json. Run once after training; requires a write token
(`huggingface-cli login`).

    python scripts/push_models_to_hub.py --user shiva-1993
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Repo naming: <user>/eurosat-<model> (vision), <user>/emotion-<model> (text).
VISION_REPO = {
    "resnet50": "eurosat-resnet50",
    "efficientnet_b0": "eurosat-efficientnet-b0",
    "vit_base": "eurosat-vit-base",
    "dinov2_base": "eurosat-dinov2-base",
}
TEXT_REPO = {
    "roberta": "emotion-roberta",
    "modernbert": "emotion-modernbert",
    "distilbert": "emotion-distilbert",
}


def push_vision(user: str, log, private: bool = False):
    import torch
    from huggingface_hub import upload_file

    from configs.vision_config import EUROSAT_CLASSES, NUM_CLASSES
    from src.utils.paths import (
        DEMO_VISION_FRACTION,
        DEMO_VISION_STRATEGY,
        vision_checkpoint_path,
    )
    from src.vision.model import build_model

    id2label = {i: c for i, c in enumerate(EUROSAT_CLASSES)}
    label2id = {c: i for i, c in enumerate(EUROSAT_CLASSES)}

    for model_key, repo_name in VISION_REPO.items():
        repo_id = f"{user}/{repo_name}"
        ckpt = vision_checkpoint_path(model_key, DEMO_VISION_STRATEGY, DEMO_VISION_FRACTION)
        if not ckpt.exists():
            log.warning("skip %s: no checkpoint at %s", model_key, ckpt)
            continue
        # One model's failure must not abort the other six pushes.
        try:
            model, processor = build_model(
                model_key=model_key, num_classes=NUM_CLASSES,
                id2label=id2label, label2id=label2id, strategy=DEMO_VISION_STRATEGY,
            )
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            log.info("pushing %s -> %s", model_key, repo_id)
            model.push_to_hub(
                repo_id, commit_message="EuroSAT fine-tuned vision model", private=private
            )
            processor.push_to_hub(repo_id, commit_message="image processor", private=private)
            result_json = ckpt.parent / "result.json"
            if result_json.exists():
                upload_file(
                    path_or_fileobj=str(result_json),
                    path_in_repo="result.json", repo_id=repo_id,
                    commit_message="training metrics",
                )
            else:
                log.warning("  no result.json for %s at %s", model_key, result_json)
            log.info("  done: https://huggingface.co/%s", repo_id)
        except Exception as exc:  # noqa: BLE001
            log.error("  FAILED to push %s -> %s: %s", model_key, repo_id, exc)
            continue


def push_text(user: str, log, private: bool = False):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from configs.text_config import EMOTION_CLASSES, TEXT_MODELS
    from src.utils.paths import text_checkpoint_path, text_temperature_path

    for model_key, repo_name in TEXT_REPO.items():
        repo_id = f"{user}/{repo_name}"
        ckpt = text_checkpoint_path(model_key)
        if not ckpt.exists():
            log.warning("skip %s: no checkpoint at %s", model_key, ckpt)
            continue
        # One model's failure must not abort the remaining text pushes.
        try:
            cfg = TEXT_MODELS[model_key]
            tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg["hf_id"], num_labels=len(EMOTION_CLASSES),
                id2label={i: c for i, c in enumerate(EMOTION_CLASSES)},
                label2id={c: i for i, c in enumerate(EMOTION_CLASSES)},
            )
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            log.info("pushing %s -> %s", model_key, repo_id)
            model.push_to_hub(
                repo_id, commit_message="dair-ai/emotion fine-tuned", private=private
            )
            tok.push_to_hub(repo_id, commit_message="tokenizer", private=private)
            # Calibration temperature travels with the model repo.
            temp = text_temperature_path(model_key)
            if temp.exists():
                from huggingface_hub import upload_file
                upload_file(
                    path_or_fileobj=str(temp), path_in_repo="temperature.json",
                    repo_id=repo_id, commit_message="calibration temperature",
                )
            else:
                log.warning("  no temperature.json for %s at %s", model_key, temp)
            log.info("  done: https://huggingface.co/%s", repo_id)
        except Exception as exc:  # noqa: BLE001
            log.error("  FAILED to push %s -> %s: %s", model_key, repo_id, exc)
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="HF username / org")
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument(
        "--private", action="store_true", help="Create the Hub repos as private."
    )
    args = parser.parse_args()

    from src.utils.logging_utils import get_logger

    log = get_logger("push_to_hub")
    if not args.skip_vision:
        push_vision(args.user, log, private=args.private)
    if not args.skip_text:
        push_text(args.user, log, private=args.private)
    log.info("ALL PUSHES DONE")


if __name__ == "__main__":
    main()
