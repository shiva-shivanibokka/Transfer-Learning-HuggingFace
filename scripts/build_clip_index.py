"""Build the CLIP retrieval index the Gradio app's "CLIP Image Search" tab needs.

Encodes a stratified sample of EuroSAT test images with CLIP and saves
{features, labels, images} to results/clip/retrieval_index.pt. The pipeline
(train_clip.py) computes zero-shot / few-shot metrics but does NOT build this
index, so it is produced here.

    python scripts/build_clip_index.py [--per-class 100]
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=100)
    args = parser.parse_args()

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import CLIPModel, CLIPProcessor

    from configs.clip_config import CLIP_MODEL_ID, DATASET_NAME
    from src.utils.logging_utils import get_logger
    from src.utils.paths import clip_index_path

    log = get_logger("build_clip_index")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading CLIP %s on %s", CLIP_MODEL_ID, device)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    log.info("Loading %s test split", DATASET_NAME)
    ds = load_dataset(DATASET_NAME, split="test")

    # Stratified sample: per_class images per label.
    rng = random.Random(42)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(ds["label"]):
        by_class[int(lbl)].append(i)
    idxs: list[int] = []
    for ids in by_class.values():
        idxs.extend(rng.sample(ids, min(args.per_class, len(ids))))
    rng.shuffle(idxs)
    log.info("Encoding %d images (%d classes)", len(idxs), len(by_class))

    images, labels, feats = [], [], []
    batch = 64
    for start in range(0, len(idxs), batch):
        chunk = idxs[start : start + batch]
        pil_imgs = [ds[i]["image"].convert("RGB") for i in chunk]
        inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            f = model.get_image_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
        for i, pim in zip(chunk, pil_imgs):
            images.append(np.asarray(pim.resize((64, 64)), dtype=np.uint8))
            labels.append(int(ds[i]["label"]))

    index = {
        "features": torch.cat(feats, dim=0),
        "labels": torch.tensor(labels),
        "images": images,
    }
    out = clip_index_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index, out)
    log.info("Saved retrieval index: %d images -> %s", index["features"].shape[0], out)


if __name__ == "__main__":
    main()
