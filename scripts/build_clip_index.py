"""Build the CLIP retrieval index the Gradio app's "CLIP Image Search" tab needs.

Encodes a stratified sample of EuroSAT test images with CLIP and saves
{features, labels, images} to results/clip/retrieval_index.safetensors (plain
tensors only — no pickle, so it loads safely at serve time). The pipeline
(train_clip.py) computes zero-shot / few-shot metrics but does NOT build this
index, so it is produced here.

    python scripts/build_clip_index.py [--per-class 100]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def build_index(per_class: int = 100):
    """Encode a stratified sample of EuroSAT test images and save the index.

    Returns the path the index was written to. Callable from the grid runner
    as well as the CLI.
    """
    if per_class < 1:
        raise ValueError(f"per_class must be >= 1, got {per_class}")

    import numpy as np
    import torch
    from datasets import load_dataset
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    from configs.clip_config import CLIP_MODEL_ID, DATASET_NAME
    from src.utils.logging_utils import get_logger
    from src.utils.paths import clip_index_path

    log = get_logger("build_clip_index")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading CLIP %s on %s", CLIP_MODEL_ID, device)
    try:
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load CLIP model/processor '{CLIP_MODEL_ID}': {exc}"
        ) from exc

    log.info("Loading %s test split", DATASET_NAME)
    try:
        ds = load_dataset(DATASET_NAME, split="test")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load dataset '{DATASET_NAME}' (test split): {exc}"
        ) from exc

    # Stratified sample: per_class images per label.
    rng = random.Random(42)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(ds["label"]):
        by_class[int(lbl)].append(i)
    idxs: list[int] = []
    for ids in by_class.values():
        idxs.extend(rng.sample(ids, min(per_class, len(ids))))
    rng.shuffle(idxs)
    log.info("Encoding %d images (%d classes)", len(idxs), len(by_class))

    images, labels, feats = [], [], []
    batch = 64
    for start in range(0, len(idxs), batch):
        chunk = idxs[start : start + batch]
        pil_imgs = []
        for i in chunk:
            img = ds[i]["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img).convert("RGB")
            else:
                img = img.convert("RGB")
            pil_imgs.append(img)
        inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            f = model.get_image_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
        for i, pim in zip(chunk, pil_imgs):
            images.append(np.asarray(pim.resize((64, 64)), dtype=np.uint8))
            labels.append(int(ds[i]["label"]))

    # All fields are plain tensors so the index can be saved/loaded as
    # safetensors (no pickle => no arbitrary-code-execution surface at the
    # serve-time torch.load the app used to do). Images are packed as a uint8
    # [N, 64, 64, 3] tensor.
    from safetensors.torch import save_file

    index = {
        "features": torch.cat(feats, dim=0).to(torch.float32).contiguous(),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "images": torch.from_numpy(np.stack(images)).to(torch.uint8).contiguous(),
    }
    out = clip_index_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic save: write to a temp path in the same dir, then os.replace so an
    # interrupted save can never leave a half-written / corrupt index behind.
    tmp = out.with_name(out.name + ".tmp")
    save_file(index, str(tmp))
    os.replace(tmp, out)
    log.info("Saved retrieval index: %d images -> %s", index["features"].shape[0], out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=100)
    args = parser.parse_args()

    if args.per_class < 1:
        parser.error("--per-class must be >= 1")

    build_index(per_class=args.per_class)


if __name__ == "__main__":
    main()
