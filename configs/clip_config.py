"""
Configuration for Notebook 3: CLIP zero-shot + prompt engineering study.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ── Model ──────────────────────────────────────────────────────────────────────

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# ── Dataset ────────────────────────────────────────────────────────────────────

DATASET_NAME = "blanchefort/eurosat_rgb"  # same as Notebook 1 — intentional
NUM_CLASSES = 10
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# ── Prompt templates ───────────────────────────────────────────────────────────
# Five templates per class, testing sensitivity of zero-shot accuracy to wording.
# Based on Radford et al. (2021) — OpenAI used 80-template ensembles for ImageNet.

PROMPT_TEMPLATES = [
    "a photo of {cls}",
    "a satellite image of {cls} land use",
    "an aerial photograph showing {cls}",
    "a remote sensing image of {cls}",
    "{cls} viewed from above",
]

# Human-readable class descriptions used in richer prompts
CLASS_DESCRIPTIONS: Dict[str, str] = {
    "AnnualCrop": "annual crop fields",
    "Forest": "dense forest",
    "HerbaceousVegetation": "herbaceous vegetation",
    "Highway": "a highway or road",
    "Industrial": "industrial buildings",
    "Pasture": "open pasture land",
    "PermanentCrop": "permanent crops like orchards",
    "Residential": "residential neighborhoods",
    "River": "a river or waterway",
    "SeaLake": "sea or lake water",
}

# ── Few-shot experiment settings ───────────────────────────────────────────────

FEW_SHOT_K_VALUES = [1, 5, 10, 25]  # examples per class

# ── Config dataclass ───────────────────────────────────────────────────────────


@dataclass
class CLIPConfig:
    model_id: str = CLIP_MODEL_ID
    batch_size: int = 64
    num_retrieval_results: int = 5  # top-k for cross-modal retrieval
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    mlflow_experiment: str = "transfer_learning_clip"
    mlflow_tracking_uri: str = "mlruns"
    output_dir: str = "results/clip"
    seed: int = 42
