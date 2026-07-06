"""
Configuration for Notebook 1: Vision transfer learning experiments.
All hyperparameters and model definitions live here — notebooks import this.
"""

from dataclasses import dataclass

# ── Dataset ────────────────────────────────────────────────────────────────────

DATASET_NAME = "timm/eurosat-rgb"
NUM_CLASSES = 10
IMAGE_SIZE = 224
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

# ── Models under comparison ────────────────────────────────────────────────────

VISION_MODELS = {
    "resnet50": {
        "hf_id": "microsoft/resnet-50",
        "family": "CNN",
        "params_M": 25.6,
        "gflops": 4.1,
        "year": 2015,
        "description": "Classic residual CNN — the standard ImageNet baseline",
    },
    "efficientnet_b0": {
        "hf_id": "google/efficientnet-b0",
        "family": "CNN",
        "params_M": 5.3,
        "gflops": 0.39,
        "year": 2019,
        "description": "Compound-scaled CNN — best accuracy/efficiency in its era",
    },
    "vit_base": {
        "hf_id": "google/vit-base-patch16-224",
        "family": "ViT",
        "params_M": 86.0,
        "gflops": 17.6,
        "year": 2020,
        "description": "Original Vision Transformer — patches of 16×16, 12 layers",
    },
    "dinov2_base": {
        "hf_id": "facebook/dinov2-base",
        "family": "DINOv2",
        "params_M": 86.0,
        "gflops": 17.6,
        "year": 2023,
        "description": "Self-supervised ViT (Meta AI) — no ImageNet labels, trained on 142M images",
    },
}

# ── Training strategies ────────────────────────────────────────────────────────

STRATEGIES = {
    "linear_probe": {
        "freeze_backbone": True,
        "unfreeze_last_n_blocks": 0,
        "description": "Freeze everything, train only the classification head",
    },
    "partial_unfreeze": {
        "freeze_backbone": True,
        "unfreeze_last_n_blocks": 2,
        "description": "Unfreeze last 2 blocks + classification head",
    },
    "full_finetune": {
        "freeze_backbone": False,
        "unfreeze_last_n_blocks": -1,
        "description": "Train all parameters end-to-end",
    },
}

# ── Data fraction experiments ──────────────────────────────────────────────────

DATA_FRACTIONS = [0.01, 0.05, 0.10, 1.0]  # 1%, 5%, 10%, 100%

# ── Training hyperparameters ───────────────────────────────────────────────────


@dataclass
class VisionTrainingConfig:
    model_key: str = "efficientnet_b0"
    strategy: str = "full_finetune"
    data_fraction: float = 1.0

    # Optimizer
    lr: float = 2e-4
    weight_decay: float = 1e-4
    lr_backbone_multiplier: float = 0.1  # backbone gets 10× lower LR than head

    # Schedule
    num_epochs: int = 10
    warmup_epochs: int = 1
    batch_size: int = 32

    # Regularisation
    label_smoothing: float = 0.1
    dropout: float = 0.1

    # Data augmentation (training only)
    use_augmentation: bool = True
    augmentation_strength: str = "medium"  # "light" | "medium" | "strong"

    # Hardware
    fp16: bool = True
    # Parallel DataLoader workers keep the GPU fed. Capped low because each worker
    # is a separate process that loads torch's CUDA DLLs into Windows commit
    # memory; too many exhaust the page file (WinError 1455).
    num_workers: int = 4

    # MLflow
    mlflow_experiment: str = "transfer_learning_vision"
    mlflow_tracking_uri: str = "mlruns"

    # HuggingFace Hub
    push_to_hub: bool = False
    hub_model_id: str = ""

    # Output
    output_dir: str = "results/vision"
    seed: int = 42
