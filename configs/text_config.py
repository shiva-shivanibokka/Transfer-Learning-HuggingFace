"""
Configuration for Notebook 2: Text transfer learning + calibration study.
"""

from dataclasses import dataclass


# ── Dataset ────────────────────────────────────────────────────────────────────

DATASET_NAME = "dair-ai/emotion"
NUM_CLASSES = 6
EMOTION_CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
MAX_SEQ_LENGTH = 128

# ── Models under comparison ────────────────────────────────────────────────────

TEXT_MODELS = {
    "distilbert": {
        "hf_id": "distilbert-base-uncased",
        "params_M": 66,
        "year": 2019,
        "description": "DistilBERT — 40% smaller, 60% faster than BERT. Efficiency reference only.",
    },
    "roberta": {
        "hf_id": "roberta-base",
        "params_M": 125,
        "year": 2019,
        "description": "RoBERTa — improved BERT pretraining (dynamic masking, no NSP, 10× more data)",
    },
    "modernbert": {
        "hf_id": "answerdotai/ModernBERT-base",
        "params_M": 149,
        "year": 2024,
        "description": "ModernBERT — 2024 BERT successor (RoPE, Flash Attention, 2T tokens, 8k context)",
    },
}

# ── Training hyperparameters ───────────────────────────────────────────────────


@dataclass
class TextTrainingConfig:
    model_key: str = "modernbert"
    num_epochs: int = 5
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_length: int = MAX_SEQ_LENGTH
    fp16: bool = True
    label_smoothing: float = 0.0

    # Calibration
    run_calibration: bool = True
    calibration_bins: int = 15  # ECE bin count
    temperature_init: float = 1.5  # Starting temperature for scaling
    temperature_lr: float = 0.01
    temperature_epochs: int = 50

    # MLflow
    mlflow_experiment: str = "transfer_learning_text"
    mlflow_tracking_uri: str = "mlruns"

    # HuggingFace Hub
    push_to_hub: bool = False
    hub_model_id: str = ""

    output_dir: str = "results/text"
    seed: int = 42
