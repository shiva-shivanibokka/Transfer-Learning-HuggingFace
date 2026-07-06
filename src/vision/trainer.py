"""
Training loop for vision models.

Uses HuggingFace Trainer under the hood — this avoids reimplementing
the training loop while keeping full control over the model, data,
and evaluation. MLflow logging is wired in via a custom callback.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import torch
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

from configs.vision_config import (
    EUROSAT_CLASSES,
    NUM_CLASSES,
    VISION_MODELS,
    VisionTrainingConfig,
)
from src.utils.data import load_eurosat
from src.utils.logging_utils import get_logger
from src.utils.metrics import (
    benchmark_latency,
    benchmark_onnx_latency,
    export_to_onnx,
)
from src.vision.model import build_model, count_trainable_params

log = get_logger(__name__)


# ── MLflow callback ───────────────────────────────────────────────────────────


class MLflowVisionCallback(TrainerCallback):
    """Log per-step and per-epoch metrics to MLflow."""

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        if logs and mlflow.active_run():
            step = state.global_step
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v, step=step)


# ── Custom dataset for HF Trainer ─────────────────────────────────────────────


class HFVisionDataset(torch.utils.data.Dataset):
    """Bridge between our EuroSATDataset and HuggingFace Trainer's expected format."""

    def __init__(self, eurosat_dataset):
        self.ds = eurosat_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pixel_values, label = self.ds[idx]
        return {"pixel_values": pixel_values, "labels": torch.tensor(label)}


# ── Compute metrics function ───────────────────────────────────────────────────


def make_compute_metrics(class_names):
    import evaluate

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)
        f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")
        return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"]}

    return compute_metrics


# ── Main training function ─────────────────────────────────────────────────────


def train_vision_model(
    cfg: VisionTrainingConfig,
    return_model: bool = False,
) -> dict:
    """
    Train a single vision model with the given config.
    Logs to MLflow, saves results JSON.

    Returns a dict of final metrics.
    """
    from src.utils.mlflow_utils import setup_mlflow

    setup_mlflow(cfg.mlflow_experiment, cfg.mlflow_tracking_uri)

    id2label = {i: c for i, c in enumerate(EUROSAT_CLASSES)}
    label2id = {c: i for i, c in enumerate(EUROSAT_CLASSES)}

    # Build model
    model, processor = build_model(
        model_key=cfg.model_key,
        num_classes=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        strategy=cfg.strategy,
        dropout=cfg.dropout,
    )

    param_info = count_trainable_params(model)
    log.info(f"\n[{cfg.model_key} | {cfg.strategy} | {cfg.data_fraction * 100:.0f}% data]")
    log.info(
        f"  Trainable: {param_info['trainable_params']:,} / {param_info['total_params']:,} "
        f"({param_info['trainable_pct']:.1f}%)"
    )

    # Load data
    train_loader, val_loader, test_loader = load_eurosat(
        data_fraction=cfg.data_fraction,
        image_size=224,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        augmentation_strength=cfg.augmentation_strength,
        seed=cfg.seed,
    )

    # Wrap for HF Trainer
    train_hf = HFVisionDataset(train_loader.dataset)
    val_hf = HFVisionDataset(val_loader.dataset)
    test_hf = HFVisionDataset(test_loader.dataset)

    # Training args
    output_dir = (
        Path(cfg.output_dir)
        / cfg.model_key
        / cfg.strategy
        / f"frac{cfg.data_fraction:.2f}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_epochs / max(cfg.num_epochs, 1),
        fp16=cfg.fp16 and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=20,
        label_smoothing_factor=cfg.label_smoothing,
        report_to="none",  # we handle MLflow ourselves
        seed=cfg.seed,
        # Parallel data loading: the image augmentation pipeline is CPU-heavy, so
        # with 0 workers the GPU starves (observed ~16% utilisation). A few worker
        # processes lift it to 50-90%. persistent_workers is kept OFF so the train
        # pool is released before eval spawns its own — otherwise both live at once
        # and 2x the torch processes exhaust the Windows page file (WinError 1455).
        dataloader_num_workers=cfg.num_workers,
        dataloader_persistent_workers=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    run_name = f"{cfg.model_key}__{cfg.strategy}__frac{cfg.data_fraction:.2f}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_key": cfg.model_key,
                "hf_model_id": VISION_MODELS[cfg.model_key]["hf_id"],
                "strategy": cfg.strategy,
                "data_fraction": cfg.data_fraction,
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "num_epochs": cfg.num_epochs,
                "fp16": cfg.fp16,
                **param_info,
            }
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            compute_metrics=make_compute_metrics(EUROSAT_CLASSES),
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                MLflowVisionCallback(),
            ],
        )

        train_result = trainer.train()
        mlflow.log_metric("train_loss_final", train_result.training_loss)

        # Test evaluation
        test_results = trainer.evaluate(test_hf)
        mlflow.log_metrics(
            {
                "test_accuracy": test_results.get("eval_accuracy", 0),
                "test_f1_macro": test_results.get("eval_f1_macro", 0),
            }
        )

        # Persist the deployable checkpoint. load_best_model_at_end=True means
        # `model` is already the best epoch's weights — save them where the app
        # (via src.utils.paths) will look for them.
        from src.utils.paths import vision_checkpoint_path

        ckpt_path = vision_checkpoint_path(cfg.model_key, cfg.strategy, cfg.data_fraction)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        log.info(f"  Saved deployable checkpoint -> {ckpt_path}")

        # Latency benchmark (CPU)
        latency = benchmark_latency(model, image_size=224, device="cpu")
        mlflow.log_metrics(
            {
                "latency_cpu_mean_ms": latency["mean_ms"],
                "latency_cpu_p95_ms": latency["p95_ms"],
                "throughput_cpu": latency["throughput_imgs_per_sec"],
            }
        )

        # ONNX export + benchmark
        onnx_path = str(output_dir / f"{cfg.model_key}.onnx")
        try:
            export_to_onnx(model, onnx_path, device="cpu")
            onnx_latency = benchmark_onnx_latency(onnx_path)
            mlflow.log_metrics(
                {
                    "latency_onnx_mean_ms": onnx_latency["mean_ms"],
                    "throughput_onnx": onnx_latency["throughput_imgs_per_sec"],
                }
            )
            mlflow.log_artifact(onnx_path)
        except Exception as e:
            log.info(f"  ONNX export failed: {e}")
            onnx_latency = {}

        # Save result JSON. Flat scalar keys so the summary CSV is clean
        # (no nested dicts rendered as Python reprs).
        result = {
            "model_key": cfg.model_key,
            "strategy": cfg.strategy,
            "data_fraction": cfg.data_fraction,
            "trainable_params": param_info["trainable_params"],
            "total_params": param_info["total_params"],
            "trainable_pct": param_info["trainable_pct"],
            "test_accuracy": test_results.get("eval_accuracy", 0),
            "test_f1_macro": test_results.get("eval_f1_macro", 0),
            "latency_cpu_mean_ms": latency.get("mean_ms"),
            "latency_cpu_p95_ms": latency.get("p95_ms"),
            "onnx_mean_ms": onnx_latency.get("mean_ms") if onnx_latency else None,
        }

        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2))
        mlflow.log_artifact(str(result_path))

        # Hub push
        if cfg.push_to_hub and cfg.hub_model_id:
            trainer.push_to_hub(
                commit_message=f"Transfer learning: {cfg.model_key} on EuroSAT ({cfg.strategy})"
            )

    if return_model:
        result["model"] = model
        result["processor"] = processor

    return result
