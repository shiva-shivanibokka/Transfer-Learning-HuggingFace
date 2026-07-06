"""
MLflow helpers shared across all three notebooks/scripts.
Wraps run creation, param/metric logging, and artifact saving.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

# MLflow >=3 moved the local filesystem tracking store into "maintenance mode"
# and raises unless the caller opts in. This project intentionally uses the
# simple ./mlruns file store (browsable via `mlflow ui`, gitignored) rather than
# a database backend, so opt in before importing mlflow. Set the env var to
# "false" to override and force the database-backend guidance instead.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402
import mlflow.pytorch  # noqa: E402
import numpy as np  # noqa: E402


def setup_mlflow(experiment_name: str, tracking_uri: str = "mlruns") -> None:
    """Initialise MLflow with the given experiment. Creates it if it doesn't exist."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_vision_run(
    model_key: str,
    strategy: str,
    data_fraction: float,
    config: Any,
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: Optional[dict] = None,
    latency_pytorch: Optional[dict] = None,
    latency_onnx: Optional[dict] = None,
    model_params: Optional[dict] = None,
    artifacts_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> str:
    """
    Log a single vision experiment run to MLflow.
    Returns the run_id.
    """
    run_name = run_name or f"{model_key}__{strategy}__frac{data_fraction:.2f}"

    with mlflow.start_run(run_name=run_name) as run:
        # Params
        mlflow.log_params(
            {
                "model_key": model_key,
                "strategy": strategy,
                "data_fraction": data_fraction,
                "lr": config.lr,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "fp16": config.fp16,
                "augmentation_strength": config.augmentation_strength,
                "label_smoothing": config.label_smoothing,
                "weight_decay": config.weight_decay,
            }
        )

        if model_params:
            mlflow.log_params(model_params)

        # Training metrics (per epoch as step)
        for epoch, metrics in enumerate(train_metrics.get("epoch_metrics", [])):
            mlflow.log_metrics(
                {f"train_{k}": v for k, v in metrics.items()}, step=epoch
            )

        # Final val metrics
        mlflow.log_metrics(
            {
                f"val_{k}": v
                for k, v in val_metrics.items()
                if isinstance(v, (int, float))
            }
        )

        # Test metrics
        if test_metrics:
            mlflow.log_metrics(
                {
                    f"test_{k}": v
                    for k, v in test_metrics.items()
                    if isinstance(v, (int, float))
                }
            )

        # Latency
        if latency_pytorch:
            mlflow.log_metrics(
                {
                    "latency_pytorch_mean_ms": latency_pytorch["mean_ms"],
                    "latency_pytorch_p95_ms": latency_pytorch["p95_ms"],
                    "throughput_pytorch": latency_pytorch["throughput_imgs_per_sec"],
                }
            )

        if latency_onnx:
            mlflow.log_metrics(
                {
                    "latency_onnx_mean_ms": latency_onnx["mean_ms"],
                    "latency_onnx_p95_ms": latency_onnx["p95_ms"],
                    "throughput_onnx": latency_onnx["throughput_imgs_per_sec"],
                }
            )

        # Artifacts
        if artifacts_dir:
            for path in Path(artifacts_dir).glob("*.png"):
                mlflow.log_artifact(str(path))
            for path in Path(artifacts_dir).glob("*.json"):
                mlflow.log_artifact(str(path))
            for path in Path(artifacts_dir).glob("*.onnx"):
                mlflow.log_artifact(str(path))

        return run.info.run_id


def log_text_run(
    model_key: str,
    config: Any,
    val_metrics: dict,
    test_metrics: Optional[dict] = None,
    calibration_metrics: Optional[dict] = None,
    artifacts_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> str:
    """Log a text classification experiment run."""
    run_name = run_name or f"{model_key}__text"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "model_key": model_key,
                "lr": config.lr,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "max_seq_length": config.max_seq_length,
                "weight_decay": config.weight_decay,
                "warmup_ratio": config.warmup_ratio,
                "fp16": config.fp16,
            }
        )

        mlflow.log_metrics(
            {
                f"val_{k}": v
                for k, v in val_metrics.items()
                if isinstance(v, (int, float))
            }
        )

        if test_metrics:
            mlflow.log_metrics(
                {
                    f"test_{k}": v
                    for k, v in test_metrics.items()
                    if isinstance(v, (int, float))
                }
            )

        if calibration_metrics:
            mlflow.log_metrics(
                {
                    "ece_before": calibration_metrics.get("ece_before", 0),
                    "ece_after": calibration_metrics.get("ece_after", 0),
                    "temperature": calibration_metrics.get("temperature", 1.0),
                }
            )

        if artifacts_dir:
            for path in Path(artifacts_dir).glob("*.png"):
                mlflow.log_artifact(str(path))

        return run.info.run_id


def log_clip_run(
    config: Any,
    zero_shot_results: dict,
    few_shot_results: dict,
    prompt_sensitivity: dict,
    artifacts_dir: Optional[str] = None,
) -> str:
    """Log a CLIP experiment run."""
    with mlflow.start_run(run_name="clip_eurosat") as run:
        mlflow.log_params(
            {
                "model_id": config.model_id,
                "num_prompt_templates": len(
                    zero_shot_results.get("per_template_accuracy", [])
                ),
                "few_shot_k_values": str(config.__class__.__name__),
            }
        )

        # Zero-shot
        mlflow.log_metrics(
            {
                "zeroshot_best_template_acc": max(
                    zero_shot_results.get("per_template_accuracy", [0])
                ),
                "zeroshot_ensemble_acc": zero_shot_results.get("ensemble_accuracy", 0),
                "zeroshot_single_template_acc": zero_shot_results.get(
                    "single_template_accuracy", 0
                ),
            }
        )

        # Few-shot
        for k, acc in few_shot_results.items():
            mlflow.log_metric(f"fewshot_k{k}_acc", acc)

        # Prompt sensitivity
        mlflow.log_metric(
            "prompt_accuracy_std",
            float(
                np.std(
                    list(prompt_sensitivity.get("per_template_accuracy", {}).values())
                )
            ),
        )

        if artifacts_dir:
            for path in Path(artifacts_dir).glob("*.png"):
                mlflow.log_artifact(str(path))

        return run.info.run_id
