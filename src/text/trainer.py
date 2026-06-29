"""
Training pipeline for Notebook 2: text classification with calibration.
Trains RoBERTa and ModernBERT on dair-ai/emotion, then applies
temperature scaling and measures ECE before/after.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from configs.text_config import (
    EMOTION_CLASSES,
    NUM_CLASSES,
    TEXT_MODELS,
    TextTrainingConfig,
)
from src.utils.metrics import (
    TemperatureScaler,
    compute_classification_metrics,
    compute_ece,
)
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


# ── Tokenisation ───────────────────────────────────────────────────────────────


def tokenise_dataset(dataset, tokenizer, max_length: int):
    def tokenise(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(tokenise, batched=True, remove_columns=["text"])


# ── Compute metrics ────────────────────────────────────────────────────────────


def make_compute_metrics():
    import evaluate

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)[
                "accuracy"
            ],
            "f1_macro": f1.compute(
                predictions=preds, references=labels, average="macro"
            )["f1"],
        }

    return compute_metrics


# ── Main training function ─────────────────────────────────────────────────────


def train_text_model(cfg: TextTrainingConfig) -> dict:
    """
    Fine-tune a text classifier. Runs temperature calibration after training.
    Logs all metrics to MLflow.
    """
    from src.utils.mlflow_utils import setup_mlflow

    setup_mlflow(cfg.mlflow_experiment, cfg.mlflow_tracking_uri)

    hf_id = TEXT_MODELS[cfg.model_key]["hf_id"]
    id2label = {i: c for i, c in enumerate(EMOTION_CLASSES)}
    label2id = {c: i for i, c in enumerate(EMOTION_CLASSES)}

    log.info(f"\n{'=' * 60}")
    log.info(f"Training: {cfg.model_key} ({hf_id})")
    log.info(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Load and tokenise dataset
    raw = load_dataset("dair-ai/emotion")
    tokenised = {
        split: tokenise_dataset(raw[split], tokenizer, cfg.max_seq_length)
        for split in ["train", "validation", "test"]
    }
    # Rename "label" to "labels" for Trainer
    for split in tokenised:
        tokenised[split] = tokenised[split].rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer)
    output_dir = Path(cfg.output_dir) / cfg.model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        label_smoothing_factor=cfg.label_smoothing,
        report_to="none",
        seed=cfg.seed,
    )

    with mlflow.start_run(run_name=cfg.model_key):
        mlflow.log_params(
            {
                "model_key": cfg.model_key,
                "hf_model_id": hf_id,
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "num_epochs": cfg.num_epochs,
                "max_seq_length": cfg.max_seq_length,
            }
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenised["train"],
            eval_dataset=tokenised["validation"],
            data_collator=collator,
            compute_metrics=make_compute_metrics(),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()

        # Test evaluation
        test_preds = trainer.predict(tokenised["test"])
        logits = test_preds.predictions
        labels = test_preds.label_ids
        preds = np.argmax(logits, axis=-1)

        cls_metrics = compute_classification_metrics(labels, preds, EMOTION_CLASSES)
        mlflow.log_metrics(
            {
                "test_accuracy": cls_metrics["accuracy"],
                "test_f1_macro": cls_metrics["f1_macro"],
            }
        )

        # ── Calibration ────────────────────────────────────────────────────────
        device = "cuda" if torch.cuda.is_available() else "cpu"
        confidences = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
        max_conf = confidences.max(dim=-1).values.numpy()

        ece_before, bin_data_before = compute_ece(
            max_conf, preds, labels, cfg.calibration_bins
        )
        mlflow.log_metric("ece_before_calibration", ece_before)
        log.info(f"  ECE before calibration: {ece_before:.4f}")

        # Fit temperature scaler on validation set
        # Build a simple DataLoader for calibration
        from torch.utils.data import DataLoader, TensorDataset

        val_logits = trainer.predict(tokenised["validation"]).predictions
        val_labels = tokenised["validation"]["labels"]
        calib_ds = TensorDataset(
            torch.tensor(val_logits, dtype=torch.float32),
            torch.tensor(val_labels),
        )

        # Wrap model for temperature scaling
        scaler = TemperatureScaler(model.cpu())
        # Fit using LBFGS on val logits directly (faster than full forward pass)
        optimizer = torch.optim.LBFGS(
            [scaler.temperature], lr=cfg.temperature_lr, max_iter=100
        )
        val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
        val_labels_t = torch.tensor(val_labels)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled = val_logits_t / scaler.temperature.clamp(min=0.05)
            loss = criterion(scaled, val_labels_t)
            loss.backward()
            return loss

        for _ in range(cfg.temperature_epochs):
            optimizer.step(closure)

        T = max(float(scaler.T), 0.05)  # guard: avoid div-by-~0 at serve time
        mlflow.log_metric("temperature", T)
        log.info(f"  Optimal temperature T={T:.4f}")

        # ECE after calibration
        scaled_logits = torch.tensor(logits) / T
        scaled_conf = torch.softmax(scaled_logits, dim=-1).max(dim=-1).values.numpy()
        ece_after, bin_data_after = compute_ece(
            scaled_conf, preds, labels, cfg.calibration_bins
        )
        mlflow.log_metric("ece_after_calibration", ece_after)
        log.info(f"  ECE after calibration:  {ece_after:.4f}")

        result = {
            "model_key": cfg.model_key,
            "test_accuracy": cls_metrics["accuracy"],
            "test_f1_macro": cls_metrics["f1_macro"],
            "f1_per_class": cls_metrics["f1_per_class"],
            "confusion_matrix": cls_metrics["confusion_matrix"],
            "ece_before": ece_before,
            "ece_after": ece_after,
            "temperature": T,
            "bin_data_before": bin_data_before,
            "bin_data_after": bin_data_after,
        }

        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2))
        mlflow.log_artifact(str(result_path))

        # Persist the deployable checkpoint + calibration temperature where the
        # app (via src.utils.paths) looks for them. model is already CPU here.
        from src.utils.paths import text_checkpoint_path, text_temperature_path

        ckpt_path = text_checkpoint_path(cfg.model_key)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

        temp_path = text_temperature_path(cfg.model_key)
        temp_path.write_text(json.dumps({"temperature": T}, indent=2))
        log.info(f"  Saved checkpoint -> {ckpt_path}; temperature={T:.4f}")

        # Hub push
        if cfg.push_to_hub and cfg.hub_model_id:
            trainer.push_to_hub(
                commit_message=f"Fine-tuned {cfg.model_key} on dair-ai/emotion"
            )

    return result
