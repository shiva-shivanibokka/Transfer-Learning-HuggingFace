"""
CLIP zero-shot, few-shot, and retrieval pipeline for Notebook 3.

Implements:
1. Zero-shot classification with 5 prompt templates per class
2. Prompt sensitivity analysis — accuracy per template
3. Prompt ensembling — average text embeddings across all templates
4. Few-shot linear probe on CLIP features at k=1,5,10,25
5. Cross-modal retrieval — text query → top-k images
6. UMAP embedding visualization
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from transformers import CLIPModel, CLIPProcessor, set_seed

from configs.clip_config import (
    CLASS_DESCRIPTIONS,
    EUROSAT_CLASSES,
    FEW_SHOT_K_VALUES,
    NUM_CLASSES,
    PROMPT_TEMPLATES,
    CLIPConfig,
)
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


# ── Feature extraction ─────────────────────────────────────────────────────────


@torch.no_grad()
def extract_image_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: list[Image.Image],
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """Extract L2-normalised image embeddings for a list of PIL images."""
    model = model.to(device).eval()
    all_features = []

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feats = model.get_image_features(**inputs)
        feats = F.normalize(feats, dim=-1)
        all_features.append(feats.cpu().numpy())

    return np.vstack(all_features)


@torch.no_grad()
def extract_text_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: list[str],
    device: str = "cpu",
) -> np.ndarray:
    """Extract L2-normalised text embeddings."""
    model = model.to(device).eval()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_text_features(**inputs)
    feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy()


# ── Prompt building ────────────────────────────────────────────────────────────


def build_prompts(template: str, class_names: list[str]) -> list[str]:
    """Apply a template to all class names, using CLASS_DESCRIPTIONS for richer prompts."""
    prompts = []
    for cls in class_names:
        desc = CLASS_DESCRIPTIONS.get(cls, cls.lower())
        # Try replacing {cls} with description, fallback to class name
        try:
            prompts.append(template.format(cls=desc))
        except KeyError:
            prompts.append(template.replace("{cls}", desc))
    return prompts


# ── Zero-shot classification ───────────────────────────────────────────────────


def zero_shot_classify(
    image_features: np.ndarray,
    text_features_per_template: list[np.ndarray],
    labels: np.ndarray,
    ensemble: bool = True,
) -> dict:
    """
    Classify images using cosine similarity to text embeddings.

    Args:
        image_features:           (N, D) image embeddings.
        text_features_per_template: List of (C, D) text embeddings, one per template.
        labels:                   (N,) ground-truth class indices.
        ensemble:                 If True, also compute ensemble accuracy.

    Returns:
        Dict with per_template_accuracy, ensemble_accuracy, predictions.
    """
    per_template_accuracy = []
    per_template_preds = []

    for text_feats in text_features_per_template:
        # (N, C) similarity matrix
        sims = image_features @ text_feats.T
        preds = np.argmax(sims, axis=-1)
        acc = (preds == labels).mean()
        per_template_accuracy.append(float(acc))
        per_template_preds.append(preds)

    result = {
        "per_template_accuracy": per_template_accuracy,
        "best_template_accuracy": max(per_template_accuracy),
        "worst_template_accuracy": min(per_template_accuracy),
        "mean_template_accuracy": float(np.mean(per_template_accuracy)),
        "std_template_accuracy": float(np.std(per_template_accuracy)),
    }

    if ensemble:
        # Average text embeddings across templates → re-normalise
        ensemble_feats = np.stack(text_features_per_template).mean(axis=0)
        ensemble_feats = ensemble_feats / (
            np.linalg.norm(ensemble_feats, axis=-1, keepdims=True) + 1e-8
        )
        sims = image_features @ ensemble_feats.T
        ensemble_preds = np.argmax(sims, axis=-1)
        result["ensemble_accuracy"] = float((ensemble_preds == labels).mean())
        result["ensemble_predictions"] = ensemble_preds.tolist()

    return result


# ── Few-shot linear probe ──────────────────────────────────────────────────────


def few_shot_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    k_values: list[int],
    seed: int = 42,
) -> dict[int, float]:
    """
    Train a logistic regression head on k examples per class.
    Returns {k: test_accuracy}.
    """
    results = {}
    rng = np.random.RandomState(seed)

    for k in k_values:
        # Sample k examples per class
        selected_idxs = []
        for cls in range(NUM_CLASSES):
            cls_idxs = np.where(train_labels == cls)[0]
            if len(cls_idxs) >= k:
                chosen = rng.choice(cls_idxs, k, replace=False)
            else:
                chosen = cls_idxs
            selected_idxs.extend(chosen.tolist())

        X_train = train_features[selected_idxs]
        y_train = train_labels[selected_idxs]

        # L2-normalise (already done for CLIP, but be safe)
        X_train_norm = normalize(X_train)
        X_test_norm = normalize(test_features)

        clf = LogisticRegression(
            max_iter=1000,
            C=0.316,  # from CLIP paper: C=1/lambda, they use 0.316 for linear probe
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train_norm, y_train)
        preds = clf.predict(X_test_norm)
        acc = (preds == test_labels).mean()
        results[k] = float(acc)
        log.info(f"  Few-shot k={k:3d}: accuracy={acc:.4f}")

    return results


# ── Cross-modal retrieval ──────────────────────────────────────────────────────


def retrieve_images(
    query_text: str,
    image_features: np.ndarray,
    image_labels: np.ndarray,
    model: CLIPModel,
    processor: CLIPProcessor,
    top_k: int = 5,
    device: str = "cpu",
) -> list[dict]:
    """
    Retrieve the top-k most similar images to a text query.

    Returns list of {rank, label, similarity_score, image_index}.
    """
    query_feats = extract_text_features(model, processor, [query_text], device)
    # (N, D) @ (D, 1) -> (N, 1); squeeze only the singleton query axis so a
    # single-image gallery (N=1) still yields a 1-D array, not a 0-D scalar.
    sims = (image_features @ query_feats.T).squeeze(axis=1)
    top_indices = np.argsort(sims)[::-1][:top_k]

    return [
        {
            "rank": i + 1,
            "image_index": int(idx),
            "label": EUROSAT_CLASSES[image_labels[idx]],
            "similarity": float(sims[idx]),
        }
        for i, idx in enumerate(top_indices)
    ]


# ── Main pipeline ──────────────────────────────────────────────────────────────


def run_clip_pipeline(cfg: CLIPConfig) -> dict:
    """
    Full CLIP experiment pipeline:
    1. Extract features for the entire EuroSAT test set
    2. Zero-shot classification with all 5 prompt templates
    3. Prompt sensitivity analysis
    4. Ensemble zero-shot
    5. Few-shot linear probe at k=1,5,10,25
    6. Return all results for notebook plotting
    """
    from src.utils.mlflow_utils import setup_mlflow

    # Global seed for reproducibility (seeds torch, numpy, and random).
    set_seed(cfg.seed)

    setup_mlflow(cfg.mlflow_experiment, cfg.mlflow_tracking_uri)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"\nLoading CLIP model: {cfg.model_id} on {device}")

    model = CLIPModel.from_pretrained(cfg.model_id).to(device)
    processor = CLIPProcessor.from_pretrained(cfg.model_id)

    # Load EuroSAT
    log.info("Loading EuroSAT dataset...")
    ds = load_dataset("timm/eurosat-rgb")

    def get_images_and_labels(split):
        images, labels = [], []
        for item in ds[split]:
            img = item["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img).convert("RGB")
            else:
                img = img.convert("RGB")
            images.append(img)
            labels.append(item["label"])
        return images, np.array(labels)

    log.info("Extracting image features (train split for few-shot)...")
    train_images, train_labels = get_images_and_labels("train")
    train_feats = extract_image_features(
        model, processor, train_images, cfg.batch_size, device
    )

    log.info("Extracting image features (test split)...")
    test_images, test_labels = get_images_and_labels("test")
    test_feats = extract_image_features(
        model, processor, test_images, cfg.batch_size, device
    )

    # ── Text features for each template ────────────────────────────────────────
    log.info("\nBuilding text embeddings for all prompt templates...")
    text_features_per_template = []
    for template in PROMPT_TEMPLATES:
        prompts = build_prompts(template, EUROSAT_CLASSES)
        feats = extract_text_features(model, processor, prompts, device)
        text_features_per_template.append(feats)
        log.info(f"  Template: '{template[:50]}...' → features shape {feats.shape}")

    # ── Zero-shot ───────────────────────────────────────────────────────────────
    log.info("\nRunning zero-shot classification...")
    zero_shot_results = zero_shot_classify(
        test_feats, text_features_per_template, test_labels, ensemble=True
    )
    zero_shot_results["templates"] = PROMPT_TEMPLATES

    log.info(
        f"  Per-template accuracies: {[f'{a:.3f}' for a in zero_shot_results['per_template_accuracy']]}"
    )
    log.info(f"  Best template:  {zero_shot_results['best_template_accuracy']:.4f}")
    log.info(f"  Ensemble:       {zero_shot_results['ensemble_accuracy']:.4f}")
    log.info(f"  Std across templates: {zero_shot_results['std_template_accuracy']:.4f}")

    # ── Few-shot ────────────────────────────────────────────────────────────────
    log.info("\nRunning few-shot linear probe...")
    few_shot_results = few_shot_linear_probe(
        train_feats,
        train_labels,
        test_feats,
        test_labels,
        FEW_SHOT_K_VALUES,
        cfg.seed,
    )

    # ── Save results ────────────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "zero_shot": zero_shot_results,
        "few_shot": {str(k): v for k, v in few_shot_results.items()},
        "model_id": cfg.model_id,
    }

    result_path = output_dir / "clip_results.json"
    result_path.write_text(json.dumps(results, indent=2))

    # ── MLflow logging ──────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="clip_eurosat"):
        mlflow.log_params(
            {"model_id": cfg.model_id, "num_templates": len(PROMPT_TEMPLATES)}
        )
        mlflow.log_metrics(
            {
                "zeroshot_best_template": zero_shot_results["best_template_accuracy"],
                "zeroshot_ensemble": zero_shot_results["ensemble_accuracy"],
                "zeroshot_std_across_templates": zero_shot_results[
                    "std_template_accuracy"
                ],
                **{f"fewshot_k{k}": v for k, v in few_shot_results.items()},
            }
        )
        mlflow.log_artifact(str(result_path))

    # Also return the raw data needed for notebook plotting
    results["_arrays"] = {
        "test_feats": test_feats,
        "test_labels": test_labels,
        "train_feats": train_feats,
        "train_labels": train_labels,
        "test_images": test_images[:200],  # subset for UMAP / display
        "model": model,
        "processor": processor,
    }

    return results
