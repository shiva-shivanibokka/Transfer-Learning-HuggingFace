"""
Shared data utilities: EuroSAT loading, stratified fraction sampling,
augmentation pipelines, DataLoader construction.
"""

from __future__ import annotations

import random

from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# ── EuroSAT label map ─────────────────────────────────────────────────────────

EUROSAT_LABEL2ID = {
    "AnnualCrop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Pasture": 5,
    "PermanentCrop": 6,
    "Residential": 7,
    "River": 8,
    "SeaLake": 9,
}
EUROSAT_ID2LABEL = {v: k for k, v in EUROSAT_LABEL2ID.items()}


# ── Augmentation pipelines ────────────────────────────────────────────────────


def get_train_transform(
    image_size: int = 224, strength: str = "medium"
) -> transforms.Compose:
    """
    Training augmentation. Strength controls how aggressive the augmentation is.
    Satellite imagery benefits from rotation/flip but not colour jitter as strongly.
    """
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]

    if strength == "light":
        extra = [transforms.RandomRotation(10)]
    elif strength == "medium":
        extra = [
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]
    else:  # strong
        extra = [
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]

    return transforms.Compose(
        base
        + extra
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ── PyTorch Dataset wrapper ────────────────────────────────────────────────────


class EuroSATDataset(Dataset):
    """Wraps a HuggingFace EuroSAT split as a PyTorch Dataset."""

    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        label = item["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


# ── Data loading ──────────────────────────────────────────────────────────────


def load_eurosat(
    dataset_name: str = "blanchefort/eurosat_rgb",
    data_fraction: float = 1.0,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    augmentation_strength: str = "medium",
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load EuroSAT, optionally subsample a stratified fraction of the training set,
    and return (train_loader, val_loader, test_loader).

    Args:
        data_fraction: Fraction of training data to use (1%, 5%, 10%, 100%).
                       Validation and test sets are always full.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    ds = load_dataset(dataset_name)

    train_ds = EuroSATDataset(
        ds["train"], transform=get_train_transform(image_size, augmentation_strength)
    )
    val_ds = EuroSATDataset(ds["validation"], transform=get_val_transform(image_size))
    test_ds = EuroSATDataset(ds["test"], transform=get_val_transform(image_size))

    # Stratified subsample of training set
    if data_fraction < 1.0:
        train_ds = _stratified_subset(train_ds, data_fraction, seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def _stratified_subset(dataset: EuroSATDataset, fraction: float, seed: int) -> Subset:
    """Return a stratified subset keeping `fraction` of each class."""
    rng = random.Random(seed)
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_indices: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        class_indices.setdefault(lbl, []).append(idx)

    selected = []
    for cls_idxs in class_indices.values():
        n = max(1, int(len(cls_idxs) * fraction))
        selected.extend(rng.sample(cls_idxs, n))

    return Subset(dataset, selected)
