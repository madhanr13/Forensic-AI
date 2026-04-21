"""
ForensicAI — Training Dataset Loader

Supports two modes:
    1. Custom folder structure:  data/train/real/  +  data/train/ai/
    2. CIFAKE-style:  wherever the user points with --data_dir

Applies forensic-aware augmentations to improve robustness.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np

from app.config import AI_INPUT_SIZE


class ForensicImageDataset(Dataset):
    """
    Binary dataset: label 0 = Real, label 1 = AI-Generated.

    Expected folder structure:
        root/
        ├── real/   (or 0/)
        │   ├── img001.jpg
        │   └── ...
        └── ai/     (or 1/ or fake/)
            ├── img001.jpg
            └── ...
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        max_samples_per_class: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        # Map folder names to labels
        label_map = {
            "real": 0, "0": 0, "authentic": 0, "original": 0,
            "ai": 1, "1": 1, "fake": 1, "generated": 1, "synthetic": 1,
        }

        for subdir in sorted(self.root_dir.iterdir()):
            if not subdir.is_dir():
                continue
            label = label_map.get(subdir.name.lower())
            if label is None:
                continue

            files = [
                f for f in sorted(subdir.iterdir())
                if f.suffix.lower() in self.VALID_EXTENSIONS
            ]

            if max_samples_per_class:
                files = files[:max_samples_per_class]

            for fp in files:
                self.samples.append((str(fp), label))

        if not self.samples:
            raise RuntimeError(
                f"No images found in {root_dir}. "
                "Expected subfolders: real/ and ai/ (or fake/)"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> dict:
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        return {"real": counts[0], "ai_generated": counts[1]}


def get_train_transforms() -> transforms.Compose:
    """Training transforms with forensic-aware augmentation."""
    return transforms.Compose([
        transforms.Resize((AI_INPUT_SIZE + 32, AI_INPUT_SIZE + 32)),
        transforms.RandomCrop(AI_INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        # Simulate JPEG compression artifacts
        transforms.RandomApply([
            transforms.Lambda(lambda img: _jpeg_compress(img, quality=np.random.randint(60, 95)))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),
    ])


def get_val_transforms() -> transforms.Compose:
    """Validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((AI_INPUT_SIZE, AI_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _jpeg_compress(pil_img: Image.Image, quality: int = 75) -> Image.Image:
    """Simulate JPEG compression artifacts."""
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0,
    max_samples_per_class: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    full_dataset = ForensicImageDataset(
        data_dir,
        transform=None,  # Applied individually below
        max_samples_per_class=max_samples_per_class,
    )

    # Split
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Wrap with transforms
    train_ds = TransformWrapper(train_subset, get_train_transforms())
    val_ds = TransformWrapper(val_subset, get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader


class TransformWrapper(Dataset):
    """Wraps a subset to apply a specific transform."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
