"""
ForensicAI — Training Script for AI-Generated Image Detector

Trains the EfficientNet-B0 based binary classifier to distinguish
real photographs from AI-generated images.

Features:
    - Two-phase training: frozen backbone → full fine-tuning
    - Cosine annealing learning rate schedule
    - Early stopping with patience
    - Model checkpointing (best validation accuracy)
    - Comprehensive metrics: accuracy, precision, recall, F1, AUC-ROC
    - CPU-optimized (no GPU required)

Usage:
    python -m training.train_ai_detector --data_dir data/dataset --epochs 20 --batch_size 16
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm
import numpy as np

from models.efficientnet_detector import ForensicNetB0
from training.dataset import create_dataloaders
from app.config import AI_MODEL_PATH, MODEL_DIR


def train(args):
    device = torch.device("cpu")
    print(f"\n{'='*60}")
    print(f"  ForensicAI — AI-Generated Image Detector Training")
    print(f"{'='*60}\n")
    print(f"  Device:        {device}")
    print(f"  Data dir:      {args.data_dir}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output model:  {AI_MODEL_PATH}")
    print(f"{'='*60}\n")

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=0,
        max_samples_per_class=args.max_samples,
    )

    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples:   {len(val_loader.dataset)}\n")

    # ── Model ───────────────────────────────────────────────────────────
    model = ForensicNetB0(num_classes=2, pretrained=True, dropout=args.dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: Frozen backbone ────────────────────────────────────────
    print("Phase 1: Training classifier head (backbone frozen)...\n")
    model.freeze_backbone()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.freeze_epochs)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.freeze_epochs):
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = _validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"  Epoch {epoch + 1:3d}/{args.freeze_epochs} │ "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} │ "
            f"Val Loss: {val_loss:.4f}  Acc: {val_metrics['accuracy']:.4f}  "
            f"F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            _save_model(model, AI_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

    # ── Phase 2: Full fine-tuning ───────────────────────────────────────
    print(f"\nPhase 2: Full fine-tuning (backbone unfrozen)...\n")
    model.unfreeze_backbone()
    optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.freeze_epochs,
    )
    patience_counter = 0

    for epoch in range(args.freeze_epochs, args.epochs):
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = _validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"  Epoch {epoch + 1:3d}/{args.epochs}  │ "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} │ "
            f"Val Loss: {val_loss:.4f}  Acc: {val_metrics['accuracy']:.4f}  "
            f"F1: {val_metrics['f1']:.4f}  AUC: {val_metrics['auc']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            _save_model(model, AI_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch + 1} (patience={args.patience})")
                break

    # ── Final report ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Model saved to: {AI_MODEL_PATH}")
    print(f"{'='*60}\n")


def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="    Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def _validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    total = len(all_labels)
    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    probs_np = np.array(all_probs)

    metrics = {
        "accuracy": accuracy_score(labels_np, preds_np),
        "precision": precision_score(labels_np, preds_np, zero_division=0),
        "recall": recall_score(labels_np, preds_np, zero_division=0),
        "f1": f1_score(labels_np, preds_np, zero_division=0),
    }

    try:
        metrics["auc"] = roc_auc_score(labels_np, probs_np)
    except ValueError:
        metrics["auc"] = 0.0

    return total_loss / total, metrics


def _save_model(model, path):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ForensicAI AI-Detection Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs")
    parser.add_argument("--freeze_epochs", type=int, default=5, help="Epochs with frozen backbone")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (use 8-16 for CPU)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per class")
    args = parser.parse_args()

    train(args)
