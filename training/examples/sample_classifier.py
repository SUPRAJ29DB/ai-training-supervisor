"""
Sample PyTorch Classifier – trains a simple feedforward network on
synthetic data. Designed to run on the RTX 3050 6GB GPU.

This script is self-contained and can be launched directly OR by the
AI Training Supervisor as a managed job.

Usage:
    python training/examples/sample_classifier.py
    python training/examples/sample_classifier.py --epochs 20 --batch-size 64

Supervisor integration:
    The script writes metrics to metrics.jsonl in its working directory.
    The supervisor reads this file to track progress.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on the path when run standalone
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sample_classifier")

METRICS_FILE = Path(".") / "metrics.jsonl"


# ── Model ─────────────────────────────────────────────────────────────────────

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Helpers ───────────────────────────────────────────────────────────────────

def emit_metric(epoch: int, **kwargs) -> None:
    """Append a metrics line for the supervisor to read."""
    record = {"epoch": epoch, **kwargs}
    with open(METRICS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    logger.info(
        "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | accuracy=%.4f",
        kwargs.get("current_epoch", epoch),
        kwargs.get("total_epochs", "?"),
        kwargs.get("train_loss", 0),
        kwargs.get("val_loss", 0),
        kwargs.get("accuracy", 0),
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info(
            "GPU: %s | VRAM: %.2f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        dev = torch.device("cpu")
        logger.warning("CUDA not found – training on CPU (this will be slow for large models).")
    return dev


# ── Data ──────────────────────────────────────────────────────────────────────

def build_dataloaders(n_samples: int, n_features: int, n_classes: int, batch_size: int):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=max(2, n_features // 2),
        random_state=42,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype("float32")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val,   dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, X_val, y_val


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sample PyTorch Classifier")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--n-samples",    type=int,   default=5000)
    parser.add_argument("--n-features",   type=int,   default=20)
    parser.add_argument("--n-classes",    type=int,   default=5)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints/sample")
    parser.add_argument("--save-every",   type=int,   default=5)
    args = parser.parse_args()

    device = get_device()
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.unlink(missing_ok=True)    # Start fresh

    train_loader, val_loader, X_val, y_val = build_dataloaders(
        args.n_samples, args.n_features, args.n_classes, args.batch_size
    )

    model     = SimpleClassifier(args.n_features, args.n_classes, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    start_time = time.time()

    logger.info("Starting training: epochs=%d, batch_size=%d, lr=%.4f", args.epochs, args.batch_size, args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        scheduler.step()

        emit_metric(
            epoch,
            current_epoch=epoch,
            total_epochs=args.epochs,
            train_loss=round(train_loss, 6),
            val_loss=round(val_loss, 6),
            accuracy=round(val_acc, 6),
            train_accuracy=round(train_acc, 6),
            lr=round(scheduler.get_last_lr()[0], 8),
        )

        # Print progress in format the log monitor can parse
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} acc={val_acc:.4f}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "optimizer_state":   optimizer.state_dict(),
                "val_loss":          val_loss,
                "val_accuracy":      val_acc,
                "config":            vars(args),
            }, ckpt_path)
            logger.info("Best checkpoint saved (val_loss=%.4f).", val_loss)

        if epoch % args.save_every == 0:
            periodic_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save({
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "val_loss":          val_loss,
                "config":            vars(args),
            }, periodic_path)

    elapsed = time.time() - start_time
    logger.info(
        "Training complete in %.1f s | best_val_loss=%.4f",
        elapsed, best_val_loss,
    )
    print(f"Training finished. Best val_loss={best_val_loss:.4f}. "
          f"Model saved to {ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
