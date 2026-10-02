"""
PyTorch training script template.
Extend this to implement your custom model training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

METRICS_FILE = Path(".") / "metrics.jsonl"


def emit_metric(epoch: int, **kwargs) -> None:
    record = {"epoch": epoch, **kwargs}
    with open(METRICS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # TODO: Replace with your model and data loading
    for epoch in range(1, args.epochs + 1):
        # --- training step placeholder ---
        train_loss = 1.0 / epoch           # dummy
        val_loss   = 1.1 / epoch           # dummy
        accuracy   = epoch / args.epochs   # dummy

        emit_metric(epoch, train_loss=train_loss, val_loss=val_loss, accuracy=accuracy)
        print(f"Epoch {epoch}/{args.epochs}")


if __name__ == "__main__":
    main()
