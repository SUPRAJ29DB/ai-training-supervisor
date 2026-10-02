"""
Scikit-learn training script template.
Trains a RandomForestClassifier and emits final metrics for the supervisor.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

METRICS_FILE = Path(".") / "metrics.jsonl"


def emit_metric(epoch: int, **kwargs) -> None:
    record = {"epoch": epoch, **kwargs}
    with open(METRICS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth",    type=int, default=None)
    parser.add_argument("--n-samples",    type=int, default=2000)
    parser.add_argument("--output-path",  default="artifacts/models/sklearn_model.pkl")
    args = parser.parse_args()

    X, y = make_classification(n_samples=args.n_samples, n_features=20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )
    # Sklearn is a single-epoch model; emit epoch=1 after fit
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_val, clf.predict(X_val))

    emit_metric(1, train_loss=0.0, val_loss=0.0, accuracy=acc)
    print(f"Epoch 1/1 | accuracy={acc:.4f}")

    # Save model
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(clf, fh)
    print(f"Model saved to {out}")


if __name__ == "__main__":
    main()
