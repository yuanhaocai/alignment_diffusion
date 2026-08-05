#!/usr/bin/env python3
"""Fit a training-only response scaler and transform train/validation/test."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


def as_2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return values.reshape(-1, 1) if values.ndim == 1 else values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = as_2d(np.load(args.data_dir / "y_train.npy", allow_pickle=False))
    scaler = StandardScaler().fit(train)
    for split in ("train", "val", "test"):
        path = args.data_dir / f"y_{split}.npy"
        if not path.is_file():
            continue
        transformed = scaler.transform(as_2d(np.load(path, allow_pickle=False)))
        np.save(args.output_dir / f"y_{split}_scaled.npy", transformed)
        print(f"y_{split}_scaled.npy: {transformed.shape}")
    joblib.dump(scaler, args.output_dir / "y_scaler.joblib")


if __name__ == "__main__":
    main()
