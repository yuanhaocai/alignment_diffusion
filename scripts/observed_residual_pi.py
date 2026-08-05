#!/usr/bin/env python3
"""Evaluate the fixed-M one-model observed-residual prediction interval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


def load_vector(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True).reshape(-1).astype(float)


def load_bank(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=True).astype(float)
    if value.ndim == 3:
        value = value.reshape(value.shape[0], value.shape[1])
    if value.ndim != 2:
        raise ValueError(f"expected a sample bank with shape (M, n), got {value.shape}")
    return value


def empirical_crps(
    y: np.ndarray,
    center: np.ndarray,
    residuals: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    residuals = np.sort(residuals.reshape(-1))
    n_residuals = residuals.size
    coefficients = 2 * np.arange(1, n_residuals + 1) - n_residuals - 1
    output = np.empty(y.size, dtype=float)
    for start in range(0, y.size, chunk_size):
        end = min(y.size, start + chunk_size)
        samples = center[start:end][None, :] + residuals[:, None]
        first = np.abs(samples - y[start:end][None, :]).mean(axis=0)
        pair_sum = 2.0 * np.sum(coefficients[:, None] * samples, axis=0)
        second = pair_sum / (2.0 * n_residuals * n_residuals)
        output[start:end] = first - second
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--val-samples", type=Path, required=True)
    parser.add_argument("--test-samples", type=Path, required=True)
    parser.add_argument("--m", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--standardize-metrics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    y_train = load_vector(args.data_dir / "y_train.npy")
    y_val = load_vector(args.data_dir / "y_val.npy")
    val_bank = load_bank(args.val_samples)
    if val_bank.shape[0] < args.m or val_bank.shape[1] != y_val.size:
        raise ValueError("validation sample bank does not match M or response size")
    val_center = val_bank[: args.m].mean(axis=0)
    residuals = y_val - val_center
    q_lower, q_upper = np.quantile(
        residuals,
        [args.alpha / 2.0, 1.0 - args.alpha / 2.0],
    )

    # The held-out data is loaded only after interval calibration is fixed.
    y_test = load_vector(args.data_dir / "y_test.npy")
    test_bank = load_bank(args.test_samples)
    if test_bank.shape[0] < args.m or test_bank.shape[1] != y_test.size:
        raise ValueError("test sample bank does not match M or response size")
    test_center = test_bank[: args.m].mean(axis=0)
    lower = test_center + q_lower
    upper = test_center + q_upper
    covered = (y_test >= lower) & (y_test <= upper)
    width = upper - lower
    winkler = width.copy()
    below = y_test < lower
    above = y_test > upper
    winkler[below] += (2.0 / args.alpha) * (lower[below] - y_test[below])
    winkler[above] += (2.0 / args.alpha) * (y_test[above] - upper[above])
    crps = empirical_crps(y_test, test_center, residuals, args.chunk_size)

    scale = (
        float(StandardScaler().fit(y_train[:, None]).scale_[0])
        if args.standardize_metrics
        else 1.0
    )
    result = {
        "method": "one-model fixed-M observed-residual prediction interval",
        "predictive_distribution": "test_center + empirical_validation_residuals",
        "crps_definition": "exact empirical-residual CRPS",
        "alpha": args.alpha,
        "M": args.m,
        "n_train": int(y_train.size),
        "n_val": int(y_val.size),
        "n_test": int(y_test.size),
        "metric_scale": scale,
        "q_lower_raw": float(q_lower),
        "q_upper_raw": float(q_upper),
        "coverage": float(covered.mean()),
        "coverage_se": float(np.sqrt(covered.mean() * (1.0 - covered.mean()) / y_test.size)),
        "width": float(width.mean() / scale),
        "winkler": float(winkler.mean() / scale),
        "empirical_residual_crps": float(crps.mean() / scale),
        "empirical_residual_crps_se": float(
            np.std(crps / scale, ddof=1) / np.sqrt(y_test.size)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
