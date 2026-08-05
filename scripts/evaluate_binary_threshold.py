#!/usr/bin/env python3
"""Evaluate diffusion ablations with an F1 threshold selected on validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

from tabsynfnn.prediction_interval_utils import expected_calibration_error
from tabsynfnn.util_functions import quadratic_weighted_kappa


def f1_at(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> float:
    pred = (proba >= threshold).astype(int)
    return 0.0 if pred.sum() == 0 else float(f1_score(y_true, pred))


def load_proba(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=True), dtype=float).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--val-proba", required=True)
    parser.add_argument("--test-proba", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    y_val = np.load(data_dir / "y_val.npy", allow_pickle=True).reshape(-1).astype(int)
    p_val = load_proba(Path(args.val_proba))

    if len(y_val) != len(p_val):
        raise ValueError(f"val size mismatch: y={len(y_val)} proba={len(p_val)}")

    grid = np.linspace(0.0, 1.0, 101)
    val_f1 = [f1_at(y_val, p_val, threshold) for threshold in grid]
    best_idx = int(np.argmax(val_f1))
    val_thr = float(grid[best_idx])

    # Load the held-out split only after threshold selection is complete.
    y_test = np.load(data_dir / "y_test.npy", allow_pickle=True).reshape(-1).astype(int)
    p_test = load_proba(Path(args.test_proba))
    if len(y_test) != len(p_test):
        raise ValueError(f"test size mismatch: y={len(y_test)} proba={len(p_test)}")
    test_pred = (p_test >= val_thr).astype(int)
    pred_05 = (p_test >= 0.5).astype(int)
    precision, recall, _ = precision_recall_curve(y_test, p_test, pos_label=1)

    result = {
        "model": args.model_name,
        "val_thr": val_thr,
        "val_bestf1": float(val_f1[best_idx]),
        "test_f1_at_val": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_f1_at_0.5": float(f1_score(y_test, pred_05, zero_division=0)),
        "test_acc_at_val": float(accuracy_score(y_test, test_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, p_test)),
        "test_pr_auc": float(auc(recall, precision)),
        "test_brier": float(brier_score_loss(y_test, p_test)),
        "test_log_loss": float(log_loss(y_test, p_test, labels=[0, 1])),
        "test_ece": float(expected_calibration_error(y_test, p_test)),
        "test_qwk_at_val": float(quadratic_weighted_kappa(y_test, test_pred, 2)),
        "test_cm_at_val": confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist(),
        "proba_mean": float(np.mean(p_test)),
        "proba_p99": float(np.quantile(p_test, 0.99)),
        "val_proba": str(Path(args.val_proba)),
        "test_proba": str(Path(args.test_proba)),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
