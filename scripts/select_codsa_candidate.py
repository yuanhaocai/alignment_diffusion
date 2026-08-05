#!/usr/bin/env python3
"""Select a CoDSA/classifier candidate on validation F1, then evaluate test.

This script does not train CoDSA or a classifier. Each manifest entry supplies
validation and test probabilities from an externally generated candidate and
may record its CoDSA ``(r, alpha, m)`` tuple and classifier settings. Only
validation probabilities are read while selecting a candidate and its F1
threshold. The selected candidate's test probabilities are loaded afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tabsynfnn.prediction_interval_utils import expected_calibration_error  # noqa: E402


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def load_binary_probability(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"probability file not found: {path}. "
            "Replace the manifest template path with a generated .npy file."
        )
    values = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if values.ndim == 2 and values.shape[1] == 2:
        values = values[:, 1]
    elif values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    elif values.ndim != 1:
        raise ValueError(
            f"{path} must have shape (n,), (n, 1), or (n, 2); got {values.shape}"
        )
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{path} contains invalid probabilities")
    return values.reshape(-1)


def binary_indices(y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    unknown = set(np.unique(y)) - set(labels)
    if unknown:
        raise ValueError(f"response contains labels absent from training: {unknown}")
    return (y == labels[1]).astype(int)


def select_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.0, 1.0, 101)
    scores = [
        f1_score(y_true, probability >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def test_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, probability, pos_label=1)
    return {
        "cross_entropy": float(log_loss(y_true, probability, labels=[0, 1])),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(auc(recall, precision)),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece": float(expected_calibration_error(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
    }


def run_selection(manifest_path: Path, data_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest.get("candidates", [])
    if not candidates:
        raise ValueError("manifest must contain at least one candidate")

    y_train = np.load(data_dir / "y_train.npy", allow_pickle=True).reshape(-1)
    labels = np.unique(y_train)
    if labels.size != 2:
        raise ValueError(f"CoDSA selection requires two response classes; got {labels}")
    y_val = binary_indices(
        np.load(data_dir / "y_val.npy", allow_pickle=True), labels
    )

    validation_records = []
    for index, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("id", f"candidate_{index}"))
        val_path = resolve_path(candidate["validation_proba"], manifest_path.parent)
        val_probability = load_binary_probability(val_path)
        if val_probability.size != y_val.size:
            raise ValueError(
                f"{candidate_id}: validation size mismatch "
                f"({val_probability.size} probabilities for {y_val.size} labels)"
            )
        threshold, validation_f1 = select_threshold(y_val, val_probability)
        validation_records.append(
            {
                "candidate_index": index,
                "id": candidate_id,
                "parameters": candidate.get("parameters", {}),
                "validation_threshold": threshold,
                "validation_f1": validation_f1,
            }
        )

    # np.argmax deterministically keeps manifest order when validation F1 ties.
    selected_index = int(
        np.argmax([record["validation_f1"] for record in validation_records])
    )
    selected = candidates[selected_index]
    selected_record = validation_records[selected_index]

    # Test data is intentionally touched only after all selection is complete.
    y_test = binary_indices(
        np.load(data_dir / "y_test.npy", allow_pickle=True), labels
    )
    test_path = resolve_path(selected["test_proba"], manifest_path.parent)
    test_probability = load_binary_probability(test_path)
    if test_probability.size != y_test.size:
        raise ValueError(
            f"{selected_record['id']}: test size mismatch "
            f"({test_probability.size} probabilities for {y_test.size} labels)"
        )

    return {
        "protocol": "candidate and threshold selected on validation F1; test final-only",
        "positive_label": labels[1].item() if hasattr(labels[1], "item") else labels[1],
        "selection_metric": "validation_f1",
        "validation_candidates": validation_records,
        "selected_candidate": selected_record,
        "test": test_metrics(
            y_test,
            test_probability,
            selected_record["validation_threshold"],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_selection(args.manifest.resolve(), args.data_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
