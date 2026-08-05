"""Small utilities shared by the released experiment entry points."""

from __future__ import annotations

import ast
from pathlib import Path

# Portions adapted from amazon-science/TabSyn (Apache-2.0) and modified.
import numpy as np


def read_written_hypars(hypar_path):
    """Read the ``name: value`` files written by the training code."""
    values = {}
    with Path(hypar_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if ": " not in line:
                continue
            name, raw_value = line.split(": ", 1)
            try:
                values[name] = ast.literal_eval(raw_value)
            except (SyntaxError, ValueError):
                values[name] = raw_value
    return values


def quadratic_weighted_kappa(y_true, y_pred, num_ratings=None):
    """Compute quadratic weighted kappa for labels ``0, ..., K-1``."""
    if num_ratings is None or num_ratings < 2:
        raise ValueError("num_ratings must be an integer of at least 2")

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if np.any((y_true < 0) | (y_true >= num_ratings)):
        raise ValueError("y_true contains a label outside 0, ..., num_ratings-1")
    if np.any((y_pred < 0) | (y_pred >= num_ratings)):
        raise ValueError("y_pred contains a label outside 0, ..., num_ratings-1")

    observed = np.zeros((num_ratings, num_ratings), dtype=np.float64)
    np.add.at(observed, (y_true, y_pred), 1.0)
    true_counts = np.bincount(y_true, minlength=num_ratings)
    pred_counts = np.bincount(y_pred, minlength=num_ratings)
    expected = np.outer(true_counts, pred_counts) / y_true.size

    indices = np.arange(num_ratings, dtype=np.float64)
    weights = ((indices[:, None] - indices[None, :]) / (num_ratings - 1)) ** 2
    expected_disagreement = float(np.sum(weights * expected))
    if expected_disagreement == 0.0:
        return 0.0
    observed_disagreement = float(np.sum(weights * observed))
    return 1.0 - observed_disagreement / expected_disagreement
