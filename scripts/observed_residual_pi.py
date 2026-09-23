#!/usr/bin/env python3
"""Fixed-M signed-residual intervals with finite-sample-corrected ranks."""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
from pathlib import Path
import numpy as np


def load_vector(path):
    value = np.load(path, allow_pickle=False).reshape(-1).astype(float)
    if not value.size or not np.isfinite(value).all():
        raise ValueError(f"empty or nonfinite responses: {path}")
    return value


def load_bank(path):
    value = np.load(path, allow_pickle=False).astype(float)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2 or not value.size or not np.isfinite(value).all():
        raise ValueError(f"expected a finite sample bank with shape (M, n): {path}")
    return value


def residual_quantiles(residuals, alpha=0.05, rule="revised"):
    """Return endpoints and one-based ranks; preserve infinite sentinels."""
    r = np.asarray(residuals, dtype=float).reshape(-1)
    if not r.size or not np.isfinite(r).all():
        raise ValueError("calibration residuals must be nonempty and finite")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if rule == "linear":
        lo, hi = np.quantile(r, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi), None, None
    if rule != "revised":
        raise ValueError(f"unknown endpoint rule: {rule}")
    a = Fraction(str(alpha)) / 2
    low_rank, high_rank = (len(r) + 1) * a, (len(r) + 1) * (1 - a)
    kl = low_rank.numerator // low_rank.denominator
    ku = -(-high_rank.numerator // high_rank.denominator)
    ordered = np.sort(r)
    lo = ordered[kl - 1] if kl else -np.inf
    hi = ordered[ku - 1] if ku <= len(r) else np.inf
    return float(lo), float(hi), kl, ku


def empirical_crps(y, center, residuals, chunk_size=1000):
    """Exact CRPS of center + empirical residuals using sorted prefix sums.

    chunk_size is retained for compatibility; no n_cal*n_test bank is allocated.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    r = np.sort(np.asarray(residuals, dtype=float).reshape(-1))
    if not r.size or not np.isfinite(r).all():
        raise ValueError("residuals must be nonempty and finite")
    error = np.asarray(y, dtype=float) - np.asarray(center, dtype=float)
    n = len(r)
    prefix = np.r_[0.0, np.cumsum(r)]
    k = np.searchsorted(r, error, side="right")
    first = (error * k - prefix[k] + prefix[-1] - prefix[k] - error * (n - k)) / n
    second = np.dot(2 * np.arange(1, n + 1) - n - 1, r) / n**2
    return first - second


def evaluate(y_cal, cal_bank, y_test, test_bank, *, m=200, alpha=0.05,
             metric_scale=1.0, endpoint_rule="revised"):
    if m <= 0 or not np.isfinite(metric_scale) or metric_scale <= 0:
        raise ValueError("M and metric_scale must be positive")
    y_cal, y_test = np.asarray(y_cal, dtype=float).reshape(-1), np.asarray(y_test, dtype=float).reshape(-1)
    for label, y, bank in [("calibration", y_cal, cal_bank), ("test", y_test, test_bank)]:
        if not y.size or not np.isfinite(y).all():
            raise ValueError(f"{label} responses must be nonempty and finite")
        if bank.ndim != 2 or bank.shape[0] < m or bank.shape[1] != len(y):
            raise ValueError(f"{label} bank does not match M or response count")
        if not np.isfinite(bank[:m]).all():
            raise ValueError(f"{label} bank contains nonfinite samples")
    residuals = y_cal - cal_bank[:m].astype(float).mean(0)
    lo, hi, kl, ku = residual_quantiles(residuals, alpha, endpoint_rule)
    center = test_bank[:m].astype(float).mean(0)
    lower, upper = center + lo, center + hi
    covered = (y_test >= lower) & (y_test <= upper)
    widths = upper - lower
    winkler = widths + (2 / alpha) * (np.maximum(lower - y_test, 0) + np.maximum(y_test - upper, 0))
    crps = empirical_crps(y_test, center, residuals)
    result = {
        "method": "one-model fixed-M observed-signed-residual prediction interval",
        "endpoint_rule": endpoint_rule,
        "predictive_distribution": "test_center + empirical_calibration_residuals",
        "crps_definition": "exact empirical-residual CRPS (not raw diffusion-draw CRPS)",
        "alpha": alpha, "M": m, "n_cal": len(y_cal), "n_test": len(y_test),
        "metric_scale": metric_scale, "k_lower": kl, "k_upper": ku,
        "q_lower_raw": lo, "q_upper_raw": hi,
        "coverage": float(covered.mean()), "n_covered": int(covered.sum()),
        "coverage_se": float(np.sqrt(covered.mean() * (1 - covered.mean()) / len(y_test))),
        "width": float(widths.mean() / metric_scale), "raw_width": float(widths.mean()),
        "winkler": float(winkler.mean() / metric_scale),
        "crps": float(crps.mean() / metric_scale), "raw_crps": float(crps.mean()),
        "empirical_residual_crps": float(crps.mean() / metric_scale),
        "empirical_residual_crps_se": float(crps.std(ddof=1) / metric_scale / np.sqrt(len(y_test))) if len(y_test) > 1 else None,
        "center_rmse": float(np.sqrt(np.mean((y_test - center)**2)) / metric_scale),
        "lower_tail_miss": float((y_test < lower).mean()),
        "upper_tail_miss": float((y_test > upper).mean()),
    }
    observations = dict(calibration_residuals=residuals, y_test=y_test, test_center=center,
                        revised_intervals=np.column_stack([lower, upper]), crps_raw=crps,
                        crps_paper_scale=crps / metric_scale, coverage=covered)
    return result, observations


def json_safe(value):
    """Encode unbounded endpoints as strings rather than invalid JSON numbers."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cal-samples", "--val-samples", dest="cal_samples", type=Path, required=True)
    parser.add_argument("--calibration-split", choices=["cal", "val"], default="cal",
                        help="val is only for explicit historical sensitivity comparisons")
    parser.add_argument("--test-samples", type=Path, required=True)
    parser.add_argument("--m", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--endpoint-rule", choices=["revised", "linear"], default="revised")
    parser.add_argument("--chunk-size", type=int, default=1000, help=argparse.SUPPRESS)
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument("--metric-scale", type=float, default=None, help="fixed reporting SD")
    scale.add_argument("--standardize-metrics", action="store_true", help="use current training SD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations-output", type=Path)
    args = parser.parse_args()
    metric_scale = args.metric_scale if args.metric_scale is not None else 1.0
    if args.standardize_metrics:
        metric_scale = float(load_vector(args.data_dir / "y_train.npy").std()) or 1.0
    result, observations = evaluate(
        load_vector(args.data_dir / f"y_{args.calibration_split}.npy"), load_bank(args.cal_samples),
        load_vector(args.data_dir / "y_test.npy"), load_bank(args.test_samples),
        m=args.m, alpha=args.alpha, metric_scale=metric_scale, endpoint_rule=args.endpoint_rule,
    )
    result["calibration_split"] = args.calibration_split
    result["independence_note"] = "The evaluator cannot certify independence; use the four-split fitting workflow."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n"
    args.output.write_text(text)
    if args.observations_output:
        args.observations_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.observations_output, **observations)
    print(text)


if __name__ == "__main__":
    main()
