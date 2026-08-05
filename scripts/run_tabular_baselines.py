#!/usr/bin/env python3
"""Run the paper's classical tabular baselines with validation-only selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ImportError:
    xgb = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "alignment"))

from data import load_data  # noqa: E402
from mdn_regression import (  # noqa: E402
    MixtureDensityNetworkRegressor,
    gaussian_mixture_interval,
)
from tabsynfnn.prediction_interval_utils import (  # noqa: E402
    coverage_and_width,
    crps_gaussian,
    crps_gaussian_mixture,
    expected_calibration_error,
    multiclass_brier_score,
    quantile_regression_baseline,
    split_conformal_prediction,
)
from tabsynfnn.util_functions import quadratic_weighted_kappa  # noqa: E402


def classification_metrics(y_true, proba, labels, threshold=None):
    labels = np.asarray(labels)
    if len(labels) == 2:
        positive = proba[:, 1]
        prediction = labels[(positive >= (0.5 if threshold is None else threshold)).astype(int)]
        binary_true = (y_true == labels[1]).astype(int)
        precision, recall, _ = precision_recall_curve(binary_true, positive)
        return {
            "accuracy": accuracy_score(y_true, prediction),
            "f1": f1_score(y_true, prediction, pos_label=labels[1], zero_division=0),
            "roc_auc": roc_auc_score(binary_true, positive),
            "pr_auc": auc(recall, precision),
            "cross_entropy": log_loss(y_true, proba, labels=labels),
            "brier": brier_score_loss(binary_true, positive),
            "ece": expected_calibration_error(binary_true, positive),
            "qwk": quadratic_weighted_kappa(
                np.searchsorted(labels, y_true),
                np.searchsorted(labels, prediction),
                2,
            ),
        }

    prediction_idx = np.argmax(proba, axis=1)
    prediction = labels[prediction_idx]
    true_idx = np.searchsorted(labels, y_true)
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "macro_f1": f1_score(y_true, prediction, average="macro"),
        "cross_entropy": log_loss(y_true, proba, labels=labels),
        "brier": multiclass_brier_score(y_true, proba, labels=labels),
        "ece": expected_calibration_error(true_idx, proba),
        "qwk": quadratic_weighted_kappa(
            true_idx,
            prediction_idx,
            len(labels),
        ),
    }


def select_binary_threshold(y_val, val_proba, labels):
    binary_val = (y_val == labels[1]).astype(int)
    grid = np.linspace(0.0, 1.0, 101)
    scores = [
        f1_score(binary_val, val_proba[:, 1] >= threshold, zero_division=0)
        for threshold in grid
    ]
    index = int(np.argmax(scores))
    return float(grid[index]), float(scores[index])


def run_classification(args, X_train, X_val, X_test, y_train, y_val, y_test):
    models = {
        "logistic": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "random_forest": RandomForestClassifier(
            random_state=args.seed,
            n_jobs=-1,
            n_estimators=100,
        ),
    }
    if xgb is not None:
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            n_jobs=-1,
            verbosity=1,
            random_state=args.seed,
            eval_metric="mlogloss" if args.num_classes > 2 else "logloss",
        )

    output = {}
    for name in args.models:
        if name not in models:
            if name == "xgboost" and xgb is None:
                raise ImportError("xgboost is required for the xgboost baseline")
            raise ValueError(f"{name} is not a classification baseline")
        model = models[name]
        model.fit(X_train, y_train)
        labels = np.asarray(model.classes_)
        val_proba = model.predict_proba(X_val)
        threshold = None
        record = {}
        if len(labels) == 2:
            threshold, validation_f1 = select_binary_threshold(y_val, val_proba, labels)
            record.update(
                validation_threshold=threshold,
                validation_f1=validation_f1,
            )
        # Test predictions are produced only after validation choices are fixed.
        test_proba = model.predict_proba(X_test)
        record["test"] = classification_metrics(
            y_test,
            test_proba,
            labels,
            threshold=threshold,
        )
        output[name] = record
    return output


def regression_point_metrics(y_true, prediction):
    return {
        "mse": mean_squared_error(y_true, prediction),
        "rmse": root_mean_squared_error(y_true, prediction),
        "r2": r2_score(y_true, prediction),
    }


def interval_metrics(y_true, prediction, intervals, crps):
    coverage, width = coverage_and_width(
        y_true,
        intervals[:, 0],
        intervals[:, 1],
    )
    return {
        **regression_point_metrics(y_true, prediction),
        "coverage": coverage,
        "width": width,
        "crps": crps,
    }


def run_regression(args, X_train, X_val, X_test, y_train, y_val, y_test):
    point_models = {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=args.seed,
            n_jobs=-1,
        ),
    }
    if xgb is not None:
        point_models["xgboost"] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            n_jobs=-1,
            verbosity=1,
            tree_method="auto",
            random_state=args.seed,
        )

    alpha = args.alpha
    z_value = stats.norm.ppf(1.0 - alpha / 2.0)
    output = {}
    for name in args.models:
        if name in point_models:
            model = point_models[name]
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            train_sigma = max(
                float(np.std(y_train - model.predict(X_train))),
                1e-6,
            )
            intervals = np.column_stack(
                [
                    prediction - z_value * train_sigma,
                    prediction + z_value * train_sigma,
                ]
            )
            output[name] = {
                "crps_definition": "Gaussian with training-residual scale",
                "test": interval_metrics(
                    y_test,
                    prediction,
                    intervals,
                    crps_gaussian(y_test, prediction, train_sigma),
                ),
            }
        elif name == "mdn":
            model = MixtureDensityNetworkRegressor(
                n_components=5,
                hidden_dims=(128, 64),
                max_epochs=200,
                batch_size=512,
                lr=1e-3,
                weight_decay=1e-4,
                validation_fraction=0.0,
                patience=20,
                random_state=args.seed,
                device=args.device,
            )
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            weights, mus, sigmas = model.predict_dist(X_test)
            prediction = np.sum(weights * mus, axis=1)
            intervals = gaussian_mixture_interval(weights, mus, sigmas, alpha)
            output[name] = {
                "crps_definition": "Gaussian-mixture CRPS",
                "test": interval_metrics(
                    y_test,
                    prediction,
                    intervals,
                    crps_gaussian_mixture(y_test, weights, mus, sigmas),
                ),
            }
        elif name in {"quantile", "split_conformal"}:
            if name == "quantile":
                prediction, intervals = quantile_regression_baseline(
                    X_train,
                    y_train,
                    X_test,
                    alpha=alpha,
                )
            else:
                X_fit, X_calib, y_fit, y_calib = train_test_split(
                    X_train,
                    y_train,
                    test_size=0.15,
                    random_state=args.seed,
                )
                prediction, intervals = split_conformal_prediction(
                    X_fit,
                    y_fit,
                    X_calib,
                    y_calib,
                    X_test,
                    alpha=alpha,
                )
            intervals = np.asarray(intervals)
            sigma = np.maximum(
                (intervals[:, 1] - intervals[:, 0]) / (2.0 * z_value),
                1e-6,
            )
            output[name] = {
                "crps_definition": "Gaussian approximation inferred from interval width",
                **(
                    {
                        "calibration_source": "15% random split of training data",
                        "calibration_fraction": 0.15,
                    }
                    if name == "split_conformal"
                    else {}
                ),
                "test": interval_metrics(
                    y_test,
                    prediction,
                    intervals,
                    crps_gaussian(y_test, prediction, sigma),
                ),
            }
        else:
            if name == "xgboost" and xgb is None:
                raise ImportError("xgboost is required for the xgboost baseline")
            raise ValueError(f"{name} is not a regression baseline")
    return output


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not (args.data_dir / "y_val.npy").is_file():
        raise FileNotFoundError("an explicit y_val.npy is required")
    loaded = load_data(
        data_path=str(args.data_dir),
        train_z_dat_path=None,
        predictor_to_use="original",
        task="cls" if args.task == "classification" else "reg",
        val_set=True,
    )
    X_train, X_test, X_val, y_train, y_test, y_val = loaded
    if args.task == "classification":
        if args.num_classes is None:
            parser.error("--num-classes is required for classification")
        results = run_classification(
            args,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )
    else:
        results = run_regression(
            args,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )

    payload = {
        "dataset": args.data_dir.name,
        "task": args.task,
        "selection_split": "validation",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_ready(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_ready(payload), indent=2))


if __name__ == "__main__":
    main()
