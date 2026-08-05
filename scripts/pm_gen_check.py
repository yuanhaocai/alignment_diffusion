"""Evaluate predictive-modeling diffusion samples."""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, brier_score_loss, classification_report, confusion_matrix,
    f1_score, log_loss, mean_squared_error, precision_recall_curve, r2_score,
    roc_auc_score, root_mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

from tabsynfnn.prediction_interval_utils import (
    expected_calibration_error, multiclass_brier_score,
    infer_class_labels, _as_1d_labels, labels_to_indices
)
from tabsynfnn.util_functions import quadratic_weighted_kappa
from tabsynfnn.utils_tuning import metric_dict_write
from tabsynfnn.tabsyn.latent_utils import compute_proba


output_name = "pm_results.txt"
json_output_name = "pm_results.json"


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "y", "1"}:
        return True
    if value in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_repeated_y_samples(sampling_path, num_repeats):
    all_samples = []
    for sample_index in range(num_repeats):
        file_path = os.path.join(sampling_path, f"y_test_{sample_index}.npy")
        syn_y = np.load(file_path, allow_pickle=True)  # [test_size, 1]
        all_samples.append(syn_y)
    return np.stack(all_samples, axis=0)


def resolve_y_num_classes(value):
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        return None
    return value


def load_target(data_path, split):
    return np.load(os.path.join(data_path, f"y_{split}.npy"), allow_pickle=True)


def as_column(value):
    return np.asarray(value).reshape(-1, 1)


def evaluate_generation(data_path, sampling_path, pm_task, y_num_classes=None, use_val=False, num_repeats=None,
                        use_soft_proba=False):
    y_train = load_target(data_path, "train")
    eval_split = "val" if use_val else "test"
    y_eval = _as_1d_labels(load_target(data_path, eval_split))

    pm_results_dict = {}
    if pm_task == "cls":
        y_num_classes = resolve_y_num_classes(y_num_classes)
        class_labels = infer_class_labels(y_eval, y_num_classes)
        y_gen_eval = np.load(
            os.path.join(sampling_path, "y_test_majority_vote.npy"),
            allow_pickle=True,
        )
        y_gen_eval = _as_1d_labels(y_gen_eval)

        # When use_soft_proba, evaluate with the soft-averaged decoded probabilities
        # (y_proba_soft.npy) instead of the hard-vote fractions (y_proba.npy). The
        # hard-vote majority labels (y_gen_eval) are still used for accuracy/confusion.
        proba_filename = "y_proba_soft.npy" if use_soft_proba else "y_proba.npy"
        try:
            y_eval_proba = np.load(
                os.path.join(sampling_path, proba_filename),
                allow_pickle=True,
            )
            if use_soft_proba:
                print(f"Using soft-averaged probabilities: {proba_filename}")
        except FileNotFoundError:
            if use_soft_proba:
                raise FileNotFoundError(
                    f"{proba_filename} not found in {sampling_path}. "
                    "Run sampling with the soft-probability option to create it."
                )
            try:
                print("Using y_test_all.npy")
                all_samples_arr = np.load(
                    os.path.join(sampling_path, "y_test_all.npy"),
                    allow_pickle=True,
                )
            except FileNotFoundError:
                if num_repeats is None:
                    raise
                print("Loading repeated y_test_{i}.npy files")
                all_samples_arr = load_repeated_y_samples(sampling_path, num_repeats)

            y_eval_proba = compute_proba(
                all_samples_arr,
                class_labels=class_labels,
            )

        if len(class_labels) == 2:
            y_eval_proba = np.asarray(y_eval_proba, dtype=float).reshape(-1)
            positive_label = class_labels[1]
            y_eval_binary = (y_eval == positive_label).astype(int)

            pm_results_dict["roc_auc"] = roc_auc_score(
                y_eval_binary,
                y_eval_proba,
            )
            pm_results_dict["f1"] = f1_score(
                y_eval,
                y_gen_eval,
                average="binary",
                pos_label=positive_label,
            )
            pm_results_dict["log_loss"] = log_loss(
                y_eval,
                y_eval_proba,
                labels=class_labels,
            )
            precision, recall, _ = precision_recall_curve(
                y_eval,
                y_eval_proba,
                pos_label=positive_label,
            )
            pm_results_dict["pr_auc"] = auc(recall, precision)
            pm_results_dict["brier"] = brier_score_loss(
                y_eval_binary,
                y_eval_proba,
            )
            pm_results_dict["ece"] = expected_calibration_error(y_eval_binary, y_eval_proba)

            pm_results_dict["f1_at_threshold_0.5"] = f1_score(
                y_eval_binary,
                (y_eval_proba >= 0.5).astype(int),
                zero_division=0,
            )
            pm_results_dict["pred_positive_rate_at_0.5"] = float((y_eval_proba >= 0.5).mean())
            pm_results_dict["proba_mean"] = float(np.mean(y_eval_proba))
            pm_results_dict["proba_p95"] = float(np.quantile(y_eval_proba, 0.95))
            pm_results_dict["proba_p99"] = float(np.quantile(y_eval_proba, 0.99))
        else:
            y_eval_indices = labels_to_indices(y_eval, class_labels)

            pm_results_dict["f1_macro"] = f1_score(y_eval, y_gen_eval, average="macro")
            pm_results_dict["log_loss"] = log_loss(
                y_eval,
                y_eval_proba,
                labels=class_labels,
            )
            pm_results_dict["brier"] = multiclass_brier_score(
                y_eval,
                y_eval_proba,
                labels=class_labels,
            )
            pm_results_dict["ece"] = expected_calibration_error(y_eval_indices, y_eval_proba)

        pm_results_dict["accuracy"] = accuracy_score(y_eval, y_gen_eval)
        pm_results_dict["confusion_matrix"] = confusion_matrix(
            y_eval,
            y_gen_eval,
            labels=class_labels,
        )
        pm_results_dict["classification_report"] = classification_report(
            y_eval,
            y_gen_eval,
            labels=class_labels,
        )
        y_eval_qwk = labels_to_indices(y_eval, class_labels)
        y_gen_eval_qwk = labels_to_indices(y_gen_eval, class_labels)
        pm_results_dict["quadratic_weighted_kappa"] = quadratic_weighted_kappa(
            y_eval_qwk,
            y_gen_eval_qwk,
            len(class_labels),
        )
    elif pm_task == "reg":
        y_gen_eval = np.load(os.path.join(sampling_path, "y_test_average.npy"), allow_pickle=True)
        scaler = StandardScaler()
        _ = scaler.fit_transform(as_column(y_train))
        y_eval = scaler.transform(as_column(y_eval))
        y_gen_eval = scaler.transform(as_column(y_gen_eval))
        pm_results_dict["mse"] = mean_squared_error(y_eval, y_gen_eval)
        pm_results_dict["rmse"] = root_mean_squared_error(y_eval, y_gen_eval)
        pm_results_dict["r2"] = r2_score(y_eval, y_gen_eval)
    else:
        raise NotImplementedError(pm_task)

    return pm_results_dict


def write_results(sampling_path, results, metadata, output_name="pm_results.txt", json_output_name="pm_results.json"):
    os.makedirs(sampling_path, exist_ok=True)
    for key, value in results.items():
        print(key, "\n", value, "\n", sep="")

    if output_name:
        with open(os.path.join(sampling_path, output_name), "w") as file:
            metric_dict_write(file, results, None, True)

    if json_output_name:
        with open(os.path.join(sampling_path, json_output_name), "w") as file:
            json.dump(
                json_ready(
                    {
                        **metadata,
                        "metrics": results,
                    }
                ),
                file,
                indent=2,
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate predictive-modeling diffusion samples.")
    parser.add_argument("--dataname", type=str, default=None)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--sampling-paths", type=str, nargs="+", required=True)
    parser.add_argument("--pm-task", choices=["reg", "cls"], required=True)
    parser.add_argument("--y-num-classes", type=int, default=None)
    parser.add_argument("--num-repeats", type=int, default=None)
    parser.add_argument("--use-val", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--use-soft-proba", type=str2bool, nargs="?", const=True, default=False,
                        help="Evaluate with soft-averaged decoded probabilities (y_proba_soft.npy) "
                             "instead of hard-vote fractions (y_proba.npy).")
    parser.add_argument("--output-name", type=str, default=output_name)
    parser.add_argument("--json-output-name", type=str, default=json_output_name)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    resolved_dataname = args.dataname or os.path.basename(os.path.normpath(args.data_path))
    for sampling_path in args.sampling_paths:
        print("dataname:", resolved_dataname)
        print("sampling_path:", sampling_path)

        results = evaluate_generation(
            data_path=args.data_path,
            sampling_path=sampling_path,
            pm_task=args.pm_task,
            y_num_classes=args.y_num_classes,
            use_val=args.use_val,
            num_repeats=args.num_repeats,
            use_soft_proba=args.use_soft_proba,
        )
        write_results(
            sampling_path=sampling_path,
            results=results,
            metadata={
                "dataname": resolved_dataname,
                "sampling_path": sampling_path,
                "data_path": args.data_path,
                "pm_task": args.pm_task,
                "use_val": args.use_val,
            },
            output_name=args.output_name,
            json_output_name=args.json_output_name,
        )


if __name__ == "__main__":
    main()
