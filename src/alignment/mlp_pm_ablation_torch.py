"""
PyTorch MLP ablation for predictive modeling on aligned tabular/text embeddings.

The command-line interface takes processed data, aligned embeddings, and an
output directory. As described in the paper, 15% of training is reserved for
early stopping. The explicit dataset validation split is used for binary
threshold selection, and test is evaluated only after those choices are fixed.
"""

from __future__ import annotations

import copy
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Iterable, Literal

_TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(_TMPDIR / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMPDIR / "xdg-cache"))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from tabsynfnn.prediction_interval_utils import (
    _as_1d_labels,
    expected_calibration_error,
    infer_class_labels,
    multiclass_brier_score,
)
from tabsynfnn.tabsyn.data_pm import PredictiveModelingDataset
from tabsynfnn.util_functions import quadratic_weighted_kappa
from tabsynfnn.utils_tuning import metric_dict_write

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    def root_mean_squared_error(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# =============================================================================
# Experiment settings
# =============================================================================
dataname = ""
task: Literal["cls", "reg"] = "cls"
y_levels = None

alignment_note = ""
alignment_subdir = ""

use_tab = True
use_text = True
text_pooling: Literal["mean", "first", "flatten"] = "mean"

random_state = 0
device_name = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 4

# MLP hyperparameters mirror mlp_pm_ablation0.py.
hidden_layer_sizes = (512, 256)
activation = "relu"
alpha = 1e-4
batch_size = 256
learning_rate_init = 1e-3
max_iter = 300
early_stopping = True
early_stopping_fraction = 0.15
n_iter_no_change = 20
tol = 1e-4
verbose = True

overwrite_existing_results = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_key(label) -> str:
    return str(label)


def labels_to_indices(y, class_labels: Iterable) -> np.ndarray:
    label_to_index = {label_key(class_label): index for index, class_label in enumerate(class_labels)}
    try:
        return np.asarray([label_to_index[label_key(value)] for value in _as_1d_labels(y)])
    except KeyError as exc:
        raise ValueError(f"Found label not present in class_labels: {exc.args[0]}") from exc


def load_raw_y(data_path: Path):
    y_train = _as_1d_labels(np.load(data_path / "y_train.npy", allow_pickle=True))
    y_val = _as_1d_labels(np.load(data_path / "y_val.npy", allow_pickle=True))
    y_test = _as_1d_labels(np.load(data_path / "y_test.npy", allow_pickle=True))
    return y_train, y_val, y_test


def prepare_dataset_y_files(
    save_path: Path,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_labels,
):
    if save_path.exists() and not overwrite_existing_results:
        raise FileExistsError(f"Result directory already exists: {save_path}")

    y_path = save_path / "dataset_y"
    y_path.mkdir(parents=True, exist_ok=True)

    if task == "cls":
        np.save(y_path / "train_z_y.npy", labels_to_indices(y_train, class_labels).reshape(-1, 1))
        np.save(y_path / "val_z_y.npy", labels_to_indices(y_val, class_labels).reshape(-1, 1))
        np.save(y_path / "test_z_y.npy", labels_to_indices(y_test, class_labels).reshape(-1, 1))
        y_type = "categorical"
    elif task == "reg":
        np.save(y_path / "y_train_scaled.npy", np.asarray(y_train, dtype=np.float32).reshape(-1))
        np.save(y_path / "y_val_scaled.npy", np.asarray(y_val, dtype=np.float32).reshape(-1))
        np.save(y_path / "y_test_scaled.npy", np.asarray(y_test, dtype=np.float32).reshape(-1))
        y_type = "continuous"
    else:
        raise ValueError("task must be either 'cls' or 'reg'.")

    return y_path, y_type


def make_dataset(y_path: Path, alignment_path: Path, y_type: str, split: str):
    return PredictiveModelingDataset(
        y_path,
        alignment_path,
        y_type,
        split,
        use_tab=use_tab,
        use_text=use_text,
        use_image=False,
    )


def count_indexed_npy_files(path: Path) -> int:
    indices = []
    for file_path in path.glob("*.npy"):
        try:
            indices.append(int(file_path.stem))
        except ValueError:
            pass
    if not indices:
        raise FileNotFoundError(f"No indexed .npy files found in {path}")
    expected = max(indices) + 1
    if len(indices) != expected:
        raise ValueError(f"Found {len(indices)} files in {path}, expected contiguous 0..{expected - 1}")
    return expected


def validate_split_sizes(dataset, alignment_path: Path, y_len: int, split: str) -> None:
    if len(dataset) != y_len:
        raise ValueError(f"{split} label size mismatch: dataset={len(dataset)}, raw_y={y_len}")
    if use_tab:
        tab_len = np.load(alignment_path / f"{split}_z_dat_proj.npy", mmap_mode="r").shape[0]
        if tab_len != y_len:
            raise ValueError(f"{split} tab size mismatch: tab={tab_len}, y={y_len}")
    if use_text:
        text_len = count_indexed_npy_files(alignment_path / "text_proj" / split)
        if text_len != y_len:
            raise ValueError(f"{split} text size mismatch: text={text_len}, y={y_len}")


def activation_layer(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "logistic":
        return nn.Sigmoid()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = d_in
        for hidden_dim in hidden_layer_sizes:
            layers.extend([nn.Linear(prev_dim, hidden_dim), activation_layer(activation)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def batch_features(tab_emb, text_emb, device: torch.device) -> torch.Tensor:
    parts = []
    if use_tab:
        tab_emb = tab_emb.float().to(device, non_blocking=True)
        parts.append(F.normalize(tab_emb, p=2, dim=-1, eps=1e-8))
    if use_text:
        text_emb = text_emb.float().to(device, non_blocking=True)
        if text_pooling == "mean":
            if text_emb.ndim == 3:
                token_mask = text_emb.norm(dim=-1).gt(0).to(text_emb.dtype)
                denominator = token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                text_emb = (
                    text_emb * token_mask.unsqueeze(-1)
                ).sum(dim=1) / denominator
        elif text_pooling == "first":
            text_emb = text_emb[:, 0, :] if text_emb.ndim == 3 else text_emb
        elif text_pooling == "flatten":
            text_emb = text_emb.reshape(text_emb.shape[0], -1)
        else:
            raise ValueError(f"Unsupported text_pooling: {text_pooling}")
        parts.append(F.normalize(text_emb, p=2, dim=-1, eps=1e-8))
    return torch.cat(parts, dim=1)


def infer_input_dim(dataset) -> int:
    _, tab_emb, text_emb, _ = dataset[0]
    dim = 0
    if use_tab:
        dim += int(np.asarray(tab_emb).reshape(-1).shape[0])
    if use_text:
        text_arr = np.asarray(text_emb).squeeze()
        if text_pooling in {"mean", "first"} and text_arr.ndim > 1:
            dim += int(text_arr.shape[-1])
        else:
            dim += int(text_arr.reshape(-1).shape[0])
    return dim


def make_loaders(train_dataset, selection_val_dataset, test_dataset):
    pin_memory = device_name.startswith("cuda")
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    if early_stopping:
        early_stop_size = max(
            1, int(round(early_stopping_fraction * len(train_dataset)))
        )
        fit_size = len(train_dataset) - early_stop_size
        if fit_size < 1:
            raise ValueError("training data is too small for a 15% early-stopping split")
        fit_dataset, early_stop_dataset = random_split(
            train_dataset,
            [fit_size, early_stop_size],
            generator=torch.Generator().manual_seed(random_state),
        )
        early_stop_loader = DataLoader(
            early_stop_dataset, shuffle=False, **loader_kwargs
        )
    else:
        fit_dataset = train_dataset
        early_stop_loader = None
    train_loader = DataLoader(fit_dataset, shuffle=True, **loader_kwargs)
    selection_val_loader = DataLoader(
        selection_val_dataset, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, early_stop_loader, selection_val_loader, test_loader


def validation_score(model, loader, criterion, device):
    model.eval()
    losses, y_true, y_pred = [], [], []
    with torch.no_grad():
        for y, tab_emb, text_emb, _ in loader:
            x = batch_features(tab_emb, text_emb, device)
            y = y.to(device, non_blocking=True)
            logits_or_pred = model(x)
            if task == "cls":
                loss = criterion(logits_or_pred, y.long())
                pred = logits_or_pred.argmax(dim=1)
                y_pred.append(pred.cpu().numpy())
                y_true.append(y.cpu().numpy())
            else:
                y = y.float().reshape(-1)
                pred = logits_or_pred.reshape(-1)
                loss = criterion(pred, y)
                y_pred.append(pred.cpu().numpy())
                y_true.append(y.cpu().numpy())
            losses.append(loss.item() * len(y))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    score = accuracy_score(y_true, y_pred) if task == "cls" else r2_score(y_true, y_pred)
    return sum(losses) / len(loader.dataset), score


def train_model(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss() if task == "cls" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init, weight_decay=alpha)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    best_epoch = 0
    patience = 0
    history = []

    start_time = time.time()
    for epoch in range(max_iter):
        model.train()
        epoch_loss = 0.0
        seen = 0
        pbar = tqdm(train_loader, total=len(train_loader), disable=not verbose)
        pbar.set_description(f"Epoch {epoch + 1}/{max_iter}")

        for y, tab_emb, text_emb, _ in pbar:
            x = batch_features(tab_emb, text_emb, device)
            y = y.to(device, non_blocking=True)
            out = model(x)
            if task == "cls":
                loss = criterion(out, y.long())
            else:
                loss = criterion(out.reshape(-1), y.float().reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y)
            seen += len(y)
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        train_loss = epoch_loss / seen
        row = {"epoch": epoch + 1, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}

        if val_loader is not None:
            val_loss, val_score = validation_score(model, val_loader, criterion, device)
            improved = val_score > best_score + tol
            if improved:
                best_state = copy.deepcopy(model.state_dict())
                best_score = val_score
                best_epoch = epoch + 1
                patience = 0
            else:
                patience += 1
            row.update({"val_loss": val_loss, "val_score": val_score, "patience": patience})
            print(
                f"Epoch {epoch + 1}/{max_iter}, train_loss={train_loss:.6f}, "
                f"val_loss={val_loss:.6f}, val_score={val_score:.6f}, "
                f"best_score={best_score:.6f}, patience={patience}"
            )
            if patience >= n_iter_no_change:
                print("Early stopping")
                history.append(row)
                break
        else:
            improved = best_score == -float("inf") or train_loss < best_score - tol
            if improved:
                best_state = copy.deepcopy(model.state_dict())
                best_score = train_loss
                best_epoch = epoch + 1
                patience = 0
            else:
                patience += 1
            row.update({"patience": patience})
            print(f"Epoch {epoch + 1}/{max_iter}, train_loss={train_loss:.6f}, patience={patience}")

        history.append(row)

    model.load_state_dict(best_state)
    elapsed = time.time() - start_time
    return history, best_epoch, elapsed


def predict(model, loader, device):
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for y, tab_emb, text_emb, _ in tqdm(loader, total=len(loader), desc="Predict", disable=not verbose):
            x = batch_features(tab_emb, text_emb, device)
            out = model(x)
            y_true.append(y.cpu().numpy())
            if task == "cls":
                proba = torch.softmax(out, dim=1).cpu().numpy()
                y_proba.append(proba)
                y_pred.append(proba.argmax(axis=1))
            else:
                y_pred.append(out.reshape(-1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    if task == "cls":
        return y_true, y_pred, np.concatenate(y_proba, axis=0)
    return y_true, y_pred, None


def select_binary_threshold(y_true_indices, positive_proba):
    grid = np.linspace(0.0, 1.0, 101)
    scores = [
        f1_score(
            y_true_indices,
            positive_proba >= threshold,
            zero_division=0,
        )
        for threshold in grid
    ]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


def evaluate_classification(
    y_test_indices,
    y_pred_indices,
    y_test_proba,
    class_labels,
    threshold=None,
):
    y_test = np.asarray(class_labels)[y_test_indices.astype(int)]
    y_pred = np.asarray(class_labels)[y_pred_indices.astype(int)]

    results = {}
    if len(class_labels) == 2:
        positive_label = class_labels[1]
        y_pos_proba = y_test_proba[:, 1]
        y_test_binary = (y_test == positive_label).astype(int)
        if threshold is None:
            raise ValueError("binary test evaluation requires a validation-selected threshold")
        y_pred_indices = (y_pos_proba >= threshold).astype(int)
        y_pred = np.asarray(class_labels)[y_pred_indices]
        results["roc_auc"] = roc_auc_score(y_test_binary, y_pos_proba)
        results["f1"] = f1_score(
            y_test, y_pred, average="binary", pos_label=positive_label
        )
        results["log_loss"] = log_loss(y_test, y_pos_proba, labels=class_labels)
        precision, recall, _ = precision_recall_curve(y_test, y_pos_proba, pos_label=positive_label)
        results["pr_auc"] = auc(recall, precision)
        results["brier"] = brier_score_loss(y_test_binary, y_pos_proba)
        results["ece"] = expected_calibration_error(y_test_binary, y_pos_proba)
    else:
        results["f1_macro"] = f1_score(y_test, y_pred, average="macro")
        results["log_loss"] = log_loss(y_test, y_test_proba, labels=class_labels)
        results["brier"] = multiclass_brier_score(y_test, y_test_proba, labels=class_labels)
        results["ece"] = expected_calibration_error(y_test_indices, y_test_proba)

    results["accuracy"] = accuracy_score(y_test, y_pred)
    results["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=class_labels)
    results["classification_report"] = classification_report(
        y_test,
        y_pred,
        labels=class_labels,
        zero_division=0,
    )
    results["quadratic_weighted_kappa"] = quadratic_weighted_kappa(
        y_test_indices.astype(int),
        y_pred_indices.astype(int),
        len(class_labels),
    )
    return results, y_pred


def evaluate_regression(y_true, y_pred, y_train_raw):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    y_train_raw = y_train_raw.reshape(-1, 1)

    scaler = StandardScaler().fit(y_train_raw)
    y_true = scaler.transform(y_true)
    y_pred = scaler.transform(y_pred)
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def save_history(save_path: Path, history: list[dict]) -> None:
    if not history:
        return
    keys = sorted({key for row in history for key in row})
    with open(save_path / "log_history.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def save_run(
    save_path: Path,
    model,
    model_config: dict,
    results: dict,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_labels: np.ndarray | None = None,
    history: list[dict] | None = None,
) -> None:
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / "pm_results.txt", "w") as file:
        metric_dict_write(file, results, title="PyTorch MLP embedding ablation", metric_sep_cl=True)

    np.save(save_path / "y_test_pred.npy", y_pred)
    if y_proba is not None:
        np.save(save_path / "y_test_proba.npy", y_proba)
        if y_proba.shape[1] == 2:
            np.save(save_path / "y_pos_proba.npy", y_proba[:, 1])
    if class_labels is not None:
        np.save(save_path / "class_labels.npy", np.asarray(class_labels))

    torch.save(model_config | {"state_dict": model.state_dict()}, save_path / "model.pt")
    if history is not None:
        save_history(save_path, history)

    config = {
        "dataname": dataname,
        "task": task,
        "y_levels": y_levels,
        "alignment_note": alignment_note,
        "alignment_subdir": alignment_subdir,
        "use_tab": use_tab,
        "use_text": use_text,
        "text_pooling": text_pooling,
        "random_state": random_state,
        "device": device_name,
        "num_workers": num_workers,
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "alpha": alpha,
        "batch_size": batch_size,
        "learning_rate_init": learning_rate_init,
        "max_iter": max_iter,
        "early_stopping": early_stopping,
        "early_stopping_fraction": early_stopping_fraction,
        "n_iter_no_change": n_iter_no_change,
        "tol": tol,
    }
    with open(save_path / "hyperparameters.json", "w") as file:
        json.dump(config | model_config, file, indent=2)


def main() -> None:
    global dataname, task, y_levels, alignment_note, alignment_subdir
    global use_tab, use_text, text_pooling, device_name, overwrite_existing_results

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--alignment-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=["cls", "reg"], required=True)
    parser.add_argument("--y-levels", type=int, default=None)
    parser.add_argument("--use-tab", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-pooling", choices=["mean", "first", "flatten"], default="mean")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_path = args.data_path.resolve()
    alignment_path = args.alignment_path.resolve()
    save_path = args.output_dir.resolve()
    dataname = data_path.name
    task = args.task
    y_levels = args.y_levels
    alignment_note = alignment_path.name
    alignment_subdir = ""
    use_tab = args.use_tab
    use_text = args.use_text
    text_pooling = args.text_pooling
    device_name = args.device
    overwrite_existing_results = args.overwrite
    if not (use_tab or use_text):
        raise ValueError("At least one of --use-tab or --use-text must be enabled.")
    if not (data_path / "y_val.npy").is_file():
        raise FileNotFoundError(f"explicit validation response is required: {data_path / 'y_val.npy'}")

    os.environ.setdefault("PYTHONHASHSEED", str(random_state))
    set_seed(random_state)

    y_train_raw, y_val_raw, y_test_raw = load_raw_y(data_path)
    class_labels = infer_class_labels(y_train_raw, y_levels) if task == "cls" else None
    y_path, y_type = prepare_dataset_y_files(
        save_path,
        y_train_raw,
        y_val_raw,
        y_test_raw,
        class_labels,
    )

    train_dataset = make_dataset(y_path, alignment_path, y_type, "train")
    val_dataset = make_dataset(y_path, alignment_path, y_type, "val")
    test_dataset = make_dataset(y_path, alignment_path, y_type, "test")
    validate_split_sizes(train_dataset, alignment_path, len(y_train_raw), "train")
    validate_split_sizes(val_dataset, alignment_path, len(y_val_raw), "val")
    validate_split_sizes(test_dataset, alignment_path, len(y_test_raw), "test")

    input_dim = infer_input_dim(train_dataset)
    output_dim = len(class_labels) if task == "cls" else 1
    device = torch.device(device_name)
    model = MLP(input_dim, output_dim).to(device)
    model_config = {"input_dim": input_dim, "output_dim": output_dim}

    print(f"Data path: {data_path}")
    print(f"Alignment path: {alignment_path}")
    print(f"Save path: {save_path}")
    print(f"Device: {device}")
    print(f"Input dim: {input_dim}, output dim: {output_dim}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_loader, early_stop_loader, selection_val_loader, test_loader = make_loaders(
        train_dataset, val_dataset, test_dataset
    )
    history, best_epoch, elapsed = train_model(
        model, train_loader, early_stop_loader, device
    )

    if task == "cls":
        validation_threshold = None
        validation_f1 = None
        if len(class_labels) == 2:
            y_val_true, _, y_val_proba = predict(
                model, selection_val_loader, device
            )
            validation_threshold, validation_f1 = select_binary_threshold(
                y_val_true.astype(int), y_val_proba[:, 1]
            )
        # Test predictions are produced only after all validation choices are fixed.
        y_true, y_pred_idx_or_value, y_proba = predict(model, test_loader, device)
        results, y_pred = evaluate_classification(
            y_true,
            y_pred_idx_or_value,
            y_proba,
            class_labels,
            threshold=validation_threshold,
        )
        if validation_threshold is not None:
            results["validation_threshold"] = validation_threshold
            results["validation_f1"] = validation_f1
        save_run(save_path, model, model_config | {"best_epoch": best_epoch, "time_used": elapsed},
                 results, y_pred, y_proba, class_labels, history)
    else:
        y_true, y_pred_idx_or_value, _ = predict(model, test_loader, device)
        results = evaluate_regression(y_true, y_pred_idx_or_value, y_train_raw)
        save_run(save_path, model, model_config | {"best_epoch": best_epoch, "time_used": elapsed},
                 results, y_pred_idx_or_value, history=history)

    print("\nTest metrics:")
    for key, value in results.items():
        print(key, "\n", value, "\n", sep="")
    print(f"Best epoch: {best_epoch}")
    print(f"Time used: {elapsed:.2f}s")
    print(f"Saved results to: {save_path}")


if __name__ == "__main__":
    main()
