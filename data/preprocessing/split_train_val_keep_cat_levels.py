import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


DATASET_DESCRIPTION = ""

MISSING_LEVEL = ("__MISSING_LEVEL__",)


def level_key(x):
    """Make categorical levels hashable and treat missing values as one level."""
    if x is None:
        return MISSING_LEVEL
    try:
        if np.isscalar(x) and bool(np.isnan(x)):
            return MISSING_LEVEL
    except (TypeError, ValueError):
        pass
    return x.item() if hasattr(x, "item") else x


def build_level_counts(X_cat):
    counts = [Counter() for _ in range(X_cat.shape[1])]
    for row in X_cat:
        for j, value in enumerate(row):
            counts[j][level_key(value)] += 1
    return counts


def can_move_to_val(row, train_counts):
    """Return True if this row can leave training without dropping a level."""
    for j, value in enumerate(row):
        if train_counts[j][level_key(value)] <= 1:
            return False
    return True


def remove_from_train_counts(row, train_counts):
    for j, value in enumerate(row):
        train_counts[j][level_key(value)] -= 1


def train_val_split_keep_cat_levels(X_cat, val_size, random_state=0):
    """
    Split row indices while guaranteeing every categorical level in validation
    is still present at least once in the resulting training set.

    The validation set may be smaller than val_size if too many rows contain
    categorical levels that occur only once in the original training data.
    """
    if X_cat.ndim != 2:
        raise ValueError("Expected X_cat to be 2D: (n_rows, n_categorical_features).")
    if not 0 < val_size < len(X_cat):
        raise ValueError(f"val_size must be between 1 and {len(X_cat) - 1}; got {val_size}.")

    rng = np.random.default_rng(random_state)
    train_counts = build_level_counts(X_cat)
    val_idx = []
    train_mask = np.ones(len(X_cat), dtype=bool)

    for idx in rng.permutation(len(X_cat)):
        if len(val_idx) == val_size:
            break
        row = X_cat[idx]
        if can_move_to_val(row, train_counts):
            val_idx.append(idx)
            train_mask[idx] = False
            remove_from_train_counts(row, train_counts)

    train_idx = np.flatnonzero(train_mask)
    return train_idx, np.array(val_idx, dtype=int)


def levels(values):
    return {level_key(x) for x in np.asarray(values).reshape(-1)}


def assert_val_levels_in_train(X_cat_train, X_cat_val):
    failures = []
    for j in range(X_cat_train.shape[1]):
        missing = levels(X_cat_val[:, j]) - levels(X_cat_train[:, j])
        if missing:
            failures.append((j, missing))
    if failures:
        details = "; ".join(f"cat_feature_{j}: {sorted(map(repr, missing))}" for j, missing in failures)
        raise AssertionError(f"Validation contains categorical levels missing from training: {details}")


def load_text_list(path, expected_len):
    with open(path, "r", encoding="utf-8") as f:
        text = json.load(f)
    if not isinstance(text, list):
        raise ValueError(f"Expected {path} to contain a JSON list, got {type(text).__name__}.")
    if len(text) != expected_len:
        raise ValueError(f"Expected {path} to have {expected_len} rows, got {len(text)}.")
    return text


def subset_list(values, indices):
    return [values[int(i)] for i in indices]


def save_text_list(path, values):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(values, f, ensure_ascii=False)


def read_description(args):
    if args.description_file:
        with open(args.description_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    if args.description:
        return args.description.strip()
    return DATASET_DESCRIPTION


def copy_split_script(new_data_path):
    script_path = Path(__file__).resolve()
    copied_script_path = new_data_path / script_path.name
    shutil.copy2(script_path, copied_script_path)
    return copied_script_path.name


def write_readme(
    new_data_path,
    new_data_name,
    source_data_name,
    description,
    copied_script_name,
    train_size,
    val_size,
    test_size,
    original_train_size,
    original_test_size,
    target_val_size,
    random_state,
    val_divisor,
    n_num_features,
    n_cat_features,
    has_text,
):
    text_note = " Text files were split with the same row indices." if has_text else ""
    if description:
        description_block = f"{description}\n\n"
    else:
        description_block = ""

    readme = f"""## {new_data_name}

(auto generated from the data splitting script)

{description_block}Constructed from the \"{source_data_name}\" dataset by splitting a validation set from the original training set. The requested validation size was 1/{val_divisor} of the original training size ({target_val_size} rows), using random_state={random_state}.

The split keeps every categorical predictor level in the validation set present in the resulting training set.{text_note}

original_train_size: {original_train_size}
original_test_size: {original_test_size}

train_size: {train_size}
test_size: {test_size}
val_size: {val_size}

number of features:
num: {n_num_features}
cat: {n_cat_features}
y: 1

Files:
- X_num_train.npy, X_cat_train.npy, y_train.npy
- X_num_val.npy, X_cat_val.npy, y_val.npy
- X_num_test.npy, X_cat_test.npy, y_test.npy
"""
    if has_text:
        readme += "- text_train.json, text_val.json, text_test.json\n"

    readme += f"""
Construction code:
The script used to construct this dataset is saved as `{copied_script_name}` in this directory.
"""

    with open(new_data_path / "readme.md", "w", encoding="utf-8") as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split train into train/val while keeping every validation "
            "categorical level present in the resulting training set."
        )
    )
    parser.add_argument("--dataname", default="wine_review3")
    parser.add_argument("--data-root", default="../tabsynfnn/data")
    parser.add_argument("--new-data-name", default=None)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--val-divisor", type=int, default=8)
    parser.add_argument(
        "--description",
        default=None,
        help="Specific description to place at the top of the generated readme.md.",
    )
    parser.add_argument(
        "--description-file",
        default=None,
        help="Path to a text/markdown file containing the readme.md description.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_root) / args.dataname
    cat_train = np.load(data_path / "X_cat_train.npy", allow_pickle=True)
    num_train = np.load(data_path / "X_num_train.npy", allow_pickle=True)
    y_train = np.load(data_path / "y_train.npy", allow_pickle=True)
    cat_test = np.load(data_path / "X_cat_test.npy", allow_pickle=True)
    num_test = np.load(data_path / "X_num_test.npy", allow_pickle=True)
    y_test = np.load(data_path / "y_test.npy", allow_pickle=True)
    text_train_path = data_path / "text_train.json"
    text_test_path = data_path / "text_test.json"
    has_text = text_train_path.exists() and text_test_path.exists()
    if has_text:
        text_train = load_text_list(text_train_path, expected_len=len(cat_train))
        text_test = load_text_list(text_test_path, expected_len=len(cat_test))

    target_val_size = len(cat_train) // args.val_divisor
    train_idx, val_idx = train_val_split_keep_cat_levels(
        cat_train,
        val_size=target_val_size,
        random_state=args.random_state,
    )

    X_cat_train = cat_train[train_idx]
    X_cat_val = cat_train[val_idx]
    X_num_train = num_train[train_idx]
    X_num_val = num_train[val_idx]
    y_train_new = y_train[train_idx]
    y_val = y_train[val_idx]
    if has_text:
        text_train_new = subset_list(text_train, train_idx)
        text_val = subset_list(text_train, val_idx)

    assert_val_levels_in_train(X_cat_train, X_cat_val)

    if len(val_idx) < target_val_size:
        print(
            f"Warning: requested {target_val_size} validation rows, but only "
            f"{len(val_idx)} rows can be moved while preserving all categorical levels in training."
        )

    new_data_name = args.new_data_name or f"{args.dataname}_wval"
    print(f"Saving new dataset with train/val split to {new_data_name}...")
    new_data_path = Path(args.data_root) / new_data_name
    print(new_data_path)
    os.makedirs(new_data_path, exist_ok=True)

    np.save(new_data_path / "X_num_train.npy", X_num_train)
    np.save(new_data_path / "X_cat_train.npy", X_cat_train)
    np.save(new_data_path / "y_train.npy", y_train_new)
    np.save(new_data_path / "X_num_val.npy", X_num_val)
    np.save(new_data_path / "X_cat_val.npy", X_cat_val)
    np.save(new_data_path / "y_val.npy", y_val)
    np.save(new_data_path / "X_num_test.npy", num_test)
    np.save(new_data_path / "X_cat_test.npy", cat_test)
    np.save(new_data_path / "y_test.npy", y_test)
    if has_text:
        save_text_list(new_data_path / "text_train.json", text_train_new)
        save_text_list(new_data_path / "text_val.json", text_val)
        save_text_list(new_data_path / "text_test.json", text_test)
    copied_script_name = copy_split_script(new_data_path)
    write_readme(
        new_data_path=new_data_path,
        new_data_name=new_data_name,
        source_data_name=args.dataname,
        description=read_description(args),
        copied_script_name=copied_script_name,
        train_size=len(train_idx),
        val_size=len(val_idx),
        test_size=len(cat_test),
        original_train_size=len(cat_train),
        original_test_size=len(cat_test),
        target_val_size=target_val_size,
        random_state=args.random_state,
        val_divisor=args.val_divisor,
        n_num_features=num_train.shape[1] if num_train.ndim == 2 else 1,
        n_cat_features=cat_train.shape[1],
        has_text=has_text,
    )

    print(f"train={len(train_idx)}, val={len(val_idx)}, test={len(cat_test)}")
    print("All categorical levels in validation are present in training.")
    if has_text:
        print("Text files split with the same train/validation indices.")
    print("Saved readme.md and construction script.")


if __name__ == "__main__":
    main()
