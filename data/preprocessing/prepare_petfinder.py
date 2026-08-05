#!/usr/bin/env python3
"""Construct ``petfinder_wval`` from the prepared PetFinder CSV files.

The AutoGluon PetFinder archive supplies ``train.csv`` and a labeled
``dev.csv``. The script uses the former as the training pool, reserves a
validation split from it, and uses the latter as the held-out test set. Every
validation categorical level is retained in the training split.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from split_train_val_keep_cat_levels import (
    assert_val_levels_in_train,
    train_val_split_keep_cat_levels,
)


DROP_COLUMNS = ["Name", "RescuerID", "VideoAmt", "PhotoAmt", "Images"]
NUMERICAL_COLUMNS = ["Age", "Fee", "Quantity"]
TEXT_COLUMN = "Description"
TARGET_COLUMN = "AdoptionSpeed"
ID_COLUMN = "PetID"
NON_ASCII = re.compile(r"[^\x00-\x7F]+")
REDUNDANT_PUNCTUATION = re.compile(r"[.-]{2,}|[=_+~>*]")


def clean_description(value: str) -> str:
    return REDUNDANT_PUNCTUATION.sub("", NON_ASCII.sub("", value))


def filter_frame(
    path: Path,
    allowed_breed1: set[int] | None = None,
    allowed_breed2: set[int] | None = None,
) -> pd.DataFrame:
    try:
        import langid
    except ImportError as error:
        raise ImportError(
            "prepare_petfinder.py requires langid==1.1.6; install the project "
            "with `python -m pip install -e .`"
        ) from error

    frame = pd.read_csv(path, index_col=0)
    frame = frame.loc[frame["PhotoAmt"] != 0]
    frame = frame.drop(columns=DROP_COLUMNS).dropna(axis=0)

    if allowed_breed1 is not None and allowed_breed2 is not None:
        frame = frame.loc[frame["Breed1"].astype(int).isin(allowed_breed1)]
        frame = frame.loc[frame["Breed2"].astype(int).isin(allowed_breed2)]

    english = frame[TEXT_COLUMN].apply(lambda value: langid.classify(value)[0] == "en")
    frame = frame.loc[english].copy()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].apply(clean_description)
    return frame


def arrays(frame: pd.DataFrame) -> dict[str, object]:
    excluded = {
        *NUMERICAL_COLUMNS,
        TARGET_COLUMN,
        TEXT_COLUMN,
        ID_COLUMN,
    }
    categorical_columns = [column for column in frame.columns if column not in excluded]
    return {
        "X_num": frame[NUMERICAL_COLUMNS].to_numpy(dtype=np.float32),
        "X_cat": frame[categorical_columns].to_numpy().astype(str),
        "y": frame[[TARGET_COLUMN]].to_numpy().astype(str),
        "text": [
            f"{TEXT_COLUMN} is {value}"
            for value in frame[TEXT_COLUMN].tolist()
        ],
        "categorical_columns": categorical_columns,
    }


def save_json(path: Path, values: list[str]) -> None:
    path.write_text(json.dumps(values), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument(
        "--test-csv",
        type=Path,
        required=True,
        help="Petfinder dev.csv; it contains labels and is the paper's test set.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--val-divisor", type=int, default=8)
    args = parser.parse_args()

    train_frame = filter_frame(args.train_csv)
    train_data = arrays(train_frame)
    x_cat_pool = np.asarray(train_data["X_cat"])
    allowed_breed1 = set(x_cat_pool[:, 1].astype(int))
    allowed_breed2 = set(x_cat_pool[:, 2].astype(int))

    test_frame = filter_frame(
        args.test_csv,
        allowed_breed1=allowed_breed1,
        allowed_breed2=allowed_breed2,
    )
    test_data = arrays(test_frame)
    if train_data["categorical_columns"] != test_data["categorical_columns"]:
        raise ValueError("train and test categorical columns differ")

    requested_val_size = len(x_cat_pool) // args.val_divisor
    train_idx, val_idx = train_val_split_keep_cat_levels(
        x_cat_pool,
        val_size=requested_val_size,
        random_state=args.random_state,
    )
    assert_val_levels_in_train(x_cat_pool[train_idx], x_cat_pool[val_idx])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for prefix in ("X_num", "X_cat", "y"):
        pool = np.asarray(train_data[prefix])
        np.save(args.output_dir / f"{prefix}_train.npy", pool[train_idx])
        np.save(args.output_dir / f"{prefix}_val.npy", pool[val_idx])
        np.save(args.output_dir / f"{prefix}_test.npy", np.asarray(test_data[prefix]))

    train_text = list(train_data["text"])
    save_json(args.output_dir / "text_train.json", [train_text[int(i)] for i in train_idx])
    save_json(args.output_dir / "text_val.json", [train_text[int(i)] for i in val_idx])
    save_json(args.output_dir / "text_test.json", list(test_data["text"]))

    manifest = {
        "dataset": "petfinder_wval",
        "train_csv": str(args.train_csv.resolve()),
        "test_csv": str(args.test_csv.resolve()),
        "test_csv_note": "dev.csv is the held-out labeled test set.",
        "random_state": args.random_state,
        "val_divisor": args.val_divisor,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_frame)),
        "numerical_columns": NUMERICAL_COLUMNS,
        "categorical_columns": train_data["categorical_columns"],
        "text_column": TEXT_COLUMN,
        "target_column": TARGET_COLUMN,
        "filters": [
            "PhotoAmt != 0",
            f"drop columns: {DROP_COLUMNS}",
            "drop rows containing missing values",
            "test Breed1/Breed2 levels must occur in the training pool",
            "langid language == en",
            "remove non-ASCII characters and redundant punctuation from Description",
        ],
    }
    (args.output_dir / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
