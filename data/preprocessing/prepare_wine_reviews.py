#!/usr/bin/env python3
"""Construct ``wine_review3_wval`` directly from the raw Wine Reviews CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from split_train_val_keep_cat_levels import (
    assert_val_levels_in_train,
    train_val_split_keep_cat_levels,
)


DROP_COLUMNS = [
    "designation",
    "region_2",
    "taster_twitter_handle",
    "title",
]
NUMERICAL_COLUMNS = ["price"]
CATEGORICAL_COLUMNS = ["country", "taster_name"]
TEXT_COLUMN = "description"
TARGET_COLUMN = "points"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--test-random-state", type=int, default=42)
    parser.add_argument("--val-divisor", type=int, default=8)
    parser.add_argument("--val-random-state", type=int, default=0)
    parser.add_argument("--pool-only", action="store_true",
                        help="write the original non-test pool as train, before any validation/calibration split")
    args = parser.parse_args()

    frame = pd.read_csv(args.raw_csv, index_col=0)
    missing_columns = (
        set(DROP_COLUMNS)
        | set(NUMERICAL_COLUMNS)
        | set(CATEGORICAL_COLUMNS)
        | {TEXT_COLUMN, TARGET_COLUMN}
    ) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"raw CSV is missing columns: {sorted(missing_columns)}")

    # Drop missing rows before selecting the paper features.
    frame = frame.drop(columns=DROP_COLUMNS).dropna(axis=0)
    x_num = frame[NUMERICAL_COLUMNS].to_numpy(dtype=np.float32)
    x_cat = frame[CATEGORICAL_COLUMNS].to_numpy().astype(str)
    y = frame[[TARGET_COLUMN]].to_numpy(dtype=np.float32)
    text = [f"{TEXT_COLUMN} is {value}" for value in frame[TEXT_COLUMN].tolist()]

    pool_idx, test_idx = train_test_split(
        list(range(len(frame))),
        test_size=args.test_size,
        random_state=args.test_random_state,
    )
    pool_idx = np.asarray(pool_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if args.pool_only:
        train_idx, val_idx = pool_idx, np.array([], dtype=int)
    else:
        local_train_idx, local_val_idx = train_val_split_keep_cat_levels(
            x_cat[pool_idx],
            val_size=len(pool_idx) // args.val_divisor,
            random_state=args.val_random_state,
        )
        train_idx = pool_idx[local_train_idx]
        val_idx = pool_idx[local_val_idx]
        assert_val_levels_in_train(x_cat[train_idx], x_cat[val_idx])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_indices = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    for split, indices in split_indices.items():
        if args.pool_only and split == "val":
            continue
        np.save(args.output_dir / f"X_num_{split}.npy", x_num[indices])
        np.save(args.output_dir / f"X_cat_{split}.npy", x_cat[indices])
        np.save(args.output_dir / f"y_{split}.npy", y[indices])
        (args.output_dir / f"text_{split}.json").write_text(
            json.dumps([text[int(index)] for index in indices]),
            encoding="utf-8",
        )

    manifest = {
        "dataset": "wine_review3" if args.pool_only else "wine_review3_wval",
        "pool_only": args.pool_only,
        "raw_csv": str(args.raw_csv.resolve()),
        "dropped_columns": DROP_COLUMNS,
        "drop_missing_rows_before_feature_selection": True,
        "numerical_columns": NUMERICAL_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "text_column": TEXT_COLUMN,
        "target_column": TARGET_COLUMN,
        "test_size": args.test_size,
        "test_random_state": args.test_random_state,
        "val_divisor": args.val_divisor,
        "val_random_state": args.val_random_state,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }
    (args.output_dir / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
