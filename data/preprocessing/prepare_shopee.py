#!/usr/bin/env python3
"""Construct the Shopee arrays and supervised image embeddings.

The script trains an AutoGluon MultiModal ResNet-18 classifier, extracts its
512-dimensional image embeddings, and writes the row-indexed layout used by
the diffusion model.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def expand_image_paths(value: str, image_root: Path) -> str:
    return ";".join(str((image_root / item).resolve()) for item in value.split(";"))


def save_rowwise(values: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values).reshape(len(values), -1)
    for index, value in enumerate(values):
        np.save(output_dir / f"{index}.npy", value.astype(np.float32).reshape(1, -1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Base directory for relative paths in the CSV image column.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictor-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-random-state", type=int, default=777)
    parser.add_argument("--time-limit", type=int, default=3000)
    args = parser.parse_args()

    try:
        from autogluon.multimodal import MultiModalPredictor
    except ImportError as error:
        raise ImportError(
            "prepare_shopee.py requires AutoGluon; install the project with "
            "`python -m pip install -e .`"
        ) from error

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if args.predictor_dir.exists() and any(args.predictor_dir.iterdir()):
        raise FileExistsError(f"predictor directory is not empty: {args.predictor_dir}")

    train = pd.read_csv(args.train_csv)
    test = pd.read_csv(args.test_csv)
    required = {"image", "label"}
    for name, frame in (("train", train), ("test", test)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} CSV is missing columns: {sorted(missing)}")
        frame["image"] = frame["image"].map(
            lambda value: expand_image_paths(str(value), args.image_root)
        )

    train_fit, validation = train_test_split(
        train,
        test_size=args.validation_fraction,
        random_state=args.validation_random_state,
        stratify=train["label"],
    )
    train_fit = train_fit.reset_index(drop=True)
    validation = validation.reset_index(drop=True)

    set_seed(args.seed)
    hyperparameters = {
        "model.names": ["timm_image", "fusion_mlp"],
        "model.timm_image.checkpoint_name": "resnet18",
        "data.categorical.convert_to_text": "False",
        "env.per_gpu_batch_size": "32",
        "optim.lr": "1e-4",
        "optim.weight_decay": "0",
        "optim.lr_decay": "1",
        "optim.max_epochs": "100",
        "optim.warmup_steps": "10",
    }
    predictor = MultiModalPredictor(
        problem_type="classification",
        label="label",
        path=str(args.predictor_dir),
    )
    predictor.fit(
        train_data=train_fit,
        hyperparameters=hyperparameters,
        seed=args.seed,
        holdout_frac=0.2,
        time_limit=args.time_limit,
    )

    split_frames = {
        "train": train_fit,
        "val": validation,
        "test": test,
    }
    split_embeddings = {
        split: np.asarray(
            predictor.extract_embedding(
                frame.drop(columns="label"), as_pandas=False
            )
        ).reshape(len(frame), -1)
        for split, frame in split_frames.items()
    }
    unexpected_shapes = {
        split: value.shape
        for split, value in split_embeddings.items()
        if value.shape[1] != 512
    }
    if unexpected_shapes:
        raise ValueError(
            f"expected 512-dimensional ResNet-18 embeddings; got {unexpected_shapes}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / f"{args.output_dir.name}_image_embd"
    for split, frame in split_frames.items():
        save_rowwise(split_embeddings[split], image_dir / split)
        np.save(
            args.output_dir / f"y_{split}.npy",
            frame["label"].astype(str).to_numpy().reshape(-1, 1),
        )

    # The diffusion data loader requires tabular placeholders even though the
    # Shopee experiment enables image conditioning only.
    rng = np.random.RandomState(42)
    for split, frame in split_frames.items():
        np.save(args.output_dir / f"X_num_{split}.npy", rng.randn(len(frame), 1))
        np.save(
            args.output_dir / f"X_cat_{split}.npy",
            rng.randint(0, 5, (len(frame), 1)).astype(str),
        )

    manifest = {
        "dataset": "shopee",
        "training_pool_rows": len(train),
        "train_rows": len(train_fit),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "embedding_dim": split_embeddings["train"].shape[1],
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "validation_random_state": args.validation_random_state,
        "time_limit": args.time_limit,
        "holdout_fraction_inside_autogluon": 0.2,
        "hyperparameters": hyperparameters,
    }
    (args.output_dir / "construction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
