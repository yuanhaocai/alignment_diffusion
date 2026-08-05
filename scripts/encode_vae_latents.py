#!/usr/bin/env python3
"""Encode train/validation/test arrays with a trained predictor or response VAE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tabsynfnn.tabsyn.vae.model_fnn import Encoder_model_fnn, Model_VAE_fnn
from tabsynfnn.tabsyn.vae.model_multimodal import Encoder_caty, VAE_caty
from tabsynfnn.util_functions import read_written_hypars
from tabsynfnn.utils.util import get_categories
from tabsynfnn.utils_train import preprocess


SPLITS = ("train", "test", "val")


def load_state_flexible(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        stripped = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }
        model.load_state_dict(stripped)


def find_hyperparameters(vae_dir: Path, target: str) -> dict[str, object]:
    candidates = [
        vae_dir / f"hyperparameters_{target}.txt",
        vae_dir / "hyperparameters.txt",
    ]
    for path in candidates:
        if path.is_file():
            return read_written_hypars(str(path))
    raise FileNotFoundError(f"no VAE hyperparameter file found under {vae_dir}")


def encode_tabular(
    data_dir: Path,
    vae_dir: Path,
    task_type: str,
    device: torch.device,
) -> None:
    x_num, x_cat, _, categories, d_numerical = preprocess(
        str(data_dir),
        task_type=task_type,
        is_y_cond=True,
    )
    hp = find_hyperparameters(vae_dir, "dat")
    model = Model_VAE_fnn(
        d_numerical,
        categories,
        int(hp["d_token"]),
        int(hp["seq_len"]),
        int(hp["latent_dim"]),
        bias=True,
    ).to(device)
    load_state_flexible(model, vae_dir / "model_dat.pt", device)

    encoder = Encoder_model_fnn(
        d_numerical,
        categories,
        int(hp["d_token"]),
        int(hp["seq_len"]),
        int(hp["latent_dim"]),
    ).to(device)
    encoder.load_weights(model)
    encoder.eval()

    with torch.no_grad():
        for index, split in enumerate(SPLITS):
            if split == "val" and not (data_dir / "y_val.npy").is_file():
                continue
            num = torch.as_tensor(x_num[index], dtype=torch.float32, device=device)
            cat = torch.as_tensor(x_cat[index], device=device)
            latent = encoder(num, cat).cpu().numpy()
            np.save(vae_dir / f"{split}_z_dat.npy", latent)
            print(f"{split}_z_dat.npy: {latent.shape}")


def encode_response(
    data_dir: Path,
    vae_dir: Path,
    task_type: str,
    device: torch.device,
) -> None:
    _, _, y, _, _ = preprocess(
        str(data_dir),
        task_type=task_type,
        is_y_cond=True,
    )
    hp = find_hyperparameters(vae_dir, "y")
    categories = get_categories(y[0])
    model = VAE_caty(categories, int(hp["d_token"]), bias=True).to(device)
    load_state_flexible(model, vae_dir / "model_y.pt", device)
    encoder = Encoder_caty(categories, int(hp["d_token"])).to(device)
    encoder.load_weights(model)
    encoder.eval()

    with torch.no_grad():
        for index, split in enumerate(SPLITS):
            if split == "val" and not (data_dir / "y_val.npy").is_file():
                continue
            values = torch.as_tensor(y[index], device=device)
            latent = encoder(values).cpu().numpy()
            np.save(vae_dir / f"{split}_z_y.npy", latent)
            print(f"{split}_z_y.npy: {latent.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--vae-dir", type=Path, required=True)
    parser.add_argument("--target", choices=["dat", "y"], required=True)
    parser.add_argument(
        "--task-type",
        choices=["regression", "binclass", "multiclass"],
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.target == "dat":
        encode_tabular(args.data_dir, args.vae_dir, args.task_type, device)
    elif args.task_type == "regression":
        raise ValueError(
            "Continuous responses use scale_continuous_response.py, not a categorical VAE."
        )
    else:
        encode_response(args.data_dir, args.vae_dir, args.task_type, device)


if __name__ == "__main__":
    main()
