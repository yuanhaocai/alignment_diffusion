#!/usr/bin/env python3
"""Train the tabular-predictor or categorical-response VAE."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tabsynfnn.tabsyn.vae.main_unitabsyn import vae_train_unitabsyn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=["dat", "y"], required=True)
    parser.add_argument(
        "--task-type",
        choices=["regression", "binclass", "multiclass"],
        required=True,
    )
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--d-token", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-beta", type=float, default=1e-2)
    parser.add_argument("--min-beta", type=float, default=1e-5)
    parser.add_argument("--beta-decay", type=float, default=0.7)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The recorded VAE trainer requires a CUDA device.")
    if args.target == "y" and args.task_type == "regression":
        raise ValueError(
            "Continuous responses use scale_continuous_response.py, not a categorical VAE."
        )

    vae_train_unitabsyn(
        data_path=str(args.data_dir),
        vae_path=str(args.output_dir),
        latent_dim=args.latent_dim if args.target == "dat" else None,
        d_token=args.d_token,
        task_type=args.task_type,
        train_target=args.target,
        device_ids=[args.gpu],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_beta=args.max_beta,
        min_beta=args.min_beta,
        lambd=args.beta_decay,
    )


if __name__ == "__main__":
    main()
