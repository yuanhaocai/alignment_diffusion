#!/usr/bin/env python3
"""Apply a trained alignment projector to tabular and token embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


RELEASE_ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT_SOURCE = RELEASE_ROOT / "src" / "alignment"
sys.path.insert(0, str(ALIGNMENT_SOURCE))

from utils_transform import load_model  # noqa: E402


def numeric_files(path: Path) -> list[Path]:
    files = list(path.glob("*.npy"))
    try:
        return sorted(files, key=lambda item: int(item.stem))
    except ValueError as error:
        raise ValueError(f"embedding filenames must be numeric in {path}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_destination(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_tabular(value: np.ndarray, train: np.ndarray, mode: str) -> np.ndarray:
    ddof = 0 if mode == "numpy-population" else 1
    return (value - train.mean(axis=0, keepdims=True)) / (
        train.std(axis=0, ddof=ddof, keepdims=True) + 1e-6
    )


def project_tabular(model, args, split: str, device: str) -> tuple[int, list[int]]:
    train = np.load(args.tabular_path / "train_z_dat.npy").astype(np.float32)
    value = np.load(args.tabular_path / f"{split}_z_dat.npy").astype(np.float32)
    value = normalize_tabular(value, train, args.tabular_normalization)
    output = args.output_dir / f"{split}_z_dat_proj.npy"
    prepare_destination(output, args.overwrite)

    projected = []
    with torch.no_grad():
        for start in range(0, value.shape[0], args.tabular_batch_size):
            batch = torch.from_numpy(value[start : start + args.tabular_batch_size]).to(device)
            projected.append(model.w_tab(batch).cpu().numpy())
    array = np.concatenate(projected, axis=0)
    np.save(output, array)
    return array.shape[0], list(array.shape[1:])


def project_text(model, args, split: str, device: str) -> tuple[int, list[int]]:
    source_dir = args.text_path / split
    files = numeric_files(source_dir)
    output_dir = args.output_dir / "text_proj" / split
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_shape: list[int] | None = None
    with torch.no_grad():
        for start in tqdm(range(0, len(files), args.text_batch_size), desc=f"text {split}"):
            chunk = [np.load(path).squeeze().astype(np.float32) for path in files[start : start + args.text_batch_size]]
            if any(value.ndim != 2 for value in chunk):
                shapes = [value.shape for value in chunk]
                raise ValueError(f"expected token matrices in {source_dir}; got {shapes}")
            batch = torch.from_numpy(np.stack(chunk)).to(device)
            projected = model.w_text(batch).cpu().numpy()
            output_shape = list(projected.shape[1:])
            for offset, value in enumerate(projected):
                np.save(output_dir / f"{start + offset}.npy", value)
    return len(files), output_shape or []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--tabular-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--y-type", choices=["continuous", "categorical"], required=True)
    parser.add_argument("--y-num-classes", type=int)
    parser.add_argument("--emb-dim", type=int, default=768)
    parser.add_argument("--projector-mode", choices=["auto", "linear", "residual", "identity"], default="auto")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--tabular-normalization",
        choices=["numpy-population", "torch-sample"],
        default="numpy-population",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tabular-batch-size", type=int, default=1024)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.y_type == "categorical" and args.y_num_classes is None:
        parser.error("--y-num-classes is required for categorical targets")

    args.checkpoint = args.checkpoint.resolve()
    args.text_path = args.text_path.resolve()
    args.tabular_path = args.tabular_path.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(
        str(args.checkpoint),
        args.emb_dim,
        args.y_num_classes,
        args.y_type,
        args.device,
        projector_mode=args.projector_mode,
        residual_scale=args.residual_scale,
    )

    completed = {}
    for split in args.splits:
        if not (args.tabular_path / f"{split}_z_dat.npy").exists():
            raise FileNotFoundError(args.tabular_path / f"{split}_z_dat.npy")
        if not (args.text_path / split).is_dir():
            raise FileNotFoundError(args.text_path / split)
        tab_n, tab_shape = project_tabular(model, args, split, args.device)
        text_n, text_shape = project_text(model, args, split, args.device)
        if tab_n != text_n:
            raise ValueError(f"{split}: tabular rows={tab_n}, text rows={text_n}")
        completed[split] = {
            "rows": tab_n,
            "tabular_shape_per_row": tab_shape,
            "text_shape_per_row": text_shape,
        }

    effective_residual_scale = getattr(model.w_tab, "residual_scale", args.residual_scale)
    if torch.is_tensor(effective_residual_scale):
        effective_residual_scale = effective_residual_scale.detach().cpu().item()
    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "projector_mode": args.projector_mode,
        "residual_scale_at_transform": float(effective_residual_scale),
        "tabular_normalization": args.tabular_normalization,
        "completed": completed,
    }
    (args.output_dir / "transformation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
