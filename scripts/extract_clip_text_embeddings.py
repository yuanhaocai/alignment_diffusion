#!/usr/bin/env python3
"""Encode text with a frozen CLIP text transformer.

The script saves one array of shape ``(1, 77, 768)`` per row. Padding-token
rows are zeroed so downstream code can perform masked mean pooling without a
separate mask file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of text rows encoded per batch.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-padding", action="store_true",
                        help="retain all 77 CLIP token states, as in the recorded Wine PI experiment")
    args = parser.parse_args()

    texts = json.loads(args.input_json.read_text(encoding="utf-8"))
    if not isinstance(texts, list) or not all(isinstance(item, str) for item in texts):
        raise ValueError("input JSON must be a list of strings")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.glob("*.npy"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} already contains embeddings; pass --overwrite"
        )

    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    model = CLIPTextModel.from_pretrained(args.model).eval().to(args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), args.batch_size)):
            batch_text = texts[start : start + args.batch_size]
            encoded = tokenizer(
                batch_text,
                truncation=True,
                max_length=args.max_length,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            hidden = model(
                input_ids=encoded["input_ids"].to(args.device),
            ).last_hidden_state
            attention_mask = encoded["attention_mask"].to(
                device=hidden.device, dtype=hidden.dtype
            )
            if not args.keep_padding:
                hidden = hidden * attention_mask.unsqueeze(-1)
            hidden = hidden.cpu().numpy()
            for offset in range(len(hidden)):
                np.save(
                    args.output_dir / f"{start + offset}.npy",
                    hidden[offset : offset + 1],
                )


if __name__ == "__main__":
    main()
