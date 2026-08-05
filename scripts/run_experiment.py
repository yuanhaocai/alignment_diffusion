#!/usr/bin/env python3
"""Expand and run an experiment workflow without invoking a shell."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

VARIANTS = (
    "full",
    "tabular-only",
    "no-diffusion",
    "no-response-awareness",
    "no-alignment",
)

DATASET_WORKFLOWS = {
    "petfinder_full.json": {
        "label": "Petfinder",
        "output_prefix": "petfinder",
        "task": "cls",
        "y_levels": "5",
    },
    "wine_full.json": {
        "label": "Wine Reviews",
        "output_prefix": "wine",
        "task": "reg",
        "y_levels": None,
    },
    "mimic_full.json": {
        "label": "MIMIC-IV",
        "output_prefix": "mimic",
        "task": "cls",
        "y_levels": "2",
    },
}

VARIANT_NOTES = {
    "tabular-only": (
        "Uses raw tabular VAE embeddings without text or alignment. "
        "Validation is evaluated before the final test split."
    ),
    "no-diffusion": (
        "Fits the alignment module, then trains an MLP on the aligned tabular "
        "and text embeddings instead of a diffusion model."
    ),
    "no-response-awareness": (
        "Fits alignment with the supervised response loss disabled and the "
        "alignment-loss weight fixed at 1. Validation is evaluated before test."
    ),
    "no-alignment": (
        "Trains diffusion directly on the raw tabular VAE and text embeddings. "
        "Validation is evaluated before the final test split."
    ),
}


def parse_override(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--set must have the form key=value")
    key, item = value.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("override key cannot be empty")
    return key, item


def expand(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        previous = value
        for _ in range(10):
            current = previous.format_map(variables)
            if current == previous:
                return current
            previous = current
        raise ValueError(f"placeholder expansion did not converge: {value!r}")
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, variables) for key, item in value.items()}
    return value


def command_line(argv: list[str]) -> str:
    return shlex.join(argv)


def command(config: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return next(item for item in config["commands"] if item["name"] == name)
    except StopIteration as error:
        raise ValueError(f"workflow has no {name!r} stage") from error


def remove_option(argv: list[str], option: str, value_count: int = 1) -> None:
    if option not in argv:
        return
    index = argv.index(option)
    del argv[index:index + value_count + 1]


def set_option(argv: list[str], option: str, value: str) -> None:
    if option in argv:
        argv[argv.index(option) + 1] = value
    else:
        argv.extend([option, value])


def raw_text_path(config: dict[str, Any]) -> str:
    dataset_key = (
        "text_embedding_dataset"
        if "text_embedding_dataset" in config.get("variables", {})
        else "dataset"
    )
    return f"{{embedding_root}}/{{{dataset_key}}}_text_clip_embd"


def apply_prediction_interval_variant(
    config: dict[str, Any], variant: str
) -> dict[str, Any]:
    if variant == "full":
        return config
    if variant != "tabular-only":
        raise ValueError(
            "wine_prediction_interval.json supports only the full and "
            "tabular-only variants"
        )
    config["name"] = "Wine Reviews tabular-only prediction-interval workflow"
    config["protocol_note"] = (
        "Uses the trained Wine Reviews tabular-only diffusion model. "
        "Validation residuals calibrate the intervals before test evaluation."
    )
    config["variables"]["diffusion_dir"] = (
        "{artifacts_root}/wine_tabular_only/diffusion"
    )
    config["variables"]["run_root"] = (
        "{artifacts_root}/wine_tabular_only_prediction_interval"
    )
    return config


def apply_variant(
    raw_config: dict[str, Any], config_path: Path, variant: str
) -> dict[str, Any]:
    config = copy.deepcopy(raw_config)
    if config_path.name == "wine_prediction_interval.json":
        return apply_prediction_interval_variant(config, variant)
    if variant == "full":
        return config

    workflow = DATASET_WORKFLOWS.get(config_path.name)
    if workflow is None:
        raise ValueError(
            f"{variant!r} is supported only for petfinder_full.json, "
            "wine_full.json, mimic_full.json, and the tabular-only Wine "
            "prediction-interval workflow"
        )

    variant_slug = variant.replace("-", "_")
    config["name"] = f"{workflow['label']} {variant} ablation workflow"
    config["protocol_note"] = VARIANT_NOTES[variant]
    config["variables"]["run_root"] = (
        f"{{artifacts_root}}/{workflow['output_prefix']}_{variant_slug}"
    )
    if config_path.name == "mimic_full.json":
        set_option(
            command(config, "evaluate_validation_threshold")["argv"],
            "--model-name",
            variant_slug,
        )

    if variant == "no-response-awareness":
        argv = command(config, "alignment_train")["argv"]
        set_option(argv, "--weight_clip_loss", "1.0")
        set_option(argv, "--weight_sup_loss", "0.0")
        remove_option(argv, "--y_class_weight", value_count=2)
        return config

    if variant == "no-diffusion":
        alignment_commands = [
            command(config, "alignment_train"),
            command(config, "alignment_transform"),
        ]
        mlp_argv = [
            "{python}",
            "src/alignment/mlp_pm_ablation_torch.py",
            "--data-path", "{data_root}/{dataset}",
            "--alignment-path", "{aligned_embedding_dir}",
            "--output-dir", "{run_root}/mlp",
            "--task", workflow["task"],
            "--use-tab",
            "--use-text",
            "--text-pooling", "mean",
        ]
        if workflow["y_levels"] is not None:
            mlp_argv.extend(["--y-levels", workflow["y_levels"]])
        config["commands"] = [
            *alignment_commands,
            {
                "name": "mlp_train_evaluate",
                "cwd": "{repo}",
                "argv": mlp_argv,
            },
        ]
        return config

    config["commands"] = [
        item
        for item in config["commands"]
        if item["name"] not in {"alignment_train", "alignment_transform"}
    ]
    diffusion_argv = command(config, "diffusion_train")["argv"]
    remove_option(diffusion_argv, "--emb_alignment_path")
    remove_option(diffusion_argv, "--alignment_mix_alpha")
    set_option(
        diffusion_argv,
        "--vae_dat_path",
        "{pretrained_root}/{dataset}/tabular_vae",
    )

    if variant == "tabular-only":
        set_option(diffusion_argv, "--use_tab", "true")
        set_option(diffusion_argv, "--use_text", "false")
        set_option(diffusion_argv, "--use_image", "false")
        set_option(diffusion_argv, "--normalize_tab_cond", "false")
        remove_option(diffusion_argv, "--text_pooling")
        remove_option(diffusion_argv, "--normalize_text_tokens")
        remove_option(diffusion_argv, "--normalize_text_pooled")
    elif variant == "no-alignment":
        set_option(diffusion_argv, "--use_tab", "true")
        set_option(diffusion_argv, "--use_text", "true")
        set_option(diffusion_argv, "--use_image", "false")
        set_option(diffusion_argv, "--text_path", raw_text_path(config))
    else:
        raise ValueError(f"unknown variant: {variant}")

    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="full",
        help="run the full method or a component ablation",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], type=parse_override)
    parser.add_argument("--only", action="append", default=[], help="run only the named stage; repeat as needed")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = apply_variant(
        json.loads(config_path.read_text(encoding="utf-8")),
        config_path,
        args.variant,
    )
    variables = {
        "repo": str(REPO_ROOT),
        "python": sys.executable,
        **{key: str(value) for key, value in config.get("variables", {}).items()},
        **dict(args.overrides),
    }
    variables = expand(variables, variables)

    commands = config.get("commands", [])
    if not commands:
        raise ValueError(f"{config_path} has no commands")

    if args.list:
        for item in commands:
            print(item["name"])
        return

    selected = set(args.only)
    known = {item["name"] for item in commands}
    unknown = selected - known
    if unknown:
        raise ValueError(f"unknown stages: {', '.join(sorted(unknown))}")

    print(f"workflow: {config.get('name', config_path.stem)}")
    if config.get("protocol_note"):
        print(f"protocol: {config['protocol_note']}")

    for raw in commands:
        if selected and raw["name"] not in selected:
            continue
        item = expand(raw, variables)
        argv = [str(part) for part in item["argv"]]
        cwd = Path(item.get("cwd", REPO_ROOT)).resolve()
        env = os.environ.copy()
        env.update({key: str(value) for key, value in item.get("env", {}).items()})

        print(f"\n[{item['name']}]")
        print(f"cwd: {cwd}")
        print(command_line(argv))
        if args.dry_run:
            continue
        if not cwd.is_dir():
            raise FileNotFoundError(f"working directory does not exist: {cwd}")
        subprocess.run(argv, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    main()
