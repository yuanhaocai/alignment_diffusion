#!/usr/bin/env python3
"""Run fast structural and safety checks on the release directory."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pt", ".ckpt", ".npy", ".npz", ".db", ".log"}
SECRET_PATTERNS = [
    re.compile(r"Diffusion15", re.IGNORECASE),
    re.compile(r"BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY"),
]


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    required = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "configs/wine_full.json",
        "configs/mimic_full.json",
        "configs/mimic_codsa_candidates.json",
        "configs/petfinder_full.json",
        "configs/shopee_image_only.json",
        "configs/wine_prediction_interval.json",
        "scripts/run_experiment.py",
        "scripts/pm_diff_train.py",
        "scripts/pm_diff_sample.py",
        "scripts/pm_gen_check.py",
        "scripts/run_tabular_baselines.py",
        "scripts/observed_residual_pi.py",
        "scripts/select_codsa_candidate.py",
        "data/preprocessing/prepare_petfinder.py",
        "data/preprocessing/prepare_shopee.py",
        "data/preprocessing/prepare_wine_reviews.py",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            fail(f"missing {relative}")

    expected_dataset_notes = {
        "mimic_adm_pt_disch_discharge_censored_wval",
        "petfinder_wval",
        "shopee",
        "wine_review3_wval",
    }
    actual_dataset_notes = {
        path.name
        for path in (ROOT / "data" / "dataset_notes").iterdir()
        if path.is_dir()
    }
    if actual_dataset_notes != expected_dataset_notes:
        fail(
            "unexpected dataset-note directories: "
            f"{sorted(actual_dataset_notes ^ expected_dataset_notes)}"
        )

    for path in ROOT.rglob("*"):
        if "__pycache__" in path.parts or path.name == ".DS_Store":
            continue
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            relative = path.relative_to(ROOT)
            fail(f"large/restricted artifact should not be committed: {relative}")

    for path in ROOT.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            fail(f"invalid JSON {path.relative_to(ROOT)}: {error}")

    text_extensions = {".py", ".sh", ".md", ".json", ".yml", ".yaml", ".toml", ".txt", ".csv", ".tex"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_extensions:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                fail(f"possible secret in {path.relative_to(ROOT)}")

    configs = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in (ROOT / "configs").glob("*.json")
    }
    for name, config in configs.items():
        obsolete_variables = {"tabsyn_python", "alignment_python"} & set(
            config.get("variables", {})
        )
        if obsolete_variables:
            fail(f"{name} contains obsolete Python variables: {sorted(obsolete_variables)}")

    for name in ("wine_full.json", "mimic_full.json", "petfinder_full.json"):
        arguments = [
            value
            for command in configs[name]["commands"]
            for value in command["argv"]
        ]
        if "--weight_sup_loss_text" in arguments or "--eval_use_train_set" in arguments:
            fail(f"{name} contains an obsolete or unsafe alignment option")

    wine_commands = {
        command["name"]: command["argv"]
        for command in configs["wine_full.json"]["commands"]
    }
    train_scale = wine_commands["alignment_train"][
        wine_commands["alignment_train"].index("--residual_scale") + 1
    ]
    transform_scale = wine_commands["alignment_transform"][
        wine_commands["alignment_transform"].index("--residual-scale") + 1
    ]
    if train_scale != transform_scale:
        fail("Wine residual scale differs between training and transformation")

    shopee_commands = configs["shopee_image_only.json"]["commands"]
    shopee_sampling = [
        command for command in shopee_commands if command["name"].startswith("sample_")
    ]
    if {command["name"] for command in shopee_sampling} != {
        "sample_validation", "sample_test"
    }:
        fail("Shopee must contain validation and final test sampling stages")
    for command in shopee_sampling:
        repeats = command["argv"][command["argv"].index("--num_repeats") + 1]
        if repeats != "100":
            fail("Shopee sampling must use the paper's M=100")

    for path in (ROOT / "src" / "tabsynfnn").rglob("*.py"):
        if path.name == "__init__.py":
            continue
        if "adapted from amazon-science/TabSyn" not in path.read_text(
            encoding="utf-8", errors="ignore"
        ):
            fail(f"missing TabSyn attribution header in {path.relative_to(ROOT)}")

    print(
        json.dumps(
            {
                "status": "ok",
                "files": sum(path.is_file() for path in ROOT.rglob("*")),
                "python_files": sum(1 for _ in ROOT.rglob("*.py")),
                "dataset_notes": sorted(expected_dataset_notes),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
