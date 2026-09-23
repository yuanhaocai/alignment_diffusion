#!/usr/bin/env python3
"""Run the recorded four-split Wine PI protocol without server-specific paths."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="original 61813-row non-test pool in *_train.npy, plus text_train.json")
    parser.add_argument("--test-source-dir", type=Path,
                        help="directory with the unchanged four *_test files; defaults to source-dir")
    parser.add_argument("--text-dir", type=Path, required=True,
                        help="frozen original-pool CLIP embeddings under train/ and test/")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["both", "full", "tabular-only"], default="both")
    parser.add_argument("--stage", choices=["all", "prepare", "train", "sample", "evaluate"], default="all")
    parser.add_argument("--gpu", default="0", help="one CUDA device; variants run sequentially and share VAE")
    args = parser.parse_args()
    paths = {"source_dir": str(args.source_dir.resolve()),
             "test_source_dir": str((args.test_source_dir or args.source_dir).resolve()),
             "text_dir": str(args.text_dir.resolve())}
    root = args.run_dir.resolve()
    for name, value in paths.items():
        source = Path(value)
        if root == source or root in source.parents or source in root.parents:
            parser.error(f"run-dir must be separate from {name}")
    root.mkdir(parents=True, exist_ok=True)
    # Lock before writing protocol/provenance or touching any generated output.
    import fcntl
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path_record = root / "input_paths.json"
        if path_record.exists() and json.loads(path_record.read_text()) != paths:
            raise ValueError("input paths changed: choose a fresh run directory")
        path_record.write_text(json.dumps(paths, indent=2) + "\n")
        os.environ.update(WINE_RUN_ROOT=str(root), WINE_POOL_DIR=paths["source_dir"],
                          WINE_TEST_DIR=paths["test_source_dir"], WINE_TEXT_DIR=paths["text_dir"])
        from wine_pi.pipeline import run
        variants = ["full", "tabular_only"] if args.variant == "both" else [args.variant.replace("-", "_")]
        run(args.stage, variants, args.gpu)


if __name__ == "__main__":
    main()
