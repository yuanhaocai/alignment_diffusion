"""Optional CUDA integration smoke test with synthetic data, never paper data.

From the repository root:
    PYTHONPATH=src:scripts python tests/smoke_wine_pi_gpu.py --output-dir /tmp/fresh-pi-smoke
Uses shortened training/sampling only inside this process, not the release protocol.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
from wine_pi import common, prepare, train, evaluate, pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    base = parser.parse_args().output_dir.resolve()
    base.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("This optional integration test requires CUDA")
    source, root, text = base / "source", base / "run", base / "text"
    source.mkdir()
    for directory in ["data", "shared", "full", "tabular_only"]:
        (root / directory).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(73)
    for split, n in [("train", 96), ("test", 16)]:
        (text / split).mkdir(parents=True)
        np.save(source / f"X_num_{split}.npy", rng.normal(size=(n, 1)).astype(np.float32))
        np.save(source / f"X_cat_{split}.npy", rng.integers(0, 3, size=(n, 2)).astype(str))
        np.save(source / f"y_{split}.npy", (88 + rng.normal(size=(n, 1))).astype(np.float32))
        (source / f"text_{split}.json").write_text(json.dumps([f"Synthetic wine {i}" for i in range(n)]))
        for i in range(n):
            np.save(text / split / f"{i}.npy", rng.normal(size=(1, 77, 768)).astype(np.float32))
    protocol = copy.deepcopy(common.PROTOCOL)
    protocol.update(n_train=64, n_val=16, n_cal=16, n_test=16, M=2, alpha=.5,
                    validation_selection_M=2, sampling_steps=3, diffusion_candidates=["model.pt"])
    protocol["vae"].update(epochs=2, batch_size=32)
    protocol["alignment"].update(epochs=1, batch_size=16)
    for variant in ["full", "tabular_only"]:
        protocol[variant].update(epochs=2, batch_size=32, dim_t=32)
    for module in [common, prepare, train, evaluate, pipeline]:
        module.ROOT = root
        module.PROTOCOL = protocol
        module.TRAIN_SOURCE = source
        module.TEST_SOURCE = source
        module.TEXT = text
    prepare.data()
    prepare.preprocess_fit()
    prepare.pack_text(["train", "val"])
    train.selfcheck()
    train.vae()
    train.encode(["train", "val"])
    train.align()
    train.project(["train", "val"])
    for variant in ["full", "tabular_only"]:
        train.diffusion(variant)
        train.select(variant)
        pipeline.freeze(variant)
    # Feature processing and all later draws happen after selection.
    for split in ["cal", "test"]:
        prepare.transformed(split)
    prepare.pack_text(["cal", "test"])
    train.encode(["cal", "test"])
    train.project(["cal", "test"])
    for variant in ["full", "tabular_only"]:
        for split in ["cal", "test"]:
            # Fail if sampling attempts to read held-out outcomes.
            label_path = root / "data" / f"y_{split}.npy"
            hidden = label_path.with_suffix(".hidden")
            label_path.rename(hidden)
            try:
                train.sample(variant, split, M=2)
            finally:
                hidden.rename(label_path)
        sys.argv = ["evaluate", "--variant", variant]
        evaluate.main()
    result = json.loads((root / "results/summary.json").read_text())
    assert set(result["models"]) == {"full", "tabular_only"}
    for v in result["models"].values():
        assert v["revised"]["n_cal"] == 16 and np.isfinite(v["revised"]["crps"])
    common.write_json(base / "passed.json", dict(all_passed=True, synthetic_data_only=True,
        stages=["prepare", "VAE", "alignment", "both diffusion variants", "validation selection",
                "calibration/test sampling without outcomes", "interval evaluation"],
        test_immutability=common.check_test()))
    print("CUDA PI integration smoke test passed")


if __name__ == "__main__":
    main()
