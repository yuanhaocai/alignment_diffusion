from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_experiment",
    ROOT / "scripts" / "run_experiment.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class RunExperimentTest(unittest.TestCase):
    @staticmethod
    def load(name: str) -> dict:
        return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))

    @staticmethod
    def stage(config: dict, name: str) -> dict:
        return next(item for item in config["commands"] if item["name"] == name)

    def test_nested_placeholders_and_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "workflow.json"
            config.write_text(
                json.dumps(
                    {
                        "name": "test",
                        "variables": {
                            "output_root": "{repo}/artifacts",
                            "output": "{output_root}/trial",
                        },
                        "commands": [
                            {
                                "name": "stage",
                                "cwd": "{repo}",
                                "argv": ["{python}", "-c", "print('ok')", "{output}"],
                            }
                        ],
                    }
                )
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_experiment.py"),
                    str(config),
                    "--dry-run",
                    "--set",
                    f"output_root={temporary}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(str(Path(temporary) / "trial"), result.stdout)

    def test_all_dataset_ablation_variants(self) -> None:
        for name in ("petfinder_full.json", "wine_full.json", "mimic_full.json"):
            path = ROOT / "configs" / name
            raw = self.load(name)

            tabular = RUNNER.apply_variant(raw, path, "tabular-only")
            self.assertNotIn(
                "alignment_train", {item["name"] for item in tabular["commands"]}
            )
            tabular_train = self.stage(tabular, "diffusion_train")["argv"]
            self.assertIn("--vae_dat_path", tabular_train)
            self.assertEqual(
                tabular_train[tabular_train.index("--use_text") + 1], "false"
            )

            no_alignment = RUNNER.apply_variant(raw, path, "no-alignment")
            no_alignment_train = self.stage(no_alignment, "diffusion_train")["argv"]
            self.assertNotIn("--emb_alignment_path", no_alignment_train)
            self.assertIn("--vae_dat_path", no_alignment_train)
            self.assertIn("--text_path", no_alignment_train)

            no_response = RUNNER.apply_variant(raw, path, "no-response-awareness")
            alignment = self.stage(no_response, "alignment_train")["argv"]
            self.assertEqual(
                alignment[alignment.index("--weight_clip_loss") + 1], "1.0"
            )
            self.assertEqual(
                alignment[alignment.index("--weight_sup_loss") + 1], "0.0"
            )

            no_diffusion = RUNNER.apply_variant(raw, path, "no-diffusion")
            self.assertEqual(
                [item["name"] for item in no_diffusion["commands"]],
                ["alignment_train", "alignment_transform", "mlp_train_evaluate"],
            )

    def test_prediction_interval_variants_retrain_with_shared_split_and_vae(self) -> None:
        name = "wine_prediction_interval.json"
        config = RUNNER.apply_variant(
            self.load(name), ROOT / "configs" / name, "tabular-only"
        )
        self.assertEqual(config["variables"]["variant"], "tabular-only")
        self.assertNotIn("diffusion_dir", config["variables"])
        self.assertEqual([c["name"] for c in config["commands"]], [
            "prepare_interval_data", "train_interval_model", "sample_calibration_and_test", "evaluate_interval"])
        full = RUNNER.apply_variant(self.load(name), ROOT / "configs" / name, "full")
        self.assertEqual(full["variables"]["run_root"], config["variables"]["run_root"])
        self.assertEqual(full["variables"]["variant"], "full")
        with self.assertRaises(ValueError):
            RUNNER.apply_variant(self.load(name), ROOT / "configs" / name, "no-alignment")


if __name__ == "__main__":
    unittest.main()
