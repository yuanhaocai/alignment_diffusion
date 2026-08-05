from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class ObservedResidualPITest(unittest.TestCase):
    def test_known_interval_and_crps_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            np.save(data / "y_train.npy", np.arange(20, dtype=float)[:, None])
            np.save(data / "y_val.npy", np.array([0.0, 1.0, 2.0])[:, None])
            np.save(data / "y_test.npy", np.array([1.0, 3.0])[:, None])

            val_bank = np.array(
                [
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                ]
            )
            test_bank = np.array(
                [
                    [1.0, 3.0],
                    [1.0, 3.0],
                ]
            )
            np.save(root / "val.npy", val_bank)
            np.save(root / "test.npy", test_bank)
            output = root / "summary.json"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "observed_residual_pi.py"),
                    "--data-dir",
                    str(data),
                    "--val-samples",
                    str(root / "val.npy"),
                    "--test-samples",
                    str(root / "test.npy"),
                    "--m",
                    "2",
                    "--alpha",
                    "0.05",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output.read_text())
            self.assertEqual(result["coverage"], 1.0)
            self.assertEqual(result["width"], 0.0)
            self.assertEqual(result["empirical_residual_crps"], 0.0)


if __name__ == "__main__":
    unittest.main()

