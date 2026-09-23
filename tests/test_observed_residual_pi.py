from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from observed_residual_pi import residual_quantiles, empirical_crps, evaluate


class ObservedResidualPITest(unittest.TestCase):
    def test_known_interval_and_crps_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            np.save(data / "y_train.npy", np.arange(20, dtype=float)[:, None])
            np.save(data / "y_cal.npy", np.arange(39, dtype=float)[:, None])
            # Deliberately incompatible validation outcomes: the default must read cal.
            np.save(data / "y_val.npy", np.array([9999.0])[:, None])
            np.save(data / "y_test.npy", np.array([1.0, 3.0])[:, None])

            val_bank = np.tile(np.arange(39, dtype=float), (2, 1))
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
                    "--cal-samples",
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
            self.assertEqual(result["calibration_split"], "cal")
            self.assertEqual((result["k_lower"], result["k_upper"]), (1, 39))

    def test_corrected_ranks_and_unbounded_small_samples(self):
        lo, hi, kl, ku = residual_quantiles(np.arange(7726))
        self.assertEqual((lo, hi, kl, ku), (192, 7533, 193, 7534))
        lo, hi, kl, ku = residual_quantiles(np.arange(3))
        self.assertEqual((lo, hi, kl, ku), (-np.inf, np.inf, 0, 4))
        # Integral decimal ranks must not shift due to binary floating point.
        self.assertEqual(residual_quantiles(np.arange(199), .14)[2:], (14, 186))

    def test_rank_coverage_by_exhaustive_exchangeable_ranks(self):
        population = np.arange(10.)
        covered = []
        for heldout in range(10):
            lo, hi, _, _ = residual_quantiles(np.delete(population, heldout), .4)
            covered.append(lo <= population[heldout] <= hi)
        self.assertEqual(sum(covered), 6)

    def test_crps_matches_pairwise_definition_including_ties(self):
        residuals = np.array([-3., -.2, -.2, 1., 4.])
        y, center = np.array([-8., -.2, 0., 9.]), np.array([1., 3., -2., .5])
        expected = np.abs(residuals[:, None] + center - y).mean(0)
        expected -= .5 * np.abs(residuals[:, None] - residuals).mean()
        np.testing.assert_allclose(empirical_crps(y, center, residuals), expected, atol=1e-12)

    def test_endpoint_change_preserves_crps_and_fixed_reporting_scale(self):
        residuals = np.arange(5.)
        test_y = np.array([1., 3., 7.])
        kwargs = dict(m=2, alpha=.5, metric_scale=2.)
        revised, obs = evaluate(residuals, np.zeros((2, 5)), test_y, np.zeros((2, 3)), **kwargs)
        linear, _ = evaluate(residuals, np.zeros((2, 5)), test_y, np.zeros((2, 3)), endpoint_rule="linear", **kwargs)
        self.assertEqual(revised["width"], 2.)
        self.assertEqual(linear["width"], 1.)
        self.assertEqual(revised["crps"], linear["crps"])
        np.testing.assert_allclose(obs["crps_paper_scale"], obs["crps_raw"] / 2)

    def test_invalid_input_rejected(self):
        for alpha in [0, 1, -1, np.nan]:
            with self.assertRaises(ValueError):
                residual_quantiles([1, 2], alpha)
        with self.assertRaises(ValueError):
            residual_quantiles([])
        with self.assertRaises(ValueError):
            evaluate([1], np.ones((1, 1)), [1], np.ones((1, 1)), m=2)


if __name__ == "__main__":
    unittest.main()
