import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from wine_pi import common, prepare


class WinePIDataTest(unittest.TestCase):
    def test_disjoint_random_splits_train_only_preprocessing_and_unchanged_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, run = base / "source", base / "run"
            source.mkdir()
            (run / "shared").mkdir(parents=True)
            protocol = copy.deepcopy(common.PROTOCOL)
            protocol.update(n_train=40, n_val=10, n_cal=10, n_test=3)
            for split, n in [("train", 60), ("test", 3)]:
                np.save(source / f"X_num_{split}.npy", np.arange(n, dtype=float)[:, None])
                np.save(source / f"X_cat_{split}.npy", np.array([[f"unique-{i}"] for i in range(n)]))
                np.save(source / f"y_{split}.npy", np.arange(n, dtype=float)[:, None])
                (source / f"text_{split}.json").write_text(json.dumps([str(i) for i in range(n)]))
            values = dict(ROOT=run, TRAIN_SOURCE=source, TEST_SOURCE=source, PROTOCOL=protocol)
            with patch.multiple(common, **values), patch.multiple(prepare, **values):
                prepare.data()
                indices = {s: np.load(run / "data" / f"source_indices_{s}.npy") for s in ["train", "val", "cal"]}
                self.assertEqual(len(np.unique(np.concatenate(list(indices.values())))), 60)
                np.testing.assert_array_equal(indices["cal"], np.random.default_rng(20260921).permutation(60)[:10])
                prepare.preprocess_fit()
                model = prepare.joblib.load(run / "preprocessed/fitted_train_only.joblib")
                self.assertAlmostEqual(model["response_scaler"].mean_[0], indices["train"].mean())
                # Even calibration outcomes may be unavailable during feature transformation.
                (run / "data/y_cal.npy").unlink()
                prepare.transformed("cal")
                cal = np.load(run / "preprocessed/cat_cal.npy")
                self.assertTrue((cal == 40).all())
                self.assertEqual(model["categories"], [41])
                self.assertTrue(all(v["copy_identical"] for v in common.check_test().values()))
                for name in ["X_num_test.npy", "X_cat_test.npy", "y_test.npy", "text_test.json"]:
                    self.assertEqual((source / name).read_bytes(), (run / "data" / name).read_bytes())
                np.save(run / "data/y_test.npy", np.zeros(3))
                with self.assertRaises(AssertionError):
                    common.check_test()


if __name__ == "__main__":
    unittest.main()
