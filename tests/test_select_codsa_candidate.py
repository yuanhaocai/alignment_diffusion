import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_codsa_candidate",
    ROOT / "scripts" / "select_codsa_candidate.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CoDSASelectionTest(unittest.TestCase):
    def test_selection_reads_only_selected_candidates_test_probabilities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            np.save(data / "y_train.npy", np.array([[0], [1], [0], [1]]))
            np.save(data / "y_val.npy", np.array([[0], [1], [0], [1]]))
            np.save(data / "y_test.npy", np.array([[0], [1], [0], [1]]))

            np.save(root / "weak_val.npy", np.array([0.5, 0.5, 0.5, 0.5]))
            # Deliberately do not create weak_test.npy. Selection must not read it.
            np.save(root / "strong_val.npy", np.array([0.1, 0.9, 0.2, 0.8]))
            np.save(root / "strong_test.npy", np.array([0.2, 0.8, 0.1, 0.9]))
            manifest = {
                "candidates": [
                    {
                        "id": "weak",
                        "parameters": {"r": 0.1, "alpha": 0.5, "m": 10},
                        "validation_proba": "weak_val.npy",
                        "test_proba": "weak_test.npy",
                    },
                    {
                        "id": "strong",
                        "parameters": {"r": 0.2, "alpha": 0.5, "m": 20},
                        "validation_proba": "strong_val.npy",
                        "test_proba": "strong_test.npy",
                    },
                ]
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = MODULE.run_selection(manifest_path, data)

            self.assertEqual(result["selected_candidate"]["id"], "strong")
            self.assertEqual(result["selected_candidate"]["parameters"]["m"], 20)
            self.assertEqual(result["test"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
