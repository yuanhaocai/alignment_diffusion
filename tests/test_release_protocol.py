import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "alignment"))
sys.path.insert(0, str(ROOT / "src"))

from model import TabTextAlign, masked_token_mean  # noqa: E402
from tabsynfnn.tabsyn.latent_utils import decode_nearest_response_prototype  # noqa: E402


class ModelProtocolTest(unittest.TestCase):
    def test_masked_mean_ignores_zero_padding(self):
        text = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [0.0, 0.0]]])
        expected = torch.tensor([[2.0, 4.0]])
        torch.testing.assert_close(masked_token_mean(text), expected)

    def test_response_head_uses_concatenated_modalities(self):
        model = TabTextAlign(
            emb_dim=2,
            num_classes=2,
            y_type="categorical",
            projector_mode="identity",
        )
        self.assertEqual(model.head.in_features, 4)
        tab = torch.tensor([[1.0, 0.0]])
        text = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
        condition = model.response_condition(tab, text)
        torch.testing.assert_close(condition, torch.tensor([[1.0, 0.0, 0.0, 1.0]]))

    def test_categorical_response_uses_nearest_prototype(self):
        samples = torch.tensor([[0.1, 0.0], [0.0, 0.9]])
        prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        decoded = decode_nearest_response_prototype(
            samples,
            prototypes,
            np.array([0, 1]),
            cat_dim=0,
            cat_inverse=lambda values: values,
        )
        np.testing.assert_array_equal(decoded.reshape(-1), np.array([0, 1]))


class ConfigurationProtocolTest(unittest.TestCase):
    @staticmethod
    def load(name):
        return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))

    def test_wine_residual_scale_is_consistent(self):
        commands = {
            command["name"]: command["argv"]
            for command in self.load("wine_full.json")["commands"]
        }
        train = commands["alignment_train"]
        transform = commands["alignment_transform"]
        self.assertEqual(
            train[train.index("--residual_scale") + 1],
            transform[transform.index("--residual-scale") + 1],
        )

    def test_alignment_configs_have_one_joint_supervised_loss(self):
        for name in ("wine_full.json", "mimic_full.json", "petfinder_full.json"):
            arguments = [
                value
                for command in self.load(name)["commands"]
                for value in command["argv"]
            ]
            self.assertIn("--weight_sup_loss", arguments)
            self.assertNotIn("--weight_sup_loss_text", arguments)
            self.assertNotIn("--eval_use_train_set", arguments)

    def test_full_multimodal_configs_use_paper_condition_normalization(self):
        for name in ("wine_full.json", "mimic_full.json", "petfinder_full.json"):
            diffusion = next(
                command["argv"]
                for command in self.load(name)["commands"]
                if command["name"] == "diffusion_train"
            )
            options = dict(zip(diffusion[:-1], diffusion[1:]))
            self.assertEqual(options["--normalize_tab_cond"], "true")
            self.assertEqual(options["--normalize_text_tokens"], "false")
            self.assertEqual(options["--normalize_text_pooled"], "true")

    def test_shopee_has_validation_and_paper_repeat_count(self):
        config = self.load("shopee_image_only.json")
        self.assertEqual(config["variables"]["dataset"], "shopee")
        commands = config["commands"]
        sampling = {
            command["name"]: command["argv"]
            for command in commands
            if command["name"].startswith("sample_")
        }
        self.assertEqual(set(sampling), {"sample_validation", "sample_test"})
        for arguments in sampling.values():
            self.assertEqual(arguments[arguments.index("--num_repeats") + 1], "100")


if __name__ == "__main__":
    unittest.main()
