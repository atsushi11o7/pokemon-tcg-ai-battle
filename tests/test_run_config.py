import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
from training.common.run_config import load_run_config, validate_algorithm  # noqa: E402


class RunConfigTest(unittest.TestCase):
    def test_generalist_configs_select_exactly_one_algorithm(self) -> None:
        ppo = load_run_config(ROOT / "configs" / "ppo_generalist.yaml")
        mcts = load_run_config(ROOT / "configs" / "mcts_generalist.yaml")

        self.assertEqual(ppo.algorithm, "ppo")
        self.assertEqual(mcts.algorithm, "mcts")
        self.assertIsNone(ppo.deck_path)
        self.assertIsNone(mcts.deck_path)
        self.assertEqual(
            mcts.model.initial_checkpoint,
            ROOT / "outputs" / "runs" / "ppo_generalist" / "checkpoints" / "final.pt",
        )

    def test_asymmetric_mode_accepts_a_fixed_deck(self) -> None:
        """generalistと同じ書式のまま、モードとデッキだけ差し替えられること。"""
        raw = self._ppo_config()
        raw["training"]["selfplay_mode"] = "asymmetric"
        raw["training"]["deck_path"] = "decks/candidates/crustle_team3.csv"

        config = self._load_temporary(raw)

        self.assertEqual(config.selfplay_mode, "asymmetric")
        self.assertEqual(config.deck_path, ROOT / "decks" / "candidates" / "crustle_team3.csv")

    def test_architecture_is_not_a_run_config_field(self) -> None:
        raw = self._ppo_config()
        raw["architecture"] = {"d_model": 128}

        with self.assertRaisesRegex(ValueError, "architecture"):
            self._load_temporary(raw)

    def test_fixed_deck_mode_requires_deck_path(self) -> None:
        raw = self._ppo_config()
        raw["training"]["selfplay_mode"] = "mirror"

        with self.assertRaisesRegex(ValueError, "deck_path"):
            self._load_temporary(raw)

    def test_invalid_hyperparameter_is_rejected_before_training(self) -> None:
        raw = self._ppo_config()
        raw["training"]["learning_rate"] = 0

        with self.assertRaisesRegex(ValueError, "learning_rate"):
            self._load_temporary(raw)

    def test_asymmetric_requires_even_games_per_round(self) -> None:
        raw = self._ppo_config()
        raw["training"]["selfplay_mode"] = "asymmetric"
        raw["training"]["deck_path"] = "decks/candidates/crustle_team3.csv"
        raw["training"]["games_per_round"] = 501

        with self.assertRaisesRegex(ValueError, "must be even"):
            self._load_temporary(raw)

    def test_evaluation_games_must_fill_four_game_blocks(self) -> None:
        raw = self._ppo_config()
        raw["training"]["eval_games_per_round"] = 10

        with self.assertRaisesRegex(ValueError, "multiple of 4"):
            self._load_temporary(raw)

    def test_trainer_algorithm_mismatch_is_rejected(self) -> None:
        config = load_run_config(ROOT / "configs" / "ppo_generalist.yaml")

        with self.assertRaisesRegex(ValueError, "mcts"):
            validate_algorithm(config, "mcts")

    def _ppo_config(self) -> dict:
        with (ROOT / "configs" / "ppo_generalist.yaml").open(encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _load_temporary(self, raw: dict):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            return load_run_config(path)


if __name__ == "__main__":
    unittest.main()
