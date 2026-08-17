import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.common.deck_pool import WeightedDeckPool, load_weighted_deck_pool
from training.common.selfplay_modes import (
    fixed_deck_seat_for_game,
    pick_decks_and_collect_seats,
)


def snapshot() -> dict:
    return {
        "builtAt": "2026-08-06T00:00:00+00:00",
        "learnerMixture": {"coverage": 1.0},
        "opponentMixture": {"meta": 1.0},
        "decks": [
            {
                "deckHash": "coverage",
                "cards": [1] * 60,
                "weights": {"coverage": 1.0, "meta": 0.0},
            },
            {
                "deckHash": "meta",
                "cards": [2] * 60,
                "weights": {"coverage": 0.0, "meta": 1.0},
            },
        ],
    }


class DeckSamplingTest(unittest.TestCase):
    def test_generalist_uses_separate_learner_and_opponent_distributions(self) -> None:
        pool = WeightedDeckPool(snapshot())

        with patch("training.common.selfplay_modes.random.randrange", return_value=0):
            decks, collect_seats = pick_decks_and_collect_seats("generalist", [9] * 60, pool)

        self.assertEqual(decks, [[1] * 60, [2] * 60])
        self.assertEqual(collect_seats, {0, 1})

    def test_generalist_randomly_swaps_sampling_roles_between_seats(self) -> None:
        pool = WeightedDeckPool(snapshot())

        with patch("training.common.selfplay_modes.random.randrange", return_value=1):
            decks, collect_seats = pick_decks_and_collect_seats("generalist", [9] * 60, pool)

        self.assertEqual(decks, [[2] * 60, [1] * 60])
        self.assertEqual(collect_seats, {0, 1})

    def test_asymmetric_places_fixed_deck_in_requested_seat_and_collects_both(self) -> None:
        fixed_deck = [9] * 60
        opponent_pool = [[2] * 60]

        first_decks, first_collect_seats = pick_decks_and_collect_seats(
            "asymmetric", fixed_deck, opponent_pool, fixed_deck_seat=0
        )
        second_decks, second_collect_seats = pick_decks_and_collect_seats(
            "asymmetric", fixed_deck, opponent_pool, fixed_deck_seat=1
        )

        self.assertEqual(first_decks, [fixed_deck, [2] * 60])
        self.assertEqual(second_decks, [[2] * 60, fixed_deck])
        self.assertEqual(first_collect_seats, {0, 1})
        self.assertEqual(second_collect_seats, {0, 1})

    def test_asymmetric_rejects_invalid_fixed_deck_seat(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed_deck_seat"):
            pick_decks_and_collect_seats(
                "asymmetric",
                [9] * 60,
                [[2] * 60],
                fixed_deck_seat=2,
            )

    def test_asymmetric_alternates_fixed_deck_seat_by_game_index(self) -> None:
        self.assertEqual(
            [fixed_deck_seat_for_game("asymmetric", index) for index in range(6)],
            [0, 1, 0, 1, 0, 1],
        )
        self.assertIsNone(fixed_deck_seat_for_game("generalist", 1))

    def test_loader_returns_deck_pool_directly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            path.write_text(json.dumps(snapshot()), encoding="utf-8")

            pool = load_weighted_deck_pool(path)

        self.assertIsInstance(pool, WeightedDeckPool)
        self.assertEqual(len(pool), 2)

    def test_weighted_pool_remains_list_compatible(self) -> None:
        pool = WeightedDeckPool(snapshot())

        self.assertEqual(len(pool), 2)
        self.assertEqual(pool[0], [1] * 60)


if __name__ == "__main__":
    unittest.main()
