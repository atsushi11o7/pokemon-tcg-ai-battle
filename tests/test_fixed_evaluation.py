import random
import unittest
from unittest.mock import patch

from evaluation.match_runner import evaluate_fixed_matchups
from training.common.parallel import build_fixed_matchups


def deck_agent_factory(deck: list[int]):
    return lambda _obs: deck


class FixedEvaluationTest(unittest.TestCase):
    def test_matchup_uses_same_seed_and_swaps_seats_and_decks(self) -> None:
        deck_a = [1] * 60
        deck_b = [2] * 60
        calls = []

        def fake_match(agent_left, agent_right):
            calls.append((agent_left({})[0], agent_right({})[0], random.random()))
            return 1.0, -1.0

        with patch("evaluation.match_runner.play_one_match", side_effect=fake_match):
            result = evaluate_fixed_matchups(
                deck_agent_factory,
                deck_agent_factory,
                [(deck_a, deck_b)],
                seed=123,
            )

        self.assertEqual([(a, b) for a, b, _ in calls], [(1, 2), (2, 1), (2, 1), (1, 2)])
        self.assertEqual(calls[0][2], calls[1][2])
        self.assertEqual(calls[2][2], calls[3][2])
        self.assertEqual(result, {"wins": 2, "losses": 2, "draws": 0, "games": 4, "win_rate": 0.5})

    def test_plan_requires_complete_four_game_blocks(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple of 4"):
            build_fixed_matchups("generalist", [], [[1] * 60], 10, 0)

    def test_asymmetric_plan_uses_fixed_deck_against_sampled_opponents(self) -> None:
        fixed_deck = [9] * 60
        opponent_deck = [2] * 60

        matchups = build_fixed_matchups("asymmetric", fixed_deck, [opponent_deck], 8, 0)

        self.assertEqual(matchups, [(fixed_deck, opponent_deck)] * 2)


if __name__ == "__main__":
    unittest.main()
