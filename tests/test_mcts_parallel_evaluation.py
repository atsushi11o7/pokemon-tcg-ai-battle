import unittest
from pathlib import Path
from unittest.mock import patch

from training.mcts.parallel_evaluation import (
    build_evaluation_games,
    evaluate_mcts_networks_parallel,
)


class FakeNetwork:
    def __init__(self, name: str) -> None:
        self.name = name

    def state_dict(self) -> dict:
        return {"name": self.name}


class MctsParallelEvaluationTest(unittest.TestCase):
    def test_plan_preserves_deck_swaps_seat_swaps_and_pair_seed(self) -> None:
        deck_a = [1] * 60
        deck_b = [2] * 60

        games = build_evaluation_games([(deck_a, deck_b)], num_opponents=2, seed=123)

        self.assertEqual(len(games), 8)
        for offset in (0, 4):
            self.assertEqual(
                [
                    (
                        game.candidate_deck[0],
                        game.opponent_deck[0],
                        game.candidate_seat,
                        game.seed,
                    )
                    for game in games[offset : offset + 4]
                ],
                [
                    (1, 2, 0, 123),
                    (1, 2, 1, 123),
                    (2, 1, 0, 124),
                    (2, 1, 1, 124),
                ],
            )

    def test_parallel_results_are_grouped_by_opponent_and_track_failures(self) -> None:
        def fake_parallel_games(**kwargs):
            games = kwargs["initargs"][-1].games
            for game_index, game in enumerate(games):
                if game_index == 0:
                    kwargs["on_failure"](game_index, "expected failure")
                    continue
                reward = 1.0 if game.candidate_seat == 0 else -1.0
                kwargs["on_result"](game_index, (game.opponent_index, reward))
            return len(games) - 1, 1

        with patch(
            "training.common.parallel_evaluation.run_parallel_games",
            side_effect=fake_parallel_games,
        ):
            results = evaluate_mcts_networks_parallel(
                FakeNetwork("candidate"),
                [("best", FakeNetwork("best")), ("random", None)],
                [[1] * 60, [2] * 60],
                [([1] * 60, [2] * 60)],
                search_count=50,
                sampling_snapshot=Path("snapshot.json"),
                seed=123,
                num_workers=7,
                game_timeout_seconds=300,
                round_timeout_seconds=900,
            )

        self.assertEqual(
            results["best"],
            {"wins": 1, "losses": 2, "draws": 0, "games": 3, "failed": 1, "win_rate": 1 / 3},
        )
        self.assertEqual(
            results["random"],
            {"wins": 2, "losses": 2, "draws": 0, "games": 4, "failed": 0, "win_rate": 0.5},
        )


if __name__ == "__main__":
    unittest.main()
