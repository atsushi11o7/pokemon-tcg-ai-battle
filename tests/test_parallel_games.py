import os
import signal
import time
import unittest

from training.common.parallel_games import run_parallel_games


def initialize() -> None:
    pass


def mixed_task(game_index: int) -> int:
    if game_index == 1:
        raise ValueError("expected test error")
    if game_index == 2:
        time.sleep(2)
    if game_index == 3:
        os.kill(os.getpid(), signal.SIGSEGV)
    return game_index * 10


class ParallelGamesTest(unittest.TestCase):
    def test_recovers_from_exception_timeout_and_signal(self) -> None:
        results = {}
        failures = {}

        completed, failed = run_parallel_games(
            num_games=6,
            num_workers=2,
            initializer=initialize,
            initargs=(),
            task=mixed_task,
            game_timeout_seconds=0.3,
            round_timeout_seconds=10,
            on_result=lambda index, result: results.__setitem__(index, result),
            on_failure=lambda index, reason: failures.__setitem__(index, reason),
        )

        self.assertEqual(completed, 3)
        self.assertEqual(failed, 3)
        self.assertEqual(results, {0: 0, 4: 40, 5: 50})
        self.assertIn("ValueError", failures[1])
        self.assertIn("timed out", failures[2])
        self.assertIn("code -11", failures[3])

    def test_zero_games_does_not_start_workers(self) -> None:
        completed, failed = run_parallel_games(
            num_games=0,
            num_workers=2,
            initializer=initialize,
            initargs=(),
            task=mixed_task,
            game_timeout_seconds=1,
            round_timeout_seconds=1,
            on_result=lambda _index, _result: None,
            on_failure=lambda _index, _reason: None,
        )
        self.assertEqual((completed, failed), (0, 0))


if __name__ == "__main__":
    unittest.main()
