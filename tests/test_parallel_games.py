import os
import signal
import tempfile
import time
import unittest
from pathlib import Path

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


def failing_initializer() -> None:
    raise RuntimeError("initializer always fails")


def unused_task(game_index: int) -> int:
    return game_index


class WorkerStartupFailureTest(unittest.TestCase):
    """初期化が必ず失敗する状況で、ワーカーを無限に作り直さないこと。

    以前はラウンド上限まで再生成し続け、8ワーカー・上限2時間の設定では
    2時間CPUを焼いたうえで0サンプルを返していた(終了コードは0なので
    CLIの再起動も働かない)。
    """

    def test_gives_up_instead_of_respawning_forever(self) -> None:
        started = time.monotonic()
        with self.assertRaises(RuntimeError) as caught:
            run_parallel_games(
                num_games=50,
                num_workers=2,
                initializer=failing_initializer,
                initargs=(),
                task=unused_task,
                game_timeout_seconds=5,
                round_timeout_seconds=60,
                on_result=lambda index, result: None,
                on_failure=lambda index, reason: None,
            )
        self.assertIn("before receiving a game", str(caught.exception))
        self.assertLess(time.monotonic() - started, 30)


class CompletionRateGateTest(unittest.TestCase):
    """大半が失敗したラウンドで学習を進めないこと。"""

    def test_low_completion_rate_raises(self) -> None:
        from training.common.parallel import run_selfplay_round

        with self.assertRaises(RuntimeError) as caught:
            run_selfplay_round(
                algorithm="test",
                round_num=1,
                mode="generalist",
                num_games=10,
                num_workers=2,
                initializer=initialize,
                initargs=(),
                task=always_failing_task,
                game_timeout_seconds=5,
                round_timeout_seconds=60,
                event_log_path=Path(tempfile.mkdtemp()) / "events.jsonl",
            )
        self.assertIn("refusing to train", str(caught.exception))


def always_failing_task(game_index: int):
    raise ValueError("boom")
