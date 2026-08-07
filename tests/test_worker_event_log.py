import json
import tempfile
import unittest
from pathlib import Path

from training.common.parallel_games import run_parallel_games


def initialize() -> None:
    pass


def task(game_index: int) -> int:
    return game_index


class WorkerEventLogTest(unittest.TestCase):
    def test_log_contains_game_pid_rss_and_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "worker_events.jsonl"
            completed, failed = run_parallel_games(
                num_games=1,
                num_workers=1,
                initializer=initialize,
                initargs=(),
                task=task,
                game_timeout_seconds=3,
                round_timeout_seconds=5,
                on_result=lambda _index, _result: None,
                on_failure=lambda _index, _reason: None,
                event_log_path=log_path,
                event_context={"algorithm": "test", "round": 7},
            )
            events = [json.loads(line) for line in log_path.read_text().splitlines()]

        self.assertEqual((completed, failed), (1, 0))
        completed_event = next(event for event in events if event["event"] == "game_completed")
        stopped_event = next(event for event in events if event["event"] == "worker_stopped")
        self.assertEqual(completed_event["game_index"], 0)
        self.assertIsInstance(completed_event["pid"], int)
        self.assertIsInstance(completed_event["rss_bytes"], int)
        self.assertEqual(stopped_event["exit_code"], 0)
        self.assertEqual(stopped_event["algorithm"], "test")
        self.assertEqual(stopped_event["round"], 7)


if __name__ == "__main__":
    unittest.main()
