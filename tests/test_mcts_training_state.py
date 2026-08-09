import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

from training.mcts.training_state import (
    restore_training_state,
    save_training_state,
    saved_training_state_round,
)


class FakeNetwork:
    def __init__(self, state=None) -> None:
        self.state = state or {}

    def state_dict(self):
        return self.state


def fake_build(state_dict, assign=False):
    return FakeNetwork(state_dict)


class MctsTrainingStateTest(unittest.TestCase):
    def test_round_trip_replay_and_checkpoint_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            pool_network = FakeNetwork({"weight": 3})
            replay = deque([(2, ["sample-a"]), (3, ["sample-b"])], maxlen=2)

            save_training_state(checkpoint_dir, "generalist", 3, replay, [(1, pool_network)])
            with patch("training.mcts.training_state.build_policy_value_net", fake_build):
                restored_replay, restored_pool = restore_training_state(
                    checkpoint_dir, "generalist", expected_round=3, replay_buffer_rounds=2
                )

            self.assertEqual(saved_training_state_round(checkpoint_dir, "generalist"), 3)
            self.assertEqual(list(restored_replay), [(2, ["sample-a"]), (3, ["sample-b"])])
            self.assertEqual(restored_pool[0][0], 1)
            self.assertEqual(restored_pool[0][1].state, {"weight": 3})

    def test_pool_weights_are_not_rewritten_when_unchanged(self) -> None:
        """poolは毎ラウンド書き直さない(107MB級の無駄書き込みを避ける)。"""
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            pool = [(1, FakeNetwork({"weight": 3}))]
            replay = deque([(1, ["a"])], maxlen=2)

            save_training_state(checkpoint_dir, "generalist", 1, replay, pool)
            pool_file = checkpoint_dir.parent / "pool" / "generalist_round1.pt"
            first_mtime = pool_file.stat().st_mtime_ns

            save_training_state(checkpoint_dir, "generalist", 2, replay, pool)

            self.assertEqual(pool_file.stat().st_mtime_ns, first_mtime)

    def test_dropped_pool_entries_are_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            replay = deque([(1, ["a"])], maxlen=2)

            save_training_state(
                checkpoint_dir, "generalist", 1, replay, [(1, FakeNetwork({"w": 1}))]
            )
            save_training_state(
                checkpoint_dir, "generalist", 2, replay, [(2, FakeNetwork({"w": 2}))]
            )

            pool_dir = checkpoint_dir.parent / "pool"
            self.assertFalse((pool_dir / "generalist_round1.pt").exists())
            self.assertTrue((pool_dir / "generalist_round2.pt").exists())


if __name__ == "__main__":
    unittest.main()
