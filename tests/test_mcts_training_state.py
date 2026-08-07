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
    def __init__(self, *_args) -> None:
        self.state = {}

    def state_dict(self):
        return self.state

    def load_state_dict(self, state, assign=False) -> None:
        self.state = state

    def eval(self) -> None:
        pass


class MctsTrainingStateTest(unittest.TestCase):
    def test_round_trip_replay_and_checkpoint_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            pool_network = FakeNetwork()
            pool_network.state = {"weight": 3}
            replay = deque([(2, ["sample-a"]), (3, ["sample-b"])], maxlen=2)

            save_training_state(checkpoint_dir, "generalist", 3, replay, [pool_network])
            with patch("training.mcts.training_state.PolicyValueNet", FakeNetwork):
                restored_replay, restored_pool = restore_training_state(
                    checkpoint_dir, "generalist", expected_round=3, replay_buffer_rounds=2
                )

            self.assertEqual(saved_training_state_round(checkpoint_dir, "generalist"), 3)
            self.assertEqual(list(restored_replay), [(2, ["sample-a"]), (3, ["sample-b"])])
            self.assertEqual(restored_pool[0].state, {"weight": 3})


if __name__ == "__main__":
    unittest.main()
