import tempfile
import unittest
from pathlib import Path

import torch

from training.common.checkpoints import latest_checkpoint_round, restore_optimizer_state


class CheckpointDiscoveryTest(unittest.TestCase):
    def test_returns_latest_model_round_and_ignores_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            for name in (
                "generalist_round2.pt",
                "generalist_round10.pt",
                "generalist_round20_optimizer.pt",
                "mirror_round30.pt",
                "final.pt",
            ):
                (checkpoint_dir / name).touch()

            latest = latest_checkpoint_round(checkpoint_dir, "generalist")

        self.assertEqual(latest, 10)

    def test_returns_none_without_matching_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(latest_checkpoint_round(Path(directory), "generalist"))

    def test_restore_optimizer_keeps_state_but_uses_configured_learning_rate(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        original = torch.optim.Adam([parameter], lr=3e-4)
        parameter.grad = torch.tensor([1.0])
        original.step()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "optimizer.pt"
            torch.save(original.state_dict(), checkpoint)

            restored = torch.optim.Adam([parameter], lr=9e-4)
            restore_optimizer_state(restored, checkpoint, learning_rate=1e-4)

        self.assertEqual(restored.param_groups[0]["lr"], 1e-4)
        self.assertTrue(restored.state[parameter])


if __name__ == "__main__":
    unittest.main()
