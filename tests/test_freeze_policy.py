"""`freeze_policy`が、方策の出力を変えずに価値だけ更新することを検証する。

模倣学習で得た方策(ラダー実測905)を壊さずに価値ヘッドだけ鍛えるための仕掛け。
`encoder_fc`はCLSトークンから価値を読むだけで、方策側は`encoder`の出力全体へ
交差注意する。したがって`encoder_fc`以外を凍結すれば方策は不変になる ——
この前提が崩れると、静かに方策が劣化するため例外にならず気付けない。
"""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.common.network import NUM_WORDS_ENCODER, build_policy_value_net  # noqa: E402
from training.common.sparse_features import SparseVector, decoder_size, encoder_size  # noqa: E402
from training.common.training_utils import collate_samples  # noqa: E402
from training.mcts.selfplay import Sample  # noqa: E402
from training.mcts.train import train_one_round, trainable_parameters  # noqa: E402


def _sample(n_actions: int, seed: int, label: float) -> Sample:
    generator = torch.Generator().manual_seed(seed)
    enc = SparseVector()
    for _ in range(NUM_WORDS_ENCODER):
        enc.word_start()
        enc.add_absolute(int(torch.randint(0, encoder_size(), (1,), generator=generator)), 1.0)
    dec = SparseVector()
    for _ in range(n_actions):
        dec.word_start()
        dec.add_absolute(int(torch.randint(0, decoder_size(), (1,), generator=generator)), 1.0)
    sample = Sample(enc, dec, [1.0 / n_actions] * n_actions, label)
    sample.label = label
    return sample


class TrainableParametersTest(unittest.TestCase):
    def test_freeze_selects_only_the_value_head(self):
        network = build_policy_value_net()
        frozen = trainable_parameters(network, freeze_policy=True)
        self.assertEqual([id(p) for p in frozen], [id(p) for p in network.encoder_fc.parameters()])

    def test_unfrozen_selects_everything(self):
        network = build_policy_value_net()
        self.assertEqual(
            len(trainable_parameters(network, freeze_policy=False)),
            len(list(network.parameters())),
        )


class FrozenPolicyIsUnchangedTest(unittest.TestCase):
    """凍結学習を1ラウンド回して、方策スコアが一致し価値だけ動くこと。"""

    def _scores_and_value(self, network, batch):
        network.eval()
        with torch.no_grad():
            values, scores = network(*batch[:6], batch[6])
        return scores.clone(), values.clone()

    def test_policy_scores_survive_a_training_round(self):
        torch.manual_seed(0)
        network = build_policy_value_net()
        samples = [_sample(3 + i % 5, seed=i, label=1.0 if i % 2 else -1.0) for i in range(16)]
        probe = collate_samples(samples[:4])

        before_scores, before_values = self._scores_and_value(network, probe)
        optimizer = torch.optim.Adam(trainable_parameters(network, freeze_policy=True), lr=1e-2)
        train_one_round(
            network,
            optimizer,
            samples,
            epochs=3,
            batch_size=4,
            device=torch.device("cpu"),
            freeze_policy=True,
        )
        after_scores, after_values = self._scores_and_value(network, probe)

        self.assertTrue(
            torch.equal(before_scores, after_scores),
            f"方策が動いた: 最大差 {(before_scores - after_scores).abs().max().item()}",
        )
        self.assertGreater(
            (before_values - after_values).abs().max().item(),
            1e-6,
            "価値ヘッドが更新されていない",
        )

    def test_unfrozen_training_does_move_the_policy(self):
        """対照。凍結しなければ方策は動く(テスト自体が空振りしていないことの確認)。"""
        torch.manual_seed(0)
        network = build_policy_value_net()
        samples = [_sample(3 + i % 5, seed=i, label=1.0 if i % 2 else -1.0) for i in range(16)]
        probe = collate_samples(samples[:4])

        before_scores, _ = self._scores_and_value(network, probe)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
        train_one_round(
            network,
            optimizer,
            samples,
            epochs=3,
            batch_size=4,
            device=torch.device("cpu"),
            freeze_policy=False,
        )
        after_scores, _ = self._scores_and_value(network, probe)
        self.assertFalse(torch.equal(before_scores, after_scores))


if __name__ == "__main__":
    unittest.main()
