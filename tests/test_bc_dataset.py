"""模倣学習用の抽出が、リプレイの記録形式を正しく解釈できているかを検証する。

守りたいのは次の2点。どちらも実データで踏んだ罠で、間違えても例外にならず
「教師がずれたまま学習が進む」ため、テストで固定する。

1. step iのobservationへの応答は、step i+1のactionへ記録される
2. 待機中(INACTIVE)の席には前回のactionが残っており、応答ではない
"""

import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.bc.dataset import episode_decks, iter_decisions, winning_seat  # noqa: E402
from training.mcts.selfplay import Sample  # noqa: E402

DECK_A = list(range(1, 61))
DECK_B = list(range(61, 121))
SETUP = [[{"action": DECK_A}, {"action": DECK_B}], [{"action": None}, {"action": None}]]


def _seat(status: str, option_count: int = 3) -> dict:
    return {
        "status": status,
        "observation": {"select": {"option": [{"index": i} for i in range(option_count)]}},
    }


class EpisodeMetadataTest(unittest.TestCase):
    def test_decks_are_read_from_the_first_steps(self):
        self.assertEqual(episode_decks(SETUP), [DECK_A, DECK_B])

    def test_incomplete_decks_are_rejected(self):
        self.assertIsNone(episode_decks([[{"action": DECK_A}, {"action": None}]]))

    def test_winning_seat(self):
        self.assertEqual(winning_seat({"rewards": [1, -1]}), 0)
        self.assertEqual(winning_seat({"rewards": [-1, 1]}), 1)
        self.assertIsNone(winning_seat({"rewards": [0, 0]}))
        self.assertIsNone(winning_seat({"rewards": []}))


class IterDecisionsTest(unittest.TestCase):
    def test_action_comes_from_the_next_step(self):
        steps = SETUP + [
            [_seat("ACTIVE"), _seat("INACTIVE")],
            [{"action": [2]}, {"action": None}],
        ]
        found = list(iter_decisions(steps, winner=0, stats=Counter()))
        self.assertEqual([(seat, action) for seat, _obs, action in found], [(0, [2])])

    def test_same_step_action_is_not_used(self):
        """同じstepのactionを読んでいたら、この構成では[9]が拾われてしまう。"""
        steps = SETUP + [
            [dict(_seat("ACTIVE"), action=[9]), _seat("INACTIVE")],
            [{"action": [1]}, {"action": None}],
        ]
        actions = [action for _seat_index, _obs, action in iter_decisions(steps, 0, Counter())]
        self.assertEqual(actions, [[1]])

    def test_inactive_seat_is_skipped(self):
        steps = SETUP + [
            [_seat("INACTIVE"), _seat("INACTIVE")],
            [{"action": [1]}, {"action": [1]}],
        ]
        stats: Counter = Counter()
        self.assertEqual(list(iter_decisions(steps, 0, stats, winner_only=False)), [])
        self.assertEqual(stats["skip_inactive"], 2)

    def test_loser_side_is_excluded_by_default(self):
        steps = SETUP + [
            [_seat("INACTIVE"), _seat("ACTIVE")],
            [{"action": None}, {"action": [0]}],
        ]
        self.assertEqual(list(iter_decisions(steps, winner=0, stats=Counter())), [])
        both = list(iter_decisions(steps, winner=0, stats=Counter(), winner_only=False))
        self.assertEqual([seat for seat, _obs, _action in both], [1])

    def test_deck_submission_is_not_a_decision(self):
        steps = SETUP + [
            [_seat("ACTIVE"), _seat("INACTIVE")],
            [{"action": DECK_A}, {"action": None}],
        ]
        self.assertEqual(list(iter_decisions(steps, 0, Counter())), [])

    def test_last_step_has_no_following_action(self):
        steps = SETUP + [[_seat("ACTIVE"), _seat("INACTIVE")]]
        self.assertEqual(list(iter_decisions(steps, 0, Counter())), [])


class SampleLabelTest(unittest.TestCase):
    """価値の教師は`label`から読まれる。`value`だけ埋めると学習時にNoneで落ちる。"""

    def test_label_is_none_until_assigned(self):
        self.assertIsNone(Sample(None, None, [1.0], 1.0).label)


class ZeroPolicyTargetTest(unittest.TestCase):
    """敗者サンプルは`policy_target`を全ゼロにして、方策の勾配を出さない。

    価値の教師(`label`)だけを効かせるための仕掛け。損失の式に依存しているので、
    式が変わったらここで落ちるようにしておく。
    """

    def test_zero_target_produces_no_policy_gradient(self):
        import torch

        from training.common.training_utils import masked_policy_loss

        scores = torch.randn(2, 4, requires_grad=True)
        mask = torch.ones(2, 4, dtype=torch.bool)
        targets = torch.zeros(2, 4)
        targets[0, 2] = 1.0  # 1件目だけone-hot、2件目は全ゼロ
        masked_policy_loss(scores, mask, targets).backward()
        self.assertGreater(scores.grad[0].abs().sum().item(), 0.0)
        self.assertEqual(scores.grad[1].abs().sum().item(), 0.0)


class PaddingInvarianceTest(unittest.TestCase):
    """行動数の異なるサンプルを同居させても、方策スコアが変わらないこと。

    デコーダに自己注意を入れた際、`key_padding_mask`を渡し忘れると空の行動トークンへ
    注意してしまい、同じ局面でもバッチ内の最大行動数でスコアが変わる。学習時と
    単独推論時で方策が食い違う形の不整合で、例外にならないため気付きにくい。
    """

    def test_scores_do_not_depend_on_batch_padding(self):
        import torch

        from training.common.network import build_policy_value_net
        from training.common.training_utils import collate_samples
        from training.mcts.selfplay import Sample

        def make(n_actions: int, seed: int) -> Sample:
            torch.manual_seed(seed)
            from training.common.network import NUM_WORDS_ENCODER
            from training.common.sparse_features import SparseVector, decoder_size, encoder_size

            enc = SparseVector()
            for _ in range(NUM_WORDS_ENCODER):
                enc.word_start()
                enc.add_absolute(int(torch.randint(0, encoder_size(), (1,))), 1.0)
            dec = SparseVector()
            for _ in range(n_actions):
                dec.word_start()
                dec.add_absolute(int(torch.randint(0, decoder_size(), (1,))), 1.0)
            sample = Sample(enc, dec, [1.0 / n_actions] * n_actions, 0.0)
            sample.label = 0.0
            return sample

        small, large = make(3, 0), make(40, 1)
        net = build_policy_value_net()
        net.eval()
        with torch.no_grad():
            alone = collate_samples([small])
            padded = collate_samples([small, large])
            score_alone = net(*alone[:6], alone[6])[1][0][:3]
            score_padded = net(*padded[:6], padded[6])[1][0][:3]
        self.assertTrue(
            torch.allclose(score_alone, score_padded, atol=1e-5),
            f"パディングの有無でスコアが変わった: {score_alone} != {score_padded}",
        )


if __name__ == "__main__":
    unittest.main()
