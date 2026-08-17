"""「自分のデッキの席」を特定する抽出ロジックを固定する。

勝った試合だけを集めていた頃は、方策の教師は常に勝者へ付けてよかった。負けた試合も
使うようになると、勝者=相手なので、そのまま教師を付けると別のデッキの操縦を覚える。
実データで確認したときは、負け試合25,866件すべてが相手側の指し手になっていた。
例外にならず教師だけがずれる種類の間違いなので、ここで固定する。
"""

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("data_extract", ROOT / "scripts" / "data_extract.py")
data_extract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data_extract)

accept_episode = data_extract.accept_episode
deck_seats = data_extract.deck_seats

GRIMM = list(range(1, 61))
OTHER = list(range(61, 121))
# 60枚中4枚だけ違う構築。同じアーキタイプの範囲で、Jaccard 0.5を超える。
GRIMM_VARIANT = list(range(1, 57)) + [200, 201, 202, 203]


def replay(seat0: list[int], seat1: list[int], rewards: list[int]) -> dict:
    steps = [[{"action": seat0}, {"action": seat1}], [{"action": None}, {"action": None}]]
    return {"steps": steps, "rewards": rewards, "info": {"TeamNames": ["t0", "t1"]}}


class DeckSeatsTest(unittest.TestCase):
    def test_finds_the_seat_holding_the_deck(self):
        deck = Counter(GRIMM)
        self.assertEqual(deck_seats(replay(GRIMM, OTHER, [1, -1]), deck, 0.5), frozenset({0}))
        self.assertEqual(deck_seats(replay(OTHER, GRIMM, [1, -1]), deck, 0.5), frozenset({1}))

    def test_mirror_match_yields_both_seats(self):
        """両者が同じデッキなら2席とも返す。片方だけ選ぶと指し手を半分捨てる。"""
        seats = deck_seats(replay(GRIMM, GRIMM_VARIANT, [1, -1]), Counter(GRIMM), 0.5)
        self.assertEqual(seats, frozenset({0, 1}))

    def test_no_match_yields_empty(self):
        self.assertEqual(
            deck_seats(replay(OTHER, OTHER, [1, -1]), Counter(GRIMM), 0.5), frozenset()
        )


class AcceptEpisodeSideTest(unittest.TestCase):
    def setUp(self) -> None:
        self.deck = Counter(GRIMM)

    def accept(self, rep: dict, side: str, ratings=None, min_rating: float = 0.0) -> bool:
        return accept_episode(rep, self.deck, 0.5, ratings or {}, min_rating, Counter(), side=side)

    def test_any_takes_both_wins_and_losses(self):
        won = replay(GRIMM, OTHER, [1, -1])
        lost = replay(GRIMM, OTHER, [-1, 1])
        self.assertTrue(self.accept(won, "any"))
        self.assertTrue(self.accept(lost, "any"))
        # 従来の勝者側だけの条件では、負けた試合が落ちる。
        self.assertTrue(self.accept(won, "winner"))
        self.assertFalse(self.accept(lost, "winner"))

    def test_any_rejects_episodes_without_the_deck(self):
        self.assertFalse(self.accept(replay(OTHER, OTHER, [1, -1]), "any"))

    def test_any_rates_the_seat_holding_the_deck(self):
        """レーティングは自分のデッキを握る側で見る。相手の強さで採否しない。"""
        lost = replay(OTHER, GRIMM, [1, -1])  # 席1がGrimmsnarlで敗北
        self.assertTrue(self.accept(lost, "any", {"t0": 500.0, "t1": 950.0}, 900.0))
        self.assertFalse(self.accept(lost, "any", {"t0": 1500.0, "t1": 800.0}, 900.0))


class PolicySeatsTest(unittest.TestCase):
    """`policy_seats`で指定した席にだけone-hotが立つ。"""

    def test_only_the_named_seat_is_taught(self):
        from training.bc.dataset import extract_episode

        rep = replay(GRIMM, OTHER, [-1, 1])  # 席0(自分)が敗北
        rep["steps"] += [
            [
                {
                    "status": "ACTIVE",
                    "observation": {"select": {"option": [{"index": i} for i in range(3)]}},
                },
                {
                    "status": "ACTIVE",
                    "observation": {"select": {"option": [{"index": i} for i in range(3)]}},
                },
            ],
            [{"action": [1]}, {"action": [2]}],
        ]
        try:
            samples = extract_episode(
                rep, Counter(), winner_only=False, policy_seats=frozenset({0})
            )
        except ImportError:
            self.skipTest("cg.api が無い環境ではobservationを組み立てられない")
        if not samples:
            self.skipTest("observationのスタブが実エンジンの形式を満たさない")
        # 席0は負けているが教師が付き、勝った席1には付かない。
        self.assertEqual([sum(s.policy_target) for s in samples], [1.0, 0.0])
        self.assertEqual([s.label for s in samples], [-1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
