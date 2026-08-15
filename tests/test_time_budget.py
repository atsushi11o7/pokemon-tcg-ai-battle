"""持ち時間を使い切らないよう、探索を段階的に浅くすることを検証する。

1手あたりの探索回数を固定すると、試合が長引いたぶんだけ超過する。持ち時間を
使い切るとエピソードごと失敗扱いになり(提出#20)、その1件は無価値になる。
ローカルからKaggleへの換算係数は実測のばらつきが大きく、事前見積もりでは守れない。

超過は例外ではなくKaggle側の失敗として現れるため、手元では気付けない。
段階の境界と、最後に必ず貪欲方策へ落ちることをここで固定する。
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.inference.agent import SubmissionAgent  # noqa: E402


class _Agent(SubmissionAgent):
    """重みやデッキを読まずに予算計算だけ試すための最小構成。"""

    def __init__(self, search_count: int, budget: float) -> None:
        self.search_count = search_count
        self.num_determinizations = 3
        self.time_budget_seconds = budget
        self.spent_seconds = 0.0


class BudgetScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = _Agent(search_count=200, budget=540.0)

    def test_full_search_early(self) -> None:
        self.assertEqual(self.agent.budgeted_search_count(0.0), 200)
        self.assertEqual(self.agent.budgeted_search_count(540 * 0.44), 200)

    def test_search_shrinks_as_time_is_consumed(self) -> None:
        counts = [self.agent.budgeted_search_count(540 * used) for used in (0.5, 0.7, 0.85)]
        self.assertEqual(counts, [100, 50, 20])
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_falls_back_to_greedy_before_the_budget_runs_out(self) -> None:
        self.assertEqual(self.agent.budgeted_search_count(540 * 0.95), 0)
        self.assertEqual(self.agent.budgeted_search_count(10_000.0), 0)

    def test_search_never_reaches_zero_while_still_searching(self) -> None:
        """倍率で0になっても、探索する段では最低1回は回す(0は貪欲方策の合図)。"""
        tiny = _Agent(search_count=4, budget=540.0)
        self.assertEqual(tiny.budgeted_search_count(540 * 0.85), 1)

    def test_zero_budget_disables_search(self) -> None:
        self.assertEqual(_Agent(200, budget=0.0).budgeted_search_count(0.0), 0)


class WorstCaseGameTest(unittest.TestCase):
    """極端に長い試合でも、合計消費が持ち時間を超えないこと。"""

    def test_long_game_stays_within_budget(self) -> None:
        budget = 540.0
        agent = _Agent(search_count=150, budget=budget)
        seconds_per_sim = 0.02  # 1シミュレーションあたりの想定コスト
        for _ in range(400):  # 通常110〜136手のところを400手引いた場合
            count = agent.budgeted_search_count(agent.spent_seconds)
            agent.spent_seconds += count * seconds_per_sim if count else 0.003
        self.assertLess(agent.spent_seconds, budget)


class MarginScaleTest(unittest.TestCase):
    """方策の迷い(top1-top2)で探索予算を配分すること。

    24試合2,035手の実測では、探索が着手を変えた割合は
    差<0.5で40.2%、1.5-3.0で1.9%、3.0以上で0.6%だった。差1.5以上は全手番の58%を
    占めながら着手変更の5%しか生まないので、そこを切って迷う局面へ回す。
    配分がずれても例外にならず「同じ予算で浅く読む」だけになるため、テストで固定する。
    """

    def setUp(self) -> None:
        self.agent = _Agent(search_count=150, budget=540.0)

    def test_uncertain_positions_get_more_search(self) -> None:
        self.assertEqual(self.agent.margin_scale(0.0), 2.0)
        self.assertEqual(self.agent.margin_scale(0.49), 2.0)

    def test_scale_decreases_as_the_policy_gets_confident(self) -> None:
        scales = [self.agent.margin_scale(m) for m in (0.2, 1.0, 2.0, 5.0)]
        self.assertEqual(scales, sorted(scales, reverse=True))
        self.assertEqual(scales[-1], 0.0)

    def test_confident_positions_skip_search(self) -> None:
        self.assertEqual(self.agent.margin_scale(3.0), 0.0)
        self.assertEqual(self.agent.margin_scale(float("inf")), 0.0)

    def test_combined_with_the_time_budget(self) -> None:
        """時間の逼迫と方策の確信は掛け合わさる。"""
        base = self.agent.budgeted_search_count(0.0)
        self.assertEqual(int(base * self.agent.margin_scale(0.1)), 300)
        self.assertEqual(int(base * self.agent.margin_scale(2.0)), 22)
        self.assertEqual(int(base * self.agent.margin_scale(9.0)), 0)
        late = self.agent.budgeted_search_count(540 * 0.7)
        self.assertEqual(int(late * self.agent.margin_scale(0.1)), 74)


if __name__ == "__main__":
    unittest.main()
