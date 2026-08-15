"""提出物が読み込む推論エージェント。

学習側と同じ`sparse_features`/`network`をそのまま使う。提出物へコードを複製すると
特徴量を変えるたびに陳腐化し、学習時と推論時で食い違う(以前はそれが起きていた)。
提出物にはこのパッケージごと同梱し、`main.py`は薄いシムにする。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ..common.deck import parse_deck_csv
from ..common.network import PolicyValueNet, load_policy_value_net
from ..common.sparse_features import get_decoder_input, get_encoder_input
from ..mcts.search import enumerate_actions

KAGGLE_AGENT_DIR = Path("/kaggle_simulations/agent")


def resource_path(name: str, base_dir: Path) -> Path:
    """提出物に同梱したファイルを、ローカル実行と本番実行の両方で解決する。"""
    local = base_dir / name
    if local.exists():
        return local
    return KAGGLE_AGENT_DIR / name


def rank_actions(network: PolicyValueNet, obs, deck: list[int]) -> tuple[list, list[int], float]:
    """盤面を1回評価し、(選択肢, 方策スコア最大の手, top1とtop2の差)を返す。

    差は「方策がどれだけ迷っているか」の指標で、探索予算の配分に使う。ここで得た
    結果を使い回すことで、貪欲な着手と迷いの判定で二重に評価せずに済む。
    """
    actions = enumerate_actions(obs.select)
    encoder_sv = get_encoder_input(obs, deck)
    decoder_sv = get_decoder_input(obs, actions)
    index_enc, value_enc, offset_enc = encoder_sv.to_tensors()
    index_dec, value_dec, offset_dec = decoder_sv.to_tensors()
    with torch.inference_mode():
        _value, scores = network(index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec)
    row = scores[0]
    if row.shape[0] < 2:
        return actions, actions[int(torch.argmax(row).item())], float("inf")
    top = torch.topk(row, 2)
    return actions, actions[int(top.indices[0])], float(top.values[0] - top.values[1])


def choose_greedy(network: PolicyValueNet, obs, deck: list[int]) -> list[int]:
    """探索せず、盤面を1回評価して方策スコア最大の選択肢を返す。"""
    _actions, best, _margin = rank_actions(network, obs, deck)
    return best


class SubmissionAgent:
    """`deck.csv`と重みを読み込み、貪欲方策または探索付きで着手を選ぶ。"""

    # 持ち時間の消費率 → 探索回数の倍率。使うほど浅くし、最後は貪欲方策だけにする。
    BUDGET_SCHEDULE = ((0.45, 1.0), (0.65, 0.5), (0.80, 0.25), (0.90, 0.1))

    # 方策のtop1とtop2のスコア差 → 探索回数の倍率。
    # 24試合2,035手の実測で、探索が着手を変えた割合は差ごとに次のとおりだった。
    #   差<0.5 で 40.2%(600手) / 0.5-1.5 で 5.2% / 1.5-3.0 で 1.9% / 3.0以上 で 0.6%
    # 差1.5以上の手番は全体の58%を占めるが、着手変更の5%しか生んでいない。
    # そこを切って迷っている局面へ回せば、同じ予算で深く読める。
    # 進行度でも測ったが 9.5%〜18.2% と差が小さく、配分の基準にならなかった。
    MARGIN_SCHEDULE = ((0.5, 2.0), (1.5, 0.5), (3.0, 0.15))

    def __init__(
        self,
        base_dir: Path,
        *,
        search_count: int = 0,
        num_determinizations: int = 1,
        time_budget_seconds: float = 540.0,
        weights_name: str = "policy.pt",
        deck_name: str = "deck.csv",
        opponent_decks_name: str = "opponent_decks.json",
    ) -> None:
        """
        Args:
            base_dir: 同梱ファイルが置かれたディレクトリ。
            search_count: 1手あたりの探索回数。0なら探索せず貪欲方策のみ。
            num_determinizations: 隠れ情報の仮説数。`search_count`をこの数で分割する。
            weights_name: 重みファイル名。
            deck_name: デッキファイル名。
            opponent_decks_name: 探索時の相手デッキ候補(探索する場合のみ必要)。
        """
        self.network = load_policy_value_net(resource_path(weights_name, base_dir))
        self.deck = parse_deck_csv(resource_path(deck_name, base_dir))
        self.search_count = search_count
        self.num_determinizations = num_determinizations
        # 本番の上限は1エージェント600秒。読み込みや例外処理の余地を残して低めに置く。
        self.time_budget_seconds = time_budget_seconds
        self.spent_seconds = 0.0
        self.opponent_decks: list[list[int]] = []
        if search_count > 0:
            path = resource_path(opponent_decks_name, base_dir)
            self.opponent_decks = json.loads(path.read_text(encoding="utf-8"))

    def margin_scale(self, margin: float) -> float:
        """方策の迷い(top1とtop2の差)から、探索回数の倍率を返す。

        差が大きい局面は探索しても着手が変わらない(実測で3.0以上なら0.6%)。
        そこを切って、迷っている局面(差<0.5で40.2%が変わる)へ予算を寄せる。
        """
        for threshold, scale in self.MARGIN_SCHEDULE:
            if margin < threshold:
                return scale
        return 0.0

    def budgeted_search_count(self, spent_seconds: float) -> int:
        """これまでの消費時間から、この手番に使う探索回数を決める。

        1手あたりの探索回数を固定すると、試合が長引いたぶんだけ超過する。持ち時間を
        使い切るとエピソードごと失敗扱いになり(提出#20で経験)、その1件は無価値になる。
        ローカルからKaggleへの換算係数は実測にばらつきが大きく、事前見積もりでは
        守り切れない。そこで、自分の消費時間を見ながら段階的に浅くする。

        貪欲方策は1手あたり数ミリ秒なので、最終段まで落ちれば残りは必ず捌ける。
        """
        if self.time_budget_seconds <= 0:
            return 0  # 持ち時間が無いなら探索しない(安全側に倒す)
        used = spent_seconds / self.time_budget_seconds
        for threshold, scale in self.BUDGET_SCHEDULE:
            if used < threshold:
                return max(1, int(self.search_count * scale))
        return 0  # 貪欲方策のみ

    def select(self, obs) -> list[int]:
        """1手分の選択を返す。探索が失敗した手番は貪欲方策へ退避する。"""
        if obs.select is None:
            return self.deck
        started = time.monotonic()
        try:
            actions, greedy, margin = rank_actions(self.network, obs, self.deck)
            count = int(self.budgeted_search_count(self.spent_seconds) * self.margin_scale(margin))
            if count > 0 and len(actions) >= 2:
                try:
                    return self._select_with_search(obs, count)
                except Exception:
                    # 1手も返せないと即敗北になるため、探索の失敗は必ず吸収する。
                    pass
            return greedy
        finally:
            self.spent_seconds += time.monotonic() - started

    def _select_with_search(self, obs, count: int) -> list[int]:
        from ..mcts.selfplay import run_determinized_mcts

        select, _policy, _value, _actions = run_determinized_mcts(
            self.network,
            obs,
            self.deck,
            self.opponent_decks,
            count,
            num_determinizations=self.num_determinizations,
        )
        return select
