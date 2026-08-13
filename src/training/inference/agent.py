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


def choose_greedy(network: PolicyValueNet, obs, deck: list[int]) -> list[int]:
    """探索せず、盤面を1回評価して方策スコア最大の選択肢を返す。"""
    actions = enumerate_actions(obs.select)
    encoder_sv = get_encoder_input(obs, deck)
    decoder_sv = get_decoder_input(obs, actions)
    index_enc, value_enc, offset_enc = encoder_sv.to_tensors()
    index_dec, value_dec, offset_dec = decoder_sv.to_tensors()
    with torch.inference_mode():
        _value, scores = network(index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec)
    return actions[int(torch.argmax(scores[0]).item())]


class SubmissionAgent:
    """`deck.csv`と重みを読み込み、貪欲方策または探索付きで着手を選ぶ。"""

    # 持ち時間の消費率 → 探索回数の倍率。使うほど浅くし、最後は貪欲方策だけにする。
    BUDGET_SCHEDULE = ((0.45, 1.0), (0.65, 0.5), (0.80, 0.25), (0.90, 0.1))

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
            if self.budgeted_search_count(self.spent_seconds) > 0:
                try:
                    return self._select_with_search(obs)
                except Exception:
                    # 1手も返せないと即敗北になるため、探索の失敗は必ず吸収する。
                    pass
            return choose_greedy(self.network, obs, self.deck)
        finally:
            self.spent_seconds += time.monotonic() - started

    def _select_with_search(self, obs) -> list[int]:
        from ..mcts.selfplay import run_determinized_mcts

        select, _policy, _value, _actions = run_determinized_mcts(
            self.network,
            obs,
            self.deck,
            self.opponent_decks,
            self.budgeted_search_count(self.spent_seconds),
            num_determinizations=self.num_determinizations,
        )
        return select
