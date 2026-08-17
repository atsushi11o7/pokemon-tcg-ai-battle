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


def rank_actions(network: PolicyValueNet, obs, deck: list[int]) -> tuple[list, list[int]]:
    """盤面を1回評価し、(選択肢, 方策スコア最大の手)を返す。"""
    actions = enumerate_actions(obs.select)
    encoder_sv = get_encoder_input(obs, deck)
    decoder_sv = get_decoder_input(obs, actions)
    index_enc, value_enc, offset_enc = encoder_sv.to_tensors()
    index_dec, value_dec, offset_dec = decoder_sv.to_tensors()
    with torch.inference_mode():
        _value, scores = network(index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec)
    return actions, actions[int(torch.argmax(scores[0]).item())]


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
        leaf_value: float | None = None,
    ) -> None:
        """
        Args:
            base_dir: 同梱ファイルが置かれたディレクトリ。
            search_count: 1手あたりの探索回数。0なら探索せず貪欲方策のみ。
            num_determinizations: 隠れ情報の仮説数。`search_count`をこの数で分割する。
            weights_name: 重みファイル名。
            deck_name: デッキファイル名。
            opponent_decks_name: 探索時の相手デッキ候補(探索する場合のみ必要)。
            leaf_value: 探索の葉を価値ヘッドではなくこの定数で評価する
                (`mcts.selfplay.make_eval_fn`に理由を書いた)。Noneなら価値ヘッドを使う。
        """
        self.network = load_policy_value_net(resource_path(weights_name, base_dir))
        self.deck = parse_deck_csv(resource_path(deck_name, base_dir))
        self.search_count = search_count
        self.num_determinizations = num_determinizations
        # 本番の上限は1エージェント600秒。読み込みや例外処理の余地を残して低めに置く。
        self.time_budget_seconds = time_budget_seconds
        self.leaf_value = leaf_value
        self.spent_seconds = 0.0
        self.opponent_decks: list[list[int]] = []
        if search_count > 0:
            path = resource_path(opponent_decks_name, base_dir)
            self.opponent_decks = json.loads(path.read_text(encoding="utf-8"))

    def budgeted_search_count(self, spent_seconds: float) -> int:
        """これまでの消費時間から、この手番に使う探索回数を決める。

        持ち時間を使い切るとエピソードごと失敗扱いになるため、消費に応じて段階的に
        浅くし、最終段では探索を打ち切って貪欲方策だけにする。
        """
        if self.time_budget_seconds <= 0:
            return 0
        used = spent_seconds / self.time_budget_seconds
        for threshold, scale in self.BUDGET_SCHEDULE:
            if used < threshold:
                return max(1, int(self.search_count * scale))
        return 0

    def select(self, obs) -> list[int]:
        """1手分の選択を返す。探索が失敗した手番は貪欲方策へ退避する。"""
        if obs.select is None:
            return self.deck
        started = time.monotonic()
        try:
            actions, greedy = rank_actions(self.network, obs, self.deck)
            count = self.budgeted_search_count(self.spent_seconds)
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
            leaf_value=self.leaf_value,
        )
        return select
