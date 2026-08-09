"""提出物が読み込む推論エージェント。

学習側と同じ`sparse_features`/`network`をそのまま使う。提出物へコードを複製すると
特徴量を変えるたびに陳腐化し、学習時と推論時で食い違う(以前はそれが起きていた)。
提出物にはこのパッケージごと同梱し、`main.py`は薄いシムにする。
"""

from __future__ import annotations

import json
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

    def __init__(
        self,
        base_dir: Path,
        *,
        search_count: int = 0,
        num_determinizations: int = 1,
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
        self.opponent_decks: list[list[int]] = []
        if search_count > 0:
            path = resource_path(opponent_decks_name, base_dir)
            self.opponent_decks = json.loads(path.read_text(encoding="utf-8"))

    def select(self, obs) -> list[int]:
        """1手分の選択を返す。探索が失敗した手番は貪欲方策へ退避する。"""
        if obs.select is None:
            return self.deck
        if self.search_count > 0:
            try:
                return self._select_with_search(obs)
            except Exception:
                # 1手も返せないと即敗北になるため、探索の失敗は必ず吸収する。
                pass
        return choose_greedy(self.network, obs, self.deck)

    def _select_with_search(self, obs) -> list[int]:
        from ..mcts.selfplay import run_determinized_mcts

        select, _policy, _value, _actions = run_determinized_mcts(
            self.network,
            obs,
            self.deck,
            self.opponent_decks,
            self.search_count,
            num_determinizations=self.num_determinizations,
        )
        return select
