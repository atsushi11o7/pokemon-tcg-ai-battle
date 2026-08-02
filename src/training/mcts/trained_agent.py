"""自己対戦で学習した`PolicyValueNet`(Transformer)のチェックポイントを使う、動作確認用エージェント。

探索の事前分布・評価値を、自己対戦(`train_mcts.py`)で学習した`PolicyValueNet`に
基づいて計算する。本番提出物として整えたものではなく、勝率を試しに測るための
最小構成(`src/evaluation/match_runner.py`での評価用)。
"""

import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "training" / "common"))
from determinize import search_begin_kwargs  # noqa: E402
from network import PolicyValueNet  # noqa: E402
from search import run_mcts  # noqa: E402
from selfplay import make_eval_fn  # noqa: E402
from train_mcts import (  # noqa: E402
    D_FEEDFORWARD,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS_DECODER,
    NUM_LAYERS_ENCODER,
)

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import (  # noqa: E402
    Observation,
    search_begin,
    search_end,
    to_observation_class,
)

DECK_PATH = ROOT / "decks" / "cynthias_garchomp_ex.csv"
CHECKPOINT_DIR = ROOT / "outputs" / "mcts_checkpoints"
SEARCH_COUNT = 10  # 1手あたりのMCTSシミュレーション回数(GPU無し・2vCPU実行のため小さめ)


MIN_CHECKPOINT_BYTES = 10_000_000  # 旧アーキテクチャ(密なMLP、数百KB)の残骸を除外する閾値


def _latest_checkpoint_path() -> Path:
    """`CHECKPOINT_DIR`の中から、ラウンド番号が最も大きいチェックポイントを探す。

    サイズが`MIN_CHECKPOINT_BYTES`未満のファイルは、別アーキテクチャの古い
    チェックポイントとして無視する。

    Returns:
        Path: 最新の`round{N}.pt`のパス。

    Raises:
        FileNotFoundError: チェックポイントが1つも見つからない場合。
    """
    checkpoints = [
        (int(m.group(1)), p)
        for p in CHECKPOINT_DIR.glob("round*.pt")
        if p.stat().st_size >= MIN_CHECKPOINT_BYTES and (m := re.match(r"round(\d+)\.pt", p.name))
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    return max(checkpoints, key=lambda x: x[0])[1]


_network = PolicyValueNet(D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER)
_network.load_state_dict(torch.load(_latest_checkpoint_path(), weights_only=True))
_network.eval()

_own_deck_cache: list[int] | None = None


def read_deck() -> list[int]:
    """デッキ提出用に、decks/配下の共通deck.csv(60行のカードID)を読み込む。

    Returns:
        list[int]: 60枚分のカードID。
    """
    return [int(x) for x in DECK_PATH.read_text().split("\n") if x.strip()]


def _own_deck() -> list[int]:
    """自分のデッキリストを遅延ロードしてキャッシュする。

    Returns:
        list[int]: `read_deck`が返す60枚分のカードID。
    """
    global _own_deck_cache
    if _own_deck_cache is None:
        _own_deck_cache = read_deck()
    return _own_deck_cache


_eval_fn = None  # `_own_deck()`確定後に`choose_with_mcts`が遅延生成する


def _get_eval_fn():
    """相手の本当のデッキは分からないため、`[自分のデッキ, 自分のデッキ]`を仮定して
    `eval_fn`を遅延生成する(相手側の`your_deck`引数は`obs.current.yourIndex`が
    相手になった場合にのみ参照されるが、本番推論では相手の手番の局面を評価しない
    ため実質使われない)。

    Returns:
        Callable: `search.run_mcts`に渡す`eval_fn`。
    """
    global _eval_fn
    if _eval_fn is None:
        _eval_fn = make_eval_fn(_network, [_own_deck(), _own_deck()])
    return _eval_fn


def choose_with_mcts(obs: Observation) -> list[int]:
    """隠れ情報を1つ仮定した上でDeterminized MCTSを実行し、最善と判断した選択を返す。

    Args:
        obs: `select`がNoneでない局面のObservation。

    Returns:
        list[int]: 選んだ選択肢のindexのリスト。
    """
    root_state = search_begin(obs, **search_begin_kwargs(obs, _own_deck()))
    try:
        select, _policy_target, _root_value, _actions = run_mcts(
            root_state, obs.current.yourIndex, _get_eval_fn(), SEARCH_COUNT
        )
    finally:
        search_end()
    return select


def agent(obs_dict: dict) -> list[int]:
    """デッキ提出以外は、あらゆる選択をDeterminized MCTSで決める。

    Args:
        obs_dict: kaggle_environmentsから渡される観測(生dict)。

    Returns:
        list[int]: 選択した選択肢のindexのリスト。
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck()
    return choose_with_mcts(obs)
