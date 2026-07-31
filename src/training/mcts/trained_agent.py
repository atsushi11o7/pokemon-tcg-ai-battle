"""自己対戦で学習した`PolicyValueNet`(Transformer)のチェックポイントを使う、動作確認用エージェント。

探索の事前分布・評価値を、自己対戦(`train_mcts.py`)で学習した`PolicyValueNet`に
基づいて計算する。本番提出物として整えたものではなく、勝率を試しに測るための
最小構成(`src/evaluation/match_runner.py`での評価用)。
"""

import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from determinize import (  # noqa: E402
    sample_opponent_active_guess,
    sample_opponent_hidden,
    sample_own_hidden,
)
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
    SelectData,
    SelectType,
    search_begin,
    search_end,
    to_observation_class,
)

DECK_PATH = ROOT / "submission" / "01_rule_based" / "deck.csv"
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
    """デッキ提出用に、01_rule_basedと共通のdeck.csv(60行のカードID)を読み込む。

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


def _search_begin_kwargs(obs: Observation) -> dict:
    """`search_begin`に渡す、隠れ情報の仮定一式を組み立てる。

    自分側は既知のデッキリストから、相手側は過去リプレイのカード出現頻度から
    それぞれサンプリングする(`determinize.py`)。相手のアクティブポケモンが伏せられている
    場合のみ、その正体の仮定も追加する。

    Args:
        obs: 探索を開始したい局面のObservation。

    Returns:
        dict: `search_begin`にそのまま`**kwargs`で渡せる引数一式。
    """
    state = obs.current
    your_index = state.yourIndex
    me = state.players[your_index]
    opp = state.players[1 - your_index]

    your_deck, your_prize = sample_own_hidden(_own_deck(), me.deckCount, len(me.prize))
    opponent_deck, opponent_hand, opponent_prize = sample_opponent_hidden(
        opp.deckCount, opp.handCount, len(opp.prize)
    )
    opponent_active: list[int] = []
    if opp.active and opp.active[0] is None:
        opponent_active = [sample_opponent_active_guess()]

    return {
        "your_deck": your_deck,
        "your_prize": your_prize,
        "opponent_deck": opponent_deck,
        "opponent_prize": opponent_prize,
        "opponent_hand": opponent_hand,
        "opponent_active": opponent_active,
    }


def choose_with_mcts(obs: Observation) -> list[int]:
    """隠れ情報を1つ仮定した上でDeterminized MCTSを実行し、最善と判断した選択を返す。

    Args:
        obs: `select.type`がMAIN/EVOLVEで、`minCount==maxCount==1`であることを
            呼び出し側で保証しておくこと。

    Returns:
        list[int]: 選んだ選択肢のindexを1つだけ含むリスト。
    """
    root_state = search_begin(obs, **_search_begin_kwargs(obs))
    try:
        select, _policy_target, _root_value, _actions = run_mcts(
            root_state, obs.current.yourIndex, _get_eval_fn(), SEARCH_COUNT
        )
    finally:
        search_end()
    return select


def fallback_random(sel: SelectData) -> list[int]:
    """探索の対象外の選択(セットアップ・サーチ等)に対して、ランダムに選ぶ。

    Args:
        sel: 今回の選択肢一式。

    Returns:
        list[int]: `minCount`以上`maxCount`以下の個数だけランダムに選んだindexのリスト。
    """
    count = random.randint(sel.minCount, sel.maxCount)
    return random.sample(range(len(sel.option)), count)


def agent(obs_dict: dict) -> list[int]:
    """MAIN/EVOLVEの単一選択はDeterminized MCTS(学習済みネットワーク)で、それ以外はランダムで応答する。

    Args:
        obs_dict: kaggle_environmentsから渡される観測(生dict)。

    Returns:
        list[int]: 選択した選択肢のindexのリスト。
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck()
    sel = obs.select
    if (
        sel.type in (SelectType.MAIN, SelectType.EVOLVE)
        and sel.minCount == sel.maxCount == 1
        and (len(sel.option) >= 2)
    ):
        return choose_with_mcts(obs)
    return fallback_random(sel)
