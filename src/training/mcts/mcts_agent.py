"""Determinized MCTSで、MAIN/EVOLVE系の意思決定を行う、動作確認用のエージェント。

MAIN/EVOLVE以外の選択(セットアップ・サーチ等)は探索コストに見合わないため、
Behavior Cloningのエージェント(`bc_agent.py`)と同様にランダムに選ぶ。
探索の方策事前分布(prior)には学習済みBCモデルを流用し、局面評価には
`evaluate.heuristic_value`を使う(専用の価値ネットワークはまだ用意していない)。
本番提出物として整えたものではなく、勝率を試しに測るための最小構成
(`src/evaluation/match_runner.py`での評価用)。
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from determinize import (  # noqa: E402
    sample_opponent_active_guess,
    sample_opponent_hidden,
    sample_own_hidden,
)
from evaluate import heuristic_value  # noqa: E402
from search import run_mcts  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "training" / "bc"))
from features import OPTION_DIM, STATE_DIM, encode_option, encode_state  # noqa: E402
from train_bc import HIDDEN_DIM, MODEL_PATH, PolicyNet  # noqa: E402

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import (  # noqa: E402
    Observation,
    SelectData,
    SelectType,
    search_begin,
    search_end,
    to_observation_class,
)

# ルールベースエージェント(01_rule_based)・BCエージェントと同じデッキ(Issue #4で決定したもの)を使う
DECK_PATH = ROOT / "submission" / "01_rule_based" / "deck.csv"
SEARCH_COUNT = 10  # 1手あたりのMCTSシミュレーション回数(GPU無し・2vCPU実行のため小さめ)

_model = PolicyNet(STATE_DIM, OPTION_DIM, HIDDEN_DIM)
_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
_model.eval()

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


def _eval_fn(obs: Observation) -> tuple[list[float] | None, float]:
    """学習済みBCモデルの事前確率と、`heuristic_value`による評価値をまとめて返す。

    BCモデルはMAIN/EVOLVEの単一選択でしか学習していないため、それ以外の選択タイプに対しては
    Noneを返し、呼び出し側(`search.create_node`)で一様分布にフォールバックさせる。

    Args:
        obs: 評価したい局面のObservation(探索中に得られるものでよい)。

    Returns:
        tuple[list[float] | None, float]: (`obs.select.option`と同じ長さの確率分布、
            またはMAIN/EVOLVE以外の場合はNone, `obs.current.yourIndex`視点の評価値)。
    """
    probs = None
    if obs.select.type in (SelectType.MAIN, SelectType.EVOLVE):
        state_vec = torch.from_numpy(encode_state(obs)).unsqueeze(0)
        option_vecs = np.stack([encode_option(obs, opt) for opt in obs.select.option])
        options = torch.from_numpy(option_vecs).unsqueeze(0)
        with torch.no_grad():
            scores = _model(state_vec, options)[0]
        probs = torch.softmax(scores, dim=-1).tolist()
    value = heuristic_value(obs, obs.current.yourIndex)
    return probs, value


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
        select, _policy_target, _root_value = run_mcts(
            root_state, obs.current.yourIndex, _eval_fn, SEARCH_COUNT
        )
    finally:
        search_end()
    return select


def fallback_random(sel: SelectData) -> list[int]:
    """学習・探索の対象外の選択(セットアップ・サーチ等)に対して、ランダムに選ぶ。

    Args:
        sel: 今回の選択肢一式。

    Returns:
        list[int]: `minCount`以上`maxCount`以下の個数だけランダムに選んだindexのリスト。
    """
    count = random.randint(sel.minCount, sel.maxCount)
    return random.sample(range(len(sel.option)), count)


def agent(obs_dict: dict) -> list[int]:
    """MAIN/EVOLVEの単一選択はDeterminized MCTSで、それ以外はランダムで応答する。

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
