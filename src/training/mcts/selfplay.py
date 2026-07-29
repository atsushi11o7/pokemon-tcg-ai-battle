"""Determinized MCTSによる自己対戦で、方策・価値ネットワークの学習データを集める。

公式サンプルコード(reinforcement-learning-and-mcts-sample-code.ipynb)と同じく、
`cg.game`のbattle_start/battle_select/battle_finishで直接対戦を回し、MAIN/EVOLVEの
単一選択のたびにMCTS探索を行う。探索結果(訪問回数分布)を方策の教師信号、
探索で得られた価値の推定値をゲーム終了後に補正した値を価値の教師信号として、
1試合分の学習サンプルを作る。

自己対戦は両プレイヤーが同じデッキを使うため、相手のデッキ構成も「既知」として扱える
(本番の対戦相手のように頻度プールから推測する必要が無い)。ただし手札の中身までは
分からないので、`determinize.sample_full_hidden`でランダムに割り当てる。
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from determinize import sample_active_guess_from_known_deck, sample_full_hidden  # noqa: E402
from search import run_mcts  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "training" / "bc"))
from features import encode_option, encode_state  # noqa: E402

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import SelectType, search_begin, search_end, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

LAMBDA = (
    0.9  # 価値ラベルを作るときの、実際の結果と探索時のvalue推定とのブレンド率(公式サンプルと同じ)
)


class Sample:
    """1回のMAIN/EVOLVE単一選択について集めた学習サンプル。"""

    def __init__(
        self,
        state_vec: np.ndarray,
        option_vecs: np.ndarray,
        policy_target: list[float],
        value: float,
    ) -> None:
        """
        Args:
            state_vec: `encode_state`が返す状態ベクトル。
            option_vecs: `encode_option`を選択肢ごとに並べた配列。
            policy_target: `run_mcts`が返す、訪問回数を正規化した方策の教師信号。
            value: `run_mcts`が返す、探索直後の価値推定(`root_value`)。
        """
        self.state_vec = state_vec
        self.option_vecs = option_vecs
        self.policy_target = policy_target
        self.value = value
        self.label: float | None = None  # 価値の教師信号。ゲーム終了後に`_assign_labels`で埋める


def make_eval_fn(network):
    """`PolicyValueNet`をラップして、`search.create_node`が要求する`eval_fn`にする。

    Args:
        network: 評価に使う`PolicyValueNet`(eval modeで呼び出し側が管理すること)。

    Returns:
        Callable[[Observation], tuple[list[float], float]]: `eval_fn`。
    """

    def eval_fn(obs):
        state_vec = torch.from_numpy(encode_state(obs)).unsqueeze(0)
        option_vecs = np.stack([encode_option(obs, opt) for opt in obs.select.option])
        options = torch.from_numpy(option_vecs).unsqueeze(0)
        with torch.no_grad():
            scores, value = network(state_vec, options)
        probs = torch.softmax(scores[0], dim=-1).tolist()
        return probs, float(value.item())

    return eval_fn


def _search_begin_kwargs(obs, known_deck: list[int]) -> dict:
    """自己対戦用に、両者とも既知の`known_deck`を前提とした隠れ情報の仮定一式を組み立てる。

    Args:
        obs: 探索を開始したい局面のObservation。
        known_deck: 両プレイヤーが使っている、既知の60枚のデッキリスト。

    Returns:
        dict: `search_begin`にそのまま`**kwargs`で渡せる引数一式。
    """
    state = obs.current
    your_index = state.yourIndex
    me = state.players[your_index]
    opp = state.players[1 - your_index]

    # 自分の手札はObservationで既知なので、hand_count=0にしてdeck/prizeだけ受け取る
    your_deck, _, your_prize = sample_full_hidden(known_deck, me.deckCount, 0, len(me.prize))
    opponent_deck, opponent_hand, opponent_prize = sample_full_hidden(
        known_deck, opp.deckCount, opp.handCount, len(opp.prize)
    )
    opponent_active: list[int] = []
    if opp.active and opp.active[0] is None:
        opponent_active = [sample_active_guess_from_known_deck(known_deck)]

    return {
        "your_deck": your_deck,
        "your_prize": your_prize,
        "opponent_deck": opponent_deck,
        "opponent_prize": opponent_prize,
        "opponent_hand": opponent_hand,
        "opponent_active": opponent_active,
    }


def _fallback_random(sel) -> list[int]:
    """探索の対象外の選択(セットアップ・サーチ等)に対して、ランダムに選ぶ。

    Args:
        sel: 今回の選択肢一式。

    Returns:
        list[int]: `minCount`以上`maxCount`以下の個数だけランダムに選んだindexのリスト。
    """
    count = random.randint(sel.minCount, sel.maxCount)
    return random.sample(range(len(sel.option)), count)


def _assign_labels(samples: list[Sample], winner: int, player_index: int) -> None:
    """1プレイヤー分のサンプルに、ゲーム終了後の結果を使って価値の教師信号を付ける。

    最終結果(勝ち+1/負け-1/引き分け0)を起点に、終盤から遡りながら探索時のvalue推定と
    ブレンドしていく(TD的な補正。公式サンプルコードと同じ式)。

    Args:
        samples: このプレイヤーの、時系列順のSampleのリスト(`label`をこの関数で埋める)。
        winner: `state.result`(0/1が勝者のplayerIndex、2は引き分け)。
        player_index: このサンプル群を作ったプレイヤーのインデックス。
    """
    if winner == 2:
        value = 0.0
    elif winner == player_index:
        value = 1.0
    else:
        value = -1.0

    for sample in reversed(samples):
        sample.label = (value + sample.value) * 0.5
        value = value * LAMBDA + sample.value * (1.0 - LAMBDA)


def play_selfplay_game(
    network, known_deck: list[int], search_count: int
) -> tuple[list[Sample], int]:
    """1試合分の自己対戦を行い、方策・価値の学習サンプルを集める。

    Args:
        network: 探索の事前分布・評価値に使う`PolicyValueNet`(eval modeにしておくこと)。
        known_deck: 両プレイヤーが使う60枚のデッキリスト。
        search_count: 1手あたりのMCTSシミュレーション回数。

    Returns:
        tuple[list[Sample], int]: (両プレイヤー分の、labelまで埋めたSampleのリスト,
            `state.result`(0/1が勝者のplayerIndex、2は引き分け))。
    """
    eval_fn = make_eval_fn(network)
    obs_dict, start_data = battle_start(known_deck, known_deck)
    if start_data.errorPlayer >= 0:
        raise ValueError(f"deck error: errorType={start_data.errorType}")

    samples_by_player: list[list[Sample]] = [[], []]
    obs = to_observation_class(obs_dict)

    while obs.current.result < 0:
        sel = obs.select
        if (
            sel.type in (SelectType.MAIN, SelectType.EVOLVE)
            and sel.minCount == sel.maxCount == 1
            and (len(sel.option) >= 2)
        ):
            your_index = obs.current.yourIndex
            root_state = search_begin(obs, **_search_begin_kwargs(obs, known_deck))
            try:
                select, policy_target, root_value = run_mcts(
                    root_state, your_index, eval_fn, search_count
                )
            finally:
                search_end()
            state_vec = encode_state(obs)
            option_vecs = np.stack([encode_option(obs, opt) for opt in sel.option])
            samples_by_player[your_index].append(
                Sample(state_vec, option_vecs, policy_target, root_value)
            )
        else:
            select = _fallback_random(sel)
        obs = to_observation_class(battle_select(select))

    battle_finish()

    winner = obs.current.result
    for player_index in range(2):
        _assign_labels(samples_by_player[player_index], winner, player_index)

    return samples_by_player[0] + samples_by_player[1], winner
