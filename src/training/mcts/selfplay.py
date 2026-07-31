"""Determinized MCTSによる自己対戦で、方策・価値ネットワークの学習データを集める。

`cg.game`で直接対戦を回し、MAIN/EVOLVEの単一選択のたびにMCTS探索を行う。探索結果
(訪問回数分布)を方策の教師信号、ゲーム終了後に補正した価値推定を価値の教師信号として、
1試合分の学習サンプルを作る。

対戦相手には実在デッキ(`determinize.load_opponent_deck_pool`)からランダムに選んだ
ものを使い、こちらの固定デッキ(`our_deck`)側の意思決定のみを学習サンプルとして集める。
探索側は自分の本当のデッキだけを知り、相手の隠れ情報は本番と同じくカード出現頻度からの
推測(`determinize.sample_opponent_hidden`)を使う(相手の本当のデッキはカンニングしない)。
"""

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from determinize import (  # noqa: E402
    sample_full_hidden,
    sample_opponent_active_guess,
    sample_opponent_hidden,
)
from search import run_mcts  # noqa: E402
from sparse_features import get_decoder_input, get_encoder_input  # noqa: E402

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import SelectType, search_begin, search_end, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

LAMBDA = (
    0.9  # 価値ラベルを作るときの、実際の結果と探索時のvalue推定とのブレンド率(公式サンプルと同じ)
)


class Sample:
    """1回のMAIN/EVOLVE単一選択について集めた学習サンプル。"""

    def __init__(self, encoder_sv, decoder_sv, policy_target: list[float], value: float) -> None:
        """
        Args:
            encoder_sv: `sparse_features.get_encoder_input`が返す盤面の疎ベクトル。
            decoder_sv: `sparse_features.get_decoder_input`が返す行動群の疎ベクトル。
            policy_target: `run_mcts`が返す、訪問回数を正規化した方策の教師信号。
            value: `run_mcts`が返す、探索直後の価値推定(`root_value`)。
        """
        self.encoder_sv = encoder_sv
        self.decoder_sv = decoder_sv
        self.policy_target = policy_target
        self.value = value
        self.label: float | None = None  # 価値の教師信号。ゲーム終了後に`_assign_labels`で埋める


def make_eval_fn(network, decks: list[list[int]]):
    """`PolicyValueNet`をラップして、`search.create_node`が要求する`eval_fn`にする。

    探索木は複数ターンにまたがりうるため、ノードごとに「今まさに手番を選んでいる側
    (`obs.current.yourIndex`)の本当のデッキ」を`decks`から引いて`get_encoder_input`に渡す
    (`decks[0]`/`decks[1]`のどちらがその局面で手番を持っているかは局面ごとに変わりうる)。

    Args:
        network: 評価に使う`PolicyValueNet`(eval modeで呼び出し側が管理すること)。
        decks: `[player0の60枚デッキ, player1の60枚デッキ]`。本番推論時は
            `[自分のデッキ, 自分のデッキ]`のように同じものを渡してよい
            (相手の本当のデッキは分からないため)。

    Returns:
        Callable[[Observation, list[list[int]]], tuple[list[float], float]]: `eval_fn`。
    """

    def eval_fn(obs, actions: list[list[int]]):
        your_deck = decks[obs.current.yourIndex]
        encoder_sv = get_encoder_input(obs, your_deck)
        decoder_sv = get_decoder_input(obs, actions)
        index_enc, value_enc, offset_enc = encoder_sv.to_tensors()
        index_dec, value_dec, offset_dec = decoder_sv.to_tensors()
        with torch.no_grad():
            value, scores = network(
                index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec
            )
        probs = torch.softmax(scores[0], dim=-1).tolist()
        return probs, float(value.item())

    return eval_fn


def _search_begin_kwargs(obs, my_true_deck: list[int]) -> dict:
    """今まさに手番を選んでいる側の視点で、隠れ情報の仮定一式を組み立てる。

    自分の本当のデッキ(`my_true_deck`)は探索側にとって既知だが、相手の本当のデッキ
    (自己対戦なので実際には`play_selfplay_game`が知っている)はカンニングさせず、
    本番の対戦相手と同じくカード出現頻度からの推測(`determinize.sample_opponent_hidden`)
    を使う。これにより、学習時の探索と本番推論時の探索が同じ前提で動く。

    Args:
        obs: 探索を開始したい局面のObservation。
        my_true_deck: 今探索している側が実際に使っている60枚のデッキリスト。

    Returns:
        dict: `search_begin`にそのまま`**kwargs`で渡せる引数一式。
    """
    state = obs.current
    your_index = state.yourIndex
    me = state.players[your_index]
    opp = state.players[1 - your_index]

    # 自分の手札はObservationで既知なので、hand_count=0にしてdeck/prizeだけ受け取る
    your_deck, _, your_prize = sample_full_hidden(my_true_deck, me.deckCount, 0, len(me.prize))
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
    network, our_deck: list[int], opponent_deck_pool: list[list[int]], search_count: int
) -> tuple[list[Sample], int]:
    """1試合分の自己対戦を行い、方策・価値の学習サンプルを集める。

    対戦相手には`opponent_deck_pool`(過去リプレイの実在デッキ)からランダムに選んだ
    ものを使う。どちらの座席が`our_deck`を使うかも毎回ランダムに決め、先手/後手の
    偏りが学習データに乗らないようにする。学習サンプルは`our_deck`を使っている側の
    意思決定のみを集める(対戦相手側のデッキを学習対象にする必要は無いため)。

    Args:
        network: 探索の事前分布・評価値に使う`PolicyValueNet`(eval modeにしておくこと)。
        our_deck: 本番でも使う、こちらの60枚のデッキリスト。
        opponent_deck_pool: 対戦相手として選ぶ、実在デッキ(60枚)のリスト。
        search_count: 1手あたりのMCTSシミュレーション回数。

    Returns:
        tuple[list[Sample], int]: (`our_deck`側の、labelまで埋めたSampleのリスト,
            `state.result`(0/1が勝者のplayerIndex、2は引き分け))。
    """
    our_seat = random.randint(0, 1)
    decks: list[list[int]] = [our_deck, our_deck]
    decks[1 - our_seat] = random.choice(opponent_deck_pool)

    eval_fn = make_eval_fn(network, decks)
    obs_dict, start_data = battle_start(decks[0], decks[1])
    if start_data.errorPlayer >= 0:
        raise ValueError(f"deck error: errorType={start_data.errorType}")

    samples: list[Sample] = []
    obs = to_observation_class(obs_dict)

    while obs.current.result < 0:
        sel = obs.select
        if (
            sel.type in (SelectType.MAIN, SelectType.EVOLVE)
            and sel.minCount == sel.maxCount == 1
            and (len(sel.option) >= 2)
        ):
            your_index = obs.current.yourIndex
            root_state = search_begin(obs, **_search_begin_kwargs(obs, decks[your_index]))
            try:
                select, policy_target, root_value, actions = run_mcts(
                    root_state, your_index, eval_fn, search_count
                )
            finally:
                search_end()
            if your_index == our_seat:
                encoder_sv = get_encoder_input(obs, our_deck)
                decoder_sv = get_decoder_input(obs, actions)
                samples.append(Sample(encoder_sv, decoder_sv, policy_target, root_value))
        else:
            select = _fallback_random(sel)
        obs = to_observation_class(battle_select(select))

    battle_finish()

    winner = obs.current.result
    _assign_labels(samples, winner, our_seat)

    return samples, winner
