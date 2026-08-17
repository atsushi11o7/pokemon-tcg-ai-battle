"""Determinized MCTSによる自己対戦で、方策・価値ネットワークの学習データを集める。

`cg.game`で直接対戦を回し、あらゆる選択(セットアップ・サーチ効果・YES/NO確認等も含む)の
たびにMCTS探索を行う。選択肢が実質1つしかない局面も同じ探索木の枠組みで扱う(子が1つの
木として自然に処理される)。探索結果(訪問回数分布)を方策の教師信号、ゲーム終了後に
補正した価値推定を価値の教師信号として、1試合分の学習サンプルを作る。

3つの自己対戦モードを切り替えられる(`mode`引数、詳細は`play_selfplay_game`のdocstring参照)。
"asymmetric"(固定デッキ対ランダム・両サイド学習)、"mirror"(両者同デッキ・両サイド学習)、
"generalist"(両者独立ランダム・両サイド学習)。いずれも両サイドを学習し、探索側は自分のデッキだけを知り、
相手の隠れ情報は本番と同じく、公開カードと整合する実在デッキ候補から推測する。
"""

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]

from ..common.deck import SelfplayMode, pick_decks_and_collect_seats  # noqa: E402
from ..common.sparse_features import get_decoder_input, get_encoder_input  # noqa: E402
from .determinize import determinize_for_search  # noqa: E402
from .search import run_mcts  # noqa: E402

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import search_begin, search_end, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

LAMBDA = 0.9
NUM_DETERMINIZATIONS = 5
ROOT_DIRICHLET_ALPHA = 0.3
ROOT_NOISE_FRACTION = 0.25
SELFPLAY_TEMPERATURE_TURNS = 20


class Sample:
    """1回の選択(2択以上)について集めた学習サンプル。"""

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


def make_eval_fn(network, decks: list[list[int]], leaf_value: float | None = None):
    """`PolicyValueNet`をラップして、`search.create_node`が要求する`eval_fn`にする。

    探索木は複数ターンにまたがりうるため、ノードごとに「今まさに手番を選んでいる側
    (`obs.current.yourIndex`)の仮定デッキ」を`decks`から引いて`get_encoder_input`に渡す
    (`decks[0]`/`decks[1]`のどちらがその局面で手番を持っているかは局面ごとに変わりうる)。

    Args:
        network: 評価に使う`PolicyValueNet`(eval modeで呼び出し側が管理すること)。
        decks: `[player0の60枚デッキ, player1の60枚デッキ]`。本番推論時は
            `[自分のデッキ, 自分のデッキ]`のように同じものを渡してよい
            (相手の本当のデッキは分からないため)。
        leaf_value: 指定すると葉の評価を価値ヘッドではなくこの定数にする。
            ラダー実績1046.1の#34の価値ヘッドは、実測で自分の勝ち局面の符号正答率94.1%・
            負け局面16.1%(平均+0.850)と、ほぼ常に+1を返すだけのものだった。
            `create_node`は手番が根の側でなければ符号を反転するので、定数+1は
            「葉が自分の手番なら+1、相手の手番なら-1」となり、実質「自分の手番が
            多く回る手順を選ぶ」低分散なヒューリスティックとして働いていた。
            価値ヘッドを正しく学習し直した#39(符号正答率68.8%)は、探索が貪欲方策を
            上書きする割合が2.7%から11.0%へ増え、ラダーは1005.4から614.0へ落ちた。
            50シミュレーション/仮説では終局までほとんど届かず、探索は実質1手先の
            価値評価にしかならない。方策(精度約80%)の方が着手の順位付けとしては
            優れているため、定数に固定して探索を#34の挙動へ戻せるようにする。
            終局ノードは`create_node`が実際の勝敗を使うので、ここでは変えない。

    Returns:
        Callable[[Observation, list[list[int]]], tuple[list[float], float]]: `eval_fn`。
    """

    def eval_fn(obs, actions: list[list[int]]):
        your_deck = decks[obs.current.yourIndex]
        encoder_sv = get_encoder_input(obs, your_deck)
        decoder_sv = get_decoder_input(obs, actions)
        index_enc, value_enc, offset_enc = encoder_sv.to_tensors()
        index_dec, value_dec, offset_dec = decoder_sv.to_tensors()
        with torch.inference_mode():
            value, scores = network(
                index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec
            )
        probs = torch.softmax(scores[0], dim=-1).tolist()
        return probs, float(value.item()) if leaf_value is None else leaf_value

    return eval_fn


def run_determinized_mcts(
    network,
    obs,
    own_deck: list[int],
    opponent_deck_pool: list[list[int]],
    search_count: int,
    *,
    num_determinizations: int = NUM_DETERMINIZATIONS,
    add_root_noise: bool = False,
    temperature: float | None = None,
    leaf_value: float | None = None,
) -> tuple[list[int], list[float], float, list[list[int]]]:
    """複数の隠れ情報仮説へ探索予算を分配し、根の方策と価値を集約する。

    `leaf_value`を渡すと葉の評価を価値ヘッドではなく定数にする(`make_eval_fn`参照)。
    """
    hypothesis_count = min(max(1, num_determinizations), max(1, search_count))
    base_budget, remainder = divmod(search_count, hypothesis_count)
    aggregate_policy: list[float] | None = None
    aggregate_value = 0.0
    total_weight = 0
    common_actions: list[list[int]] | None = None

    for hypothesis_index in range(hypothesis_count):
        budget = base_budget + int(hypothesis_index < remainder)
        kwargs, assumed_decks = determinize_for_search(obs, own_deck, opponent_deck_pool)
        eval_fn = make_eval_fn(network, assumed_decks, leaf_value)
        root_state = search_begin(obs, **kwargs)
        try:
            _select, policy, value, actions = run_mcts(
                root_state,
                obs.current.yourIndex,
                eval_fn,
                budget,
                root_dirichlet_alpha=ROOT_DIRICHLET_ALPHA if add_root_noise else None,
                root_noise_fraction=ROOT_NOISE_FRACTION if add_root_noise else 0.0,
            )
        finally:
            search_end()

        if common_actions is None:
            common_actions = actions
            aggregate_policy = [0.0] * len(policy)
        elif actions != common_actions:
            raise RuntimeError("root legal actions changed across determinizations")
        weight = max(1, budget)
        for index, probability in enumerate(policy):
            aggregate_policy[index] += probability * weight
        aggregate_value += value * weight
        total_weight += weight

    assert common_actions is not None and aggregate_policy is not None
    policy_target = [value / total_weight for value in aggregate_policy]
    root_value = aggregate_value / total_weight
    if temperature is None or temperature <= 0:
        selected_index = max(range(len(policy_target)), key=policy_target.__getitem__)
    else:
        weights = [probability ** (1.0 / temperature) for probability in policy_target]
        selected_index = random.choices(range(len(weights)), weights=weights, k=1)[0]
    return common_actions[selected_index], policy_target, root_value, common_actions


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
    network,
    our_deck: list[int],
    opponent_deck_pool: list[list[int]],
    search_count: int,
    mode: SelfplayMode = "asymmetric",
    fixed_deck_seat: int | None = None,
    num_determinizations: int = NUM_DETERMINIZATIONS,
) -> tuple[list[Sample], int]:
    """1試合分の自己対戦を行い、方策・価値の学習サンプルを集める。

    `mode="asymmetric"`(既定)では対戦相手に`opponent_deck_pool`からランダムに選んだ実在デッキを
    使い、trainerが試合番号の偶奇により`our_deck`の座席を均等に割り当てる。
    固定デッキ側とランダムデッキ側の両方を学習する。`mode="mirror"`では両者とも`our_deck`を使い、
    `mode="generalist"`では両者とも`opponent_deck_pool`から独立にランダムに選んだ実在デッキ
    (両者同じ組み合わせになることもある)を使い、`our_deck`は使わない(あらゆる実在デッキを
    乗りこなす汎用方策を狙う)。いずれのモードでも両サイドの意思決定を学習サンプルにする。

    Args:
        network: 探索の事前分布・評価値に使う`PolicyValueNet`(eval modeにしておくこと)。
        our_deck: 本番でも使う、こちらの60枚のデッキリスト。`mode="generalist"`のときは
            自己対戦のデッキ選択には使われない。
        opponent_deck_pool: 対戦相手として選ぶ、実在デッキ(60枚)のリスト。
            デッキ選択に使わない"mirror"でも、探索時の相手隠れ情報の推測に必要。
        search_count: 1手あたりのMCTSシミュレーション回数。
        mode: "asymmetric"(固定デッキ対ランダムデッキ・両サイド学習)、
            "mirror"(両者同デッキ・両サイド学習、公式サンプルコードと同じ構成)、
            "generalist"(両者とも実在デッキプールから独立ランダム・両サイド学習)。
        fixed_deck_seat: asymmetricで固定デッキを置く座席。trainerから0/1を交互に渡す。
        num_determinizations: 隠れ情報の仮説数。`search_count`はこの数へ分割されるので、
            1仮説あたりの探索の深さは`search_count // num_determinizations`になる。

    Returns:
        tuple[list[Sample], int]: (labelまで埋めた両サイド分のSample、
            `state.result`(0/1が勝者のplayerIndex、2は引き分け))。
    """
    decks, collect_seats = pick_decks_and_collect_seats(
        mode, our_deck, opponent_deck_pool, fixed_deck_seat
    )
    obs_dict, start_data = battle_start(decks[0], decks[1])
    try:
        if start_data.errorPlayer >= 0:
            raise ValueError(f"deck error: errorType={start_data.errorType}")

        samples_by_seat: list[list[Sample]] = [[], []]
        obs = to_observation_class(obs_dict)

        while obs.current.result < 0:
            your_index = obs.current.yourIndex
            select, policy_target, root_value, actions = run_determinized_mcts(
                network,
                obs,
                decks[your_index],
                opponent_deck_pool,
                search_count,
                num_determinizations=num_determinizations,
                add_root_noise=True,
                temperature=1.0 if obs.current.turn <= SELFPLAY_TEMPERATURE_TURNS else None,
            )
            if your_index in collect_seats:
                encoder_sv = get_encoder_input(obs, decks[your_index])
                decoder_sv = get_decoder_input(obs, actions)
                samples_by_seat[your_index].append(
                    Sample(encoder_sv, decoder_sv, policy_target, root_value)
                )
            obs = to_observation_class(battle_select(select))

        winner = obs.current.result
        for seat in collect_seats:
            _assign_labels(samples_by_seat[seat], winner, seat)

        return samples_by_seat[0] + samples_by_seat[1], winner
    finally:
        battle_finish()
