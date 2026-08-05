"""PPO用の自己対戦ロールアウト収集。

MCTS(`mcts/selfplay.py`)と違い、探索を行わずネットワークの方策から直接サンプリングして
1手ずつ進める。1手あたりのネット評価が1回で済むため、MCTS自己対戦(`SEARCH_COUNT`回評価)
より大幅に軽い。探索を行わないので、隠れ情報を仮定する`determinize.py`も不要で、
実際の`Observation`をそのまま使う。

`mcts/selfplay.py`と同じ"asymmetric"/"mirror"/"generalist"の3モードを切り替えられる
(詳細は同ファイルのモジュールdocstringを参照)。
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT / "src" / "training" / "common"))
sys.path.insert(0, str(ROOT / "src" / "training" / "mcts"))
from search import _enumerate_actions  # noqa: E402
from selfplay_modes import SelfplayMode, pick_decks_and_collect_seats  # noqa: E402
from sparse_features import SparseVector, get_decoder_input, get_encoder_input  # noqa: E402

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


class PPOSample:
    """1回の意思決定について集めた、PPO学習用のサンプル。"""

    def __init__(
        self, encoder_sv, decoder_sv, action_index: int, old_log_prob: float, value: float
    ) -> None:
        """
        Args:
            encoder_sv: `sparse_features.get_encoder_input`が返す盤面の疎ベクトル。
            decoder_sv: `sparse_features.get_decoder_input`が返す行動群の疎ベクトル。
            action_index: 列挙された行動のうち実際に選んだもののindex。
            old_log_prob: 収集時点の方策(behavior policy)での、選んだ行動のlog確率。
            value: 収集時点のネットワークによる価値推定`V(s)`。
        """
        self.encoder_sv = encoder_sv
        self.decoder_sv = decoder_sv
        self.action_index = action_index
        self.old_log_prob = old_log_prob
        self.value = value
        self.reward = 0.0  # 終端以外は0。ゲーム終了後に最後のサンプルだけ書き換える
        self.advantage: float | None = None  # GAEで後から埋める
        self.return_: float | None = None  # GAEで後から埋める(advantage + value)


def evaluate_policy(network, obs, deck: list[int]):
    """現局面を`PolicyValueNet`に通し、列挙済みの行動・方策ロジット・価値推定を得る。

    自己対戦(サンプリング)・本番推論(貪欲方策)の両方から共通で使う、特徴量エンコード
    〜forward計算までの処理。

    Args:
        network: 評価に使う`PolicyValueNet`(eval modeで呼び出し側が管理すること)。
        obs: 現在の`Observation`。
        deck: 今まさに手番を選んでいる側の60枚のデッキリスト。

    Returns:
        tuple[list[list[int]], torch.Tensor, float, SparseVector, SparseVector]:
            (列挙済みの行動一覧, 方策の生ロジット(形状`(n_actions,)`), 価値推定`V(s)`,
            盤面の疎ベクトル, 行動群の疎ベクトル)。
    """
    actions = _enumerate_actions(obs.select)
    encoder_sv = get_encoder_input(obs, deck)
    decoder_sv = get_decoder_input(obs, actions)
    index_enc, value_enc, offset_enc = encoder_sv.to_tensors()
    index_dec, value_dec, offset_dec = decoder_sv.to_tensors()
    with torch.inference_mode():
        value, scores = network(index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec)
    return actions, scores[0], float(value.item()), encoder_sv, decoder_sv


def _act(
    network, obs, your_deck: list[int]
) -> tuple[int, float, float, list[list[int]], SparseVector, SparseVector]:
    """現在の方策からカテゴリカルサンプリングで1つ行動を選ぶ。

    Args:
        network: 評価に使う`PolicyValueNet`(eval modeで呼び出し側が管理すること)。
        obs: 現在の`Observation`。
        your_deck: 今まさに手番を選んでいる側の60枚のデッキリスト。

    Returns:
        tuple[int, float, float, list[list[int]], SparseVector, SparseVector]:
            (選んだ行動のindex, その行動のlog確率, 価値推定`V(s)`, 列挙済みの行動一覧,
            盤面の疎ベクトル, 行動群の疎ベクトル)。
    """
    actions, scores, value, encoder_sv, decoder_sv = evaluate_policy(network, obs, your_deck)
    dist = torch.distributions.Categorical(logits=scores)
    action_index = int(dist.sample().item())
    log_prob = float(dist.log_prob(torch.tensor(action_index)).item())
    return action_index, log_prob, value, actions, encoder_sv, decoder_sv


def play_ppo_game(
    network,
    our_deck: list[int],
    opponent_deck_pool: list[list[int]] | None,
    mode: SelfplayMode = "asymmetric",
) -> tuple[list[list[PPOSample]], int]:
    """1試合分の自己対戦を行い、PPO学習用のサンプルを集める。

    `mcts/selfplay.py`の`play_selfplay_game`と同じ"asymmetric"/"mirror"/"generalist"モード
    構成だが、探索を行わず方策から直接サンプリングする点が異なる。GAE計算はサンプルの
    時系列的な隣接関係(1つのMDP軌跡)に依存するため、座席ごとに分けたまま返す
    (呼び出し側が`compute_gae`を座席ごとに適用してから結合すること。特に"mirror"/
    "generalist"では両座席分の軌跡を1つの試合から得るので、ここで結合してしまうと
    片方の軌跡の終端がもう片方の軌跡の先頭に繋がってしまいGAEが壊れる)。

    Args:
        network: 方策・価値に使う`PolicyValueNet`(eval modeにしておくこと)。
        our_deck: 本番でも使う、こちらの60枚のデッキリスト。`mode="generalist"`のときは
            自己対戦のデッキ選択には使われない。
        opponent_deck_pool: 対戦相手として選ぶ、実在デッキ(60枚)のリスト。
            `mode="mirror"`のときは使われないので`None`でよい。
        mode: "asymmetric"(相手デッキランダム・自分側のみ学習)、
            "mirror"(両者同デッキ・両サイド学習)、
            "generalist"(両者とも実在デッキプールから独立ランダム・両サイド学習)。

    Returns:
        tuple[list[list[PPOSample]], int]: (`[player0のPPOSampleのリスト,
            player1のPPOSampleのリスト]`。学習サンプルを集めなかった側は空リスト。
            rewardまで埋め済み, `state.result`(0/1が勝者のplayerIndex、2は引き分け))。
    """
    decks, collect_seats = pick_decks_and_collect_seats(mode, our_deck, opponent_deck_pool)

    obs_dict, start_data = battle_start(decks[0], decks[1])
    try:
        if start_data.errorPlayer >= 0:
            raise ValueError(f"deck error: errorType={start_data.errorType}")

        samples_by_seat: list[list[PPOSample]] = [[], []]
        obs = to_observation_class(obs_dict)

        while obs.current.result < 0:
            your_index = obs.current.yourIndex
            action_index, log_prob, value, actions, encoder_sv, decoder_sv = _act(
                network, obs, decks[your_index]
            )
            if your_index in collect_seats:
                samples_by_seat[your_index].append(
                    PPOSample(encoder_sv, decoder_sv, action_index, log_prob, value)
                )
            obs = to_observation_class(battle_select(actions[action_index]))

        winner = obs.current.result
        if winner != 2:  # 引き分けはreward=0のまま(初期値)でよい
            for seat in collect_seats:
                if samples_by_seat[seat]:
                    samples_by_seat[seat][-1].reward = 1.0 if winner == seat else -1.0

        return samples_by_seat, winner
    finally:
        battle_finish()


def compute_gae(samples: list[PPOSample], gamma: float, lam: float) -> None:
    """1試合・1座席分の`PPOSample`列に、GAE(Generalized Advantage Estimation)で
    advantage/returnを付ける。

    このゲームは対局終了時にしか報酬が無い(`samples`の最後だけ`reward`が非0)。
    `samples`は同一試合・同一座席の意思決定だけを時系列順に並べたものなので、これを1つの
    MDP軌跡とみなし、最後のサンプルは終端(それ以降のブートストラップ値は0)として扱う。

    Args:
        samples: `play_ppo_game`が返す`samples_by_seat`の1座席分(時系列順、1試合分)。
        gamma: 割引率。
        lam: GAEのλ(バイアスと分散のトレードオフ)。
    """
    last_advantage = 0.0
    for t in reversed(range(len(samples))):
        next_value = samples[t + 1].value if t + 1 < len(samples) else 0.0
        delta = samples[t].reward + gamma * next_value - samples[t].value
        last_advantage = delta + gamma * lam * last_advantage
        samples[t].advantage = last_advantage
        samples[t].return_ = last_advantage + samples[t].value
