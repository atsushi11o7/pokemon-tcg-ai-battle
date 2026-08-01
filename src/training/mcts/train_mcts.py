"""Determinized MCTSの自己対戦(selfplay.py)で集めたデータで、価値・方策ネットワークを学習する。

自己対戦→学習→(更新したネットワークで)自己対戦、を繰り返すAlphaZero的なループ。
ネットワークは完全ランダム初期化から開始する。

Usage:
    uv run python src/training/mcts/train_mcts.py
"""

import copy
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from determinize import load_opponent_deck_pool, search_begin_kwargs  # noqa: E402
from network import PolicyValueNet, SparseBatch  # noqa: E402
from search import run_mcts  # noqa: E402
from selfplay import Sample, make_eval_fn, play_selfplay_game  # noqa: E402

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import (  # noqa: E402
    Observation,
    search_begin,
    search_end,
    to_observation_class,
)

sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from match_runner import evaluate as match_evaluate  # noqa: E402

DECK_PATH = ROOT / "decks" / "cynthias_garchomp_ex.csv"
CHECKPOINT_DIR = ROOT / "outputs" / "mcts_checkpoints"

GAMES_PER_ROUND = 200  # 全選択を探索するようになった分、1試合あたり約3.5秒(以前は2.2秒)
N_ROUNDS = 30  # ランダム初期化から自己対戦のみで学習
SEARCH_COUNT = 10  # 1手あたりのMCTSシミュレーション回数
EPOCHS_PER_ROUND = 3
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SEED = 0
EVAL_GAMES_PER_ROUND = 10  # ラウンドごとのアリーナ戦・ランダム相手戦の試合数
GATING_WIN_RATE = 0.5  # 候補ネットワークをbest_networkとして採用する最低勝率

# ネットワーク構成(公式サンプルコードの`MyModel(128, 2, 256, 1, 1)`と同じ規模)
D_MODEL = 128
NUM_HEADS = 2
D_FEEDFORWARD = 256
NUM_LAYERS_ENCODER = 1
NUM_LAYERS_DECODER = 1


def read_deck() -> list[int]:
    """自己対戦に使う、decks/配下の共通deck.csv(60行のカードID)を読み込む。

    Returns:
        list[int]: 60枚分のカードID。
    """
    return [int(x) for x in DECK_PATH.read_text().split("\n") if x.strip()]


class SelfplayDataset(Dataset):
    """自己対戦で集めた`Sample`のリストをそのままDatasetにする。"""

    def __init__(self, samples: list[Sample]) -> None:
        """
        Args:
            samples: `selfplay.play_selfplay_game`が返す`Sample`のリスト(複数ゲーム分をまとめてよい)。
        """
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def _collate(batch: list[Sample]):
    """可変長の行動群を、バッチ内の最大行動数までパディング(空行動)し、疎ベクトルを連結する。

    エンコーダ側は全サンプル共通で`sparse_features.NUM_WORDS_ENCODER`個のトークンなので
    パディング不要だが、デコーダ側(行動の数)は局面ごとに異なるため、
    `SparseBatch.add_empty_word`で埋める(公式サンプルコードの`LearnInput`と同じ扱い)。

    Args:
        batch: `Sample`のリスト。

    Returns:
        tuple: (index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec,
            mask, policy_targets, value_labels)。
            mask: 形状`(batch, max_actions)`のbool。パディングした行動位置はFalse。
            policy_targets: 形状`(batch, max_actions)`の教師分布(パディング部分は0)。
            value_labels: 形状`(batch,)`の価値の教師信号。
    """
    max_actions = max(len(s.policy_target) for s in batch)

    encoder_batch = SparseBatch()
    decoder_batch = SparseBatch()
    mask = torch.zeros(len(batch), max_actions, dtype=torch.bool)
    policy_targets = torch.zeros(len(batch), max_actions, dtype=torch.float32)

    for i, sample in enumerate(batch):
        encoder_batch.add(sample.encoder_sv)
        decoder_batch.add(sample.decoder_sv)
        n = len(sample.policy_target)
        mask[i, :n] = True
        policy_targets[i, :n] = torch.tensor(sample.policy_target, dtype=torch.float32)
        for _ in range(max_actions - n):
            decoder_batch.add_empty_word()

    index_enc, value_enc, offset_enc = encoder_batch.to_tensors()
    index_dec, value_dec, offset_dec = decoder_batch.to_tensors()
    value_labels = torch.tensor([s.label for s in batch], dtype=torch.float32)
    return (
        index_enc,
        value_enc,
        offset_enc,
        index_dec,
        value_dec,
        offset_dec,
        mask,
        policy_targets,
        value_labels,
    )


def _masked_policy_loss(
    scores: torch.Tensor, mask: torch.Tensor, policy_targets: torch.Tensor
) -> torch.Tensor:
    """パディング部分を除外した上で、方策の教師分布(訪問回数の正規化)に対する交差エントロピーを計算する。

    Args:
        scores: 形状`(batch, max_actions)`の生スコア(`PolicyValueNet.forward`の出力)。
        mask: 形状`(batch, max_actions)`のbool。パディングした位置はFalse。
        policy_targets: 形状`(batch, max_actions)`の教師分布(パディング部分は0)。

    Returns:
        torch.Tensor: バッチ平均の交差エントロピー損失(スカラー)。
    """
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    log_probs = functional.log_softmax(masked_scores, dim=-1)
    log_probs = log_probs.masked_fill(~mask, 0.0)  # -inf*0のnan化を防ぐ(教師も0なので影響なし)
    return -(policy_targets * log_probs).sum(dim=-1).mean()


def train_one_round(network: PolicyValueNet, samples: list[Sample], epochs: int) -> None:
    """1ラウンド分の自己対戦データで、`network`を数エポック学習する。

    Args:
        network: 更新対象のネットワーク(このラウンドの自己対戦にも使われたもの)。
        samples: このラウンドの自己対戦で集めた`Sample`のリスト。
        epochs: 学習エポック数。
    """
    dataset = SelfplayDataset(samples)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

    network.train()
    for epoch in range(epochs):
        total_policy_loss = 0.0
        total_value_loss = 0.0
        for (
            index_enc,
            value_enc,
            offset_enc,
            index_dec,
            value_dec,
            offset_dec,
            mask,
            policy_targets,
            value_labels,
        ) in loader:
            values, scores = network(
                index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec
            )
            policy_loss = _masked_policy_loss(scores, mask, policy_targets)
            value_loss = functional.mse_loss(values.squeeze(-1), value_labels)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = value_labels.shape[0]
            total_policy_loss += policy_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size

        n = len(dataset)
        print(
            f"  epoch {epoch + 1}/{epochs}  policy_loss={total_policy_loss / n:.4f}  "
            f"value_loss={total_value_loss / n:.4f}"
        )
    network.eval()


def _random_agent_factory(deck: list[int]):
    """`match_runner.evaluate`用の、完全ランダムなagent()関数を作る。

    Args:
        deck: デッキ提出時に返す60枚のデッキリスト。

    Returns:
        Callable[[dict], list[int]]: ランダムに選ぶagent関数。
    """

    def agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        sel = obs.select
        count = random.randint(sel.minCount, sel.maxCount)
        return random.sample(range(len(sel.option)), count)

    return agent


def make_mcts_eval_agent(network: PolicyValueNet, our_deck: list[int]):
    """`network`で、`match_runner.evaluate`用のagent()関数を作る。

    本番のagent()(`trained_agent.py`)と同じ構成(自分のデッキは既知、
    相手はカード出現頻度から推測)。

    Args:
        network: 評価対象の`PolicyValueNet`(eval modeにしておくこと)。
        our_deck: こちらの60枚のデッキリスト。

    Returns:
        Callable[[dict], list[int]]: `match_runner.evaluate`に渡せるagent関数。
    """
    eval_fn = make_eval_fn(network, [our_deck, our_deck])

    def agent(obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return our_deck

        root_state = search_begin(obs, **search_begin_kwargs(obs, our_deck))
        try:
            select, _policy_target, _root_value, _actions = run_mcts(
                root_state, obs.current.yourIndex, eval_fn, SEARCH_COUNT
            )
        finally:
            search_end()
        return select

    return agent


def run_training_loop(deck: list[int]) -> PolicyValueNet:
    """自己対戦→学習→アリーナ評価を`N_ROUNDS`回繰り返すメインループ。

    各ラウンド、`best_network`で自己対戦したデータで候補ネットワークを学習し、
    `best_network`とのミラー戦(両者とも`deck`を使用)に勝ち越した場合のみ
    `best_network`を候補で置き換える(AlphaZeroのアリーナ/ゲーティング)。
    ランダム相手との勝率も、汎化の絶対的な下限の目安として毎ラウンド測る。
    対戦相手には`determinize.load_opponent_deck_pool`で集めた実在デッキをランダムに使う。

    Args:
        deck: 本番でも使う、こちらの60枚のデッキリスト。

    Returns:
        PolicyValueNet: 最終ラウンド終了時点の`best_network`。
    """
    torch.manual_seed(SEED)
    best_network = PolicyValueNet(
        D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER
    )
    best_network.eval()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    opponent_deck_pool = load_opponent_deck_pool()
    print(f"loaded {len(opponent_deck_pool)} distinct opponent decks for self-play")

    for round_index in range(N_ROUNDS):
        print(f"=== round {round_index + 1}/{N_ROUNDS}: self-play ===")
        all_samples: list[Sample] = []
        results = {0: 0, 1: 0, 2: 0}
        for game_index in range(GAMES_PER_ROUND):
            samples, winner = play_selfplay_game(
                best_network, deck, opponent_deck_pool, SEARCH_COUNT
            )
            all_samples.extend(samples)
            results[winner] += 1
            if (game_index + 1) % 20 == 0 or game_index + 1 == GAMES_PER_ROUND:
                print(f"  game {game_index + 1}/{GAMES_PER_ROUND}  results_so_far={results}")
        print(f"  round results: {results}  total_samples={len(all_samples)}")

        print(f"=== round {round_index + 1}/{N_ROUNDS}: training ===")
        candidate_network = copy.deepcopy(best_network)
        train_one_round(candidate_network, all_samples, EPOCHS_PER_ROUND)

        print(f"=== round {round_index + 1}/{N_ROUNDS}: arena vs best_network ===")
        candidate_agent = make_mcts_eval_agent(candidate_network, deck)
        best_agent = make_mcts_eval_agent(best_network, deck)
        arena_result = match_evaluate(candidate_agent, best_agent, n_episodes=EVAL_GAMES_PER_ROUND)
        accepted = arena_result["win_rate"] > GATING_WIN_RATE
        print(f"  candidate vs best: {arena_result}  accepted={accepted}")

        if accepted:
            best_network = candidate_network

        print(f"=== round {round_index + 1}/{N_ROUNDS}: eval vs random ===")
        eval_agent = make_mcts_eval_agent(best_network, deck)
        vs_random = match_evaluate(
            eval_agent, _random_agent_factory(deck), n_episodes=EVAL_GAMES_PER_ROUND
        )
        print(f"  vs random: {vs_random}")

        checkpoint_path = CHECKPOINT_DIR / f"round{round_index + 1}.pt"
        torch.save(best_network.state_dict(), checkpoint_path)
        print(f"  saved checkpoint to {checkpoint_path}")

    return best_network


if __name__ == "__main__":
    run_training_loop(read_deck())
