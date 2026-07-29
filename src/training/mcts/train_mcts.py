"""Determinized MCTSの自己対戦(selfplay.py)で集めたデータで、価値・方策ネットワークを学習する。

自己対戦→学習→(更新したネットワークで)自己対戦、を繰り返すAlphaZero的なループ。
BCの学習済み重みを初期値として使うが(`network.new_network_from_bc`)、1ラウンド目から
方策・価値の両方をこの自己対戦ループ自身が更新していく(BCの重みは出発点に過ぎず、
凍結したpriorとして使い続けるわけではない)。

Usage:
    uv run python src/training/mcts/train_mcts.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from network import PolicyValueNet, new_network_from_bc  # noqa: E402
from selfplay import Sample, play_selfplay_game  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "training" / "bc"))
from features import OPTION_DIM, STATE_DIM  # noqa: E402
from train_bc import HIDDEN_DIM  # noqa: E402

DECK_PATH = ROOT / "submission" / "01_rule_based" / "deck.csv"
CHECKPOINT_DIR = ROOT / "outputs" / "mcts_checkpoints"

GAMES_PER_ROUND = 20
N_ROUNDS = 3
SEARCH_COUNT = 10  # 1手あたりのMCTSシミュレーション回数
EPOCHS_PER_ROUND = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SEED = 0


def read_deck() -> list[int]:
    """自己対戦に使う、01_rule_based/BCと共通のdeck.csv(60行のカードID)を読み込む。

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


def _collate(
    batch: list[Sample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """可変長の選択肢群・方策教師信号を、バッチ内の最大選択肢数までゼロ埋めする。

    Args:
        batch: `Sample`のリスト。

    Returns:
        tuple: (states, options, mask, policy_targets, value_labels)。
            states: 形状`(batch, state_dim)`。
            options: 形状`(batch, max_options, option_dim)`(パディング済み)。
            mask: 形状`(batch, max_options)`のbool。パディングした位置はFalse。
            policy_targets: 形状`(batch, max_options)`の教師分布(パディング部分は0)。
            value_labels: 形状`(batch,)`の価値の教師信号。
    """
    states = torch.from_numpy(np.stack([s.state_vec for s in batch]))
    max_options = max(s.option_vecs.shape[0] for s in batch)
    option_dim = batch[0].option_vecs.shape[1]

    options = torch.zeros(len(batch), max_options, option_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_options, dtype=torch.bool)
    policy_targets = torch.zeros(len(batch), max_options, dtype=torch.float32)
    for i, sample in enumerate(batch):
        n = sample.option_vecs.shape[0]
        options[i, :n] = torch.from_numpy(sample.option_vecs)
        mask[i, :n] = True
        policy_targets[i, :n] = torch.tensor(sample.policy_target, dtype=torch.float32)

    value_labels = torch.tensor([s.label for s in batch], dtype=torch.float32)
    return states, options, mask, policy_targets, value_labels


def _masked_policy_loss(
    scores: torch.Tensor, mask: torch.Tensor, policy_targets: torch.Tensor
) -> torch.Tensor:
    """パディング部分を除外した上で、方策の教師分布(訪問回数の正規化)に対する交差エントロピーを計算する。

    BCの`masked_cross_entropy`と異なり、正解が1つのラベルではなく確率分布(MCTSの訪問回数)
    なので、one-hotではなく分布同士の交差エントロピーとして計算する。

    Args:
        scores: 形状`(batch, max_options)`の生スコア(`PolicyValueNet.forward`の出力)。
        mask: 形状`(batch, max_options)`のbool。パディングした位置はFalse。
        policy_targets: 形状`(batch, max_options)`の教師分布(パディング部分は0)。

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
        for states, options, mask, policy_targets, value_labels in loader:
            scores, values = network(states, options)
            policy_loss = _masked_policy_loss(scores, mask, policy_targets)
            value_loss = functional.mse_loss(values, value_labels)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item() * states.shape[0]
            total_value_loss += value_loss.item() * states.shape[0]

        n = len(dataset)
        print(
            f"  epoch {epoch + 1}/{epochs}  policy_loss={total_policy_loss / n:.4f}  "
            f"value_loss={total_value_loss / n:.4f}"
        )
    network.eval()


def run_training_loop(deck: list[int]) -> PolicyValueNet:
    """自己対戦→学習を`N_ROUNDS`回繰り返すメインループ。

    Args:
        deck: 自己対戦に使う60枚のデッキリスト。

    Returns:
        PolicyValueNet: 最終ラウンド終了時点のネットワーク。
    """
    torch.manual_seed(SEED)
    network = new_network_from_bc(STATE_DIM, OPTION_DIM, HIDDEN_DIM)
    network.eval()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for round_index in range(N_ROUNDS):
        print(f"=== round {round_index + 1}/{N_ROUNDS}: self-play ===")
        all_samples: list[Sample] = []
        results = {0: 0, 1: 0, 2: 0}
        for game_index in range(GAMES_PER_ROUND):
            samples, winner = play_selfplay_game(network, deck, SEARCH_COUNT)
            all_samples.extend(samples)
            results[winner] += 1
            print(
                f"  game {game_index + 1}/{GAMES_PER_ROUND}  "
                f"winner={winner}  samples={len(samples)}"
            )
        print(f"  round results: {results}  total_samples={len(all_samples)}")

        print(f"=== round {round_index + 1}/{N_ROUNDS}: training ===")
        train_one_round(network, all_samples, EPOCHS_PER_ROUND)

        checkpoint_path = CHECKPOINT_DIR / f"round{round_index + 1}.pt"
        torch.save(network.state_dict(), checkpoint_path)
        print(f"  saved checkpoint to {checkpoint_path}")

    return network


if __name__ == "__main__":
    run_training_loop(read_deck())
