"""Behavior Cloningのモデルを学習する。

extract_dataset.pyが作った(状態, 選択肢群, 正解index)のデータセットを使い、
「状態と選択肢のペアにスコアを付け、合法な選択肢の中でsoftmaxを取って
実際に選ばれたものとの交差エントロピーを最小化する」方策ネットワークを学習する。

局面ごとに選択肢の数が異なるため、`torch.utils.data.DataLoader`のcollate_fnで
バッチ内の最大選択肢数までゼロ埋め(padding)し、埋めた分はマスクでスコアを-infにして
softmax/交差エントロピーの計算から除外することで、通常のミニバッチ学習として扱う。

Usage:
    uv run python src/training/bc/train_bc.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import OPTION_DIM, STATE_DIM  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = ROOT / "outputs" / "bc_dataset.npz"
MODEL_PATH = ROOT / "outputs" / "bc_policy.pt"

HIDDEN_DIM = 128
N_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_RATIO = 0.15
SEED = 0


class PolicyNet(nn.Module):
    """状態と選択肢のペアにスコアを付けるネットワーク。

    状態エンコーダと選択肢エンコーダで別々に埋め込みを作り、連結した上で
    1つのスカラースコアに落とす。局面ごとの選択肢の数が可変でも、
    「1つの選択肢に対してスコアを1つ返す」形にしておけば、パディングした
    選択肢をまとめてバッチ処理でき、可変長の行動空間に対応できる。
    """

    def __init__(self, state_dim: int, option_dim: int, hidden_dim: int) -> None:
        """状態エンコーダ・選択肢エンコーダ・スコアラーの3つのMLPを組み立てる。

        Args:
            state_dim: `encode_state`が返す状態ベクトルの長さ(`STATE_DIM`)。
            option_dim: `encode_option`が返す選択肢ベクトルの長さ(`OPTION_DIM`)。
            hidden_dim: 各MLPの隠れ層の次元数。
        """
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.option_encoder = nn.Sequential(
            nn.Linear(option_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        """バッチ分の状態と、パディング済みの選択肢群のスコアを返す。

        Args:
            state: 形状`(batch, state_dim)`の状態ベクトル。
            options: 形状`(batch, max_options, option_dim)`の選択肢ベクトル群
                （各局面の実際の選択肢数を超える分は0埋めされている）。

        Returns:
            torch.Tensor: 形状`(batch, max_options)`の生スコア（softmax前、
                パディング部分も含む。マスクはこの関数の外で適用する）。
        """
        max_options = options.shape[1]
        state_emb = self.state_encoder(state).unsqueeze(1).expand(-1, max_options, -1)
        option_emb = self.option_encoder(options)
        combined = torch.cat([state_emb, option_emb], dim=-1)
        return self.scorer(combined).squeeze(-1)


class BCDataset(Dataset):
    """extract_dataset.pyが保存したnpzを、局面ごとの(状態, 選択肢群, 正解index)として読み込む。"""

    def __init__(self, dataset_path: Path) -> None:
        """npzファイルを読み込み、フラット化された配列を局面ごとの例に復元する。

        Args:
            dataset_path: `extract_dataset.py`が保存したnpzファイルのパス。
        """
        data = np.load(dataset_path)
        states = data["states"]
        options = data["options"]
        option_counts = data["option_counts"]
        labels = data["labels"]

        self.examples: list[tuple[np.ndarray, np.ndarray, int]] = []
        offset = 0
        for state, count, label in zip(states, option_counts, labels, strict=True):
            self.examples.append((state, options[offset : offset + count], int(label)))
            offset += count

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        return self.examples[idx]


def collate_fn(
    batch: list[tuple[np.ndarray, np.ndarray, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """可変長の選択肢群を、バッチ内の最大選択肢数までゼロ埋めしてテンソル化する。

    Args:
        batch: `BCDataset.__getitem__`が返すタプルのリスト。

    Returns:
        tuple: (states, options, mask, labels)。
            states: 形状`(batch, state_dim)`。
            options: 形状`(batch, max_options, option_dim)`（パディング済み）。
            mask: 形状`(batch, max_options)`のbool。パディングした位置はFalse。
            labels: 形状`(batch,)`の正解index。
    """
    states = torch.from_numpy(np.stack([b[0] for b in batch]))
    option_counts = [b[1].shape[0] for b in batch]
    max_options = max(option_counts)
    option_dim = batch[0][1].shape[1]

    options = torch.zeros(len(batch), max_options, option_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_options, dtype=torch.bool)
    for i, (_, opts, _) in enumerate(batch):
        n = opts.shape[0]
        options[i, :n] = torch.from_numpy(opts)
        mask[i, :n] = True

    labels = torch.tensor([b[2] for b in batch], dtype=torch.int64)
    return states, options, mask, labels


def masked_cross_entropy(
    scores: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """パディング部分を除外した上で、選択肢に対する交差エントロピー損失を計算する。

    Args:
        scores: 形状`(batch, max_options)`の生スコア（`PolicyNet.forward`の出力）。
        mask: 形状`(batch, max_options)`のbool。パディングした位置はFalse。
        labels: 形状`(batch,)`の正解index。

    Returns:
        torch.Tensor: バッチ平均の交差エントロピー損失（スカラー）。
    """
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    return functional.cross_entropy(masked_scores, labels)


def train() -> None:
    """データセットを学習/検証に分割し、`PolicyNet`をN_EPOCHS分学習して保存する。

    エポックごとに学習損失と検証損失・検証accuracyを表示し、最後に
    `MODEL_PATH`へ`state_dict()`のみを保存する（本番提出物は素のPyTorchで
    推論する前提のため、モデル定義自体は保存しない）。
    """
    dataset = BCDataset(DATASET_PATH)
    print(f"loaded {len(dataset)} examples")

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    n_val = int(len(dataset) * VAL_RATIO)
    val_dataset = Subset(dataset, indices[:n_val])
    train_dataset = Subset(dataset, indices[n_val:])
    print(f"train={len(train_dataset)} val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    torch.manual_seed(SEED)
    model = PolicyNet(STATE_DIM, OPTION_DIM, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0.0
        for states, options, mask, labels in train_loader:
            scores = model(states, options)
            loss = masked_cross_entropy(scores, mask, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * states.shape[0]

        train_loss = total_loss / len(train_dataset)
        val_loss, val_acc = evaluate(model, val_loader, len(val_dataset))
        print(
            f"epoch {epoch + 1}/{N_EPOCHS}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved model to {MODEL_PATH}")


def evaluate(model: PolicyNet, loader: DataLoader, n_examples: int) -> tuple[float, float]:
    """検証データに対する平均損失と、最尤の選択肢が正解と一致する割合(accuracy)を計算する。

    Args:
        model: 評価対象のモデル。
        loader: 検証データのDataLoader。
        n_examples: 検証データの件数（平均を取るための分母）。

    Returns:
        tuple[float, float]: (平均交差エントロピー損失, 正解率)。
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for states, options, mask, labels in loader:
            scores = model(states, options)
            loss = masked_cross_entropy(scores, mask, labels)
            total_loss += loss.item() * states.shape[0]

            masked_scores = scores.masked_fill(~mask, float("-inf"))
            predictions = torch.argmax(masked_scores, dim=-1)
            correct += int((predictions == labels).sum().item())

    return total_loss / n_examples, correct / n_examples


if __name__ == "__main__":
    train()
