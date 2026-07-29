"""自己対戦で学習する、方策(policy)と価値(value)を同時に出力するネットワーク。

`bc/train_bc.py`のPolicyNetと同じstate_encoder/option_encoder/scorer構成を再利用しつつ、
状態エンコーダの出力から局面の価値(勝敗換算のスカラー)を予測するvalue_headを追加する。
BCの学習済み重みは「あくまで最適化の出発点」として初期値に使うだけで、自己対戦の
学習ループが1ラウンド目から方策・価値の両方を更新し続ける前提になっている
(BCの重みに凍結して依存し続けるわけではない)。
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "training" / "bc"))

from train_bc import MODEL_PATH as BC_MODEL_PATH  # noqa: E402


class PolicyValueNet(nn.Module):
    """状態と選択肢のペアに方策スコアを、状態単体に価値を付けるネットワーク。

    `state_encoder`/`option_encoder`/`scorer`は`bc.train_bc.PolicyNet`と全く同じ構成にしてある
    (初期値をBCの学習済み重みからそのままコピーできるように、意図的に形を揃えている)。
    `value_head`はBCには無い、このネットワークで新規に追加した部分。
    """

    def __init__(self, state_dim: int, option_dim: int, hidden_dim: int) -> None:
        """
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
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, state: torch.Tensor, options: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """バッチ分の状態と、パディング済みの選択肢群から、方策スコアと価値を計算する。

        Args:
            state: 形状`(batch, state_dim)`の状態ベクトル。
            options: 形状`(batch, max_options, option_dim)`の選択肢ベクトル群
                (各局面の実際の選択肢数を超える分は0埋めされている)。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (policy_scores, value)。
                policy_scores: 形状`(batch, max_options)`の生スコア(softmax前、パディング部分も含む。
                    マスクは呼び出し側で適用する)。
                value: 形状`(batch,)`の価値。`tanh`で[-1, 1]にクリップ済み。
        """
        state_feat = self.state_encoder(state)
        max_options = options.shape[1]
        state_emb = state_feat.unsqueeze(1).expand(-1, max_options, -1)
        option_emb = self.option_encoder(options)
        combined = torch.cat([state_emb, option_emb], dim=-1)
        policy_scores = self.scorer(combined).squeeze(-1)
        value = torch.tanh(self.value_head(state_feat).squeeze(-1))
        return policy_scores, value


def new_network_from_bc(state_dim: int, option_dim: int, hidden_dim: int) -> PolicyValueNet:
    """BCの学習済み重みを初期値として使った`PolicyValueNet`を作る。

    `state_encoder`/`option_encoder`/`scorer`はBCの学習済み重みをそのままコピーする
    (自己対戦の学習ループが1ラウンド目から更新していくための出発点であり、以後も
    凍結はしない)。`value_head`はBCの重みに存在しない部分なので、通常のランダム
    初期化のままにする。

    Args:
        state_dim: `encode_state`が返す状態ベクトルの長さ。
        option_dim: `encode_option`が返す選択肢ベクトルの長さ。
        hidden_dim: 各MLPの隠れ層の次元数(BC学習時と同じ値にすること)。

    Returns:
        PolicyValueNet: state_encoder/option_encoder/scorerをBCの重みで初期化したネットワーク。
    """
    net = PolicyValueNet(state_dim, option_dim, hidden_dim)
    bc_state = torch.load(BC_MODEL_PATH, weights_only=True)
    net.load_state_dict(bc_state, strict=False)
    return net
