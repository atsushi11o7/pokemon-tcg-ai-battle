"""方策(policy)と価値(value)を同時に出力するTransformerネットワーク。

`sparse_features.get_encoder_input`が作る盤面の疎ベクトルを`TransformerEncoder`で
自己注意し、`sparse_features.get_decoder_input`が作る各行動の疎ベクトルを、
エンコーダ出力に交差注意(`DecoderLayer`)することで、行動ごとの方策スコアと
局面全体の価値を計算する。
"""

import torch
import torch.nn as nn
from sparse_features import SparseVector, decoder_size, encoder_size

NUM_WORDS_ENCODER = 24  # get_encoder_inputが作るトークン数(ベンチ8x2+アクティブ2+盤面2+手札1+デッキ1+スタジアム1+ターン情報1)


class DecoderLayer(nn.Module):
    """エンコーダの出力に対して交差注意を行う、デコーダの1層分。"""

    def __init__(self, d_model: int, num_heads: int, d_feedforward: int) -> None:
        """
        Args:
            d_model: 埋め込み次元数。
            num_heads: MultiheadAttentionのヘッド数。
            d_feedforward: フィードフォワード層の隠れ次元数。
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = nn.Linear(d_model, d_feedforward)
        self.fc2 = nn.Linear(d_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        """デコーダ側の系列`x`から、エンコーダの出力`encoder_out`への交差注意を1層分行う。

        Args:
            x: 形状`(n_actions, batch, d_model)`のデコーダ側の入力(行動ごとの埋め込み)。
            encoder_out: 形状`(24, batch, d_model)`のエンコーダの出力。

        Returns:
            torch.Tensor: 形状`(n_actions, batch, d_model)`の更新後のデコーダ側の系列。
        """
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = torch.relu(self.fc1(res))
        y = self.fc2(y)
        return self.norm2(res + y)


class PolicyValueNet(nn.Module):
    """盤面(疎ベクトル)から価値を、各行動(疎ベクトル)から方策スコアを計算するTransformer。

    盤面側はTransformerEncoderで自己注意し(場のポケモン同士の関係を学習させる)、
    行動側はDecoderLayerでエンコーダの出力に交差注意する(この行動が今の盤面に対して
    どれだけ有効かを学習させる)。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        num_layers_encoder: int,
        num_layers_decoder: int,
    ) -> None:
        """
        Args:
            d_model: 埋め込み次元数。
            num_heads: MultiheadAttentionのヘッド数。
            d_feedforward: フィードフォワード層の隠れ次元数。
            num_layers_encoder: TransformerEncoderの層数。
            num_layers_decoder: DecoderLayerを重ねる数。
        """
        super().__init__()
        self.d_model = d_model

        self.encoder_bag = nn.EmbeddingBag(encoder_size(), d_model, mode="sum")
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, 0)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers_encoder, enable_nested_tensor=False
        )
        self.encoder_fc = nn.Linear(d_model, 1)

        self.decoder_bag = nn.EmbeddingBag(decoder_size(), d_model, mode="sum")
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_feedforward) for _ in range(num_layers_decoder)]
        )
        self.decoder_fc = nn.Linear(d_model, 1)

    def forward(
        self,
        index_encoder: torch.Tensor,
        value_encoder: torch.Tensor,
        offset_encoder: torch.Tensor,
        index_decoder: torch.Tensor,
        value_decoder: torch.Tensor,
        offset_decoder: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """バッチ分の盤面(疎ベクトル)と行動群(疎ベクトル)から、価値と方策スコアを計算する。

        Args:
            index_encoder: バッチ全体の盤面疎ベクトルを連結したindex列(`SparseBatch`参照)。
            value_encoder: 同上のvalue列。
            offset_encoder: 同上のoffset列(`batch_size * 24`個のトークン境界)。
            index_decoder: バッチ全体の行動疎ベクトルを連結したindex列。
            value_decoder: 同上のvalue列。
            offset_decoder: 同上のoffset列(`batch_size * max_actions`個の行動境界)。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (value, policy_scores)。
                value: 形状`(batch, 1)`の価値([-1, 1]、tanh済み)。
                policy_scores: 形状`(batch, max_actions)`の方策スコア([-1, 1]、tanh済み)。
        """
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        v = v.reshape(-1, NUM_WORDS_ENCODER, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        encoder_out = self.encoder(v)
        v = self.encoder_fc(encoder_out)
        value = torch.tanh(v.mean(0))

        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        policy_scores = torch.tanh(p)
        return value, policy_scores


class SparseBatch:
    """複数の`SparseVector`を1つのバッチとして連結するヘルパー(公式サンプルの`LearnInput`相当)。"""

    def __init__(self) -> None:
        self.index: list[int] = []
        self.value: list[float] = []
        self.offset: list[int] = []

    def add(self, sv: SparseVector) -> None:
        """1つの`SparseVector`をバッチ末尾に連結する。

        Args:
            sv: 追加する`SparseVector`。
        """
        base = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for o in sv.offset:
            self.offset.append(o + base)

    def add_empty_word(self) -> None:
        """パディング用に、空の(全て0の)トークン/行動を1つ追加する。"""
        self.offset.append(len(self.index))

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`PolicyValueNet.forward`にそのまま渡せるテンソル3点を作る。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (index, value, offset)。
        """
        return (
            torch.tensor(self.index, dtype=torch.int64),
            torch.tensor(self.value, dtype=torch.float32),
            torch.tensor(self.offset, dtype=torch.int64),
        )
