"""方策(policy)と価値(value)を同時に出力するTransformerネットワーク。

`sparse_features.get_encoder_input`が作る盤面の疎ベクトルを`TransformerEncoder`で
自己注意し、`sparse_features.get_decoder_input`が作る各行動の疎ベクトルを、
エンコーダ出力に交差注意(`DecoderLayer`)することで、行動ごとの方策スコアと
局面全体の価値を計算する。
"""

from collections.abc import Callable

import torch
import torch.nn as nn

from .sparse_features import SparseVector, decoder_size, encoder_size

NUM_WORDS_ENCODER = 26  # ベンチ8x2 + アクティブ2 + プレイヤー情報2 + 手札1 + デッキ1
# + スタジアム1 + 選択の文脈カード1 + 公開中1 + ターン情報1

# 所有者(誰の情報か)とゾーン(何の種類の情報か)。ベンチ8枠はゲームルール上並び順に
# 意味が無いため、スロットごとに別の埋め込みを持たせず、所有者ごとに1つを共有する。
_OWNER_SHARED, _OWNER_OWN, _OWNER_OPP = 0, 1, 2
_NUM_OWNERS = 3
(
    _ZONE_CLS,
    _ZONE_ACTIVE,
    _ZONE_BENCH,
    _ZONE_PLAYER_INFO,
    _ZONE_HAND,
    _ZONE_DECK,
    _ZONE_STADIUM,
    _ZONE_CONTEXT_CARD,
    _ZONE_LOOKING,
    _ZONE_TURN,
) = range(10)
_NUM_ZONES = 10

# get_encoder_inputの出力順(CLSを先頭に追加した後)に対応する(owner, zone)の並び
_TOKEN_OWNER_ZONE = (
    [(_OWNER_SHARED, _ZONE_CLS)]
    + [(_OWNER_OWN, _ZONE_BENCH)] * 8
    + [(_OWNER_OPP, _ZONE_BENCH)] * 8
    + [(_OWNER_OWN, _ZONE_ACTIVE), (_OWNER_OPP, _ZONE_ACTIVE)]
    + [(_OWNER_OWN, _ZONE_PLAYER_INFO), (_OWNER_OPP, _ZONE_PLAYER_INFO)]
    + [(_OWNER_OWN, _ZONE_HAND), (_OWNER_OWN, _ZONE_DECK)]
    + [(_OWNER_SHARED, _ZONE_STADIUM), (_OWNER_SHARED, _ZONE_CONTEXT_CARD)]
    + [(_OWNER_SHARED, _ZONE_LOOKING)]
    + [(_OWNER_SHARED, _ZONE_TURN)]
)
assert len(_TOKEN_OWNER_ZONE) == NUM_WORDS_ENCODER + 1


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
            encoder_out: 形状`(NUM_WORDS_ENCODER + 1, batch, d_model)`のエンコーダの出力。
                先頭がCLSトークン、残り`NUM_WORDS_ENCODER`個が盤面スロット。

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
        self.encoder_bag_norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # ベンチはゲームルール上スロット番号に意味が無いため、個別の位置埋め込みではなく
        # 所有者(自分/相手/共通)とゾーン(ベンチ/アクティブ/手札等)の埋め込みの和で表す
        # (自分のベンチ8枠は全て同じ埋め込みを共有する)
        owner_ids, zone_ids = zip(*_TOKEN_OWNER_ZONE, strict=True)
        self.register_buffer("_owner_ids", torch.tensor(owner_ids, dtype=torch.long))
        self.register_buffer("_zone_ids", torch.tensor(zone_ids, dtype=torch.long))
        self.owner_embedding = nn.Embedding(_NUM_OWNERS, d_model)
        self.zone_embedding = nn.Embedding(_NUM_ZONES, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, 0)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers_encoder, enable_nested_tensor=False
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1)
        )

        self.decoder_bag = nn.EmbeddingBag(decoder_size(), d_model, mode="sum")
        self.decoder_bag_norm = nn.LayerNorm(d_model)
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_feedforward) for _ in range(num_layers_decoder)]
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1)
        )

        # owner/zoneは正規化済みトークンへの加算なので、内容を覆い隠さない程度に小さく始める。
        for embedding in (self.owner_embedding, self.zone_embedding):
            nn.init.normal_(embedding.weight, std=0.02)

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
            offset_encoder: 同上のoffset列(`batch_size * NUM_WORDS_ENCODER`個のトークン境界)。
            index_decoder: バッチ全体の行動疎ベクトルを連結したindex列。
            value_decoder: 同上のvalue列。
            offset_decoder: 同上のoffset列(`batch_size * max_actions`個の行動境界)。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (value, policy_scores)。
                value: 形状`(batch, 1)`の価値([-1, 1]、tanh済み)。
                policy_scores: 形状`(batch, max_actions)`の方策の生ロジット(tanhなし、
                    呼び出し側でsoftmax/log_softmaxする前提)。
        """
        # EmbeddingBagは有効な特徴の「和」なので、トークンの大きさが特徴数に比例する。
        # 盤面トークンは数十個を足すのに対し行動トークンは数個しか足さないため、
        # 正規化しないと交差注意で行動の情報が盤面に埋もれ、全行動が同じスコアになる。
        v = self.encoder_bag_norm(self.encoder_bag(index_encoder, offset_encoder, value_encoder))
        v = v.reshape(-1, NUM_WORDS_ENCODER, self.d_model).transpose(0, 1)
        batch_size = v.size(1)

        cls = self.cls_token.expand(1, batch_size, -1)
        v = torch.cat([cls, v], dim=0)  # (NUM_WORDS_ENCODER + 1, batch, d_model)
        v = v + self.owner_embedding(self._owner_ids).unsqueeze(1)
        v = v + self.zone_embedding(self._zone_ids).unsqueeze(1)

        encoder_out = self.encoder(v)
        value = torch.tanh(self.encoder_fc(encoder_out[0]))  # CLSトークンの出力だけを価値に使う

        p = self.decoder_bag_norm(self.decoder_bag(index_decoder, offset_decoder, value_decoder))
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)  # (n_actions, batch, 1)
        policy_scores = p.transpose(0, 1).squeeze(-1)  # (batch, n_actions)
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


def collate_encoder_decoder[T](
    batch: list[T],
    get_encoder_sv: Callable[[T], SparseVector],
    get_decoder_sv: Callable[[T], SparseVector],
    get_n_actions: Callable[[T], int],
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """可変長の行動群を、バッチ内の最大行動数までパディングし、疎ベクトルを連結する。

    エンコーダ側は全サンプル共通のトークン数なのでパディング不要だが、デコーダ側
    (行動の数)は局面ごとに異なるため、`SparseBatch.add_empty_word`で埋める
    (公式サンプルコードの`LearnInput`と同じ扱い)。MCTS(`Sample`)・PPO(`PPOSample`)の
    どちらの学習サンプルからも、疎ベクトル/行動数の取り出し方だけ渡せば使える。

    Args:
        batch: 学習サンプルのリスト。
        get_encoder_sv: サンプルから盤面の疎ベクトルを取り出す関数。
        get_decoder_sv: サンプルから行動群の疎ベクトルを取り出す関数。
        get_n_actions: サンプルの局面で列挙されていた行動数を取り出す関数。

    Returns:
        tuple: (index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec, mask)。
            mask: 形状`(batch, max_actions)`のbool。パディングした行動位置はFalse。
    """
    max_actions = max(get_n_actions(sample) for sample in batch)

    encoder_batch = SparseBatch()
    decoder_batch = SparseBatch()
    mask = torch.zeros(len(batch), max_actions, dtype=torch.bool)

    for i, sample in enumerate(batch):
        encoder_batch.add(get_encoder_sv(sample))
        decoder_batch.add(get_decoder_sv(sample))
        n = get_n_actions(sample)
        mask[i, :n] = True
        for _ in range(max_actions - n):
            decoder_batch.add_empty_word()

    index_enc, value_enc, offset_enc = encoder_batch.to_tensors()
    index_dec, value_dec, offset_dec = decoder_batch.to_tensors()
    return index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec, mask


def build_policy_value_net(state_dict: dict | None = None, *, assign: bool = False):
    """`model_config.py`の構成で`PolicyValueNet`を作り、必要なら重みを載せてeval modeにする。

    構成引数を5箇所に散らさないための唯一の入口。`assign=True`はspawnワーカー向けで、
    共有メモリ上のテンソルをコピーせずそのまま採用する。

    Args:
        state_dict: 読み込む重み。Noneなら初期化済みの空ネットワークを返す。
        assign: `load_state_dict`の`assign`。

    Returns:
        PolicyValueNet: eval modeのネットワーク。
    """
    from .model_config import (
        D_FEEDFORWARD,
        D_MODEL,
        NUM_HEADS,
        NUM_LAYERS_DECODER,
        NUM_LAYERS_ENCODER,
    )

    network = PolicyValueNet(
        D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER
    )
    if state_dict is not None:
        network.load_state_dict(state_dict, assign=assign)
    network.eval()
    return network


def load_policy_value_net(path, *, assign: bool = False):
    """チェックポイントファイルから`PolicyValueNet`を復元する。

    `map_location="cpu"`を必ず指定する。外部から持ち込んだGPU保存のチェックポイントでも
    CPU専用機で読めるようにするため(指定を忘れるとload時にCUDA確保を試みて失敗する)。

    Args:
        path: `state_dict`を保存した`.pt`のパス。
        assign: `load_state_dict`の`assign`。

    Returns:
        PolicyValueNet: eval modeのネットワーク。
    """
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    return build_policy_value_net(state_dict, assign=assign)
