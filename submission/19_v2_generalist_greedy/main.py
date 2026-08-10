"""入力仕様v2・shared_cardで300ラウンド学習した方策。探索なしの貪欲選択。

`src/training/`のモジュールを機械的に連結して生成している。手で書き写していないので、
特徴量やネットワーク構成が学習側と食い違うことはない。生成は`scripts/build_submission.py`。

構成: shared_card / D_MODEL=128 / 貪欲方策(argmax、探索なし)
"""

import os
import sys
from pathlib import Path

# Kaggleは`main.py`をソースとして読み込んでexecするため、この名前空間には`__file__`が無い。
# 参照するとエピソードが即失敗するので、カレントディレクトリと本番の配置先だけで解決する。
KAGGLE_AGENT_DIR = Path("/kaggle_simulations/agent")
for _candidate in (".", str(KAGGLE_AGENT_DIR)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)


def resource_path(name: str, base_dir: Path) -> Path:
    """同梱ファイルを、ローカル実行と本番実行の両方で解決する。"""
    if os.path.exists(name):
        return Path(name)
    return KAGGLE_AGENT_DIR / name



# ===== common/model_config.py ====================================
"""`PolicyValueNet`のネットワーク構成。PPO/MCTSで共有し、チェックポイントを
相互に使い回せるようにする(定義箇所を1つに保つことで、サイズを変えるときの修正漏れを防ぐ)。
"""

# カードIDのone-hotをどう持つか。詳細は`sparse_features.py`のレイアウト定義を参照。
#   "per_role"    … カードが現れる箇所ごとに独立したカード表を持つ
#   "shared_card" … 同一トークン内で同居しない役割はカード表を共有する
# 学習時と推論時で食い違うと`load_state_dict`が形状不一致で落ちるため取り違えは起きない。
FEATURE_LAYOUT = "shared_card"

# 容量診断では当てはめ残差がほぼ0で、表現力は律速ではなかった。一方MCTSラップ推論は
# 探索回数がそのまま強さに効くため、余った容量は探索深さへ回す方が期待値が高い。
# 層のパラメータはD_MODELの二乗で効くので、256→128で計算量はおよそ1/4になる。
D_MODEL = 128
NUM_HEADS = 4
D_FEEDFORWARD = 512
# 速度は制約になっていない(5+5でも1試合が持ち時間600秒の2%程度)。一方この重みは
# 後続の学習の初期値にもなるため、容量不足だと層を増やした時点でチェックポイントが
# 使えなくなり、ゼロからやり直しになる。足りない場合の損失の方が大きいので余裕を取る。
NUM_LAYERS_ENCODER = 4
NUM_LAYERS_DECODER = 4

import types as _types

model_config = _types.SimpleNamespace(
    FEATURE_LAYOUT=FEATURE_LAYOUT,
    D_MODEL=D_MODEL,
    NUM_HEADS=NUM_HEADS,
    D_FEEDFORWARD=D_FEEDFORWARD,
    NUM_LAYERS_ENCODER=NUM_LAYERS_ENCODER,
    NUM_LAYERS_DECODER=NUM_LAYERS_DECODER,
)


# ===== common/sparse_features.py =================================
"""カード1枚1枚の識別子(cardId/attackId)を、`torch.nn.EmbeddingBag`向けの疎ベクトルとして
埋め込む特徴量エンコーディング。盤面(エンコーダ)には自分の60枚のデッキの中身も含める。
"""

import bisect
import sys
from collections import Counter
from pathlib import Path

import torch


from cg.api import (  # noqa: E402
    AreaType,
    Card,
    EnergyType,
    Observation,
    OptionType,
    PlayerState,
    Pokemon,
    SelectContext,
    all_attack,
    all_card_data,
)


_card_table: dict[int, object] | None = None
_attack_table: dict[int, object] | None = None
_card_count: int | None = None
_attack_count: int | None = None
_energy_type_count: int | None = None


def card_table() -> dict[int, object]:
    """全カードのマスタデータを、cardIdをキーにした辞書として取得する(遅延ロード・キャッシュ)。

    Returns:
        dict[int, object]: cardIdをキーにした`CardData`。
    """
    global _card_table
    if _card_table is None:
        _card_table = {c.cardId: c for c in all_card_data()}
    return _card_table


def card_count() -> int:
    """カードマスタの最大cardId+1(埋め込みテーブルのサイズ)を取得する(遅延ロード・キャッシュ)。

    Returns:
        int: 最大cardId + 1。
    """
    global _card_count
    if _card_count is None:
        _card_count = max(card_table().keys()) + 1
    return _card_count


def attack_table() -> dict[int, object]:
    """全ワザのマスタデータを、attackIdをキーにした辞書として取得する(遅延ロード・キャッシュ)。

    Returns:
        dict[int, object]: attackIdをキーにした`Attack`。
    """
    global _attack_table
    if _attack_table is None:
        _attack_table = {a.attackId: a for a in all_attack()}
    return _attack_table


def attack_count() -> int:
    """ワザマスタの最大attackId+1(埋め込みテーブルのサイズ)を取得する(遅延ロード・キャッシュ)。

    Returns:
        int: 最大attackId + 1。
    """
    global _attack_count
    if _attack_count is None:
        _attack_count = max(a.attackId for a in all_attack()) + 1
    return _attack_count


def energy_type_count() -> int:
    """`EnergyType`の取りうる値の数(埋め込みテーブルのサイズ)を取得する(遅延ロード・キャッシュ)。

    Returns:
        int: 最大`EnergyType`値 + 1。
    """
    global _energy_type_count
    if _energy_type_count is None:
        _energy_type_count = max(int(e) for e in EnergyType) + 1
    return _energy_type_count


def energy_unit_counts(energies: "list[EnergyType] | None") -> dict[int, int]:
    """付与済みエネルギーを、タイプ別の個数に集計する。

    `Pokemon.energies`はカード枚数ではなく実際のエネルギー個数の並びなので、
    1枚で複数個を供給する特殊エネルギーも正しく数えられる。
    """
    counts: dict[int, int] = {}
    for energy_type in energies or []:
        key = int(energy_type)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _add_energy_counts(sv: "SparseVector", base_offset: int, counts: dict[int, int]) -> None:
    """タイプ別のエネルギー枚数を、正規化してデコーダの`base_offset`以降に書き込む。

    Args:
        sv: 書き込み先の`SparseVector`。
        base_offset: エネルギータイプ0番の絶対インデックス。
        counts: `EnergyType`の値をキーにした枚数(`_energy_type_counts`やCounterの戻り値)。
    """
    for energy_type, count in counts.items():
        sv.add(base_offset + energy_type, count / 3)


# --- デコーダ側の先頭ブロック(NUMBER/YES/NO/SPECIAL_CONDITION/空選択)のレイアウト -------
# index 0: 選択肢が空(minCount=0で何も選ばなかった場合)
# index 1: END
# index 2: YES
# index 3: NO
# index 4-8: SPECIAL_CONDITION(POISON/BURN/SLEEP/PARALYZE/CONFUSEの5種)
# index 9-13: NUMBER(0〜4以上を5段階に丸めたもの)
# デコーダ先頭ブロックの内訳。ここは行動トークンごとに書かれる。
_HEAD_EMPTY = 0  # 何も選ばない行動
_HEAD_END = 1
_HEAD_YES = 2
_HEAD_NO = 3
_HEAD_SPECIAL_CONDITION = 4  # 5種
_HEAD_NUMBER = 9  # 0〜6と7以上で8段階
NUMBER_BUCKETS = 8
_HEAD_NUMBER_RAW = _HEAD_NUMBER + NUMBER_BUCKETS
_HEAD_REMAIN_ENERGY = _HEAD_NUMBER_RAW + 1
_HEAD_REMAIN_DAMAGE = _HEAD_REMAIN_ENERGY + 1
_HEAD_ENERGY_UNITS = _HEAD_REMAIN_DAMAGE + 1
DECODER_HEAD_SIZE = _HEAD_ENERGY_UNITS + 1
# decoder_main(MAIN選択の各種カード参照)が使う特徴インデックス数
# (PLAY, ATTACH対象カード, ATTACHの装着先, EVOLVE対象カード, EVOLVEの進化先,
#  ABILITY, DISCARD, RETREATの8種)
DECODER_MAIN_FEATURE = 8

# SelectContextの種類数
N_SELECT_CONTEXT = int(SelectContext.RECOVER_SPECIAL_CONDITION) + 1


def _decoder_attack_offset() -> int:
    """デコーダの疎ベクトルにおける、ATTACK特徴(attackIdのone-hot)の先頭インデックス。

    Returns:
        int: `DECODER_HEAD_SIZE`の直後。
    """
    return DECODER_HEAD_SIZE


def _decoder_attack_numeric_offset() -> int:
    """デコーダの疎ベクトルにおける、ATTACK技の数値特徴(ダメージ・エネルギー内訳)の先頭インデックス。

    ATTACK特徴(attackIdのone-hot、attack_count個)の直後に配置する。

    Returns:
        int: ATTACKのone-hotブロックの直後のインデックス。
    """
    return _decoder_attack_offset() + attack_count()


# ATTACK技の数値特徴のブロック内レイアウト: ダメージ1 + 必要エネルギー(タイプ別) + 現在の付与エネルギー(タイプ別)
_DECODER_ATTACK_DAMAGE_INDEX = 0
# 実効ダメージ、KO可否、相手残りHPに対する割合
_DECODER_ATTACK_EFFECTIVE_INDEX = 1
_DECODER_ATTACK_KO_INDEX = 2
_DECODER_ATTACK_RATIO_INDEX = 3
_DECODER_ATTACK_EXTRA = 3


def _decoder_attack_required_energy_offset() -> int:
    return _DECODER_ATTACK_DAMAGE_INDEX + 1 + _DECODER_ATTACK_EXTRA


def _decoder_attack_attached_energy_offset() -> int:
    return _decoder_attack_required_energy_offset() + energy_type_count()


def _decoder_attack_numeric_size() -> int:
    """ATTACK技の数値特徴ブロックの幅(ダメージ1 + 必要/現在エネルギーのタイプ別内訳)。"""
    return _decoder_attack_attached_energy_offset() + energy_type_count()


def _decoder_switch_numeric_offset() -> int:
    """デコーダの疎ベクトルにおける、交代先候補の数値特徴(残りHP・エネルギー内訳)の先頭インデックス。

    ATTACK数値特徴ブロックの直後に配置する。

    Returns:
        int: ATTACK数値特徴ブロックの直後のインデックス。
    """
    return _decoder_attack_numeric_offset() + _decoder_attack_numeric_size()


# 交代先候補の数値特徴のブロック内レイアウト: 残りHP1 + 現在の付与エネルギー(タイプ別)
_DECODER_SWITCH_HP_INDEX = 0


def _decoder_switch_energy_offset() -> int:
    return _DECODER_SWITCH_HP_INDEX + 1


def _decoder_switch_numeric_size() -> int:
    """交代先候補の数値特徴ブロックの幅(残りHP1 + 現在の付与エネルギーのタイプ別内訳)。"""
    return _decoder_switch_energy_offset() + energy_type_count()


def _decoder_main_target_offset() -> int:
    """MAIN選択の対象が場のポケモンのとき、その個体の状態を書くブロックの先頭。

    SWITCH用のブロックとは分ける。1つの行動トークンにSWITCH対象とMAIN対象が
    同時に入ることは無いが、同じ添字を共有すると将来どちらか判別できなくなる。
    """
    return _decoder_switch_numeric_offset() + _decoder_switch_numeric_size()


def _decoder_main_target_size() -> int:
    """残りHP + タイプ別の付与エネルギー個数 + このターン出たばかりか + 進化段数。"""
    return 1 + energy_type_count() + 2


AREA_COUNT = 13  # AreaTypeの最大値+1


def _decoder_scope_offset() -> int:
    """選択対象が「誰の」「どの領域」かを表すブロックの先頭。

    カードIDとcontextだけでは、同じカードを自分のベンチから選ぶのか相手のベンチから
    選ぶのかを区別できない。所有者と領域が無いと、異なる合法手が同一のトークンになり、
    方策は必ず同じスコアを出すことになる。
    """
    return _decoder_main_target_offset() + _decoder_main_target_size()


def _decoder_scope_size() -> int:
    """自分か相手かの1次元 + 領域のone-hot。"""
    return 1 + AREA_COUNT


def _decoder_card_offset() -> int:
    """デコーダの疎ベクトルにおける、カード参照系特徴の先頭インデックス。

    ATTACK特徴・ATTACK数値特徴・交代先数値特徴の直後に配置する。

    Returns:
        int: それらのブロックの直後のインデックス。
    """
    return _decoder_scope_offset() + _decoder_scope_size()


# --- 特徴量レイアウト -------------------------------------------------------
# "per_role"    : カードIDのone-hotを出現箇所ごとに独立したブロックとして持つ。
#                 同じカードでも手札とトラッシュで完全に別のベクトルになる。
# "shared_card" : 出現箇所をまたいでカード表を共有し、パラメータを減らす。
#
# 共有してよいのは「同一トークン内で同時に出現しない」役割だけ。`EmbeddingBag`はsumなので、
# 1トークンに複数のカードが入ると全部足され、どのカードがどの役割だったかの対応は失われる
# (役割ベクトルを足しても加算は交換法則が成り立つため結合できない)。
# 逆に別トークンにあるものは、network側のowner/zone埋め込みがトークン単位で区別するので
# 安全に共有できる。
#
#   本体 / 道具 / エネルギー … 同じポケモンのトークン内で同居するため分ける(3ブロック)
#                              ただし4つのポケモン枠(自他 × アクティブ/ベンチ)は
#                              別トークンなので、枠をまたいで共有する
#   手札 / デッキ / トラッシュ / スタジアム … すべて別トークンなので1ブロックに統合
#
# 結果、エンコーダのカードブロックは17 → 4に減る。
# レイアウトが変わると`encoder_size`/`decoder_size`が変わるため、学習時と推論時で
# 食い違えば`load_state_dict`が形状不一致で必ず落ちる(黙って誤動作しない)。

# エンコーダのカード役割。0〜2は同一トークン内で同居しうるので別ブロック、
# 3は別トークンにしか現れない役割をまとめたもの。
(
    CARD_ROLE_POKEMON,
    CARD_ROLE_TOOL,
    CARD_ROLE_ENERGY,
    CARD_ROLE_ZONE,
) = range(4)
ENCODER_CARD_BLOCKS = 4

_layout: str = model_config.FEATURE_LAYOUT


def configure_feature_layout(layout: str) -> None:
    """特徴量レイアウトを切り替える。

    spawnワーカーはこのmoduleを再importするため、親で設定しても引き継がれない。
    ワーカー初期化でも必ず呼ぶこと。

    Args:
        layout: "per_role" または "shared_card"。
    """
    global _layout
    if layout not in ("per_role", "shared_card"):
        raise ValueError(f"unknown feature layout: {layout!r}")
    _layout = layout


def feature_layout() -> str:
    """現在の特徴量レイアウト。"""
    return _layout


def _shared() -> bool:
    return _layout == "shared_card"


def _encoder_attack_offset() -> int:
    """共有レイアウトで、ポケモンが持つ技のブロックの先頭。"""
    return ENCODER_CARD_BLOCKS * card_count()


def _encoder_skill_offset() -> int:
    """共有レイアウトで、ポケモンが持つ特性のブロックの先頭。"""
    return _encoder_attack_offset() + attack_count()


def _encoder_block_base() -> int:
    """共有レイアウトで、トークンごとにスライドする非カードブロックの開始位置。

    先頭を共有カード表4ブロックと、技・特性のブロックが占める。技も特性も
    4つのポケモン枠それぞれが別トークンなので、枠をまたいで共有できる。
    """
    return _encoder_skill_offset() + skill_count()


def _decoder_card_block_count() -> int:
    """デコーダのカード参照ブロック数。

    共有レイアウトはMAINの8種 + context共通の1つ。`per_role`はcontextごとに
    独立したブロックを持つ。
    """
    if _shared():
        return DECODER_MAIN_FEATURE + 1
    return 1 + DECODER_MAIN_FEATURE + N_SELECT_CONTEXT


def _decoder_context_role_offset() -> int:
    """SelectContextを表す役割インデックスの先頭。

    1つの選択(select)の中でcontextは常に同一なので、カード表とは別に、
    行動トークンごとに1つの添字で表せる。`per_role`はcontextごとのカード表を
    持つが、カードを伴わない選択(YES/NO/NUMBER)ではcontextが落ちるため、
    両レイアウトでここに書く。
    """
    return _decoder_card_offset() + _decoder_card_block_count() * card_count()


# 残りHP比のバケット数。スカラー1本だと「HPが2割を切ったか」のような閾値的な判断が
# 表現しにくい(埋め込み行のスカラー倍=1本の直線方向にしかならない)ため、
# 生のスカラーと併用する形で離散化した表現も与える。
HP_RATIO_BUCKETS = 10

# ターン数のバケット境界。`state.turn`は半ターン単位で進むため、等間隔で切ると
# 序盤以外がすべて最終バケットに入って定数と変わらなくなる。序盤を細かく見る。
TURN_BUCKET_EDGES = (1, 2, 3, 4, 6, 8, 11, 15, 21, 30)
TURN_BUCKETS = len(TURN_BUCKET_EDGES) + 1


def _turn_bucket(turn: int) -> int:
    """半ターン単位の`state.turn`を非線形なバケットへ量子化する。"""
    return bisect.bisect_right(TURN_BUCKET_EDGES, turn)


def _hp_ratio_bucket(ratio: float) -> int:
    """残りHP比を`HP_RATIO_BUCKETS`段階へ量子化する(0が瀕死、最大値が満タン)。"""
    if ratio <= 0.0:
        return 0
    return min(HP_RATIO_BUCKETS - 1, int(ratio * HP_RATIO_BUCKETS))


def _pokemon_extra_size() -> int:
    """`add_pokemon`が1枠に書き込む、カードID以外の特徴のブロック幅。

    `add_pokemon`の書き込みと1対1で対応している必要がある。ずれるとブロック境界が
    重なって静かに壊れるため、`tests/test_feature_layout.py`で実際の書き込み量と照合する。
    """
    # ex, megaEx, にげるコスト, 最大HP, 印刷HP, HP強化量, 残りHP比, 進化段階3種,
    # aceSpec, tera, 逃走可否, このターン出たばかりか, 進化元の枚数, 技の最大打点
    scalars = 16
    return scalars + 3 * energy_type_count() + HP_RATIO_BUCKETS


def encoder_size() -> int:
    """エンコーダ側の疎ベクトルの総次元数(EmbeddingBagの語彙数)。

    `get_encoder_input`が書き込む全ブロックの合計。`SparseVector.pos`との一致は
    `tests/test_feature_layout.py`が実局面で検証する。

    Returns:
        int: エンコーダ側の疎ベクトルの総次元数。
    """
    # プレイヤー情報16 x 2人 + ターン情報3 + ポケモン枠の存在/HP 2 x 4枠 = 43
    # ターン資源フラグ4 + turnActionCount 1 + 被KO判定1 + ターンのバケット
    non_card = 44 + 6 + TURN_BUCKETS + 4 * _pokemon_extra_size()
    traits = attack_count() + skill_count()
    if _shared():
        # カード表4 + 技 + 特性は、いずれも枠をまたいで共有する
        return _encoder_block_base() + non_card
    return non_card + 20 * card_count() + 4 * traits


def decoder_size() -> int:
    """デコーダの疎ベクトルの総次元数(EmbeddingBagの語彙数)。

    先頭ブロック + ATTACK特徴 + ATTACK数値特徴 + 交代先数値特徴 + MAIN対象の状態 +
    所有者/領域 + カード参照ブロック + contextの役割インデックス、という構成。

    Returns:
        int: デコーダの疎ベクトルの総次元数。
    """
    return _decoder_context_role_offset() + N_SELECT_CONTEXT


class SparseVector:
    """`torch.nn.EmbeddingBag`への入力を組み立てるための、疎ベクトルの構築ヘルパー。

    `index`/`value`は非ゼロ要素だけを保持し、`offset`は「1つの疎ベクトル(=1つの
    トークン、またはデコーダなら1つの行動)の境界」を示す。`pos`は現在書き込み中の
    疎ベクトル内でのオフセットで、複数の特徴ブロックを連結する際に使う。
    """

    def __init__(self) -> None:
        self.index: list[int] = []
        self.value: list[float] = []
        self.offset: list[int] = []
        self.pos: int = 0

    def add(self, index: int, value: float) -> None:
        """現在の書き込み位置(`pos`)からの相対インデックスに値を1つ加える。

        Args:
            index: `pos`からの相対インデックス。
            value: 加える値(0の場合は疎表現なので書き込まない)。
        """
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(float(value))

    def add_absolute(self, index: int, value: float) -> None:
        """`pos`に依らない絶対インデックスへ値を加える。

        共有カード表のように、トークンごとにスライドするブロックの外側にある
        グローバルな領域へ書き込むために使う。

        Args:
            index: 疎ベクトル全体での絶対インデックス。
            value: 加える値(0の場合は書き込まない)。
        """
        if value != 0.0:
            self.index.append(index)
            self.value.append(float(value))

    def add_pos(self, pos: int) -> None:
        """次の特徴ブロックのために、書き込み位置を`pos`だけ進める。

        Args:
            pos: 進める幅(直前に書いたブロックの次元数)。
        """
        self.pos += pos

    def add_single(self, value: float | int | bool) -> None:
        """現在の書き込み位置に1つだけ値を書き、位置を1つ進める(単一のスカラー特徴用)。

        Args:
            value: 書き込む値。
        """
        v = float(value)
        if v != 0.0:
            self.index.append(self.pos)
            self.value.append(v)
        self.pos += 1

    def word_start(self) -> None:
        """新しい疎ベクトル(トークン/行動)の開始位置を記録する。"""
        self.offset.append(len(self.index))

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`torch.nn.EmbeddingBag`にそのまま渡せる、バッチサイズ1つ分のテンソル3点を作る。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (index, value, offset)。
        """
        return (
            torch.tensor(self.index, dtype=torch.int64),
            torch.tensor(self.value, dtype=torch.float32),
            torch.tensor(self.offset, dtype=torch.int64),
        )


def add_card(sv: SparseVector, card: "Card | Pokemon | None", role: int) -> None:
    """1枚のカード(またはNone)を、cardIdのone-hotとしてエンコーダに書き込む。

    Args:
        sv: 書き込み先の`SparseVector`。
        card: 対象のカード。存在しない場合はNone。
        role: `CARD_ROLE_*`。共有レイアウトでどのカード表へ書くかを決める。
            per_roleレイアウトでは使用しない。
    """
    if _shared():
        if card is not None:
            sv.add_absolute(role * card_count() + card.id, 1)
        return
    if card is not None:
        sv.add(card.id, 1)
    sv.add_pos(card_count())


def add_cards(sv: SparseVector, cards: "list[Card] | None", value: float, role: int) -> None:
    """複数のカードを、cardIdごとの出現回数(重み`value`倍)としてエンコーダに書き込む。

    Args:
        sv: 書き込み先の`SparseVector`。
        cards: 対象のカード一覧(discard等)。Noneの場合は何も書き込まない。
        value: 1枚あたりの重み。
        role: `CARD_ROLE_*`。共有レイアウトでどのカード表へ書くかを決める。
    """
    if _shared():
        if cards is not None:
            base = role * card_count()
            for card in cards:
                sv.add_absolute(base + card.id, value)
        return
    if cards is not None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(card_count())


_max_damage_cache: dict[int, float] = {}


_skill_ids: dict[str, int] | None = None
_card_skill_ids: dict[int, list[int]] | None = None


def skill_count() -> int:
    """特性の種類数(埋め込みテーブルの幅)。"""
    return len(_skill_table())


def _skill_table() -> dict[str, int]:
    """特性名に通し番号を振る。`Skill`はIDを持たず名前とテキストしかないため。"""
    global _skill_ids
    if _skill_ids is None:
        names = sorted({s.name for c in all_card_data() for s in (c.skills or [])})
        _skill_ids = {name: index for index, name in enumerate(names)}
    return _skill_ids


def card_skill_ids(card_id: int) -> list[int]:
    """そのカードが持つ特性の通し番号。"""
    global _card_skill_ids
    if _card_skill_ids is None:
        table = _skill_table()
        _card_skill_ids = {
            c.cardId: [table[s.name] for s in (c.skills or []) if s.name in table]
            for c in all_card_data()
        }
    return _card_skill_ids.get(card_id, [])


def _max_attack_damage(card_id: int) -> float:
    """そのカードが持つ技の最大ダメージ。技が無ければ0。"""
    cached = _max_damage_cache.get(card_id)
    if cached is None:
        card = card_table()[card_id]
        table = attack_table()
        damages = [float(table[a].damage) for a in (card.attacks or []) if a in table]
        cached = max(damages) if damages else 0.0
        _max_damage_cache[card_id] = cached
    return cached


# 現行レギュレーションの弱点・抵抗力の補正値。
WEAKNESS_MULTIPLIER = 2.0
RESISTANCE_REDUCTION = 30.0


def effective_damage(damage: float, attacker_card, defender_card) -> float:
    """弱点・抵抗力を反映した実効ダメージ。

    素のダメージと防御側の弱点/抵抗力は別々の特徴として渡っているが、両者を突き合わせて
    倍率を適用する計算は、別タワーに分かれた注意機構に学習させるには割に合わない。
    決定的に計算できるのでここで済ませる。

    Args:
        damage: 技の素のダメージ。
        attacker_card: 攻撃側のCardData(タイプ判定に使う)。
        defender_card: 防御側のCardData(弱点/抵抗力を持つ)。

    Returns:
        float: 補正後のダメージ(0未満にはしない)。
    """
    if damage <= 0 or defender_card is None or attacker_card is None:
        return max(0.0, damage)
    attacker_type = attacker_card.energyType
    if attacker_type is None:
        return damage
    if defender_card.weakness is not None and int(defender_card.weakness) == int(attacker_type):
        damage *= WEAKNESS_MULTIPLIER
    if defender_card.resistance is not None and int(defender_card.resistance) == int(attacker_type):
        damage -= RESISTANCE_REDUCTION
    return max(0.0, damage)


def add_energy_type(sv: SparseVector, energy_type: "EnergyType | None") -> None:
    """1つのエネルギータイプ(またはNone)を、one-hotとしてエンコーダに書き込む。

    Args:
        sv: 書き込み先の`SparseVector`。
        energy_type: 対象のエネルギータイプ。無い場合はNone。
    """
    if energy_type is not None:
        sv.add(int(energy_type), 1)
    sv.add_pos(energy_type_count())


def add_pokemon(sv: SparseVector, poke: "Pokemon | None") -> None:
    """場の1枠分(ポケモン1体、または空き枠)をエンコーダに書き込む。

    存在フラグ・HP比・カードID・付いている道具/エネルギーカードに加えて、
    サイド価値(ex/megaEx)・弱点・抵抗力・にげるコストも書き込む。

    Args:
        sv: 書き込み先の`SparseVector`。
        poke: 対象のポケモン。空き枠の場合はNone。
    """
    if poke is None:
        sv.add_single(1)
        skipped_cards = 0 if _shared() else 3 * card_count()
        skipped_traits = 0 if _shared() else attack_count() + skill_count()
        sv.add_pos(1 + skipped_cards + skipped_traits + _pokemon_extra_size())
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / 400)
        add_card(sv, poke, CARD_ROLE_POKEMON)
        add_cards(sv, poke.tools, 1.0, CARD_ROLE_TOOL)
        add_cards(sv, poke.energyCards, 0.5, CARD_ROLE_ENERGY)
        card = card_table()[poke.id]
        sv.add_single(card.ex)
        sv.add_single(card.megaEx)
        add_energy_type(sv, card.weakness)
        add_energy_type(sv, card.resistance)
        sv.add_single(card.retreatCost / 4)

        # 現在HPの絶対値だけでは、同じ0.3でもHP70のポケモンが満タンなのか
        # HP340のポケモンが瀕死なのか区別できない。最大HPと残り比率を明示的に渡す。
        # 最大HPは道具や効果で印刷値から変わるため、KO判定には盤面側の`maxHp`を使う。
        # 印刷値も渡す。カードの静的属性は出現の少ないカードの汎化に効き、両者の差は
        # 「HP強化がかかっている」という盤面の状態を表す。
        printed_hp = float(card.hp or 0)
        max_hp = float(poke.maxHp or printed_hp)
        sv.add_single(max_hp / 400)
        sv.add_single(printed_hp / 400)
        sv.add_single((max_hp - printed_hp) / 400)
        ratio = (poke.hp / max_hp) if max_hp > 0 else 0.0
        sv.add_single(ratio)
        sv.add(_hp_ratio_bucket(ratio), 1)
        sv.add_pos(HP_RATIO_BUCKETS)

        # 進化段階
        sv.add_single(card.basic)
        sv.add_single(card.stage1)
        sv.add_single(card.stage2)

        # 特殊ルール
        sv.add_single(card.aceSpec)
        sv.add_single(card.tera)

        # タイプ別の付与エネルギー個数。カードIDの重み付き和だけだと「炎が2個以上あるか」
        # のような閾値がスカラー1方向でしか表現できず、技のコストを満たせるかの判断に
        # 直結する情報が潰れる。デコーダはアクティブについて既に同じ情報を渡している。
        _add_energy_counts(sv, 0, energy_unit_counts(poke.energies))
        sv.add_pos(energy_type_count())

        # にげるコストを払えるか。コストはエネルギー「個数」で払うため、カード枚数では
        # なく`energies`(タイプの並び)の長さで数える(特殊エネルギーは1枚で複数個)。
        energy_units = len(poke.energies) if poke.energies else 0
        sv.add_single(1.0 if energy_units >= card.retreatCost else 0.0)

        # このターン場に出たばかりか。進化や特性の使用可否に関わるルール上の状態。
        sv.add_single(poke.appearThisTurn)
        # 何段進化しているか(場に積まれた進化元の枚数)。
        sv.add_single(len(poke.preEvolution) / 2 if poke.preEvolution else 0.0)

        # このポケモンの技の最大打点。相手側の枠なら脅威度の指標になる。
        sv.add_single(_max_attack_damage(poke.id) / 400)

        # 持っている技そのもの。選択肢として提示された技はデコーダ側で識別できるが、
        # 盤面としては最大打点しか渡っておらず、相手が何をしてくるかが分からなかった。
        # 1体の技は同じ役割なので、1ブロックにまとめて書ける。
        if _shared():
            for attack_id in card.attacks or []:
                sv.add_absolute(_encoder_attack_offset() + attack_id, 1)
        else:
            for attack_id in card.attacks or []:
                sv.add(attack_id, 1)
            sv.add_pos(attack_count())

        # 持っている特性。ポケモンの21%が特性持ちで、使える瞬間はABILITY選択肢として
        # 現れるが、「相手のベンチに特性持ちがいる」という盤面の把握ができていなかった。
        if _shared():
            for skill_id in card_skill_ids(poke.id):
                sv.add_absolute(_encoder_skill_offset() + skill_id, 1)
        else:
            for skill_id in card_skill_ids(poke.id):
                sv.add(skill_id, 1)
            sv.add_pos(skill_count())


def add_player(sv: SparseVector, ps: PlayerState) -> None:
    """片方のプレイヤーの盤面全体の数値情報(山札/手札/トラッシュ/ベンチ/サイド枚数、
    状態異常、トラッシュの中身)をエンコーダに書き込む。

    Args:
        sv: 書き込み先の`SparseVector`。
        ps: 対象プレイヤーの`PlayerState`。
    """
    sv.add_single(ps.deckCount / 60)
    sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8)
    sv.add_single(len(ps.bench) / max(1, ps.benchMax))
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)

    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)

    add_cards(sv, ps.discard, 0.25, CARD_ROLE_ZONE)


def get_encoder_input(obs: Observation, your_deck: list[int]) -> SparseVector:
    """局面全体を、Transformerエンコーダへの入力となる疎ベクトルに変換する。

    自分・相手それぞれのベンチ(最大8枠)・アクティブ・盤面全体の数値情報に加え、
    自分の手札と「自分の60枚のデッキの中身」も直接含める(カード識別子込みで
    デッキ固有の方策を学習させるため、公式サンプルコードと同じ構成)。

    Args:
        obs: 変換対象のObservation。`obs.current`がNoneでないこと。
        your_deck: 自分の60枚のデッキリスト(カードID)。既知のデッキ構成をそのまま渡す。

    Returns:
        SparseVector: `torch.nn.EmbeddingBag`にそのまま渡せる疎ベクトル
            (`word_start()`ごとに1トークンとなるよう区切り済み)。
    """
    your_index = obs.current.yourIndex
    state = obs.current

    sv = SparseVector()
    if _shared():
        sv.add_pos(_encoder_block_base())
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8):  # ベンチ最大8枠
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                add_pokemon(sv, ps.bench[j])
            else:
                add_pokemon(sv, None)
            if j != 7:
                sv.pos = pos

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_pokemon(sv, ps.active[0] if ps.active else None)

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_player(sv, ps)

    sv.word_start()
    add_cards(sv, state.players[your_index].hand, 0.25, CARD_ROLE_ZONE)

    sv.word_start()
    if _shared():
        base = CARD_ROLE_ZONE * card_count()
        for card_id in your_deck:
            sv.add_absolute(base + card_id, 0.25)
    else:
        for card_id in your_deck:
            sv.add(card_id, 0.25)
        sv.add_pos(card_count())

    sv.word_start()
    add_cards(sv, state.stadium, 1.0, CARD_ROLE_ZONE)

    # この選択の文脈となっているカード。`effect`とは別物で、片方しか無い局面も
    # 値が食い違う局面もある。`effect`と同じトークンに置くと加算で混ざるため分ける。
    sv.word_start()
    add_card(
        sv,
        obs.select.contextCard if obs.select is not None else None,
        CARD_ROLE_ZONE,
    )

    # サーチ効果などで今まさに公開されているカード。選択肢の対象としては見えていたが、
    # 「どの候補の中から選ばされているか」という盤面情報としては渡っていなかった。
    sv.word_start()
    # `looking`は伏せ札の位置がNoneになる(`list[Card | None]`)。中身をそのまま
    # 渡すと`.id`参照で落ちるため、公開分だけ書き、伏せ枚数は別に持たせる。
    looking = state.looking or []
    add_cards(sv, [c for c in looking if c is not None], 1.0, CARD_ROLE_ZONE)
    sv.add_single(sum(1 for c in looking if c is None) / 8)

    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)
    # ターン数の離散表現(生のスカラーと併用)
    sv.add(_turn_bucket(state.turn), 1)
    sv.add_pos(TURN_BUCKETS)
    # このターンにまだ何ができるか。サポート・エネルギー付与・にげるは1ターン1回で、
    # 手順(どの順に使うか)を決める上で欠かせないが、これまで渡していなかった。
    sv.add_single(state.supporterPlayed)
    sv.add_single(state.stadiumPlayed)
    sv.add_single(state.energyAttached)
    sv.add_single(state.retreated)
    sv.add_single(min(state.turnActionCount, 20) / 20)
    # この選択を引き起こしたカード。同じcontextでも、どのカードの効果によるかで
    # 最適な選択は変わる。
    add_card(sv, obs.select.effect if obs.select is not None else None, CARD_ROLE_ZONE)
    # 相手アクティブの最大打点で、自分のアクティブが落ちるか。
    # 逃げる/入れ替えるの判断に直結するが、両者の情報は別トークンに散っている。
    me_state = state.players[your_index]
    opp_state = state.players[1 - your_index]
    me_active = me_state.active[0] if me_state.active else None
    opp_active = opp_state.active[0] if opp_state.active else None
    threatened = 0.0
    if me_active is not None and opp_active is not None:
        incoming = effective_damage(
            _max_attack_damage(opp_active.id),
            card_table()[opp_active.id],
            card_table()[me_active.id],
        )
        threatened = 1.0 if incoming >= me_active.hp else 0.0
    sv.add_single(threatened)

    return sv


def get_card(obs: Observation, area: AreaType, index: int, player_index: int):
    """選択肢が参照しているエリア/インデックスから、実際のカードオブジェクトを引き当てる。

    Args:
        obs: 対象のObservation。
        area: 参照先のエリア。
        index: エリア内でのインデックス。
        player_index: 参照先プレイヤーのインデックス。

    Returns:
        Card | Pokemon | None: 引き当てたカード。該当しない場合はNone。
    """
    ps = obs.current.players[player_index]
    if area == AreaType.DECK:
        return obs.select.deck[index]
    if area == AreaType.HAND:
        return ps.hand[index]
    if area == AreaType.DISCARD:
        return ps.discard[index]
    if area == AreaType.ACTIVE:
        return ps.active[index]
    if area == AreaType.BENCH:
        return ps.bench[index]
    if area == AreaType.PRIZE:
        return ps.prize[index]
    if area == AreaType.STADIUM:
        return obs.current.stadium[index]
    if area == AreaType.LOOKING:
        return obs.current.looking[index]
    return None


def decoder_target_state(sv: SparseVector, target) -> None:
    """MAIN選択の対象が場のポケモンなら、その個体の状態を書き添える。

    `decoder_main`はカードIDしか書かないため、同種のポケモンが2体並ぶと
    「傷ついた方」と「無傷の方」が同一トークンになり、方策が区別できなくなる。
    交代先の選択(SWITCH)では既に同じ扱いをしている。
    """
    if target is None or not hasattr(target, "hp"):
        return
    offset = _decoder_main_target_offset()
    sv.add(offset, target.hp / 400)
    _add_energy_counts(sv, offset + 1, energy_unit_counts(target.energies))
    # 今出したばかりかは進化可否に関わり、進化段数は同じカードでも状態が違う。
    # これが無いと、同じベンチ枠の同種ポケモンを区別できない局面が残る。
    tail = offset + 1 + energy_type_count()
    sv.add(tail, 1.0 if getattr(target, "appearThisTurn", False) else 0.0)
    sv.add(tail + 1, len(getattr(target, "preEvolution", None) or []) / 2)


def decoder_scope(sv: SparseVector, area, player_index, your_index: int) -> None:
    """選択対象の所有者と領域を書き込む。

    選択肢の型によっては`area`/`playerIndex`が設定されない(END/PLAY/ATTACK等)。
    その場合は何も書かず、「所有者も領域も指定されていない」ことが値の不在として
    表現される。
    """
    offset = _decoder_scope_offset()
    if player_index is not None:
        sv.add(offset, 1.0 if player_index == your_index else -1.0)
    if area is not None and 0 <= int(area) < AREA_COUNT:
        sv.add(offset + 1 + int(area), 1)


def decoder_main(sv: SparseVector, feature_index: int, card) -> None:
    """MAIN選択で参照しているカードを、`decoder_main_feature`内の該当スロットに書き込む。

    Args:
        sv: 書き込み先の`SparseVector`。
        feature_index: `DECODER_MAIN_FEATURE`内でのスロット番号(0〜7)。
        card: 参照しているカード。存在しない場合はNone(何も書き込まない)。
    """
    if card is not None:
        sv.add(_decoder_card_offset() + feature_index * card_count() + card.id, 1)


def decoder_card_id(sv: SparseVector, context, card_id: int) -> None:
    """MAIN以外の選択で参照しているカードIDを、`SelectContext`ごとのブロックに書き込む。

    Args:
        sv: 書き込み先の`SparseVector`。
        context: 選択の`SelectContext`。
        card_id: 参照しているカードのID。
    """
    if int(context) >= N_SELECT_CONTEXT:
        # エンジン更新でSelectContextが増えると、共有レイアウトでは語彙の外へ書いてしまう。
        # 黙って壊れるより、原因の分かる例外にする。
        raise ValueError(f"SelectContext {int(context)} is outside the known range")
    # 共有レイアウトはcontextごとにカード表を分けない(どのcontextかは`_mark_context`が
    # 行動トークンごとに1回だけ記録する)。`per_role`はcontextごとに独立した表を持つ。
    block = DECODER_MAIN_FEATURE if _shared() else DECODER_MAIN_FEATURE + int(context)
    sv.add(_decoder_card_offset() + block * card_count() + card_id, 1)


def _mark_context(sv: SparseVector, context) -> None:
    """この行動トークンのSelectContextを1回だけ記録する。

    共有レイアウトではcontextごとのカード表を持たないため必須。`per_role`でも、
    YES/NO/NUMBERのようにカードを伴わない選択ではcontextがどこにも入らないため、
    両レイアウトで書く。

    カード参照のたびに立てると「その行動が何枚選ぶか」がcontextの埋め込みに
    掛かった形で混入するので、行動トークンごとに1回だけにする。
    """
    sv.add(_decoder_context_role_offset() + int(context), 1)


def decoder_card(sv: SparseVector, context, card) -> None:
    """MAIN以外の選択で参照しているカードを、`SelectContext`ごとのブロックに書き込む。

    Args:
        sv: 書き込み先の`SparseVector`。
        context: 選択の`SelectContext`。
        card: 参照しているカード。存在しない場合はNone(何も書き込まない)。
    """
    if card is not None:
        decoder_card_id(sv, context, card.id)


def get_decoder_input(obs: Observation, actions: list[list[int]]) -> SparseVector:
    """列挙済みの行動それぞれを、Transformerデコーダへの入力となる疎ベクトルに変換する。

    行動1つ(選択肢のindexのリスト)につき1トークンを作る。`search.enumerate_actions`が
    返す形式(各要素が`search_step`にそのまま渡せるindexのリスト)をそのまま受け取れる。

    Args:
        obs: 対象のObservation。
        actions: 列挙済みの行動のリスト(各要素は選択するindexのリスト)。

    Returns:
        SparseVector: `torch.nn.EmbeddingBag`にそのまま渡せる疎ベクトル
            (行動ごとに`word_start()`で区切り済み)。
    """
    sv = SparseVector()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = obs.select.context

    remain_energy = obs.select.remainEnergyCost or 0
    remain_damage = obs.select.remainDamageCounter or 0
    for action in actions:
        sv.word_start()
        _mark_context(sv, context)
        # 残り支払い量。エネルギー支払いやダメカン配置の選択では、あと何個必要かが
        # 分からないと何枚選ぶべきか判断できない。
        sv.add(_HEAD_REMAIN_ENERGY, min(remain_energy, 10) / 10)
        sv.add(_HEAD_REMAIN_DAMAGE, min(remain_damage, 20) / 20)

        if len(action) == 0:
            sv.add(_HEAD_EMPTY, 1)
            continue

        for i in action:
            o = obs.select.option[i]
            if o.type == OptionType.END:
                sv.add(_HEAD_END, 1)
            elif o.type == OptionType.YES:
                sv.add(_HEAD_YES, 1)
            elif o.type == OptionType.NO:
                sv.add(_HEAD_NO, 1)
            elif o.type == OptionType.SPECIAL_CONDITION:
                sv.add(_HEAD_SPECIAL_CONDITION + int(o.specialConditionType), 1)
            elif o.type == OptionType.NUMBER:
                # 個数はバケットと生値の両方。以前は4以上を1つにまとめており、
                # 「4個捨てる」と「7個捨てる」が同じ表現になっていた。
                sv.add(_HEAD_NUMBER + min(o.number, NUMBER_BUCKETS - 1), 1)
                sv.add(_HEAD_NUMBER_RAW, min(o.number, 20) / 20)
            elif o.type == OptionType.ATTACK:
                sv.add(_decoder_attack_offset() + o.attackId, 1)
                attack = attack_table()[o.attackId]
                numeric_offset = _decoder_attack_numeric_offset()
                sv.add(numeric_offset + _DECODER_ATTACK_DAMAGE_INDEX, attack.damage / 400)
                # 素のダメージはデコーダ側、相手の残りHPはエンコーダ側にあり、
                # 両者を突き合わせる比較は注意機構で学習するしかない形になっていた。
                # 決定的に計算できるのでここで済ませる。
                opponent = obs.current.players[1 - your_index]
                defender = opponent.active[0] if opponent.active else None
                if defender is not None:
                    attacker = ps.active[0] if ps.active else None
                    attacker_card = card_table()[attacker.id] if attacker is not None else None
                    defender_card = card_table()[defender.id]
                    eff = effective_damage(float(attack.damage), attacker_card, defender_card)
                    sv.add(numeric_offset + _DECODER_ATTACK_EFFECTIVE_INDEX, eff / 400)
                    sv.add(
                        numeric_offset + _DECODER_ATTACK_KO_INDEX,
                        1.0 if eff >= defender.hp else 0.0,
                    )
                    sv.add(
                        numeric_offset + _DECODER_ATTACK_RATIO_INDEX,
                        min(1.0, eff / defender.hp) if defender.hp > 0 else 1.0,
                    )
                required_offset = numeric_offset + _decoder_attack_required_energy_offset()
                _add_energy_counts(sv, required_offset, Counter(int(e) for e in attack.energies))
                attached_offset = numeric_offset + _decoder_attack_attached_energy_offset()
                _add_energy_counts(sv, attached_offset, energy_unit_counts(ps.active[0].energies))
            elif o.type == OptionType.PLAY:
                decoder_main(sv, 0, ps.hand[o.index])
            elif o.type == OptionType.ATTACH:
                decoder_main(sv, 1, get_card(obs, o.area, o.index, your_index))
                target = get_card(obs, o.inPlayArea, o.inPlayIndex, your_index)
                decoder_main(sv, 2, target)
                decoder_target_state(sv, target)
                decoder_scope(sv, o.inPlayArea, your_index, your_index)
            elif o.type == OptionType.EVOLVE:
                decoder_main(sv, 3, get_card(obs, o.area, o.index, your_index))
                evolving = get_card(obs, o.inPlayArea, o.inPlayIndex, your_index)
                decoder_main(sv, 4, evolving)
                decoder_target_state(sv, evolving)
                decoder_scope(sv, o.inPlayArea, your_index, your_index)
            elif o.type == OptionType.ABILITY:
                target = get_card(obs, o.area, o.index, your_index)
                decoder_main(sv, 5, target)
                decoder_target_state(sv, target)
                decoder_scope(sv, o.area, your_index, your_index)
            elif o.type == OptionType.DISCARD:
                decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index))
            elif o.type == OptionType.RETREAT:
                decoder_main(sv, 7, ps.active[0])
                decoder_target_state(sv, ps.active[0] if ps.active else None)
            elif o.type == OptionType.CARD:
                target = get_card(obs, o.area, o.index, o.playerIndex)
                decoder_card(sv, context, target)
                decoder_scope(sv, o.area, o.playerIndex, your_index)
                decoder_target_state(sv, target)
                if context == SelectContext.SWITCH and isinstance(target, Pokemon):
                    numeric_offset = _decoder_switch_numeric_offset()
                    sv.add(numeric_offset + _DECODER_SWITCH_HP_INDEX, target.hp / 400)
                    energy_offset = numeric_offset + _decoder_switch_energy_offset()
                    _add_energy_counts(sv, energy_offset, energy_unit_counts(target.energies))
            elif o.type == OptionType.TOOL_CARD:
                decoder_scope(sv, o.area, o.playerIndex, your_index)
                card = get_card(obs, o.area, o.index, o.playerIndex)
                decoder_card(sv, context, card.tools[o.toolIndex])
            elif o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY):
                decoder_scope(sv, o.area, o.playerIndex, your_index)
                card = get_card(obs, o.area, o.index, o.playerIndex)
                decoder_card(sv, context, card.energyCards[o.energyIndex])
                # そのエネルギーが何個分か。コストは個数で払うため、1枚で複数個を
                # 供給する特殊エネルギーは外したときの影響が大きい。
                if o.count:
                    sv.add(_HEAD_ENERGY_UNITS, min(o.count, 5) / 5)
            elif o.type == OptionType.SKILL:
                decoder_card_id(sv, context, o.cardId)

    return sv

# ===== common/network.py =========================================
"""方策(policy)と価値(value)を同時に出力するTransformerネットワーク。

`sparse_features.get_encoder_input`が作る盤面の疎ベクトルを`TransformerEncoder`で
自己注意し、`sparse_features.get_decoder_input`が作る各行動の疎ベクトルを、
エンコーダ出力に交差注意(`DecoderLayer`)することで、行動ごとの方策スコアと
局面全体の価値を計算する。
"""

from collections.abc import Callable

import torch
import torch.nn as nn


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

# ===== common/deck.py ============================================
"""`decks/`配下のデッキcsvを読み込む。"""

from pathlib import Path


def parse_deck_csv(path: Path) -> list[int]:
    """`decks/`配下のcsv(1行1カードID)からカードIDのリストを読み込む。

    Args:
        path: デッキcsvのパス。

    Returns:
        list[int]: ファイルに書かれていたカードID。
    """
    return [int(x) for x in path.read_text().split() if x.strip()]

# ===== common/selfplay_modes.py ==================================
"""自己対戦モードに応じて、両サイドのデッキと学習サンプルを集める座席を決める。

MCTS(`mcts/selfplay.py`)・PPO(`ppo/selfplay.py`)の両方から共通で使う
(探索の有無以外、モードの意味付けは同じため)。
"""

import random
from typing import Literal

SelfplayMode = Literal["asymmetric", "mirror", "generalist"]


def fixed_deck_seat_for_game(mode: SelfplayMode, game_index: int) -> int | None:
    """asymmetricの固定デッキ座席を試合番号の偶奇で均等に割り当てる。"""
    if game_index < 0:
        raise ValueError("game_index must be non-negative")
    if mode == "asymmetric":
        return game_index % 2
    return None


def sample_deck(opponent_deck_pool: list[list[int]], role: str) -> list[int]:
    """WeightedDeckPoolなら役割別分布、通常のlistなら後方互換の一様分布から選ぶ。"""
    sampler = getattr(opponent_deck_pool, "sample", None)
    if sampler is not None:
        return sampler(role)
    return random.choice(opponent_deck_pool)


def pick_decks_and_collect_seats(
    mode: SelfplayMode,
    our_deck: list[int],
    opponent_deck_pool: list[list[int]] | None,
    fixed_deck_seat: int | None = None,
) -> tuple[list[list[int]], set[int]]:
    """モードに応じて対局両サイドのデッキと、学習サンプルを集める座席を決める。

    Args:
        mode: "asymmetric"(固定デッキ対ランダムデッキ・両サイド学習)、
            "mirror"(両者同デッキ・両サイド学習)、
            "generalist"(両者とも実在デッキプールから独立ランダム・両サイド学習)。
        our_deck: 本番でも使う、こちらの60枚のデッキリスト。"generalist"では使われない。
        opponent_deck_pool: 対戦相手として選ぶ実在デッキのリスト。"mirror"では不要。
        fixed_deck_seat: "asymmetric"で固定デッキを置く座席(0または1)。Noneなら
            ランダムに選ぶ。trainerは試合番号の偶奇を渡してラウンド内で均等化する。

    Returns:
        tuple[list[list[int]], set[int]]: ([player0の60枚デッキ, player1の60枚デッキ],
            学習サンプルを集める座席の集合)。
    """
    if mode in ("asymmetric", "generalist") and not opponent_deck_pool:
        raise ValueError(f"opponent_deck_pool is required for mode={mode!r}")

    if mode == "mirror":
        return [our_deck, our_deck], {0, 1}
    if mode == "generalist":
        learner_deck = sample_deck(opponent_deck_pool, "learner")
        opponent_deck = sample_deck(opponent_deck_pool, "opponent")
        # デッキ分布とplayer index(先攻/後攻)を相関させない。
        # 両席とも同じポリシーの学習対象なのでcollect_seatsは変えない。
        if random.randrange(2):
            return [opponent_deck, learner_deck], {0, 1}
        return [learner_deck, opponent_deck], {0, 1}
    if mode == "asymmetric":
        if fixed_deck_seat is not None and fixed_deck_seat not in (0, 1):
            raise ValueError("fixed_deck_seat must be 0, 1, or None")
        our_seat = random.randrange(2) if fixed_deck_seat is None else fixed_deck_seat
        decks = [our_deck, our_deck]
        decks[1 - our_seat] = sample_deck(opponent_deck_pool, "opponent")
        # 固定デッキと対固定デッキの両方を、同じ汎用方策の学習対象にする。
        return decks, {0, 1}
    raise ValueError(f"unknown mode: {mode!r}")

# ===== mcts/search.py ============================================
"""Determinized MCTSの探索木本体(PUCTによる選択、ノード展開、バックプロパゲーション)。

呼び出し側が`determinize.py`で隠れ情報を1つの仮説に固定した上で`search_begin`を呼び、
その結果得られる`SearchState`を根ノードとして木探索する。局面の評価(事前確率・価値)は
外部から`eval_fn`として受け取る。
"""

import itertools
import math
import random
import sys
from pathlib import Path


from cg.api import search_release, search_step  # noqa: E402

MAX_ACTIONS_PER_NODE = 64  # 多重選択(maxCount>1)での組み合わせ爆発を抑えるための上限
PUCT_C = 0.4  # 公式サンプルコードと同じ探索係数


class Child:
    """MCTSノードの子(まだ展開されていない場合は`node`がNone)。"""

    def __init__(self, select: list[int], prob: float) -> None:
        """
        Args:
            select: この子が対応する選択(`search_step`にそのまま渡すindexのリスト)。
            prob: 方策の事前分布におけるこの選択の確率(PUCTのprior)。
        """
        self.node: Node | None = None
        self.select = select
        self.prob = prob


class Node:
    """MCTS探索木のノード。1つの`SearchState`(=1つの選択局面)に対応する。"""

    def __init__(self, parent: "Node | None", state) -> None:
        """
        Args:
            parent: 親ノード(根ノードの場合はNone)。
            state: このノードに対応する`SearchState`。
        """
        self.total = 0.0  # このノードを訪問した際のvalueの合計(バックプロパゲーションで累積)
        self.visit = 0
        self.parent = parent
        self.children: list[Child] = []
        self.state = state

    def backprop(self, value: float) -> None:
        """このノードから根に向かって、valueを合計・訪問回数を加算していく。

        Args:
            value: このノードで得られた評価値、または終局結果(+1/-1/0)。
        """
        self.total += value
        self.visit += 1
        if self.parent is not None:
            self.parent.backprop(value)


def _unrank_combination(n: int, count: int, rank: int) -> list[int]:
    """辞書順の組み合わせを全列挙せず、rank番目だけを取り出す。"""
    result: list[int] = []
    start = 0
    for position in range(count):
        remaining = count - position - 1
        for candidate in range(start, n):
            suffixes = math.comb(n - candidate - 1, remaining)
            if rank < suffixes:
                result.append(candidate)
                start = candidate + 1
                break
            rank -= suffixes
    return result


def enumerate_actions(select) -> list[list[int]]:
    """合法な複数選択を列挙し、多すぎる場合はrank空間から均等に採る。"""
    n = len(select.option)
    counts = list(range(select.minCount, select.maxCount + 1))
    sizes = [math.comb(n, count) for count in counts]
    total = sum(sizes)
    if total <= MAX_ACTIONS_PER_NODE:
        return [
            list(combo) for count in counts for combo in itertools.combinations(range(n), count)
        ]

    # 先頭だけを採ると後方indexを含む合法手が消えるため、全rankから等間隔に選ぶ。
    ranks = sorted(
        {round(i * (total - 1) / (MAX_ACTIONS_PER_NODE - 1)) for i in range(MAX_ACTIONS_PER_NODE)}
    )
    actions: list[list[int]] = []
    for rank in ranks:
        for count, size in zip(counts, sizes, strict=True):
            if rank < size:
                actions.append(_unrank_combination(n, count, rank))
                break
            rank -= size
    return actions


def create_node(parent: "Node | None", search_state, your_index: int, eval_fn) -> "Node":
    """`SearchState`から新しいノードを作る。

    終局していれば勝敗(+1/-1/0)を、していなければ`eval_fn`による評価値を、
    作成直後に即座に根までバックプロパゲーションする(公式サンプルコードと同じ扱い)。

    Args:
        parent: 親ノード(根ノードを作る場合はNone)。
        search_state: このノードのもとになる`SearchState`(`search_begin`/`search_step`の戻り値)。
        your_index: 探索の根本になっているプレイヤーのインデックス(勝敗の基準)。
        eval_fn: `(obs, actions) -> (list[float] | None, float)`。`obs.current.yourIndex`
            視点での(列挙済み行動`actions`に対する事前確率, 局面の価値)を返す関数。
            事前確率がNone、または列挙した行動数と長さが合わない場合は一様分布に
            フォールバックする。(公式サンプルコードの`eval_nn`に相当。行動群を渡すのは、
            疎な特徴量エンコーディング(`sparse_features.get_decoder_input`)が
            列挙済みの行動そのものを必要とするため)。

    Returns:
        Node: 作成したノード。
    """
    node = Node(parent, search_state)
    obs = search_state.observation
    state = obs.current

    if state.result >= 0:
        if state.result == 2:
            value = 0.0
        elif state.result == your_index:
            value = 1.0
        else:
            value = -1.0
        node.backprop(value)
        return node

    actions = enumerate_actions(obs.select)
    probs, value = eval_fn(obs, actions)
    if probs is None or len(probs) != len(actions):
        probs = [1.0 / len(actions)] * len(actions)

    for action, prob in zip(actions, probs, strict=True):
        node.children.append(Child(action, prob))

    if state.yourIndex != your_index:
        value = -value
    node.backprop(value)
    return node


def _select_child(node: "Node", your_index: int) -> "Child":
    """PUCTスコアが最大の子を選ぶ。

    スコアは「その子(まだ展開されていなければ親ノード自身)の平均value」に、
    「事前確率が高く、まだ訪問回数が少ない子」ほど大きくなる探索ボーナスを加えたもの。
    平均valueは常に`your_index`視点(手番がyour_indexでないノードでは符号を反転)に
    揃えてから比較する。

    Args:
        node: 子を選ぶ対象のノード。
        your_index: 探索の根本になっているプレイヤーのインデックス。

    Returns:
        Child: 選ばれた子。
    """
    if not node.children:
        # `enumerate_actions`が空を返した(例: minCount/maxCountに対して選択肢数が
        # 足りない、といった不整合なSelectData)場合にここへ来る。Noneを黙って返すと
        # 呼び出し側が`child.node`で意味不明なAttributeErrorになるため、ここで
        # 原因が分かる形の例外にしておく(呼び出し側の1試合単位の例外キャッチで
        # 引き続き吸収される)。
        raise RuntimeError(
            "_select_child called on a node with no enumerated actions "
            f"(degenerate SelectData?); state={node.state.observation.current!r}"
        )

    c = PUCT_C * math.sqrt(node.visit)
    best_child = None
    best_score = -math.inf
    for child in node.children:
        if child.node is None:
            # 未展開の辺には価値観測がない。親のQをコピーすると、親が高評価なだけで
            # 未知の全行動まで高評価になり、priorによる探索が歪む。
            avg_value = 0.0
            visit = 0
        else:
            avg_value = child.node.total / child.node.visit
            visit = child.node.visit
        if node.state.observation.current.yourIndex != your_index:
            avg_value = -avg_value
        score = avg_value + c * child.prob / (1 + visit)
        # NaNは比較が常にFalseになるため、素通しすると`best_child`がNoneのまま返り、
        # 呼び出し側が原因の分からないAttributeErrorで落ちる。NaNの子は選ばないだけに留め、
        # 全部がNaN(=ネットワーク出力が壊れている)のときだけ下で明示的に落とす。
        if math.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_child = child
    if best_child is None:
        raise RuntimeError(
            "all PUCT scores are NaN; the policy network likely produced NaN priors "
            f"(children={len(node.children)})"
        )
    return best_child


def run_mcts(
    root_state,
    your_index: int,
    eval_fn,
    search_count: int,
    root_dirichlet_alpha: float | None = None,
    root_noise_fraction: float = 0.0,
) -> tuple[list[int], list[float], float, list[list[int]]]:
    """根の`SearchState`からPUCTで`search_count`回のシミュレーションを行う。

    Args:
        root_state: 探索の起点になる`SearchState`(通常は`search_begin`の戻り値)。
        your_index: 探索の根本になっているプレイヤーのインデックス(勝敗の基準)。
        eval_fn: `create_node`に渡す局面評価関数。
        search_count: シミュレーション回数(MCTSのイテレーション数)。

    Returns:
        tuple[list[int], list[float], float, list[list[int]]]:
            (select, policy_target, root_value, actions)。
            select: 根から見て最も訪問回数の多い子の選択(`search_step`にそのまま渡せる形式)。
            policy_target: `actions`と同じ順序に並んだ、訪問回数を正規化した分布
                (自己対戦の学習で方策の教師信号として使う)。
            root_value: 探索全体で洗練された根ノードの平均value(`your_index`視点。
                自己対戦の学習で価値の教師信号のもとになる)。
            actions: 根で列挙された行動一覧(`policy_target`と同じ順序。学習時に
                `sparse_features.get_decoder_input`へそのまま渡せる)。
    """
    root = create_node(None, root_state, your_index, eval_fn)
    actions = [child.select for child in root.children]
    if root_dirichlet_alpha is not None and root_noise_fraction > 0 and len(root.children) > 1:
        noise = [random.gammavariate(root_dirichlet_alpha, 1.0) for _ in root.children]
        noise_total = sum(noise)
        for child, sample in zip(root.children, noise, strict=True):
            child.prob = (
                1.0 - root_noise_fraction
            ) * child.prob + root_noise_fraction * sample / noise_total
    # search_stepは呼ぶたびに新しいSearchStateをエンジン側に確保するので、使い終わったら
    # search_releaseで明示的に解放する(放置するとネイティブメモリがリークする)。
    # 根ノード(root_state)はsearch_begin/search_endが管理するのでここでは対象外。
    created_search_ids: list[int] = []

    try:
        for _ in range(search_count):
            current = root
            while True:
                child = _select_child(current, your_index)
                if child.node is None:
                    next_state = search_step(current.state.searchId, child.select)
                    created_search_ids.append(next_state.searchId)
                    child.node = create_node(current, next_state, your_index, eval_fn)
                    break
                current = child.node
                if current.state.observation.current.result >= 0:
                    # 既に終局しているノードを再訪問した場合、その結果を再度加算する
                    current.backprop(current.total / current.visit)
                    break

        root_value = root.total / root.visit
        visits = [child.node.visit if child.node is not None else 0 for child in root.children]
        total_visits = sum(visits)
        if total_visits == 0:
            # 1回もシミュレーションが展開されなかった場合(search_count=0等)のフォールバック
            policy_target = [1.0 / len(root.children)] * len(root.children)
            return root.children[0].select, policy_target, root_value, actions

        policy_target = [v / total_visits for v in visits]
        best_index = max(range(len(root.children)), key=lambda i: visits[i])
        return root.children[best_index].select, policy_target, root_value, actions
    finally:
        for search_id in created_search_ids:
            search_release(search_id)

# ===== mcts/determinize.py =======================================
"""MCTS探索用の隠れ情報を、実在デッキと公開情報に整合する形でサンプリングする。"""

import random
import sys
from collections import Counter
from pathlib import Path


from cg.api import CardType, all_card_data  # noqa: E402

_pokemon_card_ids: set[int] | None = None


def _pokemon_ids() -> set[int]:
    global _pokemon_card_ids
    if _pokemon_card_ids is None:
        _pokemon_card_ids = {c.cardId for c in all_card_data() if c.cardType == CardType.POKEMON}
    return _pokemon_card_ids


def _pokemon_cards(pokemon) -> list[int]:
    """場のポケモン1体を構成する全カードIDを返す。"""
    if pokemon is None:
        return []
    return [
        pokemon.id,
        *(card.id for card in pokemon.preEvolution),
        *(card.id for card in pokemon.energyCards),
        *(card.id for card in pokemon.tools),
    ]


def visible_cards(state, player_index: int) -> list[int]:
    """山札・非公開サイドを除き、現在公開されているカードを多重集合で返す。"""
    player = state.players[player_index]
    cards: list[int] = []
    for pokemon in player.active:
        cards.extend(_pokemon_cards(pokemon))
    for pokemon in player.bench:
        cards.extend(_pokemon_cards(pokemon))
    cards.extend(card.id for card in player.discard)
    cards.extend(card.id for card in player.prize if card is not None)
    if player.hand is not None:
        cards.extend(card.id for card in player.hand)
    cards.extend(card.id for card in state.stadium if card.playerIndex == player_index)
    return cards


def _remaining_deck(full_deck: list[int], known_cards: list[int]) -> list[int] | None:
    """full_deckから既知カードを枚数込みで差し引く。不整合ならNone。"""
    remaining = Counter(full_deck)
    remaining.subtract(Counter(known_cards))
    if any(count < 0 for count in remaining.values()):
        return None
    return list(remaining.elements())


_deck_counter_cache: dict[int, list[Counter]] = {}


def _deck_counters(candidates: list[list[int]]) -> list[Counter]:
    """候補デッキのCounterを一度だけ作って使い回す。

    デッキプールはrunを通して不変なのに、以前は候補ごと・呼び出しごとに
    `Counter(deck)`を作り直しており、決定化の時間のほとんどを占めていた。
    """
    key = id(candidates)
    cached = _deck_counter_cache.get(key)
    if cached is None or len(cached) != len(candidates):
        cached = [Counter(deck) for deck in candidates]
        _deck_counter_cache[key] = cached
    return cached


def _choose_compatible_deck(
    candidates: list[list[int]], known_cards: list[int], required_hidden: int
) -> tuple[list[int], list[int]]:
    """公開カードを含み、必要な隠れ枚数を確保できる実在デッキを選ぶ。

    候補は最大159通りあるので、選ばれなかったデッキの残り札は展開しない
    (`elements()`は選んだ1つだけに対して呼ぶ)。
    """
    known = Counter(known_cards)
    known_total = len(known_cards)
    compatible: list[int] = []
    for index, deck_counter in enumerate(_deck_counters(candidates)):
        if len(candidates[index]) - known_total < required_hidden:
            continue
        if all(deck_counter[card] >= count for card, count in known.items()):
            compatible.append(index)
    if not compatible:
        raise ValueError(
            "no opponent deck candidate is compatible with the observed public cards "
            f"(known={known}, required_hidden={required_hidden})"
        )
    chosen = random.choice(compatible)
    remaining = _deck_counters(candidates)[chosen].copy()
    remaining.subtract(known)
    return candidates[chosen], list(remaining.elements())


def determinize_for_search(
    obs,
    own_deck: list[int],
    opponent_deck_pool: list[list[int]],
) -> tuple[dict, list[list[int]]]:
    """探索側から見える情報だけで隠れ状態を作る。

    `opponent_deck_pool`は必須。以前はNoneならプロセスグローバルなキャッシュから
    遅延ロードしていたが、spawnワーカーは親の`configure_sampling_snapshot`を引き継がない
    ため、そのフォールバックは「設定と違うsnapshotを黙って読む」経路になっていた。
    呼び出し側が必ず明示的に渡す形にして、取り違えを型と例外で防ぐ。

    Returns:
        (search_beginへ渡すkwargs, playerIndex順の仮定フルデッキ)。
    """
    if not opponent_deck_pool:
        raise ValueError("opponent_deck_pool is required for determinization")
    state = obs.current
    your_index = state.yourIndex
    opponent_index = 1 - your_index
    me = state.players[your_index]
    opponent = state.players[opponent_index]

    own_remaining = _remaining_deck(own_deck, visible_cards(state, your_index))
    own_unknown_prizes = sum(card is None for card in me.prize)
    own_hidden_count = me.deckCount + own_unknown_prizes
    if own_remaining is None or len(own_remaining) < own_hidden_count:
        raise ValueError(
            "own deck is inconsistent with observed cards "
            f"(remaining={None if own_remaining is None else len(own_remaining)}, "
            f"required={own_hidden_count})"
        )
    random.shuffle(own_remaining)
    your_hidden_deck = own_remaining[: me.deckCount]
    hidden_own_prizes = iter(own_remaining[me.deckCount : own_hidden_count])
    your_prize = [card.id if card is not None else next(hidden_own_prizes) for card in me.prize]

    candidates = opponent_deck_pool
    opponent_known = visible_cards(state, opponent_index)
    face_down_active = bool(opponent.active and opponent.active[0] is None)
    opponent_unknown_prizes = sum(card is None for card in opponent.prize)
    opponent_hidden_count = opponent.deckCount + opponent.handCount + opponent_unknown_prizes
    assumed_opponent_deck, opponent_remaining = _choose_compatible_deck(
        candidates, opponent_known, opponent_hidden_count + int(face_down_active)
    )

    opponent_active: list[int] = []
    if face_down_active:
        pokemon_positions = [
            index for index, card_id in enumerate(opponent_remaining) if card_id in _pokemon_ids()
        ]
        if not pokemon_positions:
            raise ValueError("compatible opponent deck has no Pokémon for the face-down Active")
        active_position = random.choice(pokemon_positions)
        opponent_active.append(opponent_remaining.pop(active_position))

    random.shuffle(opponent_remaining)
    opponent_deck = opponent_remaining[: opponent.deckCount]
    hand_start = opponent.deckCount
    prize_start = hand_start + opponent.handCount
    opponent_hand = opponent_remaining[hand_start:prize_start]
    hidden_opponent_prizes = iter(
        opponent_remaining[prize_start : prize_start + opponent_unknown_prizes]
    )
    opponent_prize = [
        card.id if card is not None else next(hidden_opponent_prizes) for card in opponent.prize
    ]

    assumed_decks = [own_deck, own_deck]
    assumed_decks[your_index] = own_deck
    assumed_decks[opponent_index] = assumed_opponent_deck
    kwargs = {
        "your_deck": your_hidden_deck,
        "your_prize": your_prize,
        "opponent_deck": opponent_deck,
        "opponent_prize": opponent_prize,
        "opponent_hand": opponent_hand,
        "opponent_active": opponent_active,
    }
    return kwargs, assumed_decks

# ===== mcts/selfplay.py ==========================================
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


def make_eval_fn(network, decks: list[list[int]]):
    """`PolicyValueNet`をラップして、`search.create_node`が要求する`eval_fn`にする。

    探索木は複数ターンにまたがりうるため、ノードごとに「今まさに手番を選んでいる側
    (`obs.current.yourIndex`)の仮定デッキ」を`decks`から引いて`get_encoder_input`に渡す
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
        with torch.inference_mode():
            value, scores = network(
                index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec
            )
        probs = torch.softmax(scores[0], dim=-1).tolist()
        return probs, float(value.item())

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
) -> tuple[list[int], list[float], float, list[list[int]]]:
    """複数の隠れ情報仮説へ探索予算を分配し、根の方策と価値を集約する。"""
    hypothesis_count = min(max(1, num_determinizations), max(1, search_count))
    base_budget, remainder = divmod(search_count, hypothesis_count)
    aggregate_policy: list[float] | None = None
    aggregate_value = 0.0
    total_weight = 0
    common_actions: list[list[int]] | None = None

    for hypothesis_index in range(hypothesis_count):
        budget = base_budget + int(hypothesis_index < remainder)
        kwargs, assumed_decks = determinize_for_search(obs, own_deck, opponent_deck_pool)
        eval_fn = make_eval_fn(network, assumed_decks)
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

# ===== inference/agent.py ========================================
"""提出物が読み込む推論エージェント。

学習側と同じ`sparse_features`/`network`をそのまま使う。提出物へコードを複製すると
特徴量を変えるたびに陳腐化し、学習時と推論時で食い違う(以前はそれが起きていた)。
提出物にはこのパッケージごと同梱し、`main.py`は薄いシムにする。
"""


import json
from pathlib import Path

import torch


KAGGLE_AGENT_DIR = Path("/kaggle_simulations/agent")


def resource_path(name: str, base_dir: Path) -> Path:
    """提出物に同梱したファイルを、ローカル実行と本番実行の両方で解決する。"""
    local = base_dir / name
    if local.exists():
        return local
    return KAGGLE_AGENT_DIR / name


def choose_greedy(network: PolicyValueNet, obs, deck: list[int]) -> list[int]:
    """探索せず、盤面を1回評価して方策スコア最大の選択肢を返す。"""
    actions = enumerate_actions(obs.select)
    encoder_sv = get_encoder_input(obs, deck)
    decoder_sv = get_decoder_input(obs, actions)
    index_enc, value_enc, offset_enc = encoder_sv.to_tensors()
    index_dec, value_dec, offset_dec = decoder_sv.to_tensors()
    with torch.inference_mode():
        _value, scores = network(index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec)
    return actions[int(torch.argmax(scores[0]).item())]


class SubmissionAgent:
    """`deck.csv`と重みを読み込み、貪欲方策または探索付きで着手を選ぶ。"""

    def __init__(
        self,
        base_dir: Path,
        *,
        search_count: int = 0,
        num_determinizations: int = 1,
        weights_name: str = "policy.pt",
        deck_name: str = "deck.csv",
        opponent_decks_name: str = "opponent_decks.json",
    ) -> None:
        """
        Args:
            base_dir: 同梱ファイルが置かれたディレクトリ。
            search_count: 1手あたりの探索回数。0なら探索せず貪欲方策のみ。
            num_determinizations: 隠れ情報の仮説数。`search_count`をこの数で分割する。
            weights_name: 重みファイル名。
            deck_name: デッキファイル名。
            opponent_decks_name: 探索時の相手デッキ候補(探索する場合のみ必要)。
        """
        self.network = load_policy_value_net(resource_path(weights_name, base_dir))
        self.deck = parse_deck_csv(resource_path(deck_name, base_dir))
        self.search_count = search_count
        self.num_determinizations = num_determinizations
        self.opponent_decks: list[list[int]] = []
        if search_count > 0:
            path = resource_path(opponent_decks_name, base_dir)
            self.opponent_decks = json.loads(path.read_text(encoding="utf-8"))

    def select(self, obs) -> list[int]:
        """1手分の選択を返す。探索が失敗した手番は貪欲方策へ退避する。"""
        if obs.select is None:
            return self.deck
        if self.search_count > 0:
            try:
                return self._select_with_search(obs)
            except Exception:
                # 1手も返せないと即敗北になるため、探索の失敗は必ず吸収する。
                pass
        return choose_greedy(self.network, obs, self.deck)

    def _select_with_search(self, obs) -> list[int]:

        select, _policy, _value, _actions = run_determinized_mcts(
            self.network,
            obs,
            self.deck,
            self.opponent_decks,
            self.search_count,
            num_determinizations=self.num_determinizations,
        )
        return select


# ===== エントリポイント =====================================================

_agent = SubmissionAgent(
    Path("."),
    search_count=0,
    num_determinizations=4,
)


def agent(obs_dict: dict) -> list[int]:
    return _agent.select(to_observation_class(obs_dict))
