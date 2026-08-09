"""カード1枚1枚の識別子(cardId/attackId)を、`torch.nn.EmbeddingBag`向けの疎ベクトルとして
埋め込む特徴量エンコーディング。盤面(エンコーダ)には自分の60枚のデッキの中身も含める。
"""

import bisect
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_SUBMISSION_DIR = ROOT / "data" / "sample_submission" / "sample_submission"
sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

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

from . import model_config  # noqa: E402

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
DECODER_HEAD_SIZE = 14
# decoder_main(MAIN選択の各種カード参照)が使う特徴インデックス数
# (PLAY, ATTACH対象カード, ATTACHの装着先, EVOLVE対象カード, EVOLVEの進化先,
#  ABILITY, DISCARD, RETREATの8種)
DECODER_MAIN_FEATURE = 8


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


def _decoder_card_offset() -> int:
    """デコーダの疎ベクトルにおける、カード参照系特徴の先頭インデックス。

    ATTACK特徴・ATTACK数値特徴・交代先数値特徴の直後に配置する。

    Returns:
        int: それらのブロックの直後のインデックス。
    """
    return _decoder_switch_numeric_offset() + _decoder_switch_numeric_size()


# --- 特徴量レイアウト -------------------------------------------------------
# "per_role"    : カードIDのone-hotを出現箇所ごとに独立したブロックとして持つ(17ブロック)。
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


def _encoder_block_base() -> int:
    """共有レイアウトで、トークンごとにスライドする非カードブロックの開始位置。

    先頭を共有カード表4ブロックが占めるため、その直後から始める。
    """
    return ENCODER_CARD_BLOCKS * card_count()


def _decoder_context_role_offset() -> int:
    """共有レイアウトで、SelectContextを表す役割インデックスの先頭。

    1つの選択(select)の中でcontextは常に同一なので、contextごとにカード表を持つ必要はなく、
    カード表1つ + contextを示す役割インデックス、で表せる。
    """
    return _decoder_card_offset() + (DECODER_MAIN_FEATURE + 1) * card_count()


# 残りHP比のバケット数。スカラー1本だと「HPが2割を切ったか」のような閾値的な判断が
# 表現しにくい(埋め込み行のスカラー倍=1本の直線方向にしかならない)ため、
# 生のスカラーと併用する形で離散化した表現も与える。
HP_RATIO_BUCKETS = 10

# ターン数のバケット境界。`state.turn`は半ターン単位で進むため、等間隔で切ると
# 4巡目以降がすべて最終バケットに入って定数と変わらなくなる(実測で83.5%が最終箱)。
# 序盤を細かく、終盤を粗く見る非線形な区切りにする。
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
    重なって静かに壊れるため、`tests/test_feature_layout.py`で実測値と突き合わせている。
    """
    # ex, megaEx, にげるコスト, 最大HP, 印刷HP, HP強化量, 残りHP比, 進化段階3種,
    # 逃走可否, このターン出たばかりか, 進化元の枚数, 技の最大打点
    scalars = 14
    return scalars + 2 * energy_type_count() + HP_RATIO_BUCKETS + 2


def encoder_size() -> int:
    """エンコーダ側の疎ベクトルの総次元数(EmbeddingBagの語彙数)。

    `get_encoder_input`が書き込む全ブロックの合計サイズ。
    値は`get_encoder_input`呼び出し後の`SparseVector.pos`と実測一致することを確認済み。

    Returns:
        int: エンコーダ側の疎ベクトルの総次元数。
    """
    # プレイヤー情報16 x 2人 + ターン情報3 + ポケモン枠の存在/HP 2 x 4枠 = 43
    # ターン資源フラグ4 + turnActionCount 1 + 被KO判定1 + ターンのバケット
    non_card = 43 + 6 + TURN_BUCKETS + 4 * _pokemon_extra_size()
    if _shared():
        return ENCODER_CARD_BLOCKS * card_count() + non_card
    return non_card + 17 * card_count()


def decoder_size() -> int:
    """デコーダの疎ベクトルの総次元数(EmbeddingBagの語彙数)。

    先頭ブロック(NUMBER/YES/NO/SPECIAL_CONDITION/空選択) + ATTACK特徴(one-hot) +
    ATTACK数値特徴 + 交代先数値特徴 + (decoder_mainの8種 + SelectContextの種類数+ 1)個の
    カード参照ブロック、という構成。

    Returns:
        int: デコーダの疎ベクトルの総次元数。
    """
    n_context = int(SelectContext.RECOVER_SPECIAL_CONDITION) + 1
    if _shared():
        # MAINの8ブロック + context共通のカード表1ブロック + contextを示す役割インデックス
        return _decoder_card_offset() + (DECODER_MAIN_FEATURE + 1) * card_count() + n_context
    return _decoder_card_offset() + (1 + DECODER_MAIN_FEATURE + n_context) * card_count()


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
        sv.add_pos(1 + skipped_cards + _pokemon_extra_size())
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
        # 最大HPは道具や効果で印刷値から変動するため、KO判定には盤面側の`maxHp`を使う
        # (実測で約20%のポケモンが印刷値と不一致)。印刷値も併せて渡す。カードの静的属性は
        # 出現回数の少ないカードでも他カードと共有された統計として効き、両者の差は
        # 「HP強化がかかっている」という盤面の状態そのものを表す。
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
    n_context = int(SelectContext.RECOVER_SPECIAL_CONDITION) + 1
    if int(context) >= n_context:
        # エンジン更新でSelectContextが増えると、共有レイアウトでは語彙の外へ書いてしまう。
        # 黙って壊れるより、原因の分かる例外にする。
        raise ValueError(f"SelectContext {int(context)} is outside the known range")
    if _shared():
        # contextは1つのselect内で常に同一なので、カード表は1つで足りる。
        # どのcontextだったかを示す役割インデックスは、行動トークンごとに1回だけ
        # `_mark_context`が立てる(ここで立てるとカード枚数ぶん重複加算されてしまう)。
        sv.add(_decoder_card_offset() + DECODER_MAIN_FEATURE * card_count() + card_id, 1)
        return
    offset = _decoder_card_offset() + (DECODER_MAIN_FEATURE + int(context)) * card_count()
    sv.add(offset + card_id, 1)


def _mark_context(sv: SparseVector, context) -> None:
    """共有レイアウトで、この行動トークンのSelectContextを1回だけ記録する。

    `EmbeddingBag`はsumなので、カード参照のたびに立てると「その行動が何枚選ぶか」が
    contextの埋め込みに掛かった形で混入し、`per_role`と情報が非等価になる。
    """
    if _shared():
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

    行動1つ(選択肢のindexのリスト)につき1トークンを作る。`search._enumerate_actions`が
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

    for action in actions:
        sv.word_start()
        _mark_context(sv, context)

        if len(action) == 0:
            sv.add(0, 1)
            continue

        for i in action:
            o = obs.select.option[i]
            if o.type == OptionType.END:
                sv.add(1, 1)
            elif o.type == OptionType.YES:
                sv.add(2, 1)
            elif o.type == OptionType.NO:
                sv.add(3, 1)
            elif o.type == OptionType.SPECIAL_CONDITION:
                sv.add(4 + int(o.specialConditionType), 1)
            elif o.type == OptionType.NUMBER:
                sv.add(9 + min(o.number, 4), 1)
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
                decoder_main(sv, 2, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
            elif o.type == OptionType.EVOLVE:
                decoder_main(sv, 3, get_card(obs, o.area, o.index, your_index))
                decoder_main(sv, 4, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
            elif o.type == OptionType.ABILITY:
                decoder_main(sv, 5, get_card(obs, o.area, o.index, your_index))
            elif o.type == OptionType.DISCARD:
                decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index))
            elif o.type == OptionType.RETREAT:
                decoder_main(sv, 7, ps.active[0])
            elif o.type == OptionType.CARD:
                target = get_card(obs, o.area, o.index, o.playerIndex)
                decoder_card(sv, context, target)
                if context == SelectContext.SWITCH and isinstance(target, Pokemon):
                    numeric_offset = _decoder_switch_numeric_offset()
                    sv.add(numeric_offset + _DECODER_SWITCH_HP_INDEX, target.hp / 400)
                    energy_offset = numeric_offset + _decoder_switch_energy_offset()
                    _add_energy_counts(sv, energy_offset, energy_unit_counts(target.energies))
            elif o.type == OptionType.TOOL_CARD:
                card = get_card(obs, o.area, o.index, o.playerIndex)
                decoder_card(sv, context, card.tools[o.toolIndex])
            elif o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY):
                card = get_card(obs, o.area, o.index, o.playerIndex)
                decoder_card(sv, context, card.energyCards[o.energyIndex])
            elif o.type == OptionType.SKILL:
                decoder_card_id(sv, context, o.cardId)

    return sv
