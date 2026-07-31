"""カード1枚1枚の識別子(cardId/attackId)を、`torch.nn.EmbeddingBag`向けの疎ベクトルとして
埋め込む特徴量エンコーディング。盤面(エンコーダ)には自分の60枚のデッキの中身も含める。
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_SUBMISSION_DIR = ROOT / "data" / "sample_submission" / "sample_submission"
sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

from cg.api import (  # noqa: E402
    AreaType,
    Card,
    Observation,
    OptionType,
    PlayerState,
    Pokemon,
    SelectContext,
    all_attack,
    all_card_data,
)

_card_table: dict[int, object] | None = None
_card_count: int | None = None
_attack_count: int | None = None


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


def attack_count() -> int:
    """ワザマスタの最大attackId+1(埋め込みテーブルのサイズ)を取得する(遅延ロード・キャッシュ)。

    Returns:
        int: 最大attackId + 1。
    """
    global _attack_count
    if _attack_count is None:
        _attack_count = max(a.attackId for a in all_attack()) + 1
    return _attack_count


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


def _decoder_card_offset() -> int:
    """デコーダの疎ベクトルにおける、カード参照系特徴の先頭インデックス。

    ATTACK特徴(attack_count個)の直後に配置する。

    Returns:
        int: ATTACK特徴ブロックの直後のインデックス。
    """
    return _decoder_attack_offset() + attack_count()


def encoder_size() -> int:
    """エンコーダ側の疎ベクトルの総次元数(EmbeddingBagの語彙数)。

    `get_encoder_input`が書き込む全ブロックの合計サイズ(`43 + 17 * card_count()`)。
    値は`get_encoder_input`呼び出し後の`SparseVector.pos`と実測一致することを確認済み。

    Returns:
        int: エンコーダ側の疎ベクトルの総次元数。
    """
    return 43 + 17 * card_count()


def decoder_size() -> int:
    """デコーダの疎ベクトルの総次元数(EmbeddingBagの語彙数)。

    先頭ブロック(NUMBER/YES/NO/SPECIAL_CONDITION/空選択) + ATTACK特徴 +
    (decoder_mainの8種 + SelectContextの種類数+ 1)個のカード参照ブロック、
    という構成(公式サンプルコードと同じレイアウト)。

    Returns:
        int: デコーダの疎ベクトルの総次元数。
    """
    n_context = int(SelectContext.RECOVER_SPECIAL_CONDITION) + 1
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


def add_card(sv: SparseVector, card: "Card | Pokemon | None") -> None:
    """1枚のカード(またはNone)を、cardIdのone-hotとしてエンコーダに書き込む。

    Args:
        sv: 書き込み先の`SparseVector`。
        card: 対象のカード。存在しない場合はNone。
    """
    if card is not None:
        sv.add(card.id, 1)
    sv.add_pos(card_count())


def add_cards(sv: SparseVector, cards: "list[Card] | None", value: float) -> None:
    """複数のカードを、cardIdごとの出現回数(重み`value`倍)としてエンコーダに書き込む。

    Args:
        sv: 書き込み先の`SparseVector`。
        cards: 対象のカード一覧(discard等)。Noneの場合は何も書き込まない。
        value: 1枚あたりの重み。
    """
    if cards is not None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(card_count())


def add_pokemon(sv: SparseVector, poke: "Pokemon | None") -> None:
    """場の1枠分(ポケモン1体、または空き枠)をエンコーダに書き込む。

    存在フラグ・HP比・カードID・付いている道具/エネルギーカードをまとめて書き込む。

    Args:
        sv: 書き込み先の`SparseVector`。
        poke: 対象のポケモン。空き枠の場合はNone。
    """
    if poke is None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * card_count())
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / 400)
        add_card(sv, poke)
        add_cards(sv, poke.tools, 1.0)
        add_cards(sv, poke.energyCards, 0.5)


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
    sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)

    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)

    add_cards(sv, ps.discard, 0.25)


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
    add_cards(sv, state.players[your_index].hand, 0.25)

    sv.word_start()
    for card_id in your_deck:
        sv.add(card_id, 0.25)
    sv.add_pos(card_count())

    sv.word_start()
    add_cards(sv, state.stadium, 1.0)

    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)

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
    offset = _decoder_card_offset() + (DECODER_MAIN_FEATURE + int(context)) * card_count()
    sv.add(offset + card_id, 1)


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
                decoder_card(sv, context, get_card(obs, o.area, o.index, o.playerIndex))
            elif o.type == OptionType.TOOL_CARD:
                card = get_card(obs, o.area, o.index, o.playerIndex)
                decoder_card(sv, context, card.tools[o.toolIndex])
            elif o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY):
                card = get_card(obs, o.area, o.index, o.playerIndex)
                decoder_card(sv, context, card.energyCards[o.energyIndex])
            elif o.type == OptionType.SKILL:
                decoder_card_id(sv, context, o.cardId)

    return sv
