"""Observationを固定長ベクトルに変換する特徴量エンジニアリング(BC・MCTS共通)。

方針(フォーラムの助言を参考): 「盤面全体」「場のポケモン(エンティティ)」「選択肢」を
別々にベクトル化する。カードの識別子そのもの(cardId)は種類が多すぎるため埋め込まず、
CardDataから引ける静的な特徴量(HP・進化段階・ex等)に落とし込む。

デッキや学習手法に依存しない汎用的な数値化なので、`bc/`(Behavior Cloning)と
`mcts/`(Determinized MCTS)の両方から共通で使う。
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SUBMISSION_DIR = ROOT / "data" / "sample_submission" / "sample_submission"
sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

from cg.api import (  # noqa: E402
    AreaType,
    Attack,
    CardData,
    Observation,
    Option,
    OptionType,
    PlayerState,
    Pokemon,
    all_attack,
    all_card_data,
)

_CARD_DATA_CACHE: dict[int, CardData] | None = None
_ATTACK_DATA_CACHE: dict[int, Attack] | None = None


def card_data() -> dict[int, CardData]:
    """全カードのマスタデータを取得する（初回のみエンジンから取得し、以降はキャッシュを返す）。

    Returns:
        dict[int, CardData]: cardId をキーにしたカードマスタデータ。
    """
    global _CARD_DATA_CACHE
    if _CARD_DATA_CACHE is None:
        _CARD_DATA_CACHE = {c.cardId: c for c in all_card_data()}
    return _CARD_DATA_CACHE


def attack_data() -> dict[int, Attack]:
    """全ワザのマスタデータを取得する（初回のみエンジンから取得し、以降はキャッシュを返す）。

    Returns:
        dict[int, Attack]: attackId をキーにしたワザのマスタデータ。
    """
    global _ATTACK_DATA_CACHE
    if _ATTACK_DATA_CACHE is None:
        _ATTACK_DATA_CACHE = {a.attackId: a for a in all_attack()}
    return _ATTACK_DATA_CACHE


N_OPTION_TYPES = 17  # OptionType の種類数 (0〜16)
CARD_FEATURE_DIM = 11
ATTACK_FEATURE_DIM = 2  # ダメージ量 + 必要エネルギー数
POKEMON_SLOT_DIM = 3 + CARD_FEATURE_DIM  # 存在フラグ + hp比 + エネルギー数 + カード静的特徴
N_SLOTS_PER_SIDE = 6  # active(1) + bench(最大5)
GLOBAL_DIM = 16  # 既存の14項目 + 先手/後手フラグ + 自分視点のターン数
STATE_DIM = GLOBAL_DIM + POKEMON_SLOT_DIM * N_SLOTS_PER_SIDE * 2
OPTION_DIM = N_OPTION_TYPES + CARD_FEATURE_DIM + ATTACK_FEATURE_DIM + 3


def card_static_features(card_id: int | None) -> np.ndarray:
    """カード1種類分の、対戦の進行に左右されない静的な特徴量を作る。

    HP・にげるコスト・弱点/抵抗力の有無・進化段階・ex系フラグをまとめた固定長ベクトルにする。
    cardIdの種類はカードプール全体で約2000あり、そのまま埋め込むと学習が難しくなるため、
    代わりにこの手作りの特徴量に落とし込む。

    Args:
        card_id: カードのID。対象が存在しない場合はNone。

    Returns:
        np.ndarray: 長さ`CARD_FEATURE_DIM`の特徴ベクトル。card_idがNone、または
            マスタデータに存在しない場合は全て0のベクトルを返す。
    """
    vec = np.zeros(CARD_FEATURE_DIM, dtype=np.float32)
    if card_id is None:
        return vec
    card = card_data().get(card_id)
    if card is None:
        return vec
    vec[0] = card.hp / 350.0
    vec[1] = card.retreatCost / 5.0
    vec[2] = 1.0 if card.weakness is not None else 0.0
    vec[3] = 1.0 if card.resistance is not None else 0.0
    vec[4] = 1.0 if card.basic else 0.0
    vec[5] = 1.0 if card.stage1 else 0.0
    vec[6] = 1.0 if card.stage2 else 0.0
    vec[7] = 1.0 if card.ex else 0.0
    vec[8] = 1.0 if card.megaEx else 0.0
    vec[9] = 1.0 if card.tera else 0.0
    vec[10] = 1.0 if card.aceSpec else 0.0
    return vec


def attack_static_features(attack_id: int | None) -> np.ndarray:
    """ワザ1つ分の、ダメージ量・必要エネルギー数を固定長ベクトルにする。

    OptionType.ATTACKの選択肢は`attackId`しか持っておらず、そのままでは
    「どのワザなのか」がベクトルに反映されない（同じポケモンが持つ複数のワザが
    区別できない）。この関数でAttackのマスタデータを引いて特徴量に加える。

    Args:
        attack_id: ワザのID。対象がATTACK以外の選択肢の場合はNone。

    Returns:
        np.ndarray: 長さ`ATTACK_FEATURE_DIM`の特徴ベクトル。attack_idがNone、
            またはマスタデータに存在しない場合は全て0のベクトルを返す。
    """
    vec = np.zeros(ATTACK_FEATURE_DIM, dtype=np.float32)
    if attack_id is None:
        return vec
    attack = attack_data().get(attack_id)
    if attack is None:
        return vec
    vec[0] = attack.damage / 350.0
    vec[1] = len(attack.energies) / 5.0
    return vec


def pokemon_slot_features(pokemon: Pokemon | None) -> np.ndarray:
    """場（アクティブ or ベンチ）の1枠分を、その時点の対戦状況込みで特徴量にする。

    card_static_featuresが「カードの種類として不変の特徴」を扱うのに対し、
    こちらは「今このポケモンがどれだけ消耗しているか(HP比)」「エネルギーが何枚付いているか」
    といった、対戦の進行によって変わる情報を追加する。

    Args:
        pokemon: その枠にいるポケモン。空き枠の場合はNone。

    Returns:
        np.ndarray: 長さ`POKEMON_SLOT_DIM`の特徴ベクトル。先頭が「その枠にポケモンが
            存在するか」を表すフラグで、pokemonがNoneの場合は全て0になる。
    """
    vec = np.zeros(POKEMON_SLOT_DIM, dtype=np.float32)
    if pokemon is None:
        return vec
    vec[0] = 1.0  # 存在フラグ
    vec[1] = pokemon.hp / max(pokemon.maxHp, 1)
    vec[2] = len(pokemon.energies) / 8.0
    vec[3:] = card_static_features(pokemon.id)
    return vec


def side_features(player_state: PlayerState) -> np.ndarray:
    """片方のプレイヤーの場（アクティブ1体 + ベンチ最大5体）をまとめてベクトル化する。

    ベンチが5体に満たない場合の残り枠は`pokemon_slot_features(None)`（全0）で埋める。

    Args:
        player_state: ベクトル化したいプレイヤー側の状態。

    Returns:
        np.ndarray: 長さ`POKEMON_SLOT_DIM * N_SLOTS_PER_SIDE`の特徴ベクトル
            （各枠の特徴量を先頭から順に連結したもの）。
    """
    slots = np.zeros((N_SLOTS_PER_SIDE, POKEMON_SLOT_DIM), dtype=np.float32)
    active = player_state.active[0] if player_state.active else None
    slots[0] = pokemon_slot_features(active)
    for i, mon in enumerate(player_state.bench[: N_SLOTS_PER_SIDE - 1]):
        slots[i + 1] = pokemon_slot_features(mon)
    return slots.flatten()


def encode_state(obs: Observation) -> np.ndarray:
    """Observation全体を、Behavior Cloningの入力となる固定長の状態ベクトルにする。

    「盤面全体を表す少数の数値(ターン数・山札/手札/トラッシュ/サイド枚数など)」と、
    「自分・相手それぞれの場の様子」を連結する。

    `state.turn`は先手/後手どちらの立場かによって意味が変わる（同じturn数でも、
    先手なら自分の方が1手多く進んでいる）ため、それだけでは局面の進み具合を
    正しく表せない。そのため「自分が先手/後手のどちらか」と「自分視点で
    何ターン目か」を別途計算して加えている。

    Args:
        obs: `to_observation_class`で変換済みの観測。`obs.current`がNoneでないことを
            呼び出し側で保証しておくこと（デッキ提出時など`current`がNoneの局面は対象外）。

    Returns:
        np.ndarray: 長さ`STATE_DIM`の状態ベクトル。
    """
    state = obs.current
    me = state.players[state.yourIndex]
    opp = state.players[1 - state.yourIndex]

    global_vec = np.zeros(GLOBAL_DIM, dtype=np.float32)
    global_vec[0] = state.turn / 60.0
    global_vec[1] = state.turnActionCount / 10.0
    global_vec[2] = 1.0 if state.supporterPlayed else 0.0
    global_vec[3] = 1.0 if state.stadiumPlayed else 0.0
    global_vec[4] = 1.0 if state.energyAttached else 0.0
    global_vec[5] = 1.0 if state.retreated else 0.0
    global_vec[6] = me.deckCount / 60.0
    global_vec[7] = opp.deckCount / 60.0
    global_vec[8] = me.handCount / 60.0
    global_vec[9] = opp.handCount / 60.0
    global_vec[10] = len(me.discard) / 60.0
    global_vec[11] = len(opp.discard) / 60.0
    global_vec[12] = len(me.prize) / 6.0
    global_vec[13] = len(opp.prize) / 6.0

    # 先手/後手（state.turnだけでは、同じturn数でも先手か後手かで意味が変わってしまう）
    if state.firstPlayer == state.yourIndex:
        global_vec[14] = 1.0  # 自分が先手
        my_turn_number = (state.turn + 1) // 2
    elif state.firstPlayer == 1 - state.yourIndex:
        global_vec[14] = 0.0  # 自分が後手
        my_turn_number = state.turn // 2
    else:
        global_vec[14] = 0.5  # 先手/後手が未確定（コイントス前）
        my_turn_number = 0
    global_vec[15] = my_turn_number / 30.0

    return np.concatenate([global_vec, side_features(me), side_features(opp)])


def resolve_target_card_id(obs: Observation, opt: Option) -> int | None:
    """選択肢(Option)が指しているカード、または場のポケモンのcardIdを可能な範囲で特定する。

    OptionのフィールドはOptionTypeごとに意味が異なる（api.pyのOptionType定義を参照）ため、
    typeごとに「カードは手札/トラッシュにあるのか、場に出ているのか」を判定して引き当てる。
    見えない情報（相手の手札など）や、対応関係が定義されていないOptionTypeの場合はNoneを返す。

    Args:
        obs: 対象の選択肢が含まれるObservation。
        opt: cardIdを特定したい選択肢。

    Returns:
        int | None: 特定できた場合はそのcardId、特定できない場合はNone。
    """
    state = obs.current
    me = state.players[state.yourIndex]

    def from_hand_or_discard(area, index) -> int | None:
        if area == AreaType.HAND:
            hand = me.hand or []
            return hand[index].id if 0 <= index < len(hand) else None
        if area == AreaType.DISCARD:
            return me.discard[index].id if 0 <= index < len(me.discard) else None
        return None

    def from_inplay(area, index) -> int | None:
        if area == AreaType.ACTIVE:
            return me.active[index].id if 0 <= index < len(me.active) and me.active[index] else None
        if area == AreaType.BENCH:
            return me.bench[index].id if 0 <= index < len(me.bench) else None
        return None

    if opt.type == OptionType.PLAY:
        return from_hand_or_discard(AreaType.HAND, opt.index)
    if opt.type in (OptionType.EVOLVE, OptionType.ATTACH):
        cid = from_hand_or_discard(opt.area, opt.index) if opt.area is not None else None
        return cid if cid is not None else from_inplay(opt.inPlayArea, opt.inPlayIndex)
    if opt.type in (OptionType.ABILITY, OptionType.DISCARD):
        return from_inplay(opt.area, opt.index) or from_hand_or_discard(opt.area, opt.index)
    if opt.type == OptionType.CARD:
        return from_hand_or_discard(opt.area, opt.index) or from_inplay(opt.area, opt.index)
    return None


def encode_option(obs: Observation, opt: Option) -> np.ndarray:
    """選択肢(Option)1つを、状態ベクトルと組み合わせてスコアリングするための特徴量にする。

    「どの種類の選択肢か(OptionTypeのone-hot)」「対象カードの静的特徴」
    「ワザの静的特徴(ATTACK以外は全0)」「回数などの補助的な数値」の4種類を連結する。

    ATTACK型の選択肢は`attackId`しか持たないため、`resolve_target_card_id`では
    対象カードを特定できない（同じポケモンの複数のワザが区別できなくなる）。
    そのため、ワザの特徴だけは`attack_static_features`で別途補う。

    Args:
        obs: 対象の選択肢が含まれるObservation。
        opt: 特徴量にしたい選択肢。

    Returns:
        np.ndarray: 長さ`OPTION_DIM`の特徴ベクトル。
    """
    type_onehot = np.zeros(N_OPTION_TYPES, dtype=np.float32)
    type_onehot[int(opt.type)] = 1.0

    card_id = resolve_target_card_id(obs, opt)
    card_vec = card_static_features(card_id)
    attack_vec = attack_static_features(opt.attackId)

    extra = np.zeros(3, dtype=np.float32)
    extra[0] = (opt.number or 0) / 10.0
    extra[1] = (opt.count or 0) / 10.0
    extra[2] = 1.0 if opt.attackId is not None else 0.0

    return np.concatenate([type_onehot, card_vec, attack_vec, extra])
