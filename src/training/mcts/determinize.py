"""Determinized MCTSの探索用に、隠れ情報(相手の手札・山札・サイド)をサンプリングする。

自分の情報は既知のデッキリストからサンプリングする。相手の情報は不明なため、
過去リプレイの実在デッキから作ったカード出現頻度プールから確率的に推測する。
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EPISODES_DIR = ROOT / "data" / "episodes"
SAMPLE_SUBMISSION_DIR = ROOT / "data" / "sample_submission" / "sample_submission"
sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

from cg.api import CardType, all_card_data  # noqa: E402

_opponent_pool: list[int] | None = None
_pokemon_card_ids: set[int] | None = None
_opponent_deck_pool: list[list[int]] | None = None


def _load_opponent_pool() -> list[int]:
    """リプレイ中の実在デッキ(`steps[0][0]["visualize"]`)からカード出現頻度プールを作る。

    Returns:
        list[int]: 出現頻度に応じて重複を含むカードIDのリスト(`random.choices`の母集団)。
    """
    counter: Counter[int] = Counter()
    for path in EPISODES_DIR.glob("*.json"):
        with path.open() as f:
            episode = json.load(f)
        for viz in episode["steps"][0][0].get("visualize") or []:
            action = viz.get("action")
            if (
                isinstance(action, list)
                and len(action) == 2
                and all(isinstance(deck, list) and len(deck) == 60 for deck in action)
            ):
                counter.update(action[0])
                counter.update(action[1])
    if not counter:
        raise RuntimeError(f"No deck data found under {EPISODES_DIR}")
    return list(counter.elements())


def _opponent_card_pool() -> list[int]:
    """出現頻度プールを遅延ロードしてキャッシュする。

    Returns:
        list[int]: `_load_opponent_pool`が返すプール。
    """
    global _opponent_pool
    if _opponent_pool is None:
        _opponent_pool = _load_opponent_pool()
    return _opponent_pool


def sample_opponent_hidden(
    deck_count: int, hand_count: int, prize_count: int
) -> tuple[list[int], list[int], list[int]]:
    """相手の山札・手札・サイドを、カード出現頻度プールからサンプリングする。

    Args:
        deck_count: 相手の残り山札枚数(`PlayerState.deckCount`)。
        hand_count: 相手の手札枚数(`PlayerState.handCount`)。
        prize_count: 相手の残りサイド枚数(`len(PlayerState.prize)`)。

    Returns:
        tuple[list[int], list[int], list[int]]: (opponent_deck, opponent_hand, opponent_prize)。
            `search_begin`にそのまま渡せる形式。
    """
    pool = _opponent_card_pool()
    total = deck_count + hand_count + prize_count
    sampled = random.choices(pool, k=total)
    random.shuffle(sampled)
    return (
        sampled[:deck_count],
        sampled[deck_count : deck_count + hand_count],
        sampled[deck_count + hand_count :],
    )


def sample_own_hidden(
    own_deck: list[int], deck_count: int, prize_count: int
) -> tuple[list[int], list[int]]:
    """自分の山札・サイドを、既知のデッキリストからランダムに割り当てる。

    Args:
        own_deck: 自分の60枚のデッキリスト(カードID)。
        deck_count: 自分の残り山札枚数。
        prize_count: 自分の残りサイド枚数。

    Returns:
        tuple[list[int], list[int]]: (your_deck, your_prize)。`search_begin`にそのまま渡せる形式。
    """
    sampled = random.sample(own_deck, deck_count + prize_count)
    return sampled[:deck_count], sampled[deck_count:]


def _pokemon_ids() -> set[int]:
    """全カードマスタから、ポケモンカードのIDだけを遅延ロードしてキャッシュする。

    Returns:
        set[int]: `CardType.POKEMON`であるカードIDの集合。
    """
    global _pokemon_card_ids
    if _pokemon_card_ids is None:
        _pokemon_card_ids = {c.cardId for c in all_card_data() if c.cardType == CardType.POKEMON}
    return _pokemon_card_ids


def sample_opponent_active_guess() -> int:
    """相手のアクティブポケモンが伏せられている場合に、その正体として仮定するカードIDを選ぶ。

    `search_begin`はポケモンカード以外のIDを渡すとエラーになるため、出現頻度プールを
    ポケモンカードのみに絞ってサンプリングする。

    Returns:
        int: 仮定するポケモンカードのID。
    """
    pool = _opponent_card_pool()
    pokemon_ids = _pokemon_ids()
    candidates = [card_id for card_id in pool if card_id in pokemon_ids]
    if not candidates:
        candidates = list(pokemon_ids)
    return random.choice(candidates)


def sample_full_hidden(
    known_deck: list[int], deck_count: int, hand_count: int, prize_count: int
) -> tuple[list[int], list[int], list[int]]:
    """デッキ構成が既知の相手(自己対戦)向けに、山札・手札・サイドをまとめて割り当てる。

    Args:
        known_deck: 既知の60枚のデッキリスト(カードID)。
        deck_count: 残り山札枚数。
        hand_count: 手札枚数。
        prize_count: 残りサイド枚数。

    Returns:
        tuple[list[int], list[int], list[int]]: (deck, hand, prize)。
            `search_begin`にそのまま渡せる形式。
    """
    sampled = random.sample(known_deck, deck_count + hand_count + prize_count)
    return (
        sampled[:deck_count],
        sampled[deck_count : deck_count + hand_count],
        sampled[deck_count + hand_count :],
    )


def load_opponent_deck_pool() -> list[list[int]]:
    """リプレイ中の実在デッキから、重複を除いた60枚デッキの一覧を集める。

    `_load_opponent_pool`がカード単位の出現頻度を集計するのに対し、こちらはデッキ単位で
    重複を除いて集める。自己対戦の対戦相手を実在デッキからランダムに選ぶために使う。

    Returns:
        list[list[int]]: 重複を除いた60枚デッキのリスト。
    """
    global _opponent_deck_pool
    if _opponent_deck_pool is not None:
        return _opponent_deck_pool

    seen: set[tuple[int, ...]] = set()
    decks: list[list[int]] = []
    for path in EPISODES_DIR.glob("*.json"):
        with path.open() as f:
            episode = json.load(f)
        for viz in episode["steps"][0][0].get("visualize") or []:
            action = viz.get("action")
            if (
                isinstance(action, list)
                and len(action) == 2
                and all(isinstance(deck, list) and len(deck) == 60 for deck in action)
            ):
                for deck in action:
                    key = tuple(sorted(deck))
                    if key not in seen:
                        seen.add(key)
                        decks.append(deck)
    if not decks:
        raise RuntimeError(f"No deck data found under {EPISODES_DIR}")
    _opponent_deck_pool = decks
    return _opponent_deck_pool
