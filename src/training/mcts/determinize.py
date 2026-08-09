"""MCTS探索用の隠れ情報を、実在デッキと公開情報に整合する形でサンプリングする。"""

import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_SUBMISSION_DIR = ROOT / "data" / "sample_submission" / "sample_submission"
sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

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


def _choose_compatible_deck(
    candidates: list[list[int]], known_cards: list[int], required_hidden: int
) -> tuple[list[int], list[int]]:
    """公開カードを含み、必要な隠れ枚数を確保できる実在デッキを選ぶ。"""
    compatible: list[tuple[list[int], list[int]]] = []
    for deck in candidates:
        remaining = _remaining_deck(deck, known_cards)
        if remaining is not None and len(remaining) >= required_hidden:
            compatible.append((deck, remaining))
    if not compatible:
        raise ValueError(
            "no opponent deck candidate is compatible with the observed public cards "
            f"(known={Counter(known_cards)}, required_hidden={required_hidden})"
        )
    return random.choice(compatible)


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
