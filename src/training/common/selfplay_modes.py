"""自己対戦モードに応じて、両サイドのデッキと学習サンプルを集める座席を決める。

MCTS(`mcts/selfplay.py`)・PPO(`ppo/ppo_selfplay.py`)の両方から共通で使う
(探索の有無以外、モードの意味付けは同じため)。
"""

import random
from typing import Literal

SelfplayMode = Literal["asymmetric", "mirror", "generalist"]


def pick_decks_and_collect_seats(
    mode: SelfplayMode,
    our_deck: list[int],
    opponent_deck_pool: list[list[int]] | None,
) -> tuple[list[list[int]], set[int]]:
    """モードに応じて対局両サイドのデッキと、学習サンプルを集める座席を決める。

    Args:
        mode: "asymmetric"(相手デッキランダム・自分側のみ学習)、
            "mirror"(両者同デッキ・両サイド学習)、
            "generalist"(両者とも実在デッキプールから独立ランダム・両サイド学習)。
        our_deck: 本番でも使う、こちらの60枚のデッキリスト。"generalist"では使われない。
        opponent_deck_pool: 対戦相手として選ぶ実在デッキのリスト。"mirror"では不要。

    Returns:
        tuple[list[list[int]], set[int]]: ([player0の60枚デッキ, player1の60枚デッキ],
            学習サンプルを集める座席の集合)。
    """
    if mode in ("asymmetric", "generalist") and not opponent_deck_pool:
        raise ValueError(f"opponent_deck_pool is required for mode={mode!r}")

    if mode == "mirror":
        return [our_deck, our_deck], {0, 1}
    if mode == "generalist":
        return [random.choice(opponent_deck_pool), random.choice(opponent_deck_pool)], {0, 1}
    if mode == "asymmetric":
        our_seat = random.randint(0, 1)
        decks = [our_deck, our_deck]
        decks[1 - our_seat] = random.choice(opponent_deck_pool)
        return decks, {our_seat}
    raise ValueError(f"unknown mode: {mode!r}")
