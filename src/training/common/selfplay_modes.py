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
