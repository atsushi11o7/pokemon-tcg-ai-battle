"""自己対戦モードに応じて、両サイドのデッキと学習サンプルを集める座席を決める。

MCTS(`mcts/selfplay.py`)・PPO(`ppo/selfplay.py`)の両方から共通で使う
(探索の有無以外、モードの意味付けは同じため)。
"""

import random
from typing import Literal

SelfplayMode = Literal["asymmetric", "mirror", "generalist"]


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
        learner_deck = sample_deck(opponent_deck_pool, "learner")
        opponent_deck = sample_deck(opponent_deck_pool, "opponent")
        # デッキ分布とplayer index(先攻/後攻)を相関させない。
        # 両席とも同じポリシーの学習対象なのでcollect_seatsは変えない。
        if random.randrange(2):
            return [opponent_deck, learner_deck], {0, 1}
        return [learner_deck, opponent_deck], {0, 1}
    if mode == "asymmetric":
        our_seat = random.randint(0, 1)
        decks = [our_deck, our_deck]
        decks[1 - our_seat] = sample_deck(opponent_deck_pool, "opponent")
        return decks, {our_seat}
    raise ValueError(f"unknown mode: {mode!r}")
