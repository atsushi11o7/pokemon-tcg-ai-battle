"""PPO/MCTSで共通利用する、低分散な評価用デッキ組合せの生成。"""

from __future__ import annotations

import random

from .selfplay_modes import SelfplayMode, sample_deck


def build_fixed_matchups(
    mode: SelfplayMode,
    deck: list[int],
    opponent_deck_pool: list[list[int]],
    n_games: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    """4試合ブロック用のmatchupを、学習側の乱数状態を変えずに先に固定する。"""
    if n_games <= 0 or n_games % 4 != 0:
        raise ValueError("evaluation games must be a positive multiple of 4")

    random_state = random.getstate()
    try:
        random.seed(seed)
        if mode == "generalist":
            return [
                (
                    sample_deck(opponent_deck_pool, "learner"),
                    sample_deck(opponent_deck_pool, "opponent"),
                )
                for _ in range(n_games // 4)
            ]
        if mode == "asymmetric":
            return [
                (deck, sample_deck(opponent_deck_pool, "opponent")) for _ in range(n_games // 4)
            ]
        return [(deck, deck) for _ in range(n_games // 4)]
    finally:
        random.setstate(random_state)
