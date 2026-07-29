"""MCTS探索木の葉ノードを、ゲーム終了までロールアウトせずに素早く評価するヒューリスティック関数。

サイド差(勝敗に直結する)を主要な指標とし、盤面のHP割合・エネルギー数といった
副次的な指標を小さい重みで加える。返り値は探索木のバックプロパゲーションで
勝敗(+1/-1/0、終局時にsearch.pyが直接設定する)と同じスケールで扱えるよう、
[-1, 1]に収まるようにクリップする。
"""

PRIZE_MAX = 6
PRIZE_WEIGHT = 1.0
HP_WEIGHT = 0.15
ENERGY_WEIGHT = 0.05


def heuristic_value(obs, your_index: int) -> float:
    """現在の局面を、your_index視点の有利さとして[-1, 1]程度のスカラーで評価する。

    Args:
        obs: 評価対象の局面のObservation(探索中に得られるものでよい。`obs.current`が
            Noneでないこと)。
        your_index: 評価の基準にするプレイヤーのインデックス(0 or 1)。

    Returns:
        float: 値が大きいほどyour_indexに有利。サイド差を中心に、HP・エネルギー数を加味する。
    """
    state = obs.current
    me = state.players[your_index]
    opp = state.players[1 - your_index]

    prize_diff = (len(opp.prize) - len(me.prize)) / PRIZE_MAX
    hp_diff = (_hp_ratio_sum(me) - _hp_ratio_sum(opp)) / 6.0
    energy_diff = (_energy_count(me) - _energy_count(opp)) / 12.0

    value = PRIZE_WEIGHT * prize_diff + HP_WEIGHT * hp_diff + ENERGY_WEIGHT * energy_diff
    return max(-1.0, min(1.0, value))


def _hp_ratio_sum(player_state) -> float:
    """場に出ている全ポケモンのHP残存率(0〜1)の合計を返す。

    Args:
        player_state: 評価対象プレイヤーの`PlayerState`。

    Returns:
        float: active + benchの各ポケモンについて`hp/maxHp`を合計した値。
    """
    pokemons = list(player_state.active) + list(player_state.bench)
    return sum(p.hp / p.maxHp for p in pokemons if p is not None and p.maxHp > 0)


def _energy_count(player_state) -> int:
    """場に出ている全ポケモンに付いているエネルギーの総数を返す。

    Args:
        player_state: 評価対象プレイヤーの`PlayerState`。

    Returns:
        int: active + benchの各ポケモンの`energies`の合計枚数。
    """
    pokemons = list(player_state.active) + list(player_state.bench)
    return sum(len(p.energies) for p in pokemons if p is not None)
