"""Evaluate two agents against each other over N episodes, using `cg.game`直接。

以前は`kaggle_environments`の"cabt"環境経由だったが、`kaggle_environments`が独自に
バンドルした別バージョンの`libcg.so`が、自己対戦側の`cg.api`と同一プロセスに
同居することでまれにネイティブクラッシュしていた。自己対戦は元々`cg.game`を直接
使っているので、評価側もこちらに揃えて二重ロードを無くす。

`agent(obs_dict) -> list[int]`のインターフェースは本番の`main.py`と同じ
(デッキ提出時は`obs_dict["select"] is None`を見て自分のデッキを返す)。

Usage:
    uv run python src/evaluation/match_runner.py
"""

import random
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

# デッキ提出フェーズのobservation(kaggle_environments側の実装と同じ形式)。
_DECK_REQUEST_OBS = {"select": None, "logs": [], "current": None, "search_begin_input": None}


def play_one_match(agent1, agent2) -> tuple[float, float]:
    """1試合実行し、(agent1の報酬, agent2の報酬)を返す。"""
    deck1 = agent1(_DECK_REQUEST_OBS)
    deck2 = agent2(_DECK_REQUEST_OBS)
    if len(deck1) != 60:
        return -1.0, 1.0
    if len(deck2) != 60:
        return 1.0, -1.0

    obs_dict, start_data = battle_start(deck1, deck2)
    try:
        if start_data.errorPlayer == 0:
            return -1.0, 1.0
        if start_data.errorPlayer == 1:
            return 1.0, -1.0

        agents = [agent1, agent2]
        while True:
            obs = to_observation_class(obs_dict)
            result = obs.current.result
            if result == 0:
                return 1.0, -1.0
            if result == 1:
                return -1.0, 1.0
            if result == 2:
                return 0.0, 0.0
            select = agents[obs.current.yourIndex](obs_dict)
            obs_dict = battle_select(select)
    finally:
        battle_finish()


def evaluate(agent1, agent2, n_episodes: int, swap_positions: bool = True) -> dict:
    """agent1視点での勝敗数・勝率を集計する。

    Args:
        swap_positions: Trueなら、先攻/後攻(playerIndex 0/1)を1試合ごとに入れ替えて、
            先攻後攻の有利不利による偏りを均す。
    """
    wins = losses = draws = 0
    for i in range(n_episodes):
        if swap_positions and i % 2 == 1:
            r2, r1 = play_one_match(agent2, agent1)  # agent1をplayerIndex=1側にする
        else:
            r1, r2 = play_one_match(agent1, agent2)

        if r1 == 1:
            wins += 1
        elif r2 == 1:
            losses += 1
        else:
            draws += 1
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_episodes,
    }


def evaluate_fixed_matchups(
    agent1_factory: Callable[[list[int]], Callable[[dict], list[int]]],
    agent2_factory: Callable[[list[int]], Callable[[dict], list[int]]],
    matchups: list[tuple[list[int], list[int]]],
    *,
    seed: int,
    swap_decks: bool = True,
) -> dict:
    """固定したデッキ組合せを、同一Python seed・席交換で評価する。

    1つのmatchup ``(A, B)`` につき、まずagent1=A/agent2=Bで両方の席を
    1回ずつ対戦する。``swap_decks=True``ならagent1=B/agent2=Aでも同じ2戦を行う。
    したがって通常は1 matchupあたり4試合になる。

    各席交換ペアは同じPython乱数seedから開始する。これはPython側のデッキ推定、
    MCTS determinization、random agentを揃えるためのもので、seed APIを公開していない
    native CG engine内部のshuffleまで同一化する保証はない。
    """
    wins = losses = draws = 0
    random_state = random.getstate()
    try:
        for matchup_index, (deck_a, deck_b) in enumerate(matchups):
            assignments = [(deck_a, deck_b)]
            if swap_decks:
                assignments.append((deck_b, deck_a))

            for assignment_index, (agent1_deck, agent2_deck) in enumerate(assignments):
                pair_seed = seed + matchup_index * 2 + assignment_index

                random.seed(pair_seed)
                agent1 = agent1_factory(agent1_deck)
                agent2 = agent2_factory(agent2_deck)
                r1, r2 = play_one_match(agent1, agent2)
                if r1 == 1:
                    wins += 1
                elif r2 == 1:
                    losses += 1
                else:
                    draws += 1

                random.seed(pair_seed)
                agent1 = agent1_factory(agent1_deck)
                agent2 = agent2_factory(agent2_deck)
                r2, r1 = play_one_match(agent2, agent1)
                if r1 == 1:
                    wins += 1
                elif r2 == 1:
                    losses += 1
                else:
                    draws += 1
    finally:
        random.setstate(random_state)

    n_games = wins + losses + draws
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": n_games,
        "win_rate": wins / n_games if n_games else 0.0,
    }


def first_index_agent_factory(deck: list[int]):
    """常に先頭の選択肢を返す`agent()`を作る(学習の進捗を測る基準線)。

    エンジンが提示する選択肢の並びには質の情報が含まれており、先頭を選ぶだけでも
    ランダムより明確に強い(実測でランダム相手に約64%)。ランダム相手の勝率は
    早々に100%へ飽和して改善も劣化も検出できなくなるため、こちらを基準線に使う。

    Args:
        deck: デッキ提出時に返す60枚のデッキリスト。

    Returns:
        Callable[[dict], list[int]]: `evaluate_fixed_matchups`に渡せるagent関数。
    """

    def agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        sel = obs.select
        count = min(max(sel.minCount, 1), len(sel.option))
        return list(range(count))

    return agent


def random_agent_factory(deck: list[int]):
    """完全ランダムに選ぶ`agent()`を作る(下限確認用)。

    Args:
        deck: デッキ提出時に返す60枚のデッキリスト。

    Returns:
        Callable[[dict], list[int]]: `evaluate`に渡せるagent関数。
    """

    def agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        sel = obs.select
        count = random.randint(sel.minCount, sel.maxCount)
        return random.sample(range(len(sel.option)), count)

    return agent


if __name__ == "__main__":
    # 動作確認用: ランダムAI同士で対戦させる
    deck = [
        int(x)
        for x in (ROOT / "decks" / "candidates" / "crustle_meta.csv").read_text().split("\n")
        if x.strip()
    ]
    random_agent = random_agent_factory(deck)
    result = evaluate(random_agent, random_agent, n_episodes=30)
    print(result)
