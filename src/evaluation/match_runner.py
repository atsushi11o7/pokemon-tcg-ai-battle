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


def random_agent_factory(deck: list[int]):
    """完全ランダムに選ぶ`agent()`を作る(「vs random」評価の対戦相手・floor-check用)。

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
        for x in (ROOT / "decks" / "cynthias_garchomp_ex.csv").read_text().split("\n")
        if x.strip()
    ]
    random_agent = random_agent_factory(deck)
    result = evaluate(random_agent, random_agent, n_episodes=30)
    print(result)
