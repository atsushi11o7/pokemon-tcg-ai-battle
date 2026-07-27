"""Evaluate two agents against each other over N episodes via kaggle_environments.

kaggle_environmentsの"cabt"環境は本番と同じハーネスなので、ここで使う`agent(obs_dict) -> list[int]`は
本番のmain.pyに書くものと全く同じ形でよい（cg.gameのように自分で対戦を進行させるコードは不要）。

Usage:
    uv run python src/evaluation/match_runner.py
"""

from kaggle_environments import make


def play_one_match(agent1, agent2) -> tuple[float, float]:
    """1試合実行し、(agent1の報酬, agent2の報酬)を返す。"""
    env = make("cabt", debug=True)
    env.run([agent1, agent2])
    final_state = env.steps[-1]
    return final_state[0].reward, final_state[1].reward


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


if __name__ == "__main__":
    # 動作確認用: kaggle_environments に組み込まれているランダムAI同士で対戦させる
    from kaggle_environments.envs.cabt.cabt import agents

    result = evaluate(agents["random"], agents["first"], n_episodes=30)
    print(result)
