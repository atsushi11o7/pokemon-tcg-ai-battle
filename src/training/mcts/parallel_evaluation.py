"""MCTSモデル向けの固定matchup並列評価アダプタ。

試合の組み立て・ワーカー管理は`common/parallel_evaluation.py`が担当し、ここでは
「MCTS探索で1手選ぶagent」の作り方だけを与える。
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from ..common.network import PolicyValueNet  # noqa: E402
from ..common.parallel_evaluation import (  # noqa: E402
    EvaluationGame,
    build_evaluation_games,
    evaluate_networks_parallel,
)
from .selfplay import run_determinized_mcts  # noqa: E402

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import Observation, to_observation_class  # noqa: E402

__all__ = ["EvaluationGame", "build_evaluation_games", "evaluate_mcts_networks_parallel"]


def mcts_agent_factory(
    network: PolicyValueNet,
    deck: list[int],
    *,
    opponent_deck_pool: list[list[int]],
    search_count: int,
    num_determinizations: int,
):
    """MCTS探索で1手選ぶagentを作る。

    module levelの関数なので、`functools.partial`にしてspawn workerへpickleできる。
    """

    def agent(obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        select, _policy, _value, _actions = run_determinized_mcts(
            network,
            obs,
            deck,
            opponent_deck_pool,
            search_count,
            num_determinizations=num_determinizations,
        )
        return select

    return agent


def evaluate_mcts_networks_parallel(
    candidate_network: PolicyValueNet,
    opponents: list[tuple[str, PolicyValueNet | None]],
    opponent_deck_pool: list[list[int]],
    matchups: list[tuple[list[int], list[int]]],
    *,
    search_count: int,
    num_determinizations: int,
    sampling_snapshot: Path,
    seed: int,
    num_workers: int,
    game_timeout_seconds: float,
    round_timeout_seconds: float,
    event_log_path: Path | None = None,
    event_context: dict | None = None,
) -> dict[str, dict[str, int | float]]:
    """候補のMCTSモデルを複数相手へ固定条件で評価し、相手別に勝敗を返す。"""
    return evaluate_networks_parallel(
        candidate_network,
        opponents,
        opponent_deck_pool,
        matchups,
        agent_factory=functools.partial(
            mcts_agent_factory,
            opponent_deck_pool=opponent_deck_pool,
            search_count=search_count,
            num_determinizations=num_determinizations,
        ),
        sampling_snapshot=sampling_snapshot,
        seed=seed,
        num_workers=num_workers,
        game_timeout_seconds=game_timeout_seconds,
        round_timeout_seconds=round_timeout_seconds,
        event_log_path=event_log_path,
        event_context=event_context,
    )
