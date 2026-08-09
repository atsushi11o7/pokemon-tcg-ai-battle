"""固定matchup評価を、監視可能なspawn workerで並列実行する。

直列版(`evaluation.match_runner.evaluate_fixed_matchups`)と同じ組み合わせ・同じseedの
試合を、1試合=1タスクに展開してワーカーへ流す。ネイティブエンジンのクラッシュやハングを
trainer本体から隔離し、1試合単位のタイムアウトを効かせるのが目的。

MCTS/PPOのどちらでも使えるよう、agentの作り方だけ`agent_factory`で差し替える。
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from .network import PolicyValueNet, build_policy_value_net  # noqa: E402
from .opponent_pool import configure_sampling_snapshot, seed_opponent_deck_pool_cache  # noqa: E402
from .parallel_games import run_parallel_games  # noqa: E402
from .training_utils import seed_game  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from match_runner import play_one_match, random_agent_factory  # noqa: E402

# agentの作り方。(network, deck) -> agent。spawn境界を越えるのでpickle可能であること
# (module levelの関数か、そのfunctools.partial)。
AgentFactory = Callable[[PolicyValueNet, list[int]], Callable[[dict], list[int]]]


@dataclass(frozen=True)
class EvaluationGame:
    """固定評価1局分の、モデル・デッキ・座席・seed指定。"""

    opponent_index: int
    candidate_deck: list[int]
    opponent_deck: list[int]
    candidate_seat: int
    seed: int


def build_evaluation_games(
    matchups: list[tuple[list[int], list[int]]],
    num_opponents: int,
    seed: int,
) -> list[EvaluationGame]:
    """直列版と同じデッキ交換・席交換・pair seedを試合単位へ展開する。"""
    games: list[EvaluationGame] = []
    for opponent_index in range(num_opponents):
        for matchup_index, (deck_a, deck_b) in enumerate(matchups):
            for assignment_index, (candidate_deck, opponent_deck) in enumerate(
                ((deck_a, deck_b), (deck_b, deck_a))
            ):
                pair_seed = seed + matchup_index * 2 + assignment_index
                games.append(
                    EvaluationGame(
                        opponent_index,
                        candidate_deck,
                        opponent_deck,
                        candidate_seat=0,
                        seed=pair_seed,
                    )
                )
                games.append(
                    EvaluationGame(
                        opponent_index,
                        candidate_deck,
                        opponent_deck,
                        candidate_seat=1,
                        seed=pair_seed,
                    )
                )
    return games


@dataclass(frozen=True)
class _EvaluationWorkerContext:
    """spawn workerが固定評価に必要とする設定一式。"""

    agent_factory: AgentFactory
    opponent_deck_pool: list[list[int]]
    sampling_snapshot: Path
    games: list[EvaluationGame]


_worker_context: _EvaluationWorkerContext | None = None
_worker_candidate: PolicyValueNet | None = None
_worker_opponents: list[PolicyValueNet | None] = []


def _init_evaluation_worker(
    candidate_state_dict: dict,
    opponent_state_dicts: list[dict | None],
    context: _EvaluationWorkerContext,
) -> None:
    """デッキプールと全評価モデルをworkerごとに一度だけロードする。"""
    global _worker_context, _worker_candidate, _worker_opponents

    import torch

    torch.set_num_threads(1)
    configure_sampling_snapshot(context.sampling_snapshot)
    seed_opponent_deck_pool_cache(context.opponent_deck_pool)
    _worker_candidate = build_policy_value_net(candidate_state_dict, assign=True)
    _worker_opponents = [
        build_policy_value_net(state_dict, assign=True) if state_dict is not None else None
        for state_dict in opponent_state_dicts
    ]
    _worker_context = context


def _play_evaluation_game(game_index: int) -> tuple[int, float]:
    context = _worker_context
    spec = context.games[game_index]
    seed_game(spec.seed)

    candidate_agent = context.agent_factory(_worker_candidate, spec.candidate_deck)
    opponent_network = _worker_opponents[spec.opponent_index]
    if opponent_network is None:
        opponent_agent = random_agent_factory(spec.opponent_deck)
    else:
        opponent_agent = context.agent_factory(opponent_network, spec.opponent_deck)

    if spec.candidate_seat == 0:
        reward, _ = play_one_match(candidate_agent, opponent_agent)
    else:
        _, reward = play_one_match(opponent_agent, candidate_agent)
    return spec.opponent_index, reward


def _empty_result() -> dict[str, int | float]:
    return {"wins": 0, "losses": 0, "draws": 0, "games": 0, "failed": 0, "win_rate": 0.0}


def evaluate_networks_parallel(
    candidate_network: PolicyValueNet,
    opponents: list[tuple[str, PolicyValueNet | None]],
    opponent_deck_pool: list[list[int]],
    matchups: list[tuple[list[int], list[int]]],
    *,
    agent_factory: AgentFactory,
    sampling_snapshot: Path,
    seed: int,
    num_workers: int,
    game_timeout_seconds: float,
    round_timeout_seconds: float,
    event_log_path: Path | None = None,
    event_context: dict | None = None,
) -> dict[str, dict[str, int | float]]:
    """候補を複数相手へ固定条件で評価し、相手別に勝敗を返す。

    `opponents`のnetworkが`None`の要素はランダムagentとして扱う(下限の目安用)。
    `win_rate`は完走した試合だけの比率で、落ちた試合数は`failed`に出る。
    採用判定に使う側は`failed == 0`も併せて確認すること(部分集合の勝率は偏るため)。
    """
    if not opponents:
        return {}

    games = build_evaluation_games(matchups, len(opponents), seed)
    results = {name: _empty_result() for name, _network in opponents}
    names = [name for name, _network in opponents]
    context = _EvaluationWorkerContext(
        agent_factory=agent_factory,
        opponent_deck_pool=opponent_deck_pool,
        sampling_snapshot=sampling_snapshot,
        games=games,
    )

    def on_result(_game_index: int, result: tuple[int, float]) -> None:
        opponent_index, reward = result
        summary = results[names[opponent_index]]
        summary["games"] += 1
        if reward == 1.0:
            summary["wins"] += 1
        elif reward == -1.0:
            summary["losses"] += 1
        else:
            summary["draws"] += 1

    def on_failure(game_index: int, reason: str) -> None:
        spec = games[game_index]
        results[names[spec.opponent_index]]["failed"] += 1
        print(f"  evaluation game {game_index + 1}/{len(games)} failed: {reason}")

    run_parallel_games(
        num_games=len(games),
        num_workers=num_workers,
        initializer=_init_evaluation_worker,
        initargs=(
            candidate_network.state_dict(),
            [network.state_dict() if network is not None else None for _name, network in opponents],
            context,
        ),
        task=_play_evaluation_game,
        game_timeout_seconds=game_timeout_seconds,
        round_timeout_seconds=round_timeout_seconds,
        on_result=on_result,
        on_failure=on_failure,
        event_log_path=event_log_path,
        event_context=event_context,
    )

    for summary in results.values():
        games_completed = int(summary["games"])
        summary["win_rate"] = int(summary["wins"]) / games_completed if games_completed else 0.0
    return results
