"""自己対戦のラウンド実行と、固定matchupによる候補評価。

どちらも`parallel_games.run_parallel_games`の上に乗る。低レベルの
プロセスプール側はtorchや評価モジュールに依存させないため、分けてある。
"""

from __future__ import annotations

import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

from .deck import SelfplayMode, sample_deck  # noqa: E402
from .deck_pool import configure_sampling_snapshot, seed_opponent_deck_pool_cache  # noqa: E402
from .network import PolicyValueNet, build_policy_value_net  # noqa: E402
from .parallel_games import run_parallel_games  # noqa: E402
from .training_utils import seed_game  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from match_runner import first_index_agent_factory, play_one_match  # noqa: E402

# ---- 自己対戦のラウンド -------------------------------------------------


PROGRESS_INTERVAL = 20  # 何試合ごとに進捗を出すか


def run_selfplay_round(
    *,
    algorithm: str,
    round_num: int,
    mode: SelfplayMode,
    num_games: int,
    num_workers: int,
    initializer: Callable[..., None],
    initargs: tuple[Any, ...],
    task: Callable[[int], Any],
    game_timeout_seconds: float,
    round_timeout_seconds: float,
    event_log_path: Path,
    min_completion_rate: float = 0.8,
) -> tuple[list, dict[int, int]]:
    """自己対戦を並列実行し、`(全試合分のサンプル, 勝敗内訳)`を返す。

    `task`は`(サンプルのリスト, 勝者のplayerIndex)`を返すこと。ハングやネイティブ
    クラッシュで落ちた試合はスキップ扱いにして、ラウンド全体は続行する。

    Args:
        algorithm: イベントログに残す識別子("mcts"/"ppo")。
        round_num: 進捗表示とイベントログ用のラウンド番号。
        mode: 自己対戦モード(イベントログ用)。
        num_games: このラウンドで行う試合数。
        num_workers: 並列プロセス数。
        initializer: ワーカー初期化関数(プロセスごとに1回)。
        initargs: `initializer`へ渡す引数。spawn境界を越えるのはここだけなので、
            ワーカーが必要とする設定はすべてこれに含めること。
        task: 試合番号を受け取り`(samples, winner)`を返す関数。
        game_timeout_seconds: 1試合あたりの上限。
        round_timeout_seconds: ラウンド全体の上限。
        event_log_path: ワーカーイベントのJSONL出力先。
        min_completion_rate: この割合を下回る完走率なら例外にする。cgエンジンは
            まれにネイティブクラッシュするため少数の失敗は許容するが、大半が
            失敗したまま更新して次のラウンドへ進むと、偏った少数サンプルで
            方策を壊したまま学習が続いてしまう。

    Returns:
        tuple[list, dict[int, int]]: (全試合分のサンプル, `{0: , 1: , 2(引分): }`)。
    """
    all_samples: list = []
    results = {0: 0, 1: 0, 2: 0}
    processed = 0

    def on_result(_game_index: int, result) -> None:
        nonlocal processed
        samples, winner = result
        all_samples.extend(samples)
        results[winner] += 1
        processed += 1
        if processed % PROGRESS_INTERVAL == 0 or processed == num_games:
            print(f"  games processed={processed}/{num_games}  results_so_far={results}")

    def on_failure(game_index: int, reason: str) -> None:
        nonlocal processed
        processed += 1
        print(f"  game {game_index + 1}/{num_games} failed: {reason}")

    _completed, skipped = run_parallel_games(
        num_games=num_games,
        num_workers=num_workers,
        initializer=initializer,
        initargs=initargs,
        task=task,
        game_timeout_seconds=game_timeout_seconds,
        round_timeout_seconds=round_timeout_seconds,
        on_result=on_result,
        on_failure=on_failure,
        event_log_path=event_log_path,
        event_context={"algorithm": algorithm, "round": round_num, "mode": mode},
    )
    if skipped:
        print(f"  skipped {skipped}/{num_games} failed or timed-out games")
    completion_rate = (num_games - skipped) / num_games if num_games else 1.0
    if completion_rate < min_completion_rate:
        raise RuntimeError(
            f"only {completion_rate:.0%} of games completed "
            f"({num_games - skipped}/{num_games}); refusing to train on a biased sample"
        )
    return all_samples, results


# ---- 固定matchupの評価 --------------------------------------------------


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
        # ネットワークを持たない相手は「常に先頭の選択肢を返す」基準線。
        opponent_agent = first_index_agent_factory(spec.opponent_deck)
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
