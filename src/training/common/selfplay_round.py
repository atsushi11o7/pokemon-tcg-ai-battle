"""1ラウンド分の自己対戦をspawnワーカーで並列実行し、サンプルと勝敗を集計する。

MCTS・PPOで収集するサンプルの中身は違うが、「試合を並列に流して結果を積み上げ、
失敗した試合だけ捨てる」という枠組みは同一なのでここに集約する。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .parallel_games import run_parallel_games
from .selfplay_modes import SelfplayMode

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
