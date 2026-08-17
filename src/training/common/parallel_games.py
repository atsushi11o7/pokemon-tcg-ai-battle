"""ネイティブ対戦エンジンを使う試合を、監視可能な常駐プロセスで並列実行する。"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any


def _worker_main(connection: Connection, initializer, initargs: tuple, task) -> None:
    """初期化を一度だけ行い、親から受け取った試合を逐次実行する。"""
    try:
        initializer(*initargs)
        del initializer, initargs
        connection.send(("ready",))
        while True:
            message = connection.recv()
            if message is None:
                return
            game_index = int(message)
            try:
                connection.send(("result", game_index, task(game_index)))
            except BaseException as exc:
                connection.send(
                    ("error", game_index, f"{type(exc).__name__}: {exc}", traceback.format_exc())
                )
    finally:
        connection.close()


def _rss_bytes(pid: int) -> int | None:
    """Linux /procからワーカー固有の常駐メモリを読む。終了済み・非LinuxならNone。

    `statm`のresident(field 1)そのままだと、親から共有されたネットワーク重み
    (約107MB、torchが共有メモリ経由で1コピーを全workerへ渡す)が全workerのRSSに
    二重計上され、実際にはリークしていなくても「8worker × 107MB」に見えてしまう。
    リーク検知が目的なので、shared(field 2)を差し引いた固有分だけを記録する。
    """
    try:
        fields = Path(f"/proc/{pid}/statm").read_text(encoding="ascii").split()
        resident_pages = int(fields[1]) - int(fields[2])
        return max(0, resident_pages) * os.sysconf("SC_PAGE_SIZE")
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


@dataclass
class _Worker:
    process: multiprocessing.Process
    connection: Connection
    game_index: int | None = None
    started_at: float | None = None
    last_rss_bytes: int | None = None
    connection_lost: bool = False


def run_parallel_games(
    *,
    num_games: int,
    num_workers: int,
    initializer: Callable[..., None],
    initargs: tuple[Any, ...],
    task: Callable[[int], Any],
    game_timeout_seconds: float,
    round_timeout_seconds: float,
    on_result: Callable[[int, Any], None],
    on_failure: Callable[[int, str], None],
    event_log_path: Path | None = None,
    event_context: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """試合を並列実行し、ハング・異常終了したワーカーだけを交換する。"""
    if num_games <= 0:
        return 0, 0
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    log_file = None
    if event_log_path is not None:
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = event_log_path.open("a", encoding="utf-8")

    def emit(event: str, **fields: Any) -> None:
        if log_file is None:
            return
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            **(event_context or {}),
            "event": event,
            **fields,
        }
        log_file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        log_file.flush()

    context = multiprocessing.get_context("spawn")
    deadline = time.monotonic() + round_timeout_seconds
    pending = iter(range(num_games))
    workers: list[_Worker] = []
    completed = 0
    failed = 0
    exhausted = False
    # 試合を受け取る前に死んだワーカーの連続数。初期化が常に失敗する状況
    # (importエラー等)では、これを数えないとラウンド上限まで再生成し続けてしまう。
    startup_failures = 0
    max_startup_failures = 2 * num_workers

    def worker_fields(worker: _Worker) -> dict[str, Any]:
        rss = _rss_bytes(worker.process.pid)
        if rss is not None:
            worker.last_rss_bytes = rss
        return {
            "pid": worker.process.pid,
            "rss_bytes": worker.last_rss_bytes,
        }

    def start_worker() -> _Worker:
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_worker_main, args=(child_connection, initializer, initargs, task)
        )
        process.start()
        child_connection.close()
        worker = _Worker(process, parent_connection)
        worker.last_rss_bytes = _rss_bytes(process.pid)
        emit("worker_started", **worker_fields(worker), game_index=None)
        return worker

    def stop_worker(worker: _Worker, *, terminate: bool, reason: str) -> None:
        if terminate and worker.process.is_alive():
            worker.process.terminate()
        worker.process.join(timeout=5)
        if worker.process.is_alive():
            worker.process.kill()
            worker.process.join()
        emit(
            "worker_stopped",
            **worker_fields(worker),
            game_index=worker.game_index,
            exit_code=worker.process.exitcode,
            reason=reason,
        )
        worker.connection.close()

    def assign(worker: _Worker) -> None:
        nonlocal exhausted
        try:
            game_index = next(pending)
        except StopIteration:
            exhausted = True
            worker.game_index = None
            worker.started_at = None
            return
        worker.game_index = game_index
        worker.started_at = time.monotonic()
        try:
            worker.connection.send(game_index)
            emit("game_assigned", **worker_fields(worker), game_index=game_index)
        except (BrokenPipeError, EOFError, OSError):
            emit("game_assignment_failed", **worker_fields(worker), game_index=game_index)
            if worker.process.is_alive():
                worker.process.terminate()

    emit("round_started", num_games=num_games, num_workers=num_workers)
    try:
        for _ in range(min(num_workers, num_games)):
            workers.append(start_worker())

        while completed + failed < num_games:
            if time.monotonic() >= deadline:
                remaining = num_games - completed - failed
                for worker in workers:
                    if worker.game_index is not None:
                        emit(
                            "game_failed",
                            **worker_fields(worker),
                            game_index=worker.game_index,
                            reason="round_timeout",
                        )
                        on_failure(worker.game_index, "round timeout")
                # まだ着手していない試合も呼び出し側へ通知する。通知しないと
                # 呼び出し側の失敗集計と`failed`がずれ、部分的な標本で判定しかねない。
                for game_index in pending:
                    on_failure(game_index, "round timeout")
                failed += remaining
                emit("round_timeout", remaining_games=remaining)
                break

            for index in range(len(workers) - 1, -1, -1):
                worker = workers[index]
                if worker.process.is_alive():
                    continue
                game_index = worker.game_index
                exitcode = worker.process.exitcode
                emit(
                    "worker_exit",
                    **worker_fields(worker),
                    game_index=game_index,
                    exit_code=exitcode,
                )
                stop_worker(worker, terminate=False, reason="unexpected_exit")
                workers.pop(index)
                if game_index is not None:
                    startup_failures = 0
                    failed += 1
                    on_failure(game_index, f"worker exited with code {exitcode}")
                else:
                    startup_failures += 1
                    if startup_failures >= max_startup_failures:
                        raise RuntimeError(
                            f"{startup_failures} workers died before receiving a game "
                            f"(last exit code {exitcode}); worker initialization is failing"
                        )
                if not exhausted and completed + failed < num_games:
                    workers.append(start_worker())

            connections = [w.connection for w in workers if not w.connection_lost]
            for connection in wait(connections, timeout=0.1) if connections else []:
                worker = next(w for w in workers if w.connection is connection)
                try:
                    message = connection.recv()
                except (EOFError, OSError):
                    # 接続だけ先に閉じ、プロセスはまだ終了しきっていない状態。
                    # ここで取り除くと死因と担当試合が失われるので、監視対象から
                    # 外すだけにして、後段のプロセス終了検出に失敗通知を任せる
                    # (放置するとwait()が即座に返り続けて空回りする)。
                    worker.connection_lost = True
                    continue
                kind = message[0]
                if kind == "ready":
                    emit("worker_ready", **worker_fields(worker))
                    assign(worker)
                elif kind == "result":
                    _, game_index, result = message
                    elapsed = time.monotonic() - worker.started_at if worker.started_at else None
                    completed += 1
                    emit(
                        "game_completed",
                        **worker_fields(worker),
                        game_index=game_index,
                        elapsed_seconds=elapsed,
                    )
                    on_result(game_index, result)
                    worker.game_index = None
                    worker.started_at = None
                    assign(worker)
                elif kind == "error":
                    _, game_index, summary, tb = message
                    elapsed = time.monotonic() - worker.started_at if worker.started_at else None
                    failed += 1
                    emit(
                        "game_failed",
                        **worker_fields(worker),
                        game_index=game_index,
                        elapsed_seconds=elapsed,
                        reason="python_exception",
                        error=summary,
                    )
                    on_failure(game_index, f"{summary}\n{tb}")
                    worker.game_index = None
                    worker.started_at = None
                    assign(worker)

            now = time.monotonic()
            for index in range(len(workers) - 1, -1, -1):
                worker = workers[index]
                if (
                    worker.game_index is None
                    or worker.started_at is None
                    or now - worker.started_at <= game_timeout_seconds
                ):
                    continue
                game_index = worker.game_index
                elapsed = now - worker.started_at
                emit(
                    "game_failed",
                    **worker_fields(worker),
                    game_index=game_index,
                    elapsed_seconds=elapsed,
                    reason="game_timeout",
                )
                stop_worker(worker, terminate=True, reason="game_timeout")
                workers.pop(index)
                failed += 1
                on_failure(game_index, f"timed out after {game_timeout_seconds}s")
                if not exhausted and completed + failed < num_games:
                    workers.append(start_worker())
    finally:
        for worker in workers:
            if worker.process.is_alive():
                try:
                    worker.connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            stop_worker(worker, terminate=False, reason="round_cleanup")
        emit("round_finished", completed=completed, failed=failed)
        if log_file is not None:
            log_file.close()

    return completed, failed
