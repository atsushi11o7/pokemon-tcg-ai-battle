"""PPO/MCTS共通の学習・状態確認CLI。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from .common.run_config import RunConfig, load_run_config  # noqa: E402
from .common.training_utils import latest_checkpoint_round  # noqa: E402

TRAINER_MODULES = {
    "ppo": "training.ppo.train",
    "mcts": "training.mcts.train",
    "bc": "training.bc.train",
}
NATIVE_CRASH_EXIT_CODES = {134, 137, 139}


def run_status(config: RunConfig) -> dict:
    latest = latest_checkpoint_round(config.checkpoint_dir, config.selfplay_mode)
    final_path = config.checkpoint_dir / "final.pt"
    initial = config.model.initial_checkpoint
    if final_path.exists():
        state = "COMPLETE"
    elif latest is not None and latest >= config.n_rounds:
        state = "FINALIZING"
    elif latest is not None:
        state = "IN_PROGRESS"
    elif initial is not None and not initial.exists():
        state = "BLOCKED"
    else:
        state = "NOT_STARTED"

    data_summary: dict = {}
    if config.sampling_snapshot.exists():
        try:
            with config.sampling_snapshot.open(encoding="utf-8") as file:
                snapshot = json.load(file)
            data_summary = {
                "replays": snapshot.get("replayCount"),
                "decks": snapshot.get("deckCount"),
                "archetypes": snapshot.get("archetypeCount"),
            }
        except (OSError, json.JSONDecodeError):
            data_summary = {"error": "sampling snapshot is unreadable"}

    return {
        "name": config.name,
        "algorithm": config.algorithm.upper(),
        "status": state,
        "progress": {"current_round": latest or 0, "total_rounds": config.n_rounds},
        "initial_checkpoint": str(initial) if initial else None,
        "initial_checkpoint_exists": initial.exists() if initial else None,
        "latest_checkpoint": (
            str(config.checkpoint_dir / f"{config.selfplay_mode}_round{latest}.pt")
            if latest is not None
            else None
        ),
        "final_checkpoint": str(final_path) if final_path.exists() else None,
        "sampling_snapshot": str(config.sampling_snapshot),
        "sampling_snapshot_exists": config.sampling_snapshot.exists(),
        "data": data_summary,
    }


def _print_status(status: dict) -> None:
    progress = status["progress"]
    print(f"Run: {status['name']}")
    print(f"Algorithm: {status['algorithm']}")
    print(f"Status: {status['status']}")
    print(f"Progress: {progress['current_round']} / {progress['total_rounds']}")
    print(f"Initial checkpoint: {status['initial_checkpoint']}")
    if status["latest_checkpoint"]:
        print(f"Latest checkpoint: {status['latest_checkpoint']}")
    if status["final_checkpoint"]:
        print(f"Final checkpoint: {status['final_checkpoint']}")
    data = status["data"]
    if data and "error" not in data:
        print(
            f"Data: {data.get('replays')} replays, {data.get('decks')} decks, "
            f"{data.get('archetypes')} archetypes"
        )


def _validate_inputs(config: RunConfig) -> None:
    if not config.sampling_snapshot.exists():
        raise FileNotFoundError(f"sampling snapshot does not exist: {config.sampling_snapshot}")
    if config.deck_path is not None and not config.deck_path.exists():
        raise FileNotFoundError(f"deck does not exist: {config.deck_path}")
    if latest_checkpoint_round(config.checkpoint_dir, config.selfplay_mode) is None:
        initial = config.model.initial_checkpoint
        if initial is not None and not initial.exists():
            raise FileNotFoundError(f"initial checkpoint does not exist: {initial}")


def _run_once(config_path: Path, config: RunConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.output_dir / "training.log"
    command = [sys.executable, "-m", TRAINER_MODULES[config.algorithm], str(config_path)]
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def train(config_path: Path) -> int:
    config = load_run_config(config_path)
    _validate_inputs(config)
    status = run_status(config)
    if status["status"] == "COMPLETE":
        print(f"{config.name} is already complete: {status['final_checkpoint']}")
        return 0

    attempts = 1 + config.runtime.max_restarts
    for attempt in range(1, attempts + 1):
        print(f"=== {config.name}: attempt {attempt}/{attempts} ===")
        return_code = _run_once(config_path, config)
        if return_code == 0:
            return 0
        native_crash = return_code < 0 or return_code in NATIVE_CRASH_EXIT_CODES
        if not native_crash:
            print(f"trainer stopped with non-retryable exit code {return_code}", file=sys.stderr)
            return return_code
        if attempt == attempts:
            break
        print(
            f"native crash (exit {return_code}); retrying from latest checkpoint in "
            f"{config.runtime.retry_delay_seconds}s"
        )
        time.sleep(config.runtime.retry_delay_seconds)
    print(f"gave up after {attempts} attempts", file=sys.stderr)
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m training.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "status", "validate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        if command == "status":
            subparser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    if args.command == "train":
        return train(config_path)
    config = load_run_config(config_path)
    if args.command == "validate":
        _validate_inputs(config)
        print(f"valid config: {config.name} ({config.algorithm})")
        return 0
    status = run_status(config)
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        _print_status(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
