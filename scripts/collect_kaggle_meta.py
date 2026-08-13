#!/usr/bin/env python3
"""Kaggleの公開ラダーから、再現可能なPTCGメタスナップショットを収集する。

収集順は leaderboard -> team submissions -> episodes -> replay。episode IDで重複を
除き、途中で停止しても既に検証済みのreplayは再ダウンロードしない。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "pokemon-tcg-ai-battle"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "meta" / "raw"
T = TypeVar("T")


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    return vars(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def _retry(label: str, operation: Callable[[], T], attempts: int = 4) -> T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:  # Kaggle SDKはHTTP例外の型が版ごとに異なる
            last_error = error
            if attempt == attempts:
                break
            delay = min(2 ** (attempt - 1), 8)
            print(f"{label}: attempt {attempt}/{attempts} failed: {error}; retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last_error


def _with_timeout(seconds: int, operation: Callable[[], T]) -> T:
    """Kaggle SDKのread待ちが無期限にならないようSIGALRMで打ち切る。"""

    def handle_timeout(_signum, _frame):
        raise TimeoutError(f"Kaggle replay download exceeded {seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return operation()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _score(record: dict[str, Any], key: str) -> float:
    try:
        return float(record.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _select_teams(
    leaderboard: list[dict[str, Any]], top_teams: int, mid_teams: int
) -> list[dict[str, Any]]:
    """上位を全採用し、残りから順位帯を均等に層化抽出する。"""
    top = leaderboard[:top_teams]
    remainder = leaderboard[top_teams:]
    if mid_teams <= 0 or not remainder:
        selected = top
    elif mid_teams >= len(remainder):
        selected = top + remainder
    else:
        indices = {
            round(i * (len(remainder) - 1) / max(mid_teams - 1, 1)) for i in range(mid_teams)
        }
        selected = top + [remainder[index] for index in sorted(indices)]
    return [dict(record, rank=leaderboard.index(record) + 1) for record in selected]


def _valid_replay(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open(encoding="utf-8") as file:
            replay = json.load(file)
        return isinstance(replay.get("steps"), list) and bool(replay["steps"])
    except (OSError, json.JSONDecodeError):
        return False


def collect(args: argparse.Namespace) -> None:
    output: Path = args.output
    replay_dir = output / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    raw_leaderboard = _retry(
        "leaderboard",
        lambda: api.competition_leaderboard_view(args.competition, page_size=200),
    )
    leaderboard = [_as_dict(item) for item in raw_leaderboard or [] if item is not None]
    selected_teams = _select_teams(leaderboard, args.top_teams, args.mid_teams)
    _atomic_json(output / "leaderboard.json", leaderboard)
    _atomic_json(output / "selected_teams.json", selected_teams)
    print(f"selected {len(selected_teams)}/{len(leaderboard)} leaderboard teams")

    submissions: dict[int, dict[str, Any]] = {}
    for index, team in enumerate(selected_teams, start=1):
        team_id = int(team["teamId"])
        records = _retry(
            f"team-submissions {team_id}",
            lambda team_id=team_id: api.competition_team_submissions(team_id),
        )
        team_submissions = [_as_dict(item) for item in records or [] if item is not None]
        team_submissions.sort(key=lambda item: _score(item, "publicScore"), reverse=True)
        for item in team_submissions[: args.submissions_per_team]:
            submission_id = int(item["id"])
            submissions[submission_id] = {
                **item,
                "submissionId": submission_id,
                "teamId": team_id,
                "teamName": team.get("teamName"),
                "leaderboardRank": team["rank"],
                "leaderboardScore": team.get("score"),
                "source": "leaderboard",
            }
        print(f"team submissions {index}/{len(selected_teams)}")

    own_records = _retry(
        "own submissions",
        lambda: api.competition_submissions(args.competition, page_size=args.own_submissions),
    )
    for item in own_records or []:
        if item is None:
            continue
        record = _as_dict(item)
        submission_id = int(record.get("ref") or record.get("id"))
        submissions.setdefault(
            submission_id,
            {**record, "submissionId": submission_id, "source": "own"},
        )

    submission_records = sorted(submissions.values(), key=lambda item: item["submissionId"])
    _atomic_json(output / "submissions.json", submission_records)
    print(f"selected {len(submission_records)} unique submissions")

    cutoff = datetime.now(UTC) - timedelta(days=args.recent_days)
    episodes: dict[int, dict[str, Any]] = {}
    for index, submission in enumerate(submission_records, start=1):
        submission_id = int(submission["submissionId"])
        records = _retry(
            f"episodes {submission_id}",
            lambda submission_id=submission_id: api.competition_list_episodes(submission_id),
        )
        candidates = [_as_dict(item) for item in records or [] if item is not None]
        candidates = [
            item
            for item in candidates
            if "COMPLETED" in str(item.get("state", ""))
            and (_parse_time(item.get("createTime")) or cutoff) >= cutoff
        ]
        candidates.sort(key=lambda item: item.get("createTime", ""), reverse=True)
        for item in candidates[: args.episodes_per_submission]:
            episode_id = int(item["id"])
            existing = episodes.setdefault(episode_id, {**item, "sources": []})
            source = {
                "submissionId": submission_id,
                "teamId": submission.get("teamId"),
                "teamName": submission.get("teamName"),
                "leaderboardRank": submission.get("leaderboardRank"),
                "leaderboardScore": submission.get("leaderboardScore"),
            }
            if source not in existing["sources"]:
                existing["sources"].append(source)
        _atomic_json(
            output / "episodes.json", sorted(episodes.values(), key=lambda item: item["id"])
        )
        print(f"episodes {index}/{len(submission_records)}; unique={len(episodes)}")

    failures: list[dict[str, Any]] = []
    throttle = 0.0
    streak = 0
    for index, episode in enumerate(
        sorted(episodes.values(), key=lambda item: item["id"]), start=1
    ):
        episode_id = int(episode["id"])
        path = replay_dir / f"episode-{episode_id}-replay.json"
        if _valid_replay(path):
            continue
        if args.dry_run:
            continue
        for attempt in range(1, args.replay_attempts + 1):
            try:
                _with_timeout(
                    args.replay_timeout_seconds,
                    lambda episode_id=episode_id: api.competition_episode_replay(
                        episode_id, path=str(replay_dir), quiet=True
                    ),
                )
                if not _valid_replay(path):
                    raise RuntimeError("downloaded replay is absent or invalid")
                streak += 1
                if throttle and streak >= 100:
                    throttle = max(throttle / 2, 0.0 if throttle < 0.2 else 0.1)
                    streak = 0
                break
            except Exception as error:
                streak = 0
                if attempt == args.replay_attempts:
                    failures.append({"episodeId": episode_id, "error": str(error)})
                    break
                if "429" in str(error):
                    # レート制限は同じエピソードを待って取り直す。失敗扱いにすると
                    # 残り全件を消化しきってしまう。
                    throttle = min(max(throttle * 2, args.replay_throttle_seconds), 5.0)
                    cooldown = min(args.replay_cooldown_seconds * attempt, 600)
                    print(
                        f"replay {episode_id}: rate limited (attempt {attempt}/"
                        f"{args.replay_attempts}); sleeping {cooldown}s, throttle={throttle:.2f}s"
                    )
                    time.sleep(cooldown)
                else:
                    time.sleep(min(2**attempt, 30))
        if throttle:
            time.sleep(throttle)
        if index % 20 == 0 or index == len(episodes):
            print(
                f"replays {index}/{len(episodes)}; failures={len(failures)}; "
                f"throttle={throttle:.2f}s"
            )
            _atomic_json(output / "failures.json", failures)

    collection = {
        "competition": args.competition,
        "collectedAt": datetime.now(UTC).isoformat(),
        "topTeams": args.top_teams,
        "midTeams": args.mid_teams,
        "submissionsPerTeam": args.submissions_per_team,
        "episodesPerSubmission": args.episodes_per_submission,
        "recentDays": args.recent_days,
        "leaderboardTeams": len(leaderboard),
        "selectedTeams": len(selected_teams),
        "submissions": len(submission_records),
        "episodes": len(episodes),
        "validReplays": sum(
            _valid_replay(replay_dir / f"episode-{episode_id}-replay.json")
            for episode_id in episodes
        ),
        "failures": len(failures),
    }
    _atomic_json(output / "collection.json", collection)
    print(json.dumps(collection, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-teams", type=int, default=30)
    parser.add_argument("--mid-teams", type=int, default=20)
    parser.add_argument("--submissions-per-team", type=int, default=2)
    parser.add_argument("--own-submissions", type=int, default=100)
    parser.add_argument("--episodes-per-submission", type=int, default=20)
    parser.add_argument("--recent-days", type=int, default=14)
    parser.add_argument("--replay-timeout-seconds", type=int, default=45)
    parser.add_argument("--replay-attempts", type=int, default=8)
    parser.add_argument("--replay-cooldown-seconds", type=int, default=60)
    parser.add_argument("--replay-throttle-seconds", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
