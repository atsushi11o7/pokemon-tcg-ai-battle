#!/usr/bin/env python3
"""収集済みreplayからdeck registryと学習用sampling snapshotを構築する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "meta" / "raw"
DEFAULT_OUTPUT = ROOT / "data" / "meta" / "derived" / "sampling_snapshot.json"


def canonical_deck(deck: list[int]) -> tuple[int, ...]:
    if len(deck) != 60 or not all(isinstance(card_id, int) for card_id in deck):
        raise ValueError("deck must contain exactly 60 integer card IDs")
    return tuple(sorted(deck))


def deck_hash(deck: tuple[int, ...]) -> str:
    payload = ",".join(map(str, deck)).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def extract_decks(replay: dict[str, Any]) -> list[list[int]] | None:
    """visualizerの初期化actionから両者の完全な60枚デッキを返す。"""
    for step in replay.get("steps") or []:
        for agent_step in step:
            for visual in agent_step.get("visualize") or []:
                action = visual.get("action")
                if (
                    isinstance(action, list)
                    and len(action) == 2
                    and all(
                        isinstance(deck, list)
                        and len(deck) == 60
                        and all(isinstance(card_id, int) for card_id in deck)
                        for deck in action
                    )
                ):
                    return action
    return None


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _normalise(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0:
        uniform = 1.0 / len(weights) if weights else 0.0
        return {key: uniform for key in weights}
    return {key: max(value, 0.0) / total for key, value in weights.items()}


def _idf_weights(decks: list[tuple[int, ...]]) -> dict[int, float]:
    document_frequency: Counter[int] = Counter()
    for deck in decks:
        document_frequency.update(set(deck))
    count = len(decks)
    # 多くのデッキに入る汎用札・基本エネルギーを弱め、固有カードを強くする。
    return {
        card_id: math.log((count + 1) / (frequency + 1)) + 0.1
        for card_id, frequency in document_frequency.items()
    }


def weighted_jaccard(
    first: tuple[int, ...], second: tuple[int, ...], weights: dict[int, float]
) -> float:
    left = Counter(first)
    right = Counter(second)
    cards = left.keys() | right.keys()
    intersection = sum(weights[card] * min(left[card], right[card]) for card in cards)
    union = sum(weights[card] * max(left[card], right[card]) for card in cards)
    return intersection / union if union else 1.0


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def cluster_decks(
    decks: list[tuple[int, ...]], threshold: float
) -> tuple[list[int], dict[int, float]]:
    weights = _idf_weights(decks)
    union_find = _UnionFind(len(decks))
    for left in range(len(decks)):
        for right in range(left + 1, len(decks)):
            if weighted_jaccard(decks[left], decks[right], weights) >= threshold:
                union_find.union(left, right)
    root_to_cluster: dict[int, int] = {}
    labels: list[int] = []
    for index in range(len(decks)):
        root = union_find.find(index)
        labels.append(root_to_cluster.setdefault(root, len(root_to_cluster)))
    return labels, weights


def build_snapshot(raw_dir: Path, output: Path, threshold: float, half_life_days: float) -> dict:
    with (raw_dir / "episodes.json").open(encoding="utf-8") as file:
        episode_metadata = {int(item["id"]): item for item in json.load(file)}

    replay_paths = [
        path
        for path in sorted((raw_dir / "replays").glob("episode-*-replay.json"))
        if int(path.name.split("-")[1]) in episode_metadata
    ]

    occurrences: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for path in replay_paths:
        try:
            with path.open(encoding="utf-8") as file:
                replay = json.load(file)
            decks = extract_decks(replay)
            if decks is None:
                skipped += 1
                continue
            episode_id = int(replay.get("info", {}).get("EpisodeId") or path.name.split("-")[1])
            metadata = episode_metadata.get(episode_id, {})
            created_at = metadata.get("createTime")
            team_names = replay.get("info", {}).get("TeamNames") or [None, None]
            rewards = replay.get("rewards") or [None, None]
            for seat, deck in enumerate(decks):
                occurrences[canonical_deck(deck)].append(
                    {
                        "episodeId": episode_id,
                        "createdAt": created_at,
                        "teamName": team_names[seat] if seat < len(team_names) else None,
                        "reward": rewards[seat] if seat < len(rewards) else None,
                    }
                )
        except (OSError, ValueError, json.JSONDecodeError, IndexError):
            skipped += 1

    if not occurrences:
        raise RuntimeError(f"no valid 60-card decks found under {raw_dir / 'replays'}")

    canonical_decks = sorted(occurrences, key=deck_hash)
    labels, idf = cluster_decks(canonical_decks, threshold)
    now = datetime.now(UTC)
    raw_meta: dict[str, float] = {}
    raw_exploration: dict[str, float] = {}
    records: list[dict[str, Any]] = []

    for deck, label in zip(canonical_decks, labels, strict=True):
        hash_ = deck_hash(deck)
        seen_team_days: dict[tuple[str, str], float] = {}
        teams: set[str] = set()
        last_seen: datetime | None = None
        for occurrence in occurrences[deck]:
            timestamp = _parse_time(occurrence["createdAt"])
            if timestamp is None:
                timestamp = now
            team = occurrence["teamName"] or f"unknown-{occurrence['episodeId']}"
            teams.add(team)
            day_key = (team, timestamp.date().isoformat())
            age_days = max((now - timestamp).total_seconds() / 86400, 0.0)
            recency = 0.5 ** (age_days / half_life_days)
            seen_team_days[day_key] = max(seen_team_days.get(day_key, 0.0), recency)
            last_seen = max(last_seen, timestamp) if last_seen else timestamp

        team_day_weight = sum(seen_team_days.values())
        raw_meta[hash_] = team_day_weight
        raw_exploration[hash_] = 1.0 / math.sqrt(1.0 + len(seen_team_days))
        records.append(
            {
                "deckHash": hash_,
                "archetypeId": label,
                "cards": list(deck),
                "episodeCount": len(occurrences[deck]),
                "uniqueTeams": len(teams),
                "uniqueTeamDays": len(seen_team_days),
                "lastSeen": last_seen.isoformat() if last_seen else None,
            }
        )

    meta_weights = _normalise(raw_meta)
    exploration_weights = _normalise(raw_exploration)
    archetype_members: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        archetype_members[record["archetypeId"]].append(record)

    raw_coverage: dict[str, float] = {}
    for members in archetype_members.values():
        variant_weights = {
            member["deckHash"]: math.sqrt(1.0 + member["uniqueTeamDays"]) for member in members
        }
        variants = _normalise(variant_weights)
        for hash_, weight in variants.items():
            raw_coverage[hash_] = weight / len(archetype_members)
    coverage_weights = _normalise(raw_coverage)

    for record in records:
        hash_ = record["deckHash"]
        record["weights"] = {
            "meta": meta_weights[hash_],
            "coverage": coverage_weights[hash_],
            "exploration": exploration_weights[hash_],
            "hard": 0.0,
        }

    archetypes = []
    for label, members in sorted(archetype_members.items()):
        representative = max(
            members,
            key=lambda item: (item["uniqueTeamDays"], item["episodeCount"], item["deckHash"]),
        )
        archetypes.append(
            {
                "archetypeId": label,
                "representativeDeckHash": representative["deckHash"],
                "variantCount": len(members),
                "uniqueTeamDays": sum(item["uniqueTeamDays"] for item in members),
            }
        )

    snapshot = {
        "schemaVersion": 1,
        "builtAt": now.isoformat(),
        "source": str(raw_dir.relative_to(ROOT) if raw_dir.is_relative_to(ROOT) else raw_dir),
        "clustering": {
            "method": "connected-components-weighted-jaccard-idf",
            "threshold": threshold,
            "idfCardCount": len(idf),
        },
        "recencyHalfLifeDays": half_life_days,
        "opponentMixture": {"meta": 0.55, "coverage": 0.35, "exploration": 0.10, "hard": 0.0},
        "learnerMixture": {"meta": 0.0, "coverage": 0.90, "exploration": 0.10, "hard": 0.0},
        "replayCount": len(replay_paths),
        "skippedReplayCount": skipped,
        "deckCount": len(records),
        "archetypeCount": len(archetypes),
        "archetypes": archetypes,
        "decks": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, output)
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archetype-threshold", type=float, default=0.60)
    parser.add_argument("--half-life-days", type=float, default=7.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = build_snapshot(
        arguments.raw,
        arguments.output,
        arguments.archetype_threshold,
        arguments.half_life_days,
    )
    print(
        f"built {arguments.output}: {result['deckCount']} decks, "
        f"{result['archetypeCount']} archetypes from {result['replayCount']} replays"
    )
