"""PPO checkpointと提出候補デッキを固定条件で比較する。"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.match_runner import evaluate_fixed_matchups  # noqa: E402
from training.common.evaluation_plan import build_fixed_matchups  # noqa: E402
from training.common.meta_deck_pool import load_weighted_deck_pool  # noqa: E402
from training.common.model_config import (  # noqa: E402
    D_FEEDFORWARD,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS_DECODER,
    NUM_LAYERS_ENCODER,
)
from training.common.network import PolicyValueNet  # noqa: E402
from training.ppo.selfplay import make_ppo_eval_agent  # noqa: E402

CANDIDATE_CARDS = {
    "crustle": 345,
    "marnies_grimmsnarl_ex": 648,
    "mega_lucario_ex": 678,
    "mega_lopunny_ex": 849,
}


def _load_network(checkpoint: Path) -> PolicyValueNet:
    network = PolicyValueNet(
        D_MODEL,
        NUM_HEADS,
        D_FEEDFORWARD,
        NUM_LAYERS_ENCODER,
        NUM_LAYERS_DECODER,
    )
    network.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    network.eval()
    return network


def _factory(network: PolicyValueNet):
    return lambda deck: make_ppo_eval_agent(network, deck)


def _wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games == 0:
        return 0.0, 0.0
    probability = wins / games
    denominator = 1 + z * z / games
    centre = (probability + z * z / (2 * games)) / denominator
    margin = (
        z
        * math.sqrt(probability * (1 - probability) / games + z * z / (4 * games * games))
        / denominator
    )
    return centre - margin, centre + margin


def _with_interval(result: dict) -> dict:
    low, high = _wilson_interval(result["wins"], result["games"])
    return {**result, "win_rate_ci95": [low, high]}


def _best_candidate_records(snapshot_path: Path) -> dict[str, dict]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    selected: dict[str, dict] = {}
    for name, card_id in CANDIDATE_CARDS.items():
        records = [record for record in snapshot["decks"] if card_id in record["cards"]]
        if not records:
            raise RuntimeError(f"no snapshot deck contains {name} (card ID {card_id})")
        selected[name] = max(
            records,
            key=lambda record: (
                float(record["weights"].get("meta", 0.0)),
                int(record.get("uniqueTeamDays", 0)),
                int(record.get("episodeCount", 0)),
            ),
        )
    return selected


def _file_candidate_record(name: str, path: Path) -> dict:
    cards = [
        int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(cards) != 60:
        raise ValueError(f"candidate deck {name!r} must contain exactly 60 cards: {path}")
    return {
        "archetypeId": None,
        "cards": cards,
        "deckHash": f"file:{path}",
        "episodeCount": 0,
        "uniqueTeamDays": 0,
        "weights": {"meta": 0.0},
    }


def _snapshot_hash_candidate_record(snapshot_path: Path, name: str, deck_hash: str) -> dict:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    for record in snapshot["decks"]:
        if record["deckHash"] == deck_hash:
            return record
    raise ValueError(f"unknown snapshot deck hash for {name!r}: {deck_hash}")


def _compare_checkpoints(
    candidate: PolicyValueNet,
    baselines: list[tuple[str, PolicyValueNet]],
    deck_pool,
    games: int,
    seed: int,
) -> dict[str, dict]:
    if games == 0:
        return {}
    matchups = build_fixed_matchups("generalist", [], deck_pool, games, seed)
    results = {}
    for name, baseline in baselines:
        result = evaluate_fixed_matchups(
            _factory(candidate),
            _factory(baseline),
            matchups,
            seed=seed,
        )
        results[name] = _with_interval(result)
        print(f"checkpoint final vs {name}: {results[name]}", flush=True)
    return results


def _compare_decks(
    network: PolicyValueNet,
    candidate_records: dict[str, dict],
    deck_pool,
    games: int,
    seed: int,
) -> dict[str, dict]:
    if games == 0:
        return {}
    if games <= 0 or games % 2:
        raise ValueError("deck comparison games must be a positive multiple of 2")

    random_state = random.getstate()
    try:
        random.seed(seed)
        opponents = [deck_pool.sample("opponent") for _ in range(games // 2)]
    finally:
        random.setstate(random_state)

    results = {}
    for name, record in candidate_records.items():
        matchups = [(record["cards"], opponent) for opponent in opponents]
        result = evaluate_fixed_matchups(
            _factory(network),
            _factory(network),
            matchups,
            seed=seed,
            swap_decks=False,
        )
        results[name] = {
            **_with_interval(result),
            "deck_hash": record["deckHash"],
            "archetype_id": record["archetypeId"],
            "episode_count": record["episodeCount"],
            "unique_team_days": record["uniqueTeamDays"],
            "meta_weight": record["weights"]["meta"],
            "cards": record["cards"],
        }
        print(f"deck {name} vs fixed meta opponents: {results[name]}", flush=True)
    return results


def _compare_candidate_pairs(
    network: PolicyValueNet,
    candidate_records: dict[str, dict],
    games: int,
    seed: int,
) -> dict[str, dict]:
    if games == 0:
        return {}
    if games <= 0 or games % 2:
        raise ValueError("pairwise comparison games must be a positive multiple of 2")

    results = {}
    for pair_index, ((name_a, record_a), (name_b, record_b)) in enumerate(
        itertools.combinations(candidate_records.items(), 2)
    ):
        pair_seed = seed + pair_index * 100_000
        matchups = [(record_a["cards"], record_b["cards"])] * (games // 2)
        result = evaluate_fixed_matchups(
            _factory(network),
            _factory(network),
            matchups,
            seed=pair_seed,
            swap_decks=False,
        )
        key = f"{name_a}_vs_{name_b}"
        results[key] = {
            **_with_interval(result),
            "deck_a": name_a,
            "deck_b": name_b,
        }
        print(f"pairwise {name_a} vs {name_b}: {results[key]}", flush=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--baseline", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "data/meta/derived/sampling_snapshot.json",
    )
    parser.add_argument("--checkpoint-games", type=int, default=200)
    parser.add_argument("--deck-games", type=int, default=200)
    parser.add_argument("--pairwise-games", type=int, default=0)
    parser.add_argument("--deck-file", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--snapshot-hash", action="append", default=[], metavar="NAME=HASH")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--seed", type=int, default=910_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.checkpoint_games < 0 or args.checkpoint_games % 4:
        raise ValueError("checkpoint comparison games must be zero or a positive multiple of 4")
    torch.set_num_threads(1)

    deck_pool = load_weighted_deck_pool(args.snapshot)
    final_network = _load_network(args.final)
    baselines = []
    for value in args.baseline:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--baseline must use NAME=PATH")
        baselines.append((name, _load_network(Path(raw_path))))

    candidate_records = _best_candidate_records(args.snapshot)
    for value in args.deck_file:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError("--deck-file must use NAME=PATH")
        if name in candidate_records:
            raise ValueError(f"duplicate candidate deck name: {name}")
        candidate_records[name] = _file_candidate_record(name, Path(raw_path))
    for value in args.snapshot_hash:
        name, separator, deck_hash = value.partition("=")
        if not separator or not name or not deck_hash:
            raise ValueError("--snapshot-hash must use NAME=HASH")
        if name in candidate_records:
            raise ValueError(f"duplicate candidate deck name: {name}")
        candidate_records[name] = _snapshot_hash_candidate_record(args.snapshot, name, deck_hash)
    if args.only:
        unknown = set(args.only) - set(candidate_records)
        if unknown:
            raise ValueError(f"unknown --only candidate(s): {', '.join(sorted(unknown))}")
        candidate_records = {name: candidate_records[name] for name in args.only}

    report = {
        "final_checkpoint": str(args.final),
        "snapshot": str(args.snapshot),
        "seed": args.seed,
        "checkpoint_comparisons": _compare_checkpoints(
            final_network,
            baselines,
            deck_pool,
            args.checkpoint_games,
            args.seed,
        ),
        "deck_comparisons": _compare_decks(
            final_network,
            candidate_records,
            deck_pool,
            args.deck_games,
            args.seed + 100_000,
        ),
        "pairwise_comparisons": _compare_candidate_pairs(
            final_network,
            candidate_records,
            args.pairwise_games,
            args.seed + 200_000,
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
