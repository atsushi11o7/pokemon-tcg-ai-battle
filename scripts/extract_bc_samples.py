#!/usr/bin/env python3
"""日次エピソードzipから模倣学習用サンプルを抽出し、シャードへ書き出す。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))

from training.bc.dataset import extract_episode, iter_replays  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zips", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--max-episodes", type=int, default=0, help="0なら全件")
    parser.add_argument(
        "--winner-only", action="store_true", help="敗者の局面を集めない(価値の教師が+1に偏る)"
    )
    parser.add_argument("--imitate-loser", action="store_true", help="敗者の手もone-hotで模倣する")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    buffer: list = []
    shard_index = 0

    for replay in iter_replays(args.zips, stats):
        if args.max_episodes and stats["episodes"] >= args.max_episodes:
            break
        buffer.extend(
            extract_episode(
                replay,
                stats,
                winner_only=args.winner_only,
                imitate_loser=args.imitate_loser,
            )
        )
        while len(buffer) >= args.shard_size:
            torch.save(buffer[: args.shard_size], args.output / f"shard_{shard_index:05d}.pt")
            buffer = buffer[args.shard_size :]
            shard_index += 1
            print(
                f"shard {shard_index}: episodes={stats['episodes']} samples={stats['samples']}",
                flush=True,
            )

    if buffer:
        torch.save(buffer, args.output / f"shard_{shard_index:05d}.pt")
        shard_index += 1

    summary = {"shards": shard_index, **dict(stats)}
    (args.output / "extract_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
