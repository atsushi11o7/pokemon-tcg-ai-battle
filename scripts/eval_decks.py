#!/usr/bin/env python3
"""同一のネットワークで複数デッキを操縦し、共通のメタ相手集合に対する勝率を比較する。

提出枠は1日5件しかないので、どのデッキを出すかを実測で決めるための道具。
全デッキに同じ相手デッキ列を割り当てる(対応のある比較)ことで、相手の引きによる
分散を消し、少ない試合数でもデッキ間の差を見やすくする。

Usage:
    uv run python scripts/eval_decks.py \
        --checkpoint outputs/runs/bc_toplayers/checkpoints/final.pt \
        --games 200 --workers 12 --top-archetypes 8
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))

SNAPSHOT = ROOT / "data" / "meta" / "derived" / "sampling_snapshot.json"


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    if games == 0:
        return 0.0, 0.0
    p = wins / games
    denominator = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denominator
    margin = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denominator
    return centre - margin, centre + margin


def candidate_decks(snapshot: dict, top_archetypes: int, extra: list[Path]) -> list[dict]:
    """メタ上位アーキタイプの代表デッキと、明示指定のCSVを候補にする。

    代表は「そのアーキタイプで最もエピソード数が多い60枚デッキ」。上位帯が実際に
    最も多く握った構築で、模倣学習の教師データも厚い。
    """
    by_archetype: dict[int, dict] = {}
    weight: dict[int, float] = {}
    for record in snapshot["decks"]:
        if len(record["cards"]) != 60:
            continue
        archetype = record["archetypeId"]
        weight[archetype] = weight.get(archetype, 0.0) + record["weights"]["meta"]
        best = by_archetype.get(archetype)
        if best is None or record["episodeCount"] > best["episodeCount"]:
            by_archetype[archetype] = record

    ranked = sorted(weight.items(), key=lambda kv: -kv[1])[:top_archetypes]
    candidates = []
    for archetype, meta_weight in ranked:
        record = by_archetype[archetype]
        candidates.append(
            {
                "name": f"archetype{archetype}",
                "cards": record["cards"],
                "meta_weight": meta_weight,
                "episodes": record["episodeCount"],
            }
        )
    for path in extra:
        cards = [int(x) for x in path.read_text().split() if x.strip()]
        if len(cards) != 60:
            raise ValueError(f"{path} must contain exactly 60 cards, got {len(cards)}")
        candidates.append({"name": path.stem, "cards": cards, "meta_weight": 0.0, "episodes": 0})
    return candidates


_NETWORK = None


def _init_worker(checkpoint: str) -> None:
    global _NETWORK
    import torch

    from training.common.network import load_policy_value_net

    torch.set_num_threads(1)  # ワーカー数ぶん並ぶので、1プロセス1スレッドに固定する
    _NETWORK = load_policy_value_net(checkpoint)
    _NETWORK.eval()


def _play(task: tuple[str, list[int], list[int], int, bool]) -> tuple[str, float]:
    name, our_deck, opponent_deck, seed, we_go_first = task
    import random as rnd

    from evaluation.match_runner import play_one_match
    from training.ppo.selfplay import make_ppo_eval_agent

    rnd.seed(seed)
    ours = make_ppo_eval_agent(_NETWORK, our_deck)
    theirs = make_ppo_eval_agent(_NETWORK, opponent_deck)
    try:
        if we_go_first:
            reward, _ = play_one_match(ours, theirs)
        else:
            _, reward = play_one_match(theirs, ours)
    except Exception:
        return name, float("nan")
    return name, reward


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200, help="1デッキあたりの試合数(偶数)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--top-archetypes", type=int, default=8)
    parser.add_argument("--deck", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.games % 2:
        raise ValueError("--games must be even (先手/後手を半々にするため)")

    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    decks = candidate_decks(snapshot, args.top_archetypes, args.deck)

    # 相手は meta 重みで抽選し、全候補デッキで同じ列を使い回す(対応のある比較)。
    pool = [r for r in snapshot["decks"] if len(r["cards"]) == 60]
    weights = [r["weights"]["meta"] for r in pool]
    rng = random.Random(args.seed)
    opponents = [rng.choices(pool, weights=weights)[0]["cards"] for _ in range(args.games // 2)]

    tasks = [
        (deck["name"], deck["cards"], opponent, args.seed + i, first)
        for deck in decks
        for i, opponent in enumerate(opponents)
        for first in (True, False)
    ]
    rng.shuffle(tasks)  # デッキごとに難しい相手が固まらないよう混ぜる
    print(f"候補デッキ {len(decks)}件 × {args.games}試合 = {len(tasks)}試合", flush=True)

    tally: dict[str, list[int]] = {d["name"]: [0, 0, 0] for d in decks}  # 勝/引分/負
    failed = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        args.workers, initializer=_init_worker, initargs=(str(args.checkpoint),)
    ) as pool_:
        for done, (name, reward) in enumerate(pool_.imap_unordered(_play, tasks, chunksize=1), 1):
            if reward != reward:
                failed += 1
            elif reward > 0:
                tally[name][0] += 1
            elif reward == 0:
                tally[name][1] += 1
            else:
                tally[name][2] += 1
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)} 試合完了 (失敗{failed})", flush=True)

    print(
        f"\n{'デッキ':16s} {'勝率':>7s} {'95%CI':>16s} {'勝-分-負':>12s} {'meta':>7s} {'教師':>6s}"
    )
    rows = []
    for deck in decks:
        wins, draws, losses = tally[deck["name"]]
        games = wins + draws + losses
        rate = (wins + 0.5 * draws) / games if games else 0.0
        low, high = wilson_interval(wins, games)
        rows.append(
            {
                **deck,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate": rate,
                "ci95": [low, high],
            }
        )
    for row in sorted(rows, key=lambda r: -r["win_rate"]):
        print(
            f"{row['name']:16s} {row['win_rate'] * 100:>6.1f}% "
            f"[{row['ci95'][0] * 100:>5.1f},{row['ci95'][1] * 100:>5.1f}] "
            f"{row['wins']:>4d}-{row['draws']:>3d}-{row['losses']:>3d} "
            f"{row['meta_weight'] * 100:>6.1f}% {row['episodes']:>6d}"
        )
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
