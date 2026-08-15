#!/usr/bin/env python3
"""探索が着手を変えた局面を記録し、どこに予算を割くべきかを測る。

探索が貪欲方策と同じ手を選んだなら、そのシミュレーションは捨てたのと同じ。
違う手を選んだ局面こそ探索が仕事をした場所なので、その分布を見れば
「序盤か終盤か」「方策が迷っているときか」のどちらで予算を使うべきか決まる。

推測で段階を切ると外す。実測してから配分を変える。

Usage:
    uv run python scripts/measure_search_value.py --checkpoint <ckpt> --deck <csv> --games 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))

SNAPSHOT = ROOT / "data" / "meta" / "derived" / "sampling_snapshot.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--search-count", type=int, default=150)
    parser.add_argument("--determinizations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import random

    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    from training.common.deck import parse_deck_csv
    from training.common.network import load_policy_value_net
    from training.common.sparse_features import get_decoder_input, get_encoder_input
    from training.mcts.search import enumerate_actions
    from training.mcts.selfplay import run_determinized_mcts

    torch.set_num_threads(4)
    network = load_policy_value_net(args.checkpoint)
    network.eval()
    our_deck = parse_deck_csv(args.deck)
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    pool = [r["cards"] for r in snapshot["decks"] if len(r["cards"]) == 60]
    weights = [r["weights"]["meta"] for r in snapshot["decks"] if len(r["cards"]) == 60]

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for game in range(args.games):
        opponent = rng.choices(pool, weights=weights)[0]
        obs_dict, start = battle_start(our_deck, opponent)
        if start.errorPlayer in (0, 1):
            battle_finish()
            continue
        move = 0
        try:
            while True:
                obs = to_observation_class(obs_dict)
                if obs.current.result in (0, 1, 2):
                    break
                seat = obs.current.yourIndex
                deck = our_deck if seat == 0 else opponent
                actions = enumerate_actions(obs.select)
                if seat == 0 and len(actions) >= 2:
                    encoder_sv = get_encoder_input(obs, deck)
                    decoder_sv = get_decoder_input(obs, actions)
                    with torch.inference_mode():
                        _v, scores = network(*encoder_sv.to_tensors(), *decoder_sv.to_tensors())
                    ranked = torch.topk(scores[0], min(2, scores.shape[1]))
                    greedy = actions[int(ranked.indices[0])]
                    margin = float(ranked.values[0] - ranked.values[1])
                    picked, _p, _v2, _a = run_determinized_mcts(
                        network,
                        obs,
                        deck,
                        pool,
                        args.search_count,
                        num_determinizations=args.determinizations,
                    )
                    rows.append(
                        {
                            "game": game,
                            "move": move,
                            "n_actions": len(actions),
                            "margin": margin,
                            "changed": picked != greedy,
                        }
                    )
                    select = picked
                else:
                    encoder_sv = get_encoder_input(obs, deck)
                    decoder_sv = get_decoder_input(obs, actions)
                    with torch.inference_mode():
                        _v, scores = network(*encoder_sv.to_tensors(), *decoder_sv.to_tensors())
                    select = actions[int(torch.argmax(scores[0]))]
                obs_dict = battle_select(select)
                move += 1
        finally:
            battle_finish()
        print(f"game {game + 1}/{args.games}: {move} moves", flush=True)

    if not rows:
        print("no decisions recorded")
        return 1
    if args.output:
        args.output.write_text(json.dumps(rows), encoding="utf-8")

    changed = sum(r["changed"] for r in rows)
    print(
        f"\n探索した手番 {len(rows)}  着手が変わった {changed} ({changed / len(rows) * 100:.1f}%)\n"
    )

    # 局面の進行度で分ける。1試合の長さが違うので、自分の手番数で正規化する。
    per_game = {}
    for r in rows:
        per_game.setdefault(r["game"], []).append(r)
    print(f"{'進行度':>10s} {'手番数':>7s} {'着手が変わった':>12s}")
    buckets: dict[int, list[bool]] = {}
    for moves in per_game.values():
        for i, r in enumerate(moves):
            buckets.setdefault(int(i / max(1, len(moves)) * 5), []).append(r["changed"])
    for b in sorted(buckets):
        v = buckets[b]
        print(f"{b * 20:>4d}-{b * 20 + 20:>3d}% {len(v):>7d} {sum(v) / len(v) * 100:>11.1f}%")

    print(f"\n{'方策の差(top1-top2)':>20s} {'手番数':>7s} {'着手が変わった':>12s}")
    edges = [(0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 1e9)]
    for lo, hi in edges:
        v = [r["changed"] for r in rows if lo <= r["margin"] < hi]
        if v:
            label = f"{lo:.1f}-{hi:.1f}" if hi < 1e9 else f"{lo:.1f}+"
            print(f"{label:>20s} {len(v):>7d} {sum(v) / len(v) * 100:>11.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
