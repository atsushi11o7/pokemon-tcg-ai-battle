#!/usr/bin/env python3
"""探索の根で訪問がどれだけ集中しているかを測る。

`run_mcts`は最終手を「訪問回数が最大の子」で選ぶ。未展開の子のQを0.0で初期化して
いるため(`_select_child`)、価値ヘッドが多くの子へ負の評価を返すようになると、
未展開の子(Q=0)が常に既訪問の子(Q<0)より高く見える。すると各シミュレーションは
新しい子を1つ開くだけで深まらず、全ての子が訪問1回で並ぶ。`max`は同点で最初の
要素を返すので、着手は列挙順の先頭へ潰れ、方策も価値も効かなくなる。

その状態になっているかは、根の訪問分布の集中度で分かる。全児が1回なら
max(policy)は1/n_actionsに一致する。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))

SNAPSHOT = ROOT / "data" / "meta" / "derived" / "sampling_snapshot.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--search-count", type=int, default=150)
    parser.add_argument("--determinizations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--leaf-value",
        type=float,
        default=None,
        help="葉の評価を価値ヘッドではなくこの定数にする(#34の挙動の再現用)",
    )
    args = parser.parse_args()

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
        try:
            while True:
                obs = to_observation_class(obs_dict)
                if obs.current.result in (0, 1, 2):
                    break
                seat = obs.current.yourIndex
                deck = our_deck if seat == 0 else opponent
                actions = enumerate_actions(obs.select)
                encoder_sv = get_encoder_input(obs, deck)
                decoder_sv = get_decoder_input(obs, actions)
                with torch.inference_mode():
                    value, scores = network(*encoder_sv.to_tensors(), *decoder_sv.to_tensors())
                greedy = actions[int(torch.argmax(scores[0]))]
                select = greedy
                if seat == 0 and len(actions) >= 2:
                    picked, policy, root_value, _a = run_determinized_mcts(
                        network,
                        obs,
                        deck,
                        pool,
                        args.search_count,
                        num_determinizations=args.determinizations,
                        leaf_value=args.leaf_value,
                    )
                    top = max(policy)
                    rows.append(
                        {
                            "n_actions": len(actions),
                            "max_visit_share": top,
                            # 全児が1回ずつなら max は 1/n に一致する。
                            "flat": abs(top - 1.0 / len(actions)) < 1e-6,
                            "support": sum(1 for p in policy if p > 0),
                            "changed": picked != greedy,
                            "leaf_value": float(value.squeeze().item()),
                            "root_value": root_value,
                        }
                    )
                    select = picked
                obs_dict = battle_select(select)
        finally:
            battle_finish()
        print(f"game {game + 1}/{args.games}: {len(rows)} decisions so far", flush=True)

    if not rows:
        print("no decisions recorded")
        return 1

    n = len(rows)
    multi = [r for r in rows if r["n_actions"] >= 3]
    print(f"\n{args.checkpoint}")
    print(f"  探索した手番            {n}")
    print(f"  平均 選択肢数           {sum(r['n_actions'] for r in rows) / n:.1f}")
    print(f"  平均 最大訪問シェア     {sum(r['max_visit_share'] for r in rows) / n:.3f}")
    print(f"  訪問が完全に一様        {sum(r['flat'] for r in rows) / n:.1%}")
    if multi:
        m = len(multi)
        print(f"  (選択肢3以上に限る {m}手)")
        print(f"     最大訪問シェア     {sum(r['max_visit_share'] for r in multi) / m:.3f}")
        print(f"     完全に一様         {sum(r['flat'] for r in multi) / m:.1%}")
    print(f"  貪欲方策と違う手        {sum(r['changed'] for r in rows) / n:.1%}")
    print(f"  平均 葉のvalue          {sum(r['leaf_value'] for r in rows) / n:+.3f}")
    print(f"  平均 根のvalue          {sum(r['root_value'] for r in rows) / n:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
