#!/usr/bin/env python3
"""チェックポイントを「自分のデッキ側/相手側」「勝ち/負け」に分けて評価する。

`bc.train.evaluate`は方策の指標を勝者の手だけで測る(`mask_loser_targets(...,0.0)`が
固定で入っている)。`--policy-on-deck`で負けた試合の指し手を学べるようにした今、
その効果は既定の指標に一切現れない。ここでは内訳を出す。

自分のデッキ側かどうかは`policy_target`が非ゼロかで判別する。抽出時に席を特定して
one-hotを立てているので、これがそのまま「自分のデッキを握っていた席」の印になる。
ミラー戦では両席が該当し、どちらも自分側として数える(実際どちらも同じ構築なので正しい)。

価値ヘッドはMCTSの葉で自分の手番・相手の手番の両方に使われる(search.create_nodeは
`obs.current.yourIndex`視点で評価する)ので、相手側の指標も併せて出す。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torch.utils.data import DataLoader  # noqa: E402

from training.bc.dataset import load_shard  # noqa: E402
from training.bc.train import (  # noqa: E402
    ListDataset,
    collate_samples,
    masked_policy_loss,
)
from training.common.network import load_policy_value_net  # noqa: E402


def accumulate(bucket: dict, scores, mask, targets, values, labels, rows) -> None:
    """`rows`(bool tensor)が立つ行だけを集計する。"""
    n = int(rows.sum().item())
    if not n:
        return
    bucket["n"] += n
    v = values.squeeze(-1)[rows]
    lab = labels[rows]
    bucket["value_mse"] += float(functional.mse_loss(v, lab, reduction="sum").item())
    # 符号正答率。価値ヘッドが「勝っている/負けている」を当てられているか。
    # MCTSは葉の評価をそのまま使うので、絶対値より符号が効く。
    bucket["sign_hit"] += int((torch.sign(v) == torch.sign(lab)).sum().item())
    bucket["confident"] += int((v.abs() > 0.9).sum().item())

    supervised = rows & (targets.sum(dim=-1) > 0)
    m = int(supervised.sum().item())
    if not m:
        return
    bucket["policy_n"] += m
    pred = scores.masked_fill(~mask, float("-inf")).argmax(dim=-1)[supervised]
    exp = targets[supervised].argmax(dim=-1)
    bucket["policy_hit"] += int((pred == exp).sum().item())
    sub = (scores[supervised], mask[supervised], targets[supervised])
    bucket["policy_loss"] += float(masked_policy_loss(*sub).item()) * m


def new_bucket() -> dict:
    return dict(
        n=0, value_mse=0.0, sign_hit=0, confident=0, policy_n=0, policy_hit=0, policy_loss=0.0
    )


def report(name: str, b: dict) -> dict:
    if not b["n"]:
        return {}
    out = {
        "samples": b["n"],
        "value_mse": b["value_mse"] / b["n"],
        "value_sign_acc": b["sign_hit"] / b["n"],
        "confident_frac": b["confident"] / b["n"],
    }
    if b["policy_n"]:
        out |= {
            "policy_samples": b["policy_n"],
            "accuracy": b["policy_hit"] / b["policy_n"],
            "policy_loss": b["policy_loss"] / b["policy_n"],
        }
    print(
        f"  {name:18s} "
        + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items())
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--day", default=None, help="この日のシャードだけを使う")
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    paths = sorted(args.shard_dir.glob("shard_*.pt"))
    if args.day:
        paths = [p for p in paths if p.name.split("_")[1] == args.day]
    paths = paths[: args.shards]
    if not paths:
        raise SystemExit(f"シャードが見つからない: {args.shard_dir} day={args.day}")

    samples: list = []
    for p in paths:
        samples.extend(load_shard(p))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = load_policy_value_net(args.checkpoint)
    network.to(device).eval()

    ours_win, ours_loss, opp = new_bucket(), new_bucket(), new_bucket()
    loader = DataLoader(
        ListDataset(samples), batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples
    )
    with torch.no_grad():
        for batch in loader:
            idx_e, val_e, off_e, idx_d, val_d, off_d, mask, targets, labels = [
                t.to(device) for t in batch
            ]
            values, scores = network(idx_e, val_e, off_e, idx_d, val_d, off_d, mask)
            # 方策の教師を持つ行=自分のデッキを握っていた席。
            mine = targets.sum(dim=-1) > 0
            accumulate(ours_win, scores, mask, targets, values, labels, mine & (labels > 0))
            accumulate(ours_loss, scores, mask, targets, values, labels, mine & (labels < 0))
            accumulate(opp, scores, mask, targets, values, labels, ~mine)

    print(f"{args.checkpoint}  ({len(samples)}サンプル / {len(paths)}シャード)")
    result = {
        "checkpoint": str(args.checkpoint),
        "自分・勝ち": report("自分・勝ち", ours_win),
        "自分・負け": report("自分・負け", ours_loss),
        "相手側": report("相手側", opp),
    }
    total_n = ours_win["n"] + ours_loss["n"] + opp["n"]
    total_hit = ours_win["sign_hit"] + ours_loss["sign_hit"] + opp["sign_hit"]
    print(f"  {'全体':18s} value_sign_acc={total_hit / total_n:.4f}  samples={total_n}")
    result["全体_value_sign_acc"] = total_hit / total_n
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
