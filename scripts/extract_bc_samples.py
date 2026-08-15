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

from training.bc.dataset import (  # noqa: E402
    episode_decks,
    extract_episode,
    iter_replays,
    winning_seat,
)


def deck_similarity(a: Counter, b: Counter) -> float:
    """2つの60枚デッキのJaccard係数(枚数込み)。

    同じアーキタイプでも数枚は違うので、完全一致で絞ると取りこぼす。実測では
    閾値0.5と0.9で該当率が31.1%と27.8%しか変わらず、Grimmsnarl系は構築の
    ばらつきが小さい。緩めても別デッキはほとんど混ざらない。
    """
    union = sum((a | b).values())
    return sum((a & b).values()) / union if union else 0.0


def load_ratings(path: Path | None) -> dict[str, float]:
    """リーダーボードCSVからチーム名→スコアの表を作る。

    リプレイJSONにレーティングは入っていないが、チーム名は入っている。実測で97.8%が
    照合できる。ただし取れるのは「現在」のスコアで、その試合の時点の値ではない。
    """
    if path is None:
        return {}
    import csv

    with path.open(encoding="utf-8") as handle:
        return {r["TeamName"]: float(r["Score"]) for r in csv.DictReader(handle)}


def accept_episode(
    replay: dict,
    deck: Counter | None,
    min_jaccard: float,
    ratings: dict[str, float],
    min_rating: float,
    stats: Counter,
    *,
    loser_side: bool = False,
) -> bool:
    """デッキ類似度とレーティングでエピソードを採否する。

    既定では勝者側を見る。`loser_side`のときは「そのデッキで負けた試合」を採る。
    勝った試合しか集めないと、自分のデッキが劣勢にある局面をモデルが一度も見ない。
    価値ヘッドにその局面を教えるための集合をここで切り出す(方策の教師には使わない)。
    """
    if deck is None and min_rating <= 0:
        return True
    winner = winning_seat(replay)
    if winner is None:
        return True  # 引き分けはextract_episode側で落ちる
    seat = (1 - winner) if loser_side else winner
    if min_rating > 0:
        score = ratings.get((replay.get("info") or {}).get("TeamNames", [None, None])[seat])
        if score is None:
            stats["skip_rating_unknown"] += 1
            return False
        if score < min_rating:
            stats["skip_rating_low"] += 1
            return False
    if deck is not None:
        decks = episode_decks(replay.get("steps") or [])
        if decks is None:
            return True
        if deck_similarity(deck, Counter(decks[seat])) < min_jaccard:
            stats["skip_deck_mismatch"] += 1
            return False
    return True


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
    parser.add_argument(
        "--deck-like", type=Path, default=None, help="このデッキCSVに似た構築の勝利だけを採る"
    )
    parser.add_argument("--min-jaccard", type=float, default=0.7)
    parser.add_argument(
        "--ratings", type=Path, default=None, help="リーダーボードCSV(チーム名→スコア)"
    )
    parser.add_argument("--min-winner-rating", type=float, default=0.0)
    parser.add_argument(
        "--deck-spec",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help="複数デッキを1回の走査で振り分ける。出力は<output>/<NAME>/。--deck-likeとは併用しない",
    )
    parser.add_argument(
        "--shard-prefix", default=None, help="シャード名に入れる識別子(例: 20260813)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ratings = load_ratings(args.ratings)

    # 候補デッキごとに振り分ける。1エピソードのJSON解析は重いので、デッキの数だけ
    # 走査を繰り返すと候補6件で8時間かかる。1回の走査で全バケツへ配る。
    buckets: list[tuple[str, Counter | None, Path, float, float, bool]] = []
    if args.deck_spec:
        for spec in args.deck_spec:
            # NAME=CSV[:jaccard[:rating]]。バケツごとに条件を変えられるようにして、
            # 量を稼ぐ広い集合と仕上げ用の狭い集合を1回の走査で同時に作る。
            name, _, rest = spec.partition("=")
            parts = rest.split(":")
            loser = len(parts) > 3 and parts[3] == "loser"
            jaccard = float(parts[1]) if len(parts) > 1 and parts[1] else args.min_jaccard
            rating = float(parts[2]) if len(parts) > 2 and parts[2] else args.min_winner_rating
            deck = Counter(int(x) for x in Path(parts[0]).read_text().split() if x.strip())
            buckets.append((name, deck, args.output / name, jaccard, rating, loser))
    else:
        reference = None
        if args.deck_like is not None:
            reference = Counter(int(x) for x in args.deck_like.read_text().split() if x.strip())
        buckets.append(("", reference, args.output, args.min_jaccard, args.min_winner_rating))

    for _name, _deck, out, _j, _r, _l in buckets:
        out.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    per: dict[str, Counter] = {b[0]: Counter() for b in buckets}
    buffers: dict[str, list] = {b[0]: [] for b in buckets}
    indices: dict[str, int] = {b[0]: 0 for b in buckets}

    def shard_name(index: int) -> str:
        if args.shard_prefix:
            return f"shard_{args.shard_prefix}_{index:04d}.pt"
        return f"shard_{index:05d}.pt"

    for replay in iter_replays(args.zips, stats):
        if args.max_episodes and stats["episodes"] >= args.max_episodes:
            break
        samples = None
        for name, deck, out, jaccard, rating, loser in buckets:
            if not accept_episode(
                replay, deck, jaccard, ratings, rating, per[name], loser_side=loser
            ):
                continue
            # 採用されたバケツが1つでもあれば局面を作る。複数に入る場合も作り直さない。
            if samples is None:
                samples = extract_episode(
                    replay, stats, winner_only=args.winner_only, imitate_loser=args.imitate_loser
                )
            buffers[name].extend(samples)
            per[name]["episodes"] += 1
            while len(buffers[name]) >= args.shard_size:
                torch.save(buffers[name][: args.shard_size], out / shard_name(indices[name]))
                buffers[name] = buffers[name][args.shard_size :]
                indices[name] += 1
                print(
                    f"{name or 'all'}: shard {indices[name]} (episodes={per[name]['episodes']})",
                    flush=True,
                )

    for name, _deck, out, _j, _r, _l in buckets:
        if buffers[name]:
            torch.save(buffers[name], out / shard_name(indices[name]))
            indices[name] += 1
        summary = {"shards": indices[name], **dict(stats), **dict(per[name])}
        (out / "extract_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"{name or 'all'}: {json.dumps(summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
