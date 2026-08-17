"""ラウンドごとの指標を機械可読な形で追記する。

これまで勝敗・損失・ゲーティング結果はすべて標準出力にしか出ておらず、50ラウンド回した
あとに系列を比較しようとするとログをgrepするしかなかった。1ラウンド1行のJSON Linesとして
残しておけば、runをまたいだ比較やプロットがそのままできる。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def append_round_metrics(path: Path, round_num: int, **fields: Any) -> None:
    """1ラウンド分の指標を1行のJSONとして追記する。

    学習を止めたくないので、書き込みに失敗しても例外は投げず警告だけ出す。

    Args:
        path: 追記先の`metrics.jsonl`。
        round_num: ラウンド番号。
        **fields: 記録したい値(勝敗内訳、損失、採用可否など)。
    """
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round": round_num,
        **fields,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        print(f"  warning: could not append metrics to {path}: {exc}")
