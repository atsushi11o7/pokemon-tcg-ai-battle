"""学習済みチェックポイントから提出用パッケージを生成する。

提出物へ特徴量コードを手でコピーすると、`sparse_features.py`を変えるたびに陳腐化し、
学習時と推論時で食い違う。ここでは`src/training/`のモジュールをそのまま同梱し、
`main.py`は薄いシムにすることで複製そのものを無くす。

Usage:
    uv run python scripts/build_submission.py \
        --name 19_v2_generalist \
        --checkpoint outputs/runs/xxx/checkpoints/final.pt \
        --deck decks/candidates/crustle_meta.csv \
        --search-count 60 --determinizations 4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CG_SOURCE = ROOT / "data" / "sample_submission" / "sample_submission" / "cg"

# 推論に必要なモジュールだけを同梱する(学習専用のモジュールは持ち込まない)。
BUNDLED_MODULES = (
    "__init__.py",
    "common/__init__.py",
    "common/deck.py",
    "common/model_config.py",
    "common/network.py",
    "common/sparse_features.py",
    "common/selfplay_modes.py",
    "inference/__init__.py",
    "inference/agent.py",
    "mcts/__init__.py",
    "mcts/determinize.py",
    "mcts/search.py",
    "mcts/selfplay.py",
)

MAIN_TEMPLATE = '''"""{description}

学習側と同じ`training/`モジュールをそのまま同梱しているため、特徴量やネットワーク構成が
提出物側で古くなることはない。生成は`scripts/build_submission.py`。

構成: {layout} / D_MODEL={d_model} / {inference}
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from cg.api import to_observation_class  # noqa: E402
from training.inference.agent import SubmissionAgent  # noqa: E402

_agent = SubmissionAgent(
    BASE_DIR,
    search_count={search_count},
    num_determinizations={determinizations},
)


def agent(obs_dict: dict) -> list[int]:
    return _agent.select(to_observation_class(obs_dict))
'''


def _opponent_decks(snapshot: Path) -> list[list[int]]:
    """探索時の隠れ情報推測に使う、実在デッキ候補を抜き出す。"""
    snapshot_data = json.loads(snapshot.read_text(encoding="utf-8"))
    return [record["cards"] for record in snapshot_data["decks"] if len(record["cards"]) == 60]


def build(args: argparse.Namespace) -> Path:
    sys.path.insert(0, str(ROOT / "src"))
    from training.common import model_config

    out_dir = ROOT / "submission" / args.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "training").mkdir(parents=True)

    shutil.copytree(CG_SOURCE, out_dir / "cg", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(args.checkpoint, out_dir / "policy.pt")
    shutil.copy(args.deck, out_dir / "deck.csv")

    for relative in BUNDLED_MODULES:
        destination = out_dir / "training" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "src" / "training" / relative, destination)

    if args.search_count > 0:
        decks = _opponent_decks(args.snapshot)
        (out_dir / "opponent_decks.json").write_text(json.dumps(decks, separators=(",", ":")))

    inference = (
        f"Determinized MCTS(探索{args.search_count}回 / 仮説{args.determinizations}通り)"
        if args.search_count > 0
        else "貪欲方策(argmax、探索なし)"
    )
    (out_dir / "main.py").write_text(
        MAIN_TEMPLATE.format(
            description=args.description,
            layout=model_config.FEATURE_LAYOUT,
            d_model=model_config.D_MODEL,
            inference=inference,
            search_count=args.search_count,
            determinizations=args.determinizations,
        ),
        encoding="utf-8",
    )

    archive = ROOT / "submission" / f"{args.name}.tar.gz"
    archive.unlink(missing_ok=True)
    subprocess.run(["tar", "-czf", str(archive), "-C", str(out_dir), "."], check=True)
    return archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="提出物のディレクトリ名")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--search-count", type=int, default=0, help="0なら探索せず貪欲方策のみ")
    parser.add_argument("--determinizations", type=int, default=4)
    parser.add_argument(
        "--snapshot", type=Path, default=ROOT / "data/meta/derived/sampling_snapshot.json"
    )
    parser.add_argument(
        "--description", default="学習済み方策・価値ネットワークによるエージェント。"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.search_count < 0:
        raise ValueError("--search-count must be zero or positive")
    archive = build(args)
    size_mib = archive.stat().st_size / 1024 / 1024
    print(f"built {archive} ({size_mib:.1f} MiB)")
    if size_mib > 197.7:
        raise SystemExit("提出サイズ上限197.7MiBを超えています")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
