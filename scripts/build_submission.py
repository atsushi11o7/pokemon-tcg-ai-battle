"""学習済みチェックポイントから提出用パッケージを生成する。

提出物へ特徴量コードを手でコピーすると、`sparse_features.py`を変えるたびに陳腐化し、
学習時と推論時で食い違う。ここでは`src/training/`のモジュールを機械的に連結して
1つの`main.py`を生成することで、複製を手で書かずに単体完結の提出物を作る。

パッケージとして同梱する形(`training/`ディレクトリ + 薄い`main.py`)は本番で通らない。
Kaggle側は`main.py`をソースとして読み込んでexecするため、`__file__`が無く、
importの起点も安定しない。実績のある単体完結型に揃えている。

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

# `main.py`へ連結するモジュール(依存順)。探索なしなら前半だけで足りるが、
# 分岐させると片方だけ壊れるので常に同じ順で全部入れる。
CONCATENATED_MODULES = (
    "common/model_config.py",
    "common/sparse_features.py",
    "common/network.py",
    "common/deck.py",
    "common/selfplay_modes.py",
    "mcts/search.py",
    "mcts/determinize.py",
    "mcts/selfplay.py",
    "inference/agent.py",
)

# 連結時に落とす行。パッケージ内の相対importは連結後には同じ名前空間に並ぶので不要で、
# `ROOT`まわりはリポジトリの階層に依存するため提出物では成立しない。
_DROPPED_PREFIXES = ("from .", "from __future__ import")
_DROPPED_STATEMENTS = ("ROOT = Path(", "SAMPLE_SUBMISSION_DIR = ", "sys.path.insert(")

# 連結元は構成値を`model_config.X`の形でも参照する。1ファイルになると
# そのモジュールは存在しないので、同じ名前で見えるようにしておく。
MODEL_CONFIG_SHIM = """
import types as _types

model_config = _types.SimpleNamespace(
    FEATURE_LAYOUT=FEATURE_LAYOUT,
    D_MODEL=D_MODEL,
    NUM_HEADS=NUM_HEADS,
    D_FEEDFORWARD=D_FEEDFORWARD,
    NUM_LAYERS_ENCODER=NUM_LAYERS_ENCODER,
    NUM_LAYERS_DECODER=NUM_LAYERS_DECODER,
)
"""

HEADER_TEMPLATE = '''"""{description}

`src/training/`のモジュールを機械的に連結して生成している。手で書き写していないので、
特徴量やネットワーク構成が学習側と食い違うことはない。生成は`scripts/build_submission.py`。

構成: {layout} / D_MODEL={d_model} / {inference}
"""

import os
import sys
from pathlib import Path

# Kaggleは`main.py`をソースとして読み込んでexecするため、この名前空間には`__file__`が無い。
# 参照するとエピソードが即失敗するので、カレントディレクトリと本番の配置先だけで解決する。
KAGGLE_AGENT_DIR = Path("/kaggle_simulations/agent")
for _candidate in (".", str(KAGGLE_AGENT_DIR)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)


def resource_path(name: str, base_dir: Path) -> Path:
    """同梱ファイルを、ローカル実行と本番実行の両方で解決する。"""
    if os.path.exists(name):
        return Path(name)
    return KAGGLE_AGENT_DIR / name

'''

ENTRYPOINT_TEMPLATE = """

# ===== エントリポイント =====================================================

_agent = SubmissionAgent(
    Path("."),
    search_count={search_count},
    num_determinizations={determinizations},
)


def agent(obs_dict: dict) -> list[int]:
    return _agent.select(to_observation_class(obs_dict))
"""


def _strip_module(source: str) -> str:
    """1ファイルへ連結できるよう、パッケージ前提の行を落とす。

    括弧で複数行に跨るimportは閉じ括弧まで落とす。連結後は全てが同じ名前空間に
    並ぶので、相対importは不要になる。
    """
    kept: list[str] = []
    skipping = False
    for line in source.splitlines():
        if skipping:
            skipping = ")" not in line
            continue
        stripped = line.strip()
        if stripped.startswith(_DROPPED_PREFIXES):
            skipping = stripped.endswith("(")
            continue
        if stripped.startswith(_DROPPED_STATEMENTS):
            continue
        kept.append(line)
    return "\n".join(kept)


def _generate_main(args: argparse.Namespace, layout: str, d_model: int, inference: str) -> str:
    """モジュールを連結して、単体で動く`main.py`を組み立てる。"""
    parts = [
        HEADER_TEMPLATE.format(
            description=args.description, layout=layout, d_model=d_model, inference=inference
        )
    ]
    for relative in CONCATENATED_MODULES:
        body = _strip_module((ROOT / "src" / "training" / relative).read_text(encoding="utf-8"))
        parts.append(f"\n# ===== {relative} " + "=" * max(4, 58 - len(relative)) + f"\n{body}")
        if relative == "common/model_config.py":
            parts.append(MODEL_CONFIG_SHIM)
    parts.append(
        ENTRYPOINT_TEMPLATE.format(
            search_count=args.search_count, determinizations=args.determinizations
        )
    )
    return "\n".join(parts)


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
    out_dir.mkdir(parents=True)

    shutil.copytree(CG_SOURCE, out_dir / "cg", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(args.checkpoint, out_dir / "policy.pt")
    shutil.copy(args.deck, out_dir / "deck.csv")

    if args.search_count > 0:
        decks = _opponent_decks(args.snapshot)
        (out_dir / "opponent_decks.json").write_text(json.dumps(decks, separators=(",", ":")))

    inference = (
        f"Determinized MCTS(探索{args.search_count}回 / 仮説{args.determinizations}通り)"
        if args.search_count > 0
        else "貪欲方策(argmax、探索なし)"
    )
    (out_dir / "main.py").write_text(
        _generate_main(args, model_config.FEATURE_LAYOUT, model_config.D_MODEL, inference),
        encoding="utf-8",
    )

    archive = ROOT / "submission" / f"{args.name}.tar.gz"
    archive.unlink(missing_ok=True)
    # `-C dir .`だと全エントリが"./"始まりになる。中身を名前で明示する。
    entries = sorted(path.name for path in out_dir.iterdir())
    subprocess.run(["tar", "-czf", str(archive), "-C", str(out_dir), *entries], check=True)
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
