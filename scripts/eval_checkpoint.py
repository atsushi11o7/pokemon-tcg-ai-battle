#!/usr/bin/env python3
"""任意のチェックポイントを、指定した検証シャードで測る。

学習構成を変えると検証セットも変わるため、そのままでは新旧の数字を比べられない。
古い重みを新しい検証セットへ通し直すのがこのスクリプトの用途。
`--d-feedforward`のように次元を明示できるのは、`model_config`を変更した後でも
変更前の重みを読めるようにするため(構成が食い違うとstate_dictの読み込みが失敗する)。

Usage:
    uv run python scripts/eval_checkpoint.py \
        --checkpoint outputs/runs/bc_toplayers/checkpoints/final.pt \
        --shard-dir data/bc/shards_dated --val-shards 15 --d-feedforward 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.bc.dataset import load_shard_paths  # noqa: E402
from training.bc.train import evaluate  # noqa: E402
from training.common import model_config  # noqa: E402
from training.common.network import PolicyValueNet  # noqa: E402
from training.common.training_utils import training_device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--val-shards", type=int, required=True)
    parser.add_argument(
        "--holdout-shards",
        type=int,
        default=0,
        help="学習から外す枚数。既定はval-shardsと同数。train.pyと同じ分割にするために指定する",
    )
    parser.add_argument("--d-model", type=int, default=model_config.D_MODEL)
    parser.add_argument("--num-heads", type=int, default=model_config.NUM_HEADS)
    parser.add_argument("--d-feedforward", type=int, default=model_config.D_FEEDFORWARD)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    shards = sorted(args.shard_dir.glob("shard_*.pt"))
    holdout = max(args.holdout_shards, args.val_shards)
    val_paths = shards[-holdout:][: args.val_shards]
    samples = load_shard_paths(val_paths)
    print(f"検証: {len(samples)}サンプル  {val_paths[0].name} .. {val_paths[-1].name}")

    network = PolicyValueNet(
        args.d_model,
        args.num_heads,
        args.d_feedforward,
        model_config.NUM_LAYERS_ENCODER,
        model_config.NUM_LAYERS_DECODER,
    )
    network.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    result = evaluate(network, samples, args.batch_size, training_device())
    print(
        f"accuracy={result['accuracy'] * 100:.1f}%  "
        f"policy_loss={result['policy_loss']:.4f}  value_loss={result['value_loss']:.4f}  "
        f"(policy_samples={result['policy_samples']:,} / {result['samples']:,})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
