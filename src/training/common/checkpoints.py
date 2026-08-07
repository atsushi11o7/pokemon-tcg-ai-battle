"""PPO/MCTS/CLIで共通利用するcheckpointヘルパー。"""

from __future__ import annotations

import re
from pathlib import Path

import torch


def latest_checkpoint_round(checkpoint_dir: Path, selfplay_mode: str) -> int | None:
    """``{mode}_roundN.pt``のうち最大のNを返す。optimizer/finalは除外する。"""
    pattern = re.compile(rf"{re.escape(selfplay_mode)}_round(\d+)\.pt$")
    rounds = [
        int(match.group(1))
        for path in checkpoint_dir.glob(f"{selfplay_mode}_round*.pt")
        if (match := pattern.match(path.name))
    ]
    return max(rounds) if rounds else None


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    *,
    learning_rate: float,
) -> None:
    """optimizerの履歴を復元し、実行中の設定でlearning rateを上書きする。

    PyTorchのoptimizer stateには保存時のlearning rateも含まれる。
    ``load_state_dict``後に上書きしないと、run configの変更が再開時に
    反映されないため、モーメントは引き継ぎつつ値だけ更新する。
    """
    optimizer.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
