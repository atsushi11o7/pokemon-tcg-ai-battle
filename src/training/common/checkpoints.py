"""PPO/MCTS/CLIで共通利用するcheckpointヘルパー。"""

from __future__ import annotations

import re
from dataclasses import dataclass
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


def checkpoint_path(checkpoint_dir: Path, selfplay_mode: str, round_num: int) -> Path:
    """ラウンド末に保存するネットワーク重みのパス。"""
    return checkpoint_dir / f"{selfplay_mode}_round{round_num}.pt"


def optimizer_path(checkpoint_dir: Path, selfplay_mode: str, round_num: int) -> Path:
    """`checkpoint_path`に対応するoptimizer状態のパス。"""
    return checkpoint_dir / f"{selfplay_mode}_round{round_num}_optimizer.pt"


@dataclass(frozen=True)
class ResumePoint:
    """再開時にどの重み・どのラウンドから始めるか。"""

    initial_checkpoint: Path | None
    optimizer_checkpoint: Path | None
    start_round: int


def resolve_resume_point(
    checkpoint_dir: Path,
    selfplay_mode: str,
    run_name: str,
    configured_initial: Path | None,
    *,
    rollback_to: int | None = None,
) -> ResumePoint:
    """保存済みチェックポイントがあれば再開点を、無ければ初期チェックポイントを返す。

    Args:
        checkpoint_dir: `{mode}_round{N}.pt`を保存しているディレクトリ。
        selfplay_mode: チェックポイント名に含まれる自己対戦モード。
        run_name: ログ出力に使うrun名。
        configured_initial: run configの`model.initial_checkpoint`。
        rollback_to: これより新しいチェックポイントを信用せず巻き戻す先のラウンド。
            MCTSはチェックポイント保存とtraining state保存の間で落ちうるため、
            training state側の完了ラウンドを渡して整合するところまで戻す。

    Returns:
        ResumePoint: 再開に使う重み・optimizer・開始ラウンド。
    """
    latest_round = latest_checkpoint_round(checkpoint_dir, selfplay_mode)
    if (
        latest_round is not None
        and rollback_to is not None
        and rollback_to < latest_round
        and checkpoint_path(checkpoint_dir, selfplay_mode, rollback_to).exists()
    ):
        print(
            f"warning: round {latest_round} checkpoint has no matching training state; "
            f"rolling back resume to atomic round {rollback_to}"
        )
        latest_round = rollback_to

    if latest_round is not None:
        optimizer_candidate = optimizer_path(checkpoint_dir, selfplay_mode, latest_round)
        print(f"resuming {run_name} from round {latest_round} (start_round={latest_round + 1})")
        return ResumePoint(
            initial_checkpoint=checkpoint_path(checkpoint_dir, selfplay_mode, latest_round),
            optimizer_checkpoint=optimizer_candidate if optimizer_candidate.exists() else None,
            start_round=latest_round + 1,
        )

    if configured_initial is not None and not configured_initial.exists():
        raise FileNotFoundError(f"initial checkpoint does not exist: {configured_initial}")
    print(f"starting {run_name} from initial checkpoint {configured_initial}")
    return ResumePoint(
        initial_checkpoint=configured_initial, optimizer_checkpoint=None, start_round=1
    )


def prune_checkpoints(checkpoint_dir: Path, selfplay_mode: str, keep_last: int) -> int:
    """直近`keep_last`ラウンド分だけ残し、それより古いラウンドの重みを削除する。

    1ラウンドあたり重み+optimizerで300MB超になるため、放置すると1 runで数十GBに達する。
    再開に必要なのは最新ラウンドだけなので、数ラウンド分の保険を残して残りは捨てる。
    `final.pt`とtraining state(replay/pool)は対象外。

    Args:
        checkpoint_dir: チェックポイントの保存先。
        selfplay_mode: チェックポイント名に含まれる自己対戦モード。
        keep_last: 残す最新ラウンド数(0以下なら何も削除しない)。

    Returns:
        int: 削除したファイル数。
    """
    if keep_last <= 0:
        return 0
    pattern = re.compile(rf"{re.escape(selfplay_mode)}_round(\d+)\.pt$")
    rounds = sorted(
        int(match.group(1))
        for path in checkpoint_dir.glob(f"{selfplay_mode}_round*.pt")
        if (match := pattern.match(path.name))
    )
    removed = 0
    for round_num in rounds[:-keep_last]:
        for path in (
            checkpoint_path(checkpoint_dir, selfplay_mode, round_num),
            optimizer_path(checkpoint_dir, selfplay_mode, round_num),
        ):
            if path.exists():
                path.unlink()
                removed += 1
    return removed
