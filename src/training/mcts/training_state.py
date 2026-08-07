"""MCTSのreplay bufferとarena checkpoint poolを再開可能に保存する。"""

from __future__ import annotations

import os
import re
from collections import deque
from pathlib import Path

import torch

from ..common.model_config import (
    D_FEEDFORWARD,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS_DECODER,
    NUM_LAYERS_ENCODER,
)
from ..common.network import PolicyValueNet
from .selfplay import Sample

_STATE_VERSION = 1


def _state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "training_state.pt"


def _replay_path(checkpoint_dir: Path, mode: str, round_num: int) -> Path:
    return checkpoint_dir.parent / "replay" / f"{mode}_round{round_num}.pt"


def saved_training_state_round(checkpoint_dir: Path, mode: str) -> int | None:
    """読めるMCTS状態があれば、その完了ラウンドを返す。"""
    path = _state_path(checkpoint_dir)
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"warning: cannot read MCTS training state {path}: {exc}")
        return None
    if state.get("version") != _STATE_VERSION or state.get("selfplay_mode") != mode:
        print(f"warning: ignoring incompatible MCTS training state {path}")
        return None
    return int(state["completed_round"])


def restore_training_state(
    checkpoint_dir: Path,
    mode: str,
    expected_round: int,
    replay_buffer_rounds: int,
) -> tuple[deque[tuple[int, list[Sample]]], list[PolicyValueNet]]:
    """保存状態を復元する。不整合時は重みだけの後方互換再開として空を返す。"""
    replay_buffer: deque[tuple[int, list[Sample]]] = deque(maxlen=replay_buffer_rounds)
    checkpoint_pool: list[PolicyValueNet] = []
    path = _state_path(checkpoint_dir)
    if expected_round <= 0 or not path.exists():
        return replay_buffer, checkpoint_pool

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (
            state.get("version") != _STATE_VERSION
            or state.get("selfplay_mode") != mode
            or int(state.get("completed_round", -1)) != expected_round
        ):
            raise ValueError("state does not match the checkpoint round/mode")

        for round_num in state["replay_rounds"][-replay_buffer_rounds:]:
            samples = torch.load(
                _replay_path(checkpoint_dir, mode, int(round_num)),
                map_location="cpu",
                weights_only=False,
            )
            replay_buffer.append((int(round_num), samples))

        for state_dict in state["checkpoint_pool"]:
            network = PolicyValueNet(
                D_MODEL,
                NUM_HEADS,
                D_FEEDFORWARD,
                NUM_LAYERS_ENCODER,
                NUM_LAYERS_DECODER,
            )
            network.load_state_dict(state_dict, assign=True)
            network.eval()
            checkpoint_pool.append(network)
    except Exception as exc:
        print(f"warning: MCTS replay/pool restore failed; continuing empty: {exc}")
        replay_buffer.clear()
        checkpoint_pool.clear()
        return replay_buffer, checkpoint_pool

    print(
        f"restored MCTS state: {len(replay_buffer)} replay round(s), "
        f"{len(checkpoint_pool)} pooled checkpoint(s)"
    )
    return replay_buffer, checkpoint_pool


def save_training_state(
    checkpoint_dir: Path,
    mode: str,
    completed_round: int,
    replay_buffer: deque[tuple[int, list[Sample]]],
    checkpoint_pool: list[PolicyValueNet],
) -> None:
    """replayを分割保存し、manifest/poolをatomic replaceする。"""
    replay_dir = checkpoint_dir.parent / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_rounds: list[int] = []
    for round_num, samples in replay_buffer:
        replay_rounds.append(round_num)
        replay_path = _replay_path(checkpoint_dir, mode, round_num)
        if not replay_path.exists():
            torch.save(samples, replay_path)

    state = {
        "version": _STATE_VERSION,
        "selfplay_mode": mode,
        "completed_round": completed_round,
        "replay_rounds": replay_rounds,
        "checkpoint_pool": [network.state_dict() for network in checkpoint_pool],
    }
    path = _state_path(checkpoint_dir)
    temporary_path = path.with_suffix(".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)

    pattern = re.compile(rf"{re.escape(mode)}_round(\d+)\.pt$")
    retained = set(replay_rounds)
    for replay_path in replay_dir.glob(f"{mode}_round*.pt"):
        match = pattern.match(replay_path.name)
        if match and int(match.group(1)) not in retained:
            replay_path.unlink()
