"""trainer(MCTS/PPO/BC)とspawnワーカーが共通で使う、学習まわりのユーティリティ。

デバイスと乱数、バッチ化と損失、チェックポイントの入出力、ラウンド指標の記録をまとめる。
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset, Sampler

from .network import PolicyValueNet, collate_encoder_decoder


def training_device() -> torch.device:
    """学習に使うデバイスを返す。

    spawnワーカーはtrainerモジュールを毎回再importするため、import時ではなく
    呼び出し時に判定して、使わないCUDA初期化を避ける。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_game(seed: int) -> None:
    """1試合分の乱数を、Python側とtorch側の両方で再シードする。"""
    random.seed(seed)
    torch.manual_seed(seed)


def move_optimizer_state_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """optimizerのモーメント推定を`device`へ移す。

    checkpointから復元した状態は保存時のデバイスに乗っていることがあり、
    networkのパラメータとデバイスが揃っていないとstep()で失敗する。
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


class ListDataset(Dataset):
    """学習サンプルのリストをそのまま`DataLoader`へ渡すためのDataset。"""

    def __init__(self, samples: list) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_samples(batch: list):
    """`Sample`のリストをバッチ化する。MCTS・BCで共通。

    Args:
        batch: `Sample`のリスト。`label`(価値の教師)が埋まっている必要がある。

    Returns:
        tuple: (index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec,
            mask, policy_targets, value_labels)。
    """
    index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec, mask = (
        collate_encoder_decoder(
            batch,
            lambda s: s.encoder_sv,
            lambda s: s.decoder_sv,
            lambda s: len(s.policy_target),
        )
    )
    max_actions = mask.shape[1]
    policy_targets = torch.zeros(len(batch), max_actions, dtype=torch.float32)
    for i, sample in enumerate(batch):
        n = len(sample.policy_target)
        policy_targets[i, :n] = torch.tensor(sample.policy_target, dtype=torch.float32)
    value_labels = torch.tensor([s.label for s in batch], dtype=torch.float32)
    return (
        index_enc,
        value_enc,
        offset_enc,
        index_dec,
        value_dec,
        offset_dec,
        mask,
        policy_targets,
        value_labels,
    )


def masked_policy_loss(
    scores: torch.Tensor, mask: torch.Tensor, policy_targets: torch.Tensor
) -> torch.Tensor:
    """パディングを除外した上で、方策の教師分布に対する交差エントロピーを返す。

    教師が全ゼロのサンプル(価値だけ学ばせる局面)は分母から外す。

    Args:
        scores: 形状`(batch, max_actions)`の生スコア。
        mask: 形状`(batch, max_actions)`のbool。パディング位置はFalse。
        policy_targets: 形状`(batch, max_actions)`の教師分布(パディング部分は0)。
    """
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    log_probs = functional.log_softmax(masked_scores, dim=-1)
    # -inf * 0 のnan化を防ぐ。教師も0なので値には影響しない。
    log_probs = log_probs.masked_fill(~mask, 0.0)
    per_sample = -(policy_targets * log_probs).sum(dim=-1)
    supervised = policy_targets.sum(dim=-1) > 0
    n = int(supervised.sum().item())
    if n == 0:
        return per_sample.sum() * 0.0
    return per_sample.sum() / n


class LengthBucketSampler(Sampler):
    """行動数が近いサンプルを同じバッチにまとめ、デコーダのパディングを減らす。

    バッチ内の順序とバッチ自体の出現順はエポックごとにシャッフルするので、
    確率的勾配降下としての性質は保たれる。
    """

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        order = list(range(len(self.lengths)))
        if self.shuffle:
            # 同じ長さの中での並びを毎エポック変える。
            random.shuffle(order)
        order.sort(key=lambda i: self.lengths[i])
        batches = [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def configure_trainable_parameters(
    network: PolicyValueNet, freeze_policy: bool
) -> list[torch.nn.Parameter]:
    """`requires_grad`を設定し、更新対象のパラメータを返す。

    `freeze_policy`のときは`encoder_fc`(価値ヘッド)だけを更新する。方策側は
    `encoder`の出力全体へ交差注意し、`encoder_fc`はCLSトークンから価値を読むだけなので、
    ここだけ動かせば方策の出力はビット単位で不変になる。

    `requires_grad`を落とさずoptimizerへ渡す集合を絞るだけでは、autogradが
    encoder/decoderの計算グラフを保持したまま逆伝播し、計算とメモリを無駄に払う。
    設定と選択を1つの関数に閉じ込めて、両方が必ず揃うようにしている。
    """
    for parameter in network.parameters():
        parameter.requires_grad_(not freeze_policy)
    if not freeze_policy:
        return list(network.parameters())
    for parameter in network.encoder_fc.parameters():
        parameter.requires_grad_(True)
    return list(network.encoder_fc.parameters())


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

    保存されたoptimizer stateには当時のlearning rateも含まれるため、
    上書きしないとrun configの変更が再開時に反映されない。
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


def append_round_metrics(path: Path, round_num: int, **fields: Any) -> None:
    """1ラウンド分の指標を1行のJSONとして`metrics.jsonl`へ追記する。

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
