"""上位プレイヤーのリプレイを模倣して、価値・方策ネットワークを学習する。

自己対戦は行わない。`scripts/data_extract.py`が作ったシャードを読み、
方策は「実際に選ばれた手」のone-hot、価値は「その試合の勝敗」を教師にする。

MCTSとはSampleの作り方が違うだけで、ネットワーク・バッチ化・損失は共通のものを使う
(`common.training_utils`)。したがってBCで得た重みは、そのままMCTSの初期値にも
提出物にも使える。

このmoduleは内部trainer。表向きの実行入口は `python -m training.cli`。
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from ..common.network import PolicyValueNet, build_policy_value_net, load_policy_value_net
from ..common.run_config import RunConfig, load_run_config, save_config_snapshot, validate_algorithm
from ..common.training_utils import (
    LengthBucketSampler,
    ListDataset,
    append_round_metrics,
    checkpoint_path,
    collate_samples,
    configure_trainable_parameters,
    masked_policy_loss,
    move_optimizer_state_to,
    optimizer_path,
    prune_checkpoints,
    resolve_resume_point,
    restore_optimizer_state,
    training_device,
)
from .dataset import load_shard, load_shard_paths, shard_sample_counts

# 統一CLIは`<selfplay_mode>_roundN.pt`を探して進捗表示と再開判定を行う。BCは自己対戦を
# しないがモードは"generalist"を返すので、それに合わせないと再開もstatusも効かない。
CHECKPOINT_PREFIX = "generalist"


@dataclass(frozen=True)
class BcSettings:
    run_name: str
    checkpoint_dir: Path
    output_dir: Path
    shard_dir: Path
    val_shards: int
    holdout_shards: int
    min_shard_day: str | None
    val_day: str | None
    loser_policy_weight: float
    freeze_policy: bool
    n_rounds: int
    batch_size: int
    learning_rate: float
    value_loss_coef: float
    warmup_steps: int
    seed: int
    loader_workers: int
    keep_last_checkpoints: int

    @classmethod
    def from_run_config(cls, config: RunConfig) -> BcSettings:
        settings = config.training
        return cls(
            run_name=config.name,
            checkpoint_dir=config.checkpoint_dir,
            output_dir=config.output_dir,
            shard_dir=Path(settings["shard_dir"]),
            val_shards=int(settings["val_shards"]),
            holdout_shards=int(settings.get("holdout_shards", settings["val_shards"])),
            min_shard_day=settings.get("min_shard_day"),
            val_day=settings.get("val_day"),
            loser_policy_weight=float(settings.get("loser_policy_weight", 1.0)),
            freeze_policy=bool(settings.get("freeze_policy", False)),
            n_rounds=config.n_rounds,
            batch_size=int(settings["batch_size"]),
            learning_rate=float(settings["learning_rate"]),
            value_loss_coef=float(settings.get("value_loss_coef", 1.0)),
            warmup_steps=int(settings.get("warmup_steps", 0)),
            seed=config.runtime.seed,
            loader_workers=config.runtime.workers,
            keep_last_checkpoints=config.runtime.keep_last_checkpoints,
        )


def policy_accuracy(scores: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor) -> tuple:
    """top-1が教師と一致した件数と、評価件数を返す。

    ディスカッションで上位勢が報告している「policy accuracy」と同じ定義
    (合法手の中でスコア最大の手が、実際に選ばれた手と一致するか)。

    教師が全ゼロのサンプル(価値だけ学ばせる敗者の局面)は除外する。`argmax`は
    全ゼロ行に対して0を返すため、含めると「index 0を選べば正解」という
    無意味な判定になり、精度が実態とかけ離れる。
    """
    supervised = targets.sum(dim=-1) > 0
    if not bool(supervised.any()):
        return 0, 0
    predicted = scores.masked_fill(~mask, float("-inf")).argmax(dim=-1)[supervised]
    expected = targets[supervised].argmax(dim=-1)
    return int((predicted == expected).sum().item()), int(predicted.shape[0])


def mask_loser_targets(targets: torch.Tensor, labels: torch.Tensor, weight: float) -> torch.Tensor:
    """敗者の局面の方策教師を`weight`倍する。

    敗者かどうかは価値の教師で判別する(`label`は勝ち+1・負け-1で、引き分けは抽出時に除外)。
    0.0なら勝者の手だけ、1.0なら敗者の手も同等に模倣する。
    """
    if weight == 1.0:
        return targets
    scale = torch.where(labels < 0, torch.full_like(labels, weight), torch.ones_like(labels))
    return targets * scale.unsqueeze(1)


def evaluate(network: PolicyValueNet, samples: list, batch_size: int, device) -> dict:
    """検証シャードで方策精度と損失を測る。

    `loser_policy_weight`の値によらず、方策の指標は常に勝者の手だけで測る。
    """
    if not samples:
        return {}
    loader = DataLoader(
        ListDataset(samples), batch_size=batch_size, shuffle=False, collate_fn=collate_samples
    )
    network.to(device)
    network.eval()
    # 方策は教師を持つサンプルだけ、価値は全サンプルで集計する。混ぜると、敗者の割合が
    # バッチごとに違うぶんだけ平均が歪む。
    correct = policy_seen = value_seen = 0
    policy_sum = value_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            tensors = [tensor.to(device) for tensor in batch]
            (idx_e, val_e, off_e, idx_d, val_d, off_d, mask, targets, labels) = tensors
            targets = mask_loser_targets(targets, labels, 0.0)
            values, scores = network(idx_e, val_e, off_e, idx_d, val_d, off_d, mask)
            hit, count = policy_accuracy(scores, mask, targets)
            correct += hit
            policy_seen += count
            policy_sum += masked_policy_loss(scores, mask, targets).item() * count
            value_seen += labels.shape[0]
            value_sum += functional.mse_loss(values.squeeze(-1), labels).item() * labels.shape[0]
    return {
        "accuracy": correct / policy_seen if policy_seen else 0.0,
        "policy_loss": policy_sum / policy_seen if policy_seen else 0.0,
        "value_loss": value_sum / value_seen if value_seen else 0.0,
        "policy_samples": policy_seen,
        "samples": value_seen,
    }


def run_training_loop(settings: BcSettings, initial_checkpoint: Path | None) -> PolicyValueNet:
    device = training_device()
    shards = sorted(settings.shard_dir.glob("shard_*.pt"))
    if settings.min_shard_day:
        # シャード名は`shard_YYYYMMDD_NNNN.pt`。日付は固定幅なので文字列比較でよい。
        # 直近だけで仕上げるとき、古い期間を丸ごと落とすために使う。
        shards = [p for p in shards if p.name.split("_")[1] >= settings.min_shard_day]
    if len(shards) <= settings.val_shards:
        raise RuntimeError(f"not enough shards in {settings.shard_dir}: {len(shards)}")

    # シャード名に日付が入っており`sorted`が時系列順になるので、末尾が最新日になる。
    # `holdout_shards`は学習から外す枚数、`val_shards`はそのうち実際に読む枚数。
    # 最新日を丸ごと学習から外しつつ、検証は一部だけ読むために分けてある。
    if settings.val_day:
        # 検証日を明示する。最新日まで学習に使いたいときに、少し前の日を検証へ回す。
        # 検証日より後のデータで学習するため、汎化性能の指標にはならない。
        val_paths = [p for p in shards if p.name.split("_")[1] == settings.val_day][
            : settings.val_shards
        ]
        if not val_paths:
            raise RuntimeError(f"val_day={settings.val_day} に該当するシャードが無い")
        chosen = set(val_paths)
        train_paths = [p for p in shards if p not in chosen]
    else:
        holdout = max(settings.holdout_shards, settings.val_shards)
        if holdout >= len(shards):
            # 素通りさせると`train_paths`が空になり、1ステップも学習しないまま
            # 「完了」してしまう。例外にならないので、ここで止める。
            raise RuntimeError(
                f"holdout_shards={holdout} leaves no training shards ({len(shards)} available)"
            )
        # `holdout=0`は検証せず全シャードを学習に使う。`shards[:-0]`は空リストに
        # なるので、スライスに任せず分岐する。
        val_paths = shards[-holdout:][: settings.val_shards] if holdout else []
        train_paths = shards[:-holdout] if holdout else list(shards)
    if not train_paths:
        raise RuntimeError("no training shards left after the split")
    # 検証データは常駐させず、エポック末の評価時だけ読む。20万サンプルを親プロセスに
    # 抱えたままだと、fork したDataLoaderワーカーがコピーオンライトで複製してしまう。
    span = f" ({val_paths[0].name}..{val_paths[-1].name})" if val_paths else " (検証なし)"
    print(f"shards: train={len(train_paths)} val={len(val_paths)}{span}")

    # 保存済みエポックがあればそこから、無ければ設定の初期重みから始める。
    # PPO/MCTSと同じ共通処理を使う(ネイティブクラッシュ後にCLIが再起動したとき、
    # 最初からやり直して既存チェックポイントを上書きしないため)。
    resume = resolve_resume_point(
        settings.checkpoint_dir,
        CHECKPOINT_PREFIX,
        settings.run_name,
        initial_checkpoint,
    )
    if resume.initial_checkpoint is not None:
        network = load_policy_value_net(resume.initial_checkpoint)
    else:
        network = build_policy_value_net()
    # `freeze_policy`のときは`encoder_fc`だけを更新する。方策側はエンコーダ出力全体へ
    # 交差注意し、`encoder_fc`はCLSトークンから価値を読むだけなので、方策の出力は
    # ビット単位で不変になり、方策を壊さずに価値だけ鍛えられる。
    trainable = configure_trainable_parameters(network, settings.freeze_policy)
    optimizer = torch.optim.Adam(trainable, lr=settings.learning_rate)
    if resume.optimizer_checkpoint is not None:
        restore_optimizer_state(
            optimizer,
            resume.optimizer_checkpoint,
            learning_rate=settings.learning_rate,
        )
    resume_round = resume.start_round - 1
    # 全ステップ数が事前に分かるので、warmup後にコサインで下げ切る。
    steps_per_epoch = sum(
        (count + settings.batch_size - 1) // settings.batch_size
        for count in shard_sample_counts(train_paths, settings.shard_dir)
    )
    total_steps = max(1, steps_per_epoch * settings.n_rounds)
    warmup = min(settings.warmup_steps, total_steps // 10)

    def lr_scale(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min((step - warmup) / max(1, total_steps - warmup), 1.0)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    for _ in range(resume_round * steps_per_epoch):
        scheduler.step()  # 再開時は消化済みステップ分だけ進めておく
    print(f"lr schedule: warmup={warmup} steps, cosine to 5% over {total_steps} steps")

    for round_num in range(resume_round + 1, settings.n_rounds + 1):
        print(f"=== epoch {round_num}/{settings.n_rounds} ===", flush=True)
        network.to(device)
        move_optimizer_state_to(optimizer, device)
        if settings.freeze_policy:
            # dropoutを切る。エンコーダの出力が揺れると価値ヘッドが見る特徴が安定しない。
            network.eval()
        else:
            network.train()
        policy_seen = value_seen = 0
        policy_sum = value_sum = 0.0
        # シャードは日付順に並ぶので、固定順だとエポックの終端が常に最新日になる。
        # 毎エポック順序を変える(シャード内は`LengthBucketSampler`がシャッフルする)。
        epoch_paths = list(train_paths)
        random.Random(settings.seed + round_num).shuffle(epoch_paths)
        # シャードを1つずつ載せる。全部同時に持つと数十GBになる。
        for shard_index, path in enumerate(epoch_paths, start=1):
            samples = load_shard(path)
            lengths = [len(sample.policy_target) for sample in samples]
            loader = DataLoader(
                ListDataset(samples),
                batch_sampler=LengthBucketSampler(lengths, settings.batch_size),
                collate_fn=collate_samples,
                # `persistent_workers`は使わない。ローダはシャードごとに作り直して
                # 1回しか回さないので、常駐化すると破棄の遅れたワーカーが積み上がる。
                num_workers=settings.loader_workers,
            )
            for batch in loader:
                tensors = [tensor.to(device) for tensor in batch]
                (idx_e, val_e, off_e, idx_d, val_d, off_d, mask, targets, labels) = tensors
                targets = mask_loser_targets(targets, labels, settings.loser_policy_weight)
                values, scores = network(idx_e, val_e, off_e, idx_d, val_d, off_d, mask)
                value_loss = functional.mse_loss(values.squeeze(-1), labels)
                if settings.freeze_policy:
                    policy_loss = masked_policy_loss(scores, mask, targets).detach()
                    loss = settings.value_loss_coef * value_loss
                else:
                    policy_loss = masked_policy_loss(scores, mask, targets)
                    loss = policy_loss + settings.value_loss_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()

                supervised = int((targets.sum(dim=-1) > 0).sum().item())
                policy_seen += supervised
                policy_sum += policy_loss.item() * supervised
                value_seen += labels.shape[0]
                value_sum += value_loss.item() * labels.shape[0]
            # `del samples`だけでは`loader.dataset`が参照を握ったままになる。
            # 最後のシャードではepoch末の評価まで生き残り、検証20万件と同居する。
            del loader
            del samples
            if shard_index % 10 == 0:
                print(
                    f"  shard {shard_index}/{len(train_paths)}  "
                    f"policy_loss={policy_sum / max(policy_seen, 1):.4f}  "
                    f"value_loss={value_sum / max(value_seen, 1):.4f}",
                    flush=True,
                )

        validation = load_shard_paths(val_paths)
        metrics = evaluate(network, validation, settings.batch_size, device)
        del validation
        if not metrics:
            metrics = {"accuracy": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "samples": 0}
        print(
            f"  train policy_loss={policy_sum / max(policy_seen, 1):.4f} "
            f"value_loss={value_sum / max(value_seen, 1):.4f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}\n"
            f"  val accuracy={metrics['accuracy'] * 100:.1f}%  "
            f"policy_loss={metrics['policy_loss']:.4f}  value_loss={metrics['value_loss']:.4f}",
            flush=True,
        )

        network.to("cpu")
        network.eval()
        saved = checkpoint_path(settings.checkpoint_dir, CHECKPOINT_PREFIX, round_num)
        torch.save(network.state_dict(), saved)
        torch.save(
            optimizer.state_dict(),
            optimizer_path(settings.checkpoint_dir, CHECKPOINT_PREFIX, round_num),
        )
        prune_checkpoints(
            settings.checkpoint_dir, CHECKPOINT_PREFIX, settings.keep_last_checkpoints
        )
        append_round_metrics(
            settings.output_dir / "metrics.jsonl",
            round_num,
            algorithm="bc",
            train_samples=value_seen,
            train_policy_samples=policy_seen,
            train_policy_loss=policy_sum / max(policy_seen, 1),
            train_value_loss=value_sum / max(value_seen, 1),
            **{f"val_{k}": v for k, v in metrics.items()},
        )
        print(f"  saved checkpoint to {saved}", flush=True)

    return network


def main(config_path: Path) -> int:
    config = load_run_config(config_path)
    validate_algorithm(config, "bc")
    settings = BcSettings.from_run_config(config)
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config_path, settings.output_dir)

    summary_path = settings.shard_dir / "extract_summary.json"
    if summary_path.exists():
        print(
            "extract summary:", json.dumps(json.loads(summary_path.read_text()), ensure_ascii=False)
        )

    network = run_training_loop(settings, config.model.initial_checkpoint)
    final_path = settings.checkpoint_dir / "final.pt"
    torch.save(network.state_dict(), final_path)
    print(f"saved final checkpoint to {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
