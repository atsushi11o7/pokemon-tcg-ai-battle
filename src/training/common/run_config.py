"""単一の学習run(PPOまたはMCTS)を表す厳密なYAML設定。"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from .deck import parse_deck_csv
from .selfplay_modes import SelfplayMode

ROOT = Path(__file__).resolve().parents[3]
Algorithm = Literal["ppo", "mcts", "bc"]


@dataclass(frozen=True)
class ModelConfig:
    # Noneならランダム初期化から学習する。入力仕様やD_MODELを変えると
    # 既存チェックポイントは形状不一致で読めなくなるため、その場合はNoneにする。
    initial_checkpoint: Path | None


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    workers: int
    game_timeout_seconds: float
    round_timeout_seconds: float
    max_restarts: int
    retry_delay_seconds: float
    keep_last_checkpoints: int


@dataclass(frozen=True)
class RunConfig:
    name: str
    algorithm: Algorithm
    output_dir: Path
    model: ModelConfig
    sampling_snapshot: Path
    runtime: RuntimeConfig
    selfplay_mode: SelfplayMode
    deck_path: Path | None
    training: dict[str, Any]

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def worker_event_log_path(self) -> Path:
        return self.output_dir / "worker_events.jsonl"

    @property
    def deck(self) -> list[int]:
        # generalistは両席ともsnapshotから抽選するため、固定デッキを使用しない。
        return parse_deck_csv(self.deck_path) if self.deck_path is not None else []

    @property
    def games_per_round(self) -> int:
        return int(self.training["games_per_round"])

    @property
    def n_rounds(self) -> int:
        return int(self.training["rounds"])


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _reject_unknown(raw: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown {label} field(s): {', '.join(sorted(unknown))}")


def _positive(value: Any, label: str) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def _non_negative(value: Any, label: str) -> float:
    number = float(value)
    if number < 0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _positive_int(value: Any, label: str) -> int:
    number = int(value)
    if number <= 0 or number != float(value):
        raise ValueError(f"{label} must be a positive integer")
    return number


def _non_negative_int(value: Any, label: str) -> int:
    number = int(value)
    if number < 0 or number != float(value):
        raise ValueError(f"{label} must be a non-negative integer")
    return number


def _probability(value: Any, label: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise ValueError(f"{label} must be between 0 and 1")
    return number


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _resolve(path: str | None) -> Path | None:
    return ROOT / path if path else None


def _validate_training(training: dict[str, Any], algorithm: Algorithm) -> SelfplayMode:
    # BCはリプレイからの教師あり学習で自己対戦を行わないため、対局まわりの項目を持たない。
    common_fields = (
        {"rounds"}
        if algorithm == "bc"
        else {"selfplay_mode", "deck_path", "games_per_round", "rounds"}
    )
    algorithm_fields = {
        "ppo": {
            "learning_rate",
            "minibatch_size",
            "epochs_per_round",
            "target_kl",
            "gamma",
            "gae_lambda",
            "clip_epsilon",
            "value_loss_coef",
            "entropy_coef",
            "max_grad_norm",
            "eval_games_per_round",
        },
        "mcts": {
            "search_count",
            "num_determinizations",
            "learning_rate",
            "batch_size",
            "epochs_per_round",
            "eval_games_per_round",
            "baseline_eval_games",
            "gating_win_rate",
            "checkpoint_pool_size",
            "gating_pool_sample",
            "replay_buffer_rounds",
            "freeze_policy",
        },
        "bc": {
            "shard_dir",
            "val_shards",
            "holdout_shards",
            "min_shard_day",
            "val_day",
            "loser_policy_weight",
            "freeze_policy",
            "learning_rate",
            "batch_size",
            "value_loss_coef",
            "warmup_steps",
        },
    }[algorithm]
    _reject_unknown(training, common_fields | algorithm_fields, "training")
    # 省略可能な項目。既定はeval_games_per_roundと同数(従来どおりの挙動)。
    optional_fields = {
        "deck_path",
        "baseline_eval_games",
        "value_loss_coef",
        "warmup_steps",
        "freeze_policy",
        "holdout_shards",
        "min_shard_day",
        "val_day",
    }
    required = (common_fields | algorithm_fields) - optional_fields
    missing = required - set(training)
    if missing:
        raise ValueError(f"missing training field(s): {', '.join(sorted(missing))}")

    _positive_int(training["rounds"], "training.rounds")
    _positive(training["learning_rate"], "training.learning_rate")

    if algorithm == "bc":
        _positive_int(training["batch_size"], "training.batch_size")
        # 0は「検証せず全シャードを学習に使う」。汎化の指標は取れなくなるので、
        # 既に指標が揃っていてデータ量を優先する場合にだけ選ぶこと。
        _non_negative_int(training["val_shards"], "training.val_shards")
        _required_text(training["shard_dir"], "training.shard_dir")
        if "value_loss_coef" in training:
            _non_negative(training["value_loss_coef"], "training.value_loss_coef")
        if "warmup_steps" in training:
            _non_negative_int(training["warmup_steps"], "training.warmup_steps")
        # 敗者の手を模倣する強さ。必須にしているのは、敗者のone-hotを含むシャードへ
        # 設定を向けたまま書き忘れると、黙って全面模倣に変わってしまうため。
        # 何を学ばせているかは設定ファイルだけで読み取れる必要がある。
        _probability(training["loser_policy_weight"], "training.loser_policy_weight")
        return "generalist"

    mode = training["selfplay_mode"]
    if mode not in ("generalist", "asymmetric", "mirror"):
        raise ValueError(f"invalid selfplay_mode: {mode!r}")
    if mode != "generalist" and not training.get("deck_path"):
        raise ValueError(f"training.deck_path is required for selfplay_mode={mode!r}")

    games_per_round = _positive_int(training["games_per_round"], "training.games_per_round")
    if mode == "asymmetric" and games_per_round % 2 != 0:
        raise ValueError("training.games_per_round must be even for selfplay_mode='asymmetric'")
    _positive_int(training["epochs_per_round"], "training.epochs_per_round")
    eval_games = _positive_int(training["eval_games_per_round"], "training.eval_games_per_round")
    if eval_games % 4 != 0:
        raise ValueError("training.eval_games_per_round must be a multiple of 4")

    if algorithm == "mcts" and "baseline_eval_games" in training:
        baseline_games = training["baseline_eval_games"]
        if not isinstance(baseline_games, int) or baseline_games < 0 or baseline_games % 4 != 0:
            raise ValueError("training.baseline_eval_games must be 0 (disabled) or a multiple of 4")
        if baseline_games > eval_games:
            raise ValueError(
                "training.baseline_eval_games must not exceed training.eval_games_per_round"
            )

    if algorithm == "ppo":
        _positive_int(training["minibatch_size"], "training.minibatch_size")
        _positive(training["target_kl"], "training.target_kl")
        _probability(training["gamma"], "training.gamma")
        _probability(training["gae_lambda"], "training.gae_lambda")
        _probability(training["clip_epsilon"], "training.clip_epsilon")
        _non_negative(training["value_loss_coef"], "training.value_loss_coef")
        _non_negative(training["entropy_coef"], "training.entropy_coef")
        _positive(training["max_grad_norm"], "training.max_grad_norm")
    else:
        search_count = _positive_int(training["search_count"], "training.search_count")
        hypotheses = _positive_int(
            training["num_determinizations"], "training.num_determinizations"
        )
        # 予算は仮説へ分割されるので、1仮説あたり1回未満になる組み合わせは弾く。
        if hypotheses > search_count:
            raise ValueError("training.num_determinizations cannot exceed search_count")
        _positive_int(training["batch_size"], "training.batch_size")
        _probability(training["gating_win_rate"], "training.gating_win_rate")
        pool_size = _non_negative_int(
            training["checkpoint_pool_size"], "training.checkpoint_pool_size"
        )
        pool_sample = _non_negative_int(
            training["gating_pool_sample"], "training.gating_pool_sample"
        )
        if pool_sample > pool_size:
            raise ValueError("training.gating_pool_sample cannot exceed checkpoint_pool_size")
        _positive_int(training["replay_buffer_rounds"], "training.replay_buffer_rounds")
        if "freeze_policy" in training and not isinstance(training["freeze_policy"], bool):
            # bool()で受けると、YAMLに引用符付きで"false"と書いた場合にTrueになる。
            raise ValueError("training.freeze_policy must be a boolean")
    return mode


def load_run_config(path: Path) -> RunConfig:
    """1つのalgorithmだけを実行するrun configを読み、未知フィールドも拒否する。"""
    with path.open(encoding="utf-8") as file:
        raw = _mapping(yaml.safe_load(file), "config")
    _reject_unknown(raw, {"run", "algorithm", "model", "data", "runtime", "training"}, "top-level")

    run = _mapping(raw.get("run"), "run")
    model = _mapping(raw.get("model"), "model")
    data = _mapping(raw.get("data"), "data")
    runtime = _mapping(raw.get("runtime", {}), "runtime")
    training = _mapping(raw.get("training"), "training")
    _reject_unknown(run, {"name", "output_dir"}, "run")
    _reject_unknown(model, {"initial_checkpoint"}, "model")
    _reject_unknown(data, {"sampling_snapshot"}, "data")
    _reject_unknown(
        runtime,
        {
            "seed",
            "workers",
            "game_timeout_seconds",
            "round_timeout_seconds",
            "max_restarts",
            "retry_delay_seconds",
            "keep_last_checkpoints",
        },
        "runtime",
    )

    algorithm = raw.get("algorithm")
    if algorithm not in ("ppo", "mcts", "bc"):
        raise ValueError("algorithm must be one of: ppo, mcts, bc")
    mode = _validate_training(training, algorithm)

    # 入力仕様やD_MODELを変えると既存チェックポイントは形状不一致で読めなくなる。
    # その場合はnullを指定してランダム初期化から学習する。
    checkpoint_value = model.get("initial_checkpoint")
    if checkpoint_value is not None:
        checkpoint_value = _required_text(checkpoint_value, "model.initial_checkpoint")
    checkpoint = ROOT / checkpoint_value if checkpoint_value is not None else None

    workers = _positive_int(
        runtime.get("workers", max(1, (os.cpu_count() or 4) - 2)), "runtime.workers"
    )
    max_restarts = _non_negative_int(runtime.get("max_restarts", 50), "runtime.max_restarts")
    retry_delay = _non_negative(
        runtime.get("retry_delay_seconds", 5), "runtime.retry_delay_seconds"
    )
    # 1ラウンドあたり重み+optimizerで300MB超。既定では直近数ラウンドだけ残す(0で無制限)。
    keep_last_checkpoints = _non_negative_int(
        runtime.get("keep_last_checkpoints", 3), "runtime.keep_last_checkpoints"
    )
    deck_value = training.get("deck_path")
    if deck_value is not None:
        deck_value = _required_text(deck_value, "training.deck_path")

    return RunConfig(
        name=_required_text(run.get("name"), "run.name"),
        algorithm=algorithm,
        output_dir=ROOT / _required_text(run.get("output_dir"), "run.output_dir"),
        model=ModelConfig(checkpoint),
        sampling_snapshot=ROOT
        / _required_text(data.get("sampling_snapshot"), "data.sampling_snapshot"),
        runtime=RuntimeConfig(
            seed=int(runtime.get("seed", 0)),
            workers=workers,
            game_timeout_seconds=_positive(
                runtime.get("game_timeout_seconds", 300), "runtime.game_timeout_seconds"
            ),
            round_timeout_seconds=_positive(
                runtime.get("round_timeout_seconds", 1800), "runtime.round_timeout_seconds"
            ),
            max_restarts=max_restarts,
            retry_delay_seconds=retry_delay,
            keep_last_checkpoints=keep_last_checkpoints,
        ),
        selfplay_mode=mode,
        deck_path=_resolve(deck_value),
        training=training,
    )


def validate_algorithm(config: RunConfig, expected: Algorithm) -> None:
    if config.algorithm != expected:
        raise ValueError(
            f"config algorithm is {config.algorithm!r}, but {expected!r} trainer was selected"
        )


def save_config_snapshot(config_path: Path, output_dir: Path) -> None:
    """run configと、そのrunで使ったネットワーク構成を出力先に記録する。

    入力仕様やD_MODELを変えるとチェックポイントの互換性が切れるため、
    「この重みはどの構成で学習されたか」を後から辿れるようにする。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output_dir / "run_config.yaml")

    from . import model_config

    architecture = {
        name: getattr(model_config, name)
        for name in (
            "D_MODEL",
            "NUM_HEADS",
            "D_FEEDFORWARD",
            "NUM_LAYERS_ENCODER",
            "NUM_LAYERS_DECODER",
            "HAND_TOKENS",
            "DROPOUT",
        )
    }
    (output_dir / "architecture.yaml").write_text(
        yaml.safe_dump(architecture, sort_keys=False), encoding="utf-8"
    )
