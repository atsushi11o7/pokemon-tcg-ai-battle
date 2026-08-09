"""Determinized MCTSの自己対戦(selfplay.py)で集めたデータで、価値・方策ネットワークを学習する。

自己対戦→学習→(更新したネットワークで)自己対戦、を繰り返すAlphaZero的なループ。
自己対戦と固定matchup評価はspawn workerで並列実行し、1試合単位のタイムアウト/例外は
その試合だけ諦める(cgエンジンはまれにネイティブクラッシュすることがあるため)。
プロセスごと落ちた場合は統一training CLIが再起動し、保存済みの最新ラウンドの
チェックポイントから自動で再開する。

設定はすべて`MctsSettings`に入れて引数で引き回す。spawn workerはこのmoduleを
まっさらに再importするため、module levelの可変状態に設定を置くと、workerだけが
定義時の既定値を見るという追跡困難なズレが生じる。workerへ渡すものは
`_SelfplayWorkerContext`に集約し、spawn境界を1箇所に閉じ込める。

このmoduleは内部trainer。表向きの実行入口は `python -m training.cli`。
"""

import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from ..common import model_config
from ..common.checkpoints import (
    checkpoint_path,
    optimizer_path,
    prune_checkpoints,
    resolve_resume_point,
    restore_optimizer_state,
)
from ..common.evaluation_plan import build_fixed_matchups
from ..common.network import (
    PolicyValueNet,
    build_policy_value_net,
    collate_encoder_decoder,
    load_policy_value_net,
)
from ..common.opponent_pool import (
    configure_sampling_snapshot,
    load_opponent_deck_pool,
    seed_opponent_deck_pool_cache,
)
from ..common.run_config import (
    RunConfig,
    load_run_config,
    save_config_snapshot,
    validate_algorithm,
)
from ..common.selfplay_modes import SelfplayMode, fixed_deck_seat_for_game
from ..common.selfplay_round import run_selfplay_round
from ..common.sparse_features import configure_feature_layout
from ..common.training_utils import (
    ListDataset,
    move_optimizer_state_to,
    seed_game,
    training_device,
)
from .parallel_evaluation import evaluate_mcts_networks_parallel
from .selfplay import Sample, play_selfplay_game
from .training_state import (
    restore_training_state,
    save_training_state,
    saved_training_state_round,
)


@dataclass(frozen=True)
class MctsSettings:
    """1 runぶんのMCTS学習設定。run configから作り、以降は読み取り専用で引き回す。"""

    run_name: str
    checkpoint_dir: Path
    output_dir: Path
    event_log_path: Path
    sampling_snapshot: Path
    feature_layout: str
    selfplay_mode: SelfplayMode
    games_per_round: int
    n_rounds: int
    search_count: int
    num_determinizations: int
    epochs_per_round: int
    batch_size: int
    learning_rate: float
    seed: int
    workers: int
    game_timeout_seconds: float
    round_timeout_seconds: float
    keep_last_checkpoints: int
    eval_games_per_round: int
    gating_win_rate: float
    checkpoint_pool_size: int
    gating_pool_sample: int
    replay_buffer_rounds: int

    @classmethod
    def from_run_config(cls, config: RunConfig) -> "MctsSettings":
        settings = config.training
        return cls(
            run_name=config.name,
            checkpoint_dir=config.checkpoint_dir,
            output_dir=config.output_dir,
            event_log_path=config.worker_event_log_path,
            sampling_snapshot=config.sampling_snapshot,
            selfplay_mode=config.selfplay_mode,
            games_per_round=config.games_per_round,
            n_rounds=config.n_rounds,
            search_count=int(settings["search_count"]),
            num_determinizations=int(settings["num_determinizations"]),
            epochs_per_round=int(settings["epochs_per_round"]),
            batch_size=int(settings["batch_size"]),
            learning_rate=float(settings["learning_rate"]),
            seed=config.runtime.seed,
            workers=config.runtime.workers,
            game_timeout_seconds=config.runtime.game_timeout_seconds,
            round_timeout_seconds=config.runtime.round_timeout_seconds,
            keep_last_checkpoints=config.runtime.keep_last_checkpoints,
            eval_games_per_round=int(settings["eval_games_per_round"]),
            gating_win_rate=float(settings["gating_win_rate"]),
            checkpoint_pool_size=int(settings["checkpoint_pool_size"]),
            gating_pool_sample=int(settings["gating_pool_sample"]),
            replay_buffer_rounds=int(settings["replay_buffer_rounds"]),
        )


def should_accept_candidate(
    current_best_win_rate: float,
    pooled_win_rate: float,
    gating_win_rate: float = 0.5,
) -> bool:
    """現bestへの直接対戦と過去モデルを含む全体の両方で勝ち越した場合だけ採用する。"""
    return current_best_win_rate > gating_win_rate and pooled_win_rate > gating_win_rate


def _collate(batch: list[Sample]):
    """`Sample`のリストを、`collate_encoder_decoder`にMCTS固有の教師信号を加えてバッチ化する。

    Args:
        batch: `Sample`のリスト。

    Returns:
        tuple: (index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec,
            mask, policy_targets, value_labels)。
            policy_targets: 形状`(batch, max_actions)`の教師分布(パディング部分は0)。
            value_labels: 形状`(batch,)`の価値の教師信号。
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


def _masked_policy_loss(
    scores: torch.Tensor, mask: torch.Tensor, policy_targets: torch.Tensor
) -> torch.Tensor:
    """パディング部分を除外した上で、方策の教師分布(訪問回数の正規化)に対する交差エントロピーを計算する。

    Args:
        scores: 形状`(batch, max_actions)`の生スコア(`PolicyValueNet.forward`の出力)。
        mask: 形状`(batch, max_actions)`のbool。パディングした位置はFalse。
        policy_targets: 形状`(batch, max_actions)`の教師分布(パディング部分は0)。

    Returns:
        torch.Tensor: バッチ平均の交差エントロピー損失(スカラー)。
    """
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    log_probs = functional.log_softmax(masked_scores, dim=-1)
    log_probs = log_probs.masked_fill(~mask, 0.0)  # -inf*0のnan化を防ぐ(教師も0なので影響なし)
    return -(policy_targets * log_probs).sum(dim=-1).mean()


def train_one_round(
    network: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    samples: list[Sample],
    *,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """1ラウンド分の自己対戦データで、`network`を数エポック学習する。

    Args:
        network: 更新対象のネットワーク(このラウンドの自己対戦にも使われたもの)。
        optimizer: ラウンドをまたいで保持するAdam。
        samples: このラウンドの自己対戦で集めた`Sample`のリスト。
        epochs: 学習エポック数。
        batch_size: ミニバッチサイズ。
        device: 学習に使うデバイス。
    """
    if not samples:
        print("  no self-play samples collected; skipping MCTS update")
        return

    dataset = ListDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)

    network.to(device)
    move_optimizer_state_to(optimizer, device)
    network.train()
    for epoch in range(epochs):
        total_policy_loss = 0.0
        total_value_loss = 0.0
        for batch in loader:
            (
                index_enc,
                value_enc,
                offset_enc,
                index_dec,
                value_dec,
                offset_dec,
                mask,
                policy_targets,
                value_labels,
            ) = (tensor.to(device) for tensor in batch)

            values, scores = network(
                index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec
            )
            policy_loss = _masked_policy_loss(scores, mask, policy_targets)
            value_loss = functional.mse_loss(values.squeeze(-1), value_labels)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size_actual = value_labels.shape[0]
            total_policy_loss += policy_loss.item() * batch_size_actual
            total_value_loss += value_loss.item() * batch_size_actual

        n = len(dataset)
        print(
            f"  epoch {epoch + 1}/{epochs}  policy_loss={total_policy_loss / n:.4f}  "
            f"value_loss={total_value_loss / n:.4f}"
        )
    network.to("cpu")
    network.eval()


@dataclass(frozen=True)
class _SelfplayWorkerContext:
    """spawn workerが自己対戦に必要とする設定一式。

    workerはこのmoduleをまっさらに再importするので、module levelの値は`main()`が
    書き換える前の既定値になる。workerが参照してよい設定はすべてここに集めて
    `initargs`で明示的に渡す。
    """

    deck: list[int]
    opponent_deck_pool: list[list[int]]
    search_count: int
    num_determinizations: int
    mode: SelfplayMode
    seed: int
    sampling_snapshot: Path


_worker_context: _SelfplayWorkerContext | None = None
_worker_network: PolicyValueNet | None = None


def _init_selfplay_worker(state_dict: dict, context: _SelfplayWorkerContext) -> None:
    """自己対戦ワーカープロセスの初期化(プロセスごとに1回だけ呼ばれる)。

    `cg`エンジンはプロセスグローバルな状態を持つため、対局はプロセスをまたいで並列化する
    必要がある(1プロセス内では同時に複数対局を進められない)。torchのスレッド内並列は
    プロセス間並列と競合するため無効化する。デッキプールは注入して、workerごとに
    snapshotを読み直さない。snapshotのパス自体も設定して、万一の遅延ロードが
    親と同じデータを見るようにしておく。

    Args:
        state_dict: この自己対戦ラウンドで使う`PolicyValueNet`の`state_dict()`。
        context: workerが必要とする設定一式。
    """
    global _worker_context, _worker_network
    torch.set_num_threads(1)
    # spawnワーカーはこのmoduleを再importするため、親の設定は引き継がれない。
    # レイアウトが食い違うと埋め込み行列の形状が変わり、意味の違う重みを読むことになる。
    configure_feature_layout(context.feature_layout)
    configure_sampling_snapshot(context.sampling_snapshot)
    seed_opponent_deck_pool_cache(context.opponent_deck_pool)
    _worker_network = build_policy_value_net(state_dict, assign=True)
    _worker_context = context


def _play_one_selfplay_game(game_index: int) -> tuple[list[Sample], int]:
    """ワーカープロセス内で1試合分の自己対戦を行う(`_init_selfplay_worker`の初期化後に呼ばれる)。

    Args:
        game_index: run内で一意な試合番号。乱数seedにも使用する。

    Returns:
        tuple[list[Sample], int]: `selfplay.play_selfplay_game`の戻り値そのまま。
    """
    context = _worker_context
    seed_game(context.seed + game_index)
    return play_selfplay_game(
        _worker_network,
        context.deck,
        context.opponent_deck_pool,
        context.search_count,
        context.mode,
        fixed_deck_seat_for_game(context.mode, game_index),
        context.num_determinizations,
    )


def _evaluate(
    settings: MctsSettings,
    candidate: PolicyValueNet,
    opponents: list[tuple[str, PolicyValueNet | None]],
    opponent_deck_pool: list[list[int]],
    matchups: list[tuple[list[int], list[int]]],
    round_num: int,
    stage: str,
) -> dict[str, dict[str, int | float]]:
    """固定matchup評価をワーカー並列で実行する(ゲーティング/ランダム相手戦の共通入口)。"""
    return evaluate_mcts_networks_parallel(
        candidate,
        opponents,
        opponent_deck_pool,
        matchups,
        search_count=settings.search_count,
        num_determinizations=settings.num_determinizations,
        sampling_snapshot=settings.sampling_snapshot,
        seed=settings.seed + 100_000,
        num_workers=settings.workers,
        game_timeout_seconds=settings.game_timeout_seconds,
        round_timeout_seconds=settings.round_timeout_seconds,
        event_log_path=settings.event_log_path,
        event_context={
            "algorithm": "mcts",
            "round": round_num,
            "mode": settings.selfplay_mode,
            "stage": stage,
        },
    )


def run_training_loop(
    settings: MctsSettings,
    deck: list[int],
    initial_checkpoint: Path | None = None,
    optimizer_checkpoint: Path | None = None,
    start_round: int = 1,
) -> PolicyValueNet:
    """自己対戦、replay学習、固定matchup arena評価を繰り返す。"""
    torch.manual_seed(settings.seed)
    # 過去モデルのサンプリングもrunをまたいで再現できるよう、Python側もseedする。
    random.seed(settings.seed)
    device = training_device()
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    opponent_deck_pool = load_opponent_deck_pool()
    print(f"loaded {len(opponent_deck_pool)} distinct opponent decks for self-play")
    print(f"self-play mode: {settings.selfplay_mode}")

    # replayの大きなI/Oはnative card masterを触るPolicyValueNet構築より先に済ませる。
    replay_buffer, checkpoint_pool = restore_training_state(
        settings.checkpoint_dir,
        settings.selfplay_mode,
        expected_round=start_round - 1,
        replay_buffer_rounds=settings.replay_buffer_rounds,
    )

    if initial_checkpoint is not None:
        best_network = load_policy_value_net(initial_checkpoint)
        print(f"loaded initial best_network weights from {initial_checkpoint}")
    else:
        best_network = build_policy_value_net()
    best_optimizer = torch.optim.Adam(best_network.parameters(), lr=settings.learning_rate)
    if optimizer_checkpoint is not None:
        restore_optimizer_state(
            best_optimizer, optimizer_checkpoint, learning_rate=settings.learning_rate
        )
        print(
            f"restored optimizer state from {optimizer_checkpoint} "
            f"with configured learning_rate={settings.learning_rate}"
        )

    # round間のモデル差だけを比較したいので、matchupはrunを通して固定する。
    matchups = build_fixed_matchups(
        settings.selfplay_mode,
        deck,
        opponent_deck_pool,
        settings.eval_games_per_round,
        settings.seed + 100_000,
    )

    for round_num in range(start_round, settings.n_rounds + 1):
        print(
            f"=== round {round_num}/{settings.n_rounds}: self-play ({settings.workers} workers) ==="
        )
        worker_context = _SelfplayWorkerContext(
            deck=deck,
            opponent_deck_pool=opponent_deck_pool,
            search_count=settings.search_count,
            num_determinizations=settings.num_determinizations,
            mode=settings.selfplay_mode,
            seed=settings.seed,
            sampling_snapshot=settings.sampling_snapshot,
            feature_layout=model_config.FEATURE_LAYOUT,
        )
        all_samples, results = run_selfplay_round(
            algorithm="mcts",
            round_num=round_num,
            mode=settings.selfplay_mode,
            num_games=settings.games_per_round,
            num_workers=settings.workers,
            initializer=_init_selfplay_worker,
            initargs=(best_network.state_dict(), worker_context),
            task=_play_one_selfplay_game,
            game_timeout_seconds=settings.game_timeout_seconds,
            round_timeout_seconds=settings.round_timeout_seconds,
            event_log_path=settings.event_log_path,
        )
        print(f"  round results: {results}  total_samples={len(all_samples)}")

        replay_buffer.append((round_num, all_samples))
        train_samples = [
            sample for _replay_round, round_samples in replay_buffer for sample in round_samples
        ]

        print(f"=== round {round_num}/{settings.n_rounds}: training ===")
        print(f"  training on {len(train_samples)} samples from last {len(replay_buffer)} round(s)")
        candidate_network = copy.deepcopy(best_network)
        candidate_optimizer = torch.optim.Adam(
            candidate_network.parameters(), lr=settings.learning_rate
        )
        candidate_optimizer.load_state_dict(best_optimizer.state_dict())
        train_one_round(
            candidate_network,
            candidate_optimizer,
            train_samples,
            epochs=settings.epochs_per_round,
            batch_size=settings.batch_size,
            device=device,
        )

        print(f"=== round {round_num}/{settings.n_rounds}: fixed-matchup arena ===")
        sampled_past = random.sample(
            checkpoint_pool, min(len(checkpoint_pool), settings.gating_pool_sample)
        )
        gating_opponents: list[tuple[str, PolicyValueNet | None]] = [("current_best", best_network)]
        gating_opponents += [(f"past_round{r}", net) for r, net in sampled_past]

        gating_results = _evaluate(
            settings,
            candidate_network,
            gating_opponents,
            opponent_deck_pool,
            matchups,
            round_num,
            "gating",
        )
        total_wins = 0
        total_games = 0
        gating_complete = True
        for name, _opponent_network in gating_opponents:
            result = gating_results[name]
            print(f"  candidate vs {name}: {result}")
            total_wins += int(result["wins"])
            total_games += int(result["games"])
            gating_complete &= (
                result["failed"] == 0 and result["games"] == settings.eval_games_per_round
            )
        current_best_win_rate = float(gating_results["current_best"]["win_rate"])
        pooled_win_rate = total_wins / total_games if total_games else 0.0
        # 失敗した試合があると勝率が偏った部分集合の平均になるため、完走したときだけ採用判定する。
        accepted = gating_complete and should_accept_candidate(
            current_best_win_rate, pooled_win_rate, settings.gating_win_rate
        )
        print(
            f"  pooled_win_rate={pooled_win_rate:.3f} ({total_wins}/{total_games})  "
            f"complete={gating_complete}  accepted={accepted}"
        )

        if accepted:
            checkpoint_pool.append((round_num, copy.deepcopy(best_network)))
            if len(checkpoint_pool) > settings.checkpoint_pool_size:
                checkpoint_pool.pop(0)
            best_network = candidate_network
            best_optimizer = candidate_optimizer

        print(f"=== round {round_num}/{settings.n_rounds}: fixed-matchup eval vs first-index ===")
        vs_baseline = _evaluate(
            settings,
            best_network,
            [("first_index", None)],
            opponent_deck_pool,
            matchups,
            round_num,
            "random_eval",
        )["first_index"]
        print(f"  vs first-index: {vs_baseline}")

        saved_path = checkpoint_path(settings.checkpoint_dir, settings.selfplay_mode, round_num)
        torch.save(best_network.state_dict(), saved_path)
        torch.save(
            best_optimizer.state_dict(),
            optimizer_path(settings.checkpoint_dir, settings.selfplay_mode, round_num),
        )
        save_training_state(
            settings.checkpoint_dir,
            settings.selfplay_mode,
            round_num,
            replay_buffer,
            checkpoint_pool,
        )
        # training state保存後に消す。先に消すと、直後にクラッシュしたとき
        # 巻き戻し先のチェックポイントが無くなる。
        prune_checkpoints(
            settings.checkpoint_dir, settings.selfplay_mode, settings.keep_last_checkpoints
        )
        print(f"  saved checkpoint and MCTS training state to {saved_path}")

    return best_network


def main(config_path: Path) -> int:
    """MCTS trainerを1つのrun configで実行する。再起動時は最新roundから再開する。"""
    config = load_run_config(config_path)
    validate_algorithm(config, "mcts")
    settings = MctsSettings.from_run_config(config)

    configure_sampling_snapshot(settings.sampling_snapshot)
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config_path, settings.output_dir)

    resume = resolve_resume_point(
        settings.checkpoint_dir,
        settings.selfplay_mode,
        settings.run_name,
        config.model.initial_checkpoint,
        # チェックポイント保存とtraining state保存の間で落ちた場合、両者が揃う所まで戻す。
        rollback_to=saved_training_state_round(settings.checkpoint_dir, settings.selfplay_mode),
    )

    network = run_training_loop(
        settings,
        config.deck,
        initial_checkpoint=resume.initial_checkpoint,
        optimizer_checkpoint=resume.optimizer_checkpoint,
        start_round=resume.start_round,
    )
    final_path = settings.checkpoint_dir / "final.pt"
    torch.save(network.state_dict(), final_path)
    print(f"saved final checkpoint to {final_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python train.py CONFIG_PATH")
    raise SystemExit(main(Path(sys.argv[1])))


__all__ = [
    "MctsSettings",
    "main",
    "run_training_loop",
    "should_accept_candidate",
    "train_one_round",
]
