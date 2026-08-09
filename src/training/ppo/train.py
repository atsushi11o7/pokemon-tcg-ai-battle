"""PPOによる自己対戦で、価値・方策ネットワークを学習する(MCTSを使わない軽量な代替経路)。

自己対戦→GAE計算→PPO更新、を繰り返す。1手あたりのネット評価が1回で済むため、
MCTS自己対戦(mcts/train.py、1手ごとにsearch_count回評価)よりスループットが高い。
ネットワークはMCTS側と共通の`PolicyValueNet`をそのまま使うので、既存PPO/MCTSの
チェックポイントを初期重みとして相互利用できる。

AlphaZero式のゲーティング(勝率が閾値を超えたら採用)は行わず、毎ラウンドそのまま
方策を更新し続ける(PPOはオンポリシー更新のため、探索で洗練した教師信号を前提とする
ゲーティングとは相性が良くない)。ラウンドごとのvs random評価は、悪化していないかの
モニタリング用に残す。

自己対戦も固定matchup評価もspawn workerで並列実行し、1試合単位のタイムアウト/例外は
その試合だけ諦める(cgエンジンはまれにネイティブクラッシュすることがあるため)。
プロセスごと落ちた場合は統一training CLIが再起動し、保存済みの最新ラウンドの
チェックポイントから自動で再開する。

設定はすべて`PpoSettings`に入れて引数で引き回す。spawn workerはこのmoduleを
まっさらに再importするため、module levelの可変状態に設定を置くと、workerだけが
定義時の既定値を見るという追跡困難なズレが生じる。workerへ渡すものは
`_SelfplayWorkerContext`に集約し、spawn境界を1箇所に閉じ込める。

このmoduleは内部trainer。表向きの実行入口は `python -m training.cli`。
"""

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
from ..common.opponent_pool import configure_sampling_snapshot, load_opponent_deck_pool
from ..common.parallel_evaluation import evaluate_networks_parallel
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
from .selfplay import (
    PPOSample,
    compute_gae,
    make_ppo_eval_agent,
    play_ppo_game,
)


@dataclass(frozen=True)
class PpoSettings:
    """1 runぶんのPPO学習設定。run configから作り、以降は読み取り専用で引き回す。"""

    run_name: str
    checkpoint_dir: Path
    output_dir: Path
    event_log_path: Path
    sampling_snapshot: Path
    feature_layout: str
    selfplay_mode: SelfplayMode
    games_per_round: int
    n_rounds: int
    epochs_per_round: int
    minibatch_size: int
    learning_rate: float
    target_kl: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float
    eval_games_per_round: int
    seed: int
    workers: int
    game_timeout_seconds: float
    round_timeout_seconds: float
    keep_last_checkpoints: int

    @classmethod
    def from_run_config(cls, config: RunConfig) -> "PpoSettings":
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
            epochs_per_round=int(settings["epochs_per_round"]),
            minibatch_size=int(settings["minibatch_size"]),
            learning_rate=float(settings["learning_rate"]),
            target_kl=float(settings["target_kl"]),
            gamma=float(settings["gamma"]),
            gae_lambda=float(settings["gae_lambda"]),
            clip_epsilon=float(settings["clip_epsilon"]),
            value_loss_coef=float(settings["value_loss_coef"]),
            entropy_coef=float(settings["entropy_coef"]),
            max_grad_norm=float(settings["max_grad_norm"]),
            eval_games_per_round=int(settings["eval_games_per_round"]),
            seed=config.runtime.seed,
            workers=config.runtime.workers,
            game_timeout_seconds=config.runtime.game_timeout_seconds,
            round_timeout_seconds=config.runtime.round_timeout_seconds,
            keep_last_checkpoints=config.runtime.keep_last_checkpoints,
        )


def _collate(batch: list[PPOSample]):
    """`PPOSample`のリストを、`collate_encoder_decoder`にPPO固有の教師信号を加えてバッチ化する。

    PPO固有のフィールドは、選んだ行動のindex/behavior policyのlog確率/advantage/return。

    Args:
        batch: `PPOSample`のリスト。

    Returns:
        tuple: (index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec,
            mask, action_indices, old_log_probs, advantages, returns)。
    """
    index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec, mask = (
        collate_encoder_decoder(
            batch,
            lambda s: s.encoder_sv,
            lambda s: s.decoder_sv,
            lambda s: len(s.decoder_sv.offset),
        )
    )

    action_indices = torch.tensor([s.action_index for s in batch], dtype=torch.int64)
    old_log_probs = torch.tensor([s.old_log_prob for s in batch], dtype=torch.float32)
    advantages = torch.tensor([s.advantage for s in batch], dtype=torch.float32)
    returns = torch.tensor([s.return_ for s in batch], dtype=torch.float32)

    return (
        index_enc,
        value_enc,
        offset_enc,
        index_dec,
        value_dec,
        offset_dec,
        mask,
        action_indices,
        old_log_probs,
        advantages,
        returns,
    )


def _ppo_loss(
    network: PolicyValueNet,
    batch,
    device: torch.device,
    settings: PpoSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """クリップ付きサロゲート目的関数+価値損失+エントロピー項を計算する。

    Args:
        network: 更新対象の`PolicyValueNet`。
        batch: `_collate`が返すバッチ(1ミニバッチ分)。
        device: 計算に使うデバイス。
        settings: クリップ幅・各損失係数を持つ設定。

    Returns:
        tuple[torch.Tensor, ...]: (合計損失, 方策損失, 価値損失, エントロピー,
            approx_kl, clip_fraction)。いずれもスカラー。
            approx_kl: 更新前後の方策のずれの目安(大きすぎる場合は学習率/エポック数が
                強すぎる兆候)。
            clip_fraction: クリップが実際に効いたサンプルの割合。
    """
    (
        index_enc,
        value_enc,
        offset_enc,
        index_dec,
        value_dec,
        offset_dec,
        mask,
        action_indices,
        old_log_probs,
        advantages,
        returns,
    ) = (t.to(device) for t in batch)

    values, scores = network(index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec)

    masked_scores = scores.masked_fill(~mask, float("-inf"))
    log_probs = functional.log_softmax(masked_scores, dim=-1)
    new_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)

    log_probs_safe = log_probs.masked_fill(~mask, 0.0)  # -inf*0のnan化を防ぐ(エントロピー計算用)
    probs_safe = log_probs_safe.exp()
    entropy = -(probs_safe * log_probs_safe).sum(dim=-1).mean()

    # advantageはtrain_one_round側でラウンド全体を通して一度だけ正規化済み
    # (ミニバッチごとに正規化すると、同じサンプルの値がシャッフルのたびに変わってしまう)。
    clip_epsilon = settings.clip_epsilon
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = functional.mse_loss(values.squeeze(-1), returns)

    loss = policy_loss + settings.value_loss_coef * value_loss - settings.entropy_coef * entropy

    with torch.no_grad():
        log_ratio = new_log_probs - old_log_probs
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()

    return loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction


def train_one_round(
    network: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    samples: list[PPOSample],
    settings: PpoSettings,
    device: torch.device,
) -> None:
    """1ラウンド分のロールアウトで、`network`をPPOで数エポック学習する。

    ロールアウト収集時の方策(behavior policy)に対するクリップ付き目的関数のため、
    同じデータを複数エポック再利用できる(AlphaZero式のTD学習と異なり、PPOは
    オンポリシー更新をこの範囲内で複数回行うのが特徴)。advantageはミニバッチ分割前に
    ラウンド全体で一度だけ正規化する(ミニバッチごとの正規化だと、シャッフルのたびに
    同じサンプルの正規化後advantageが変わってしまうため)。

    Args:
        network: 更新対象のネットワーク(このラウンドの自己対戦にも使われたもの)。
        optimizer: ラウンドをまたいで保持する`Adam`(モーメント推定をリセットしないため)。
        samples: このラウンドの自己対戦で集めた`PPOSample`のリスト(GAE計算済み)。
        settings: 学習ハイパーパラメータ。
        device: 学習に使うデバイス。
    """
    if not samples:
        print("  no rollout samples collected; skipping PPO update")
        return

    advantage_tensor = torch.tensor([s.advantage for s in samples], dtype=torch.float32)
    advantage_mean = advantage_tensor.mean()
    advantage_std = advantage_tensor.std(unbiased=False)
    for sample in samples:
        sample.advantage = float((sample.advantage - advantage_mean) / (advantage_std + 1e-8))

    dataset = ListDataset(samples)
    loader = DataLoader(
        dataset, batch_size=settings.minibatch_size, shuffle=True, collate_fn=_collate
    )
    network.to(device)
    move_optimizer_state_to(optimizer, device)

    network.train()
    epochs = settings.epochs_per_round
    for epoch in range(epochs):
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        n = 0
        for batch in loader:
            loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction = _ppo_loss(
                network, batch, device, settings
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), settings.max_grad_norm)
            optimizer.step()

            batch_size = batch[6].shape[0]  # mask
            total_policy_loss += policy_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size
            total_entropy += entropy.item() * batch_size
            total_approx_kl += approx_kl.item() * batch_size
            total_clip_fraction += clip_fraction.item() * batch_size
            n += batch_size

        print(
            f"  epoch {epoch + 1}/{epochs}  policy_loss={total_policy_loss / n:.4f}  "
            f"approx_kl={total_approx_kl / n:.4f}  clip_fraction={total_clip_fraction / n:.4f}  "
            f"value_loss={total_value_loss / n:.4f}  entropy={total_entropy / n:.4f}"
        )
        mean_kl = total_approx_kl / n
        if mean_kl > settings.target_kl:
            print(
                f"  early stopping PPO update: approx_kl={mean_kl:.4f} > {settings.target_kl:.4f}"
            )
            break
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
    mode: SelfplayMode
    gamma: float
    gae_lambda: float
    seed: int
    sampling_snapshot: Path


_worker_context: _SelfplayWorkerContext | None = None
_worker_network: PolicyValueNet | None = None


def _init_selfplay_worker(state_dict: dict, context: _SelfplayWorkerContext) -> None:
    """自己対戦ワーカープロセスの初期化(プロセスごとに1回だけ呼ばれる)。

    `cg`エンジンはプロセスグローバルな状態を持つため、対局はプロセスをまたいで並列化する
    必要がある(1プロセス内では同時に複数対局を進められない)。torchのスレッド内並列は
    プロセス間並列と競合するため無効化する。

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
    _worker_network = build_policy_value_net(state_dict, assign=True)
    _worker_context = context


def _play_one_ppo_game(game_index: int) -> tuple[list[PPOSample], int]:
    """ワーカープロセス内で1試合分の自己対戦を行い、GAEまで計算する。

    `play_ppo_game`は座席ごとに分けたサンプル列を返すので、GAEはそれぞれの軌跡に対して
    独立に計算してから結合する(結合してから計算すると、片方の軌跡の終端が
    もう片方の先頭に繋がってGAEが壊れる)。

    Args:
        game_index: run内で一意な試合番号。乱数seedにも使用する。

    Returns:
        tuple[list[PPOSample], int]: (advantage/return計算済みのPPOSampleのリスト
            (座席を結合済み), `state.result`)。
    """
    context = _worker_context
    seed_game(context.seed + game_index)
    samples_by_seat, winner = play_ppo_game(
        _worker_network,
        context.deck,
        context.opponent_deck_pool,
        context.mode,
        fixed_deck_seat_for_game(context.mode, game_index),
    )
    samples: list[PPOSample] = []
    for seat_samples in samples_by_seat:
        compute_gae(seat_samples, context.gamma, context.gae_lambda)
        samples.extend(seat_samples)
    return samples, winner


def run_training_loop(
    settings: PpoSettings,
    deck: list[int],
    initial_checkpoint: Path | None = None,
    optimizer_checkpoint: Path | None = None,
    start_round: int = 1,
) -> PolicyValueNet:
    """自己対戦→PPO更新を`settings.n_rounds`回繰り返すメインループ。

    Args:
        settings: 1 runぶんの学習設定。
        deck: 本番でも使う、こちらの60枚のデッキリスト。
        initial_checkpoint: 指定があれば、ランダム初期化の代わりにこのチェックポイントを
            初期値として使う(共通`PolicyValueNet`のcheckpointを想定)。
        optimizer_checkpoint: 指定があれば、Adamのモーメント推定もこのチェックポイントから
            復元する(クラッシュ後に同じラウンドの続きから再開する用途)。
        start_round: 学習を開始するラウンド番号(1始まり)。

    Returns:
        PolicyValueNet: 最終ラウンド終了時点のネットワーク。
    """
    torch.manual_seed(settings.seed)
    device = training_device()
    # PolicyValueNet構築(AllCard/AllAttackを呼ぶ)の直後に大量のファイルI/Oを行うと
    # ネイティブ側でクラッシュするため、load_opponent_deck_pool()を先に呼ぶ
    # (mcts/train.pyのrun_training_loopと同じ理由)。
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    opponent_deck_pool = load_opponent_deck_pool()
    print(f"loaded {len(opponent_deck_pool)} distinct opponent decks for self-play")
    print(f"self-play mode: {settings.selfplay_mode}")

    if initial_checkpoint is not None:
        network = load_policy_value_net(initial_checkpoint)
        print(f"loaded initial network weights from {initial_checkpoint}")
    else:
        network = build_policy_value_net()
    # ラウンドをまたいで同じoptimizerを使う(毎ラウンド作り直すとAdamのモーメント推定が
    # リセットされてしまうため)。
    optimizer = torch.optim.Adam(network.parameters(), lr=settings.learning_rate)
    if optimizer_checkpoint is not None:
        restore_optimizer_state(
            optimizer, optimizer_checkpoint, learning_rate=settings.learning_rate
        )
        print(
            f"restored optimizer state from {optimizer_checkpoint} "
            f"with configured learning_rate={settings.learning_rate}"
        )

    # round間のモデル差だけを比較したいので、matchupはrunを通して固定する。
    evaluation_seed = settings.seed + 100_000
    matchups = build_fixed_matchups(
        settings.selfplay_mode,
        deck,
        opponent_deck_pool,
        settings.eval_games_per_round,
        evaluation_seed,
    )

    for round_num in range(start_round, settings.n_rounds + 1):
        print(
            f"=== round {round_num}/{settings.n_rounds}: self-play ({settings.workers} workers) ==="
        )
        worker_context = _SelfplayWorkerContext(
            deck=deck,
            opponent_deck_pool=opponent_deck_pool,
            mode=settings.selfplay_mode,
            gamma=settings.gamma,
            gae_lambda=settings.gae_lambda,
            seed=settings.seed,
            sampling_snapshot=settings.sampling_snapshot,
            feature_layout=model_config.FEATURE_LAYOUT,
        )
        all_samples, results = run_selfplay_round(
            algorithm="ppo",
            round_num=round_num,
            mode=settings.selfplay_mode,
            num_games=settings.games_per_round,
            num_workers=settings.workers,
            initializer=_init_selfplay_worker,
            initargs=(network.state_dict(), worker_context),
            task=_play_one_ppo_game,
            game_timeout_seconds=settings.game_timeout_seconds,
            round_timeout_seconds=settings.round_timeout_seconds,
            event_log_path=settings.event_log_path,
        )
        print(f"  round results: {results}  total_samples={len(all_samples)}")

        print(f"=== round {round_num}/{settings.n_rounds}: PPO update ===")
        train_one_round(network, optimizer, all_samples, settings, device)

        print(f"=== round {round_num}/{settings.n_rounds}: fixed-matchup eval vs random ===")
        vs_random = evaluate_networks_parallel(
            network,
            [("random", None)],
            opponent_deck_pool,
            matchups,
            agent_factory=make_ppo_eval_agent,
            sampling_snapshot=settings.sampling_snapshot,
            seed=evaluation_seed,
            num_workers=settings.workers,
            game_timeout_seconds=settings.game_timeout_seconds,
            round_timeout_seconds=settings.round_timeout_seconds,
            event_log_path=settings.event_log_path,
            event_context={
                "algorithm": "ppo",
                "round": round_num,
                "mode": settings.selfplay_mode,
                "stage": "random_eval",
            },
        )["random"]
        print(f"  vs random: {vs_random}")

        saved_path = checkpoint_path(settings.checkpoint_dir, settings.selfplay_mode, round_num)
        torch.save(network.state_dict(), saved_path)
        # optimizer(Adamのモーメント推定)は別ファイルに保存する。初期重み読み込みとの
        # 互換性を保つため、本体のcheckpointにはネットワークのstate_dictだけを入れる。
        torch.save(
            optimizer.state_dict(),
            optimizer_path(settings.checkpoint_dir, settings.selfplay_mode, round_num),
        )
        prune_checkpoints(
            settings.checkpoint_dir, settings.selfplay_mode, settings.keep_last_checkpoints
        )
        print(f"  saved checkpoint to {saved_path}")

    return network


def main(config_path: Path) -> int:
    """PPO trainerを1つのrun configで実行する。再起動時は最新roundから再開する。"""
    config = load_run_config(config_path)
    validate_algorithm(config, "ppo")
    settings = PpoSettings.from_run_config(config)

    configure_sampling_snapshot(settings.sampling_snapshot)
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config_path, settings.output_dir)

    resume = resolve_resume_point(
        settings.checkpoint_dir,
        settings.selfplay_mode,
        settings.run_name,
        config.model.initial_checkpoint,
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
        raise SystemExit("Usage: python ppo/train.py CONFIG_PATH")
    raise SystemExit(main(Path(sys.argv[1])))


__all__ = [
    "PpoSettings",
    "main",
    "make_ppo_eval_agent",
    "run_training_loop",
    "train_one_round",
]
