"""PPOによる自己対戦で、価値・方策ネットワークを学習する(MCTSを使わない軽量な代替経路)。

自己対戦→GAE計算→PPO更新、を繰り返す。1手あたりのネット評価が1回で済むため、
MCTS自己対戦(mcts/train.py、1手ごとにSEARCH_COUNT回評価)よりスループットが高い。
ネットワークはMCTS側と共通の`PolicyValueNet`をそのまま使うので、既存PPO/MCTSの
チェックポイントを初期重みとして相互利用できる。

AlphaZero式のゲーティング(勝率が閾値を超えたら採用)は行わず、毎ラウンドそのまま
方策を更新し続ける(PPOはオンポリシー更新のため、探索で洗練した教師信号を前提とする
ゲーティングとは相性が良くない)。ラウンドごとのvs random評価は、悪化していないかの
モニタリング用に残す。

`_run_selfplay_round`は1試合単位のタイムアウト/例外を捕まえてその試合だけ諦める
(cgエンジンはまれにネイティブクラッシュすることがあるため)。プロセスごと落ちた場合は統一training CLIが再起動し、保存済みの最新ラウンドの
チェックポイントから自動で再開する。

このmoduleは内部trainer。表向きの実行入口は `python -m training.cli`。
"""

import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]

from ..common.checkpoints import latest_checkpoint_round, restore_optimizer_state  # noqa: E402
from ..common.evaluation_plan import build_fixed_matchups  # noqa: E402
from ..common.model_config import (  # noqa: E402
    D_FEEDFORWARD,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS_DECODER,
    NUM_LAYERS_ENCODER,
)
from ..common.network import PolicyValueNet, collate_encoder_decoder  # noqa: E402
from ..common.opponent_pool import (  # noqa: E402
    configure_sampling_snapshot,
    load_opponent_deck_pool,
)
from ..common.parallel_games import run_parallel_games  # noqa: E402
from ..common.run_config import (  # noqa: E402
    load_run_config,
    save_config_snapshot,
    validate_algorithm,
)
from ..common.selfplay_modes import (  # noqa: E402
    SelfplayMode,
    fixed_deck_seat_for_game,
)
from .selfplay import PPOSample, compute_gae, evaluate_policy, play_ppo_game  # noqa: E402

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import to_observation_class  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from match_runner import evaluate_fixed_matchups, random_agent_factory  # noqa: E402

# 以下は関数単体利用時の既定値。通常実行ではmainがrun configの値で上書きする。
# ネットワーク構造だけはcommon/model_config.pyで一元管理する。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_SELFPLAY_WORKERS = max(1, (os.cpu_count() or 4) - 2)
SELFPLAY_GAME_TIMEOUT_SECONDS = 300  # 1試合がこれを超えて終わらなければフリーズとみなして諦める
# ラウンド全体の上限。ワーカーが軒並みハングするとPool自体が壊れ、`.get(timeout=...)`が
# 機能しなくなることがある(1試合ごとのタイムアウトが効かず、逐次`num_games`回分の
# タイムアウトを律儀に待ち続けてしまう)。それを避けるため、ラウンド開始からこの時間を
# 超えたらPoolごと強制終了し、残りの試合は諦めて次のラウンドに進む。
SELFPLAY_ROUND_TIMEOUT_SECONDS = 1800
EPOCHS_PER_ROUND = 4  # 収集したロールアウトを複数エポック再利用する(PPOの特徴)
MINIBATCH_SIZE = 64
LEARNING_RATE = 3e-4
GAMMA = 1.0  # 1試合内で完結する決着のみが報酬なので、割引はほぼ効かせない
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 1.0
TARGET_KL = 0.02
EVAL_GAMES_PER_ROUND = 12
SEED = 0


class PPODataset(Dataset):
    """自己対戦で集めた`PPOSample`のリストをそのままDatasetにする。"""

    def __init__(self, samples: list[PPOSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PPOSample:
        return self.samples[idx]


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
    network: PolicyValueNet, batch, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """クリップ付きサロゲート目的関数+価値損失+エントロピー項を計算する。

    Args:
        network: 更新対象の`PolicyValueNet`。
        batch: `_collate`が返すバッチ(1ミニバッチ分)。
        device: 計算に使うデバイス。

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
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = functional.mse_loss(values.squeeze(-1), returns)

    loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy

    with torch.no_grad():
        log_ratio = new_log_probs - old_log_probs
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > CLIP_EPS).float().mean()

    return loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction


def train_one_round(
    network: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    samples: list[PPOSample],
    epochs: int,
) -> None:
    """1ラウンド分のロールアウトで、`network`をPPOで数エポック学習する。

    ロールアウト収集時の方策(behavior policy)に対するクリップ付き目的関数のため、
    同じデータを複数エポック(`epochs`)再利用できる(AlphaZero式のTD学習と異なり、
    PPOはオンポリシー更新をこの範囲内で複数回行うのが特徴)。advantageはミニバッチ分割前に
    ラウンド全体で一度だけ正規化する(ミニバッチごとの正規化だと、シャッフルのたびに
    同じサンプルの正規化後advantageが変わってしまうため)。

    Args:
        network: 更新対象のネットワーク(このラウンドの自己対戦にも使われたもの)。
        optimizer: ラウンドをまたいで保持する`Adam`(モーメント推定をリセットしないため)。
        samples: このラウンドの自己対戦で集めた`PPOSample`のリスト(GAE計算済み)。
        epochs: 学習エポック数。
    """
    if not samples:
        print("  no rollout samples collected; skipping PPO update")
        return

    advantage_tensor = torch.tensor([s.advantage for s in samples], dtype=torch.float32)
    advantage_mean = advantage_tensor.mean()
    advantage_std = advantage_tensor.std(unbiased=False)
    for sample in samples:
        sample.advantage = float((sample.advantage - advantage_mean) / (advantage_std + 1e-8))

    dataset = PPODataset(samples)
    network.to(DEVICE)
    # optimizer_checkpointから復元した状態(Adamのモーメント推定)が、保存時と異なる
    # デバイスに乗っていることがある(例: GPU上で保存 → 一度CPU上のnetworkで
    # optimizerを作ってからload_state_dict)。network側のパラメータと合わせておく。
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(DEVICE)

    network.train()
    for epoch in range(epochs):
        loader = DataLoader(dataset, batch_size=MINIBATCH_SIZE, shuffle=True, collate_fn=_collate)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        n = 0
        for batch in loader:
            loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction = _ppo_loss(
                network, batch, DEVICE
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), MAX_GRAD_NORM)
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
        if mean_kl > TARGET_KL:
            print(f"  early stopping PPO update: approx_kl={mean_kl:.4f} > {TARGET_KL:.4f}")
            break
    network.to("cpu")
    network.eval()


_worker_network: PolicyValueNet | None = None
_worker_deck: list[int] | None = None
_worker_opponent_pool: list[list[int]] | None = None
_worker_mode: SelfplayMode | None = None
_worker_gamma = GAMMA
_worker_gae_lambda = GAE_LAMBDA
_worker_seed = SEED


def _init_selfplay_worker(
    state_dict: dict,
    deck: list[int],
    opponent_deck_pool: list[list[int]],
    mode: SelfplayMode,
    gamma: float,
    gae_lambda: float,
    seed: int,
) -> None:
    """自己対戦ワーカープロセスの初期化(プロセスごとに1回だけ呼ばれる)。

    `cg`エンジンはプロセスグローバルな状態を持つため、対局はプロセスをまたいで並列化する
    必要がある(1プロセス内では同時に複数対局を進められない)。各試合は
    `run seed + game index`でPython/torchの乱数を再シードし、worker配置に依存しない。
    torchのスレッド内並列はプロセス間並列と競合するため無効化する。

    Args:
        state_dict: この自己対戦ラウンドで使う`PolicyValueNet`の`state_dict()`。
        deck: こちらの60枚のデッキリスト。
        opponent_deck_pool: 対戦相手として選ぶ実在デッキのリスト。
        mode: "asymmetric"、"mirror"、"generalist"。
        gamma: GAEに使う割引率。
        gae_lambda: GAEのlambda。
        seed: run全体の乱数seed。
    """
    global _worker_network, _worker_deck, _worker_opponent_pool, _worker_mode
    global _worker_gamma, _worker_gae_lambda, _worker_seed
    torch.set_num_threads(1)
    network = PolicyValueNet(
        D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER
    )
    network.load_state_dict(state_dict, assign=True)
    network.eval()
    _worker_network = network
    _worker_deck = deck
    _worker_opponent_pool = opponent_deck_pool
    _worker_mode = mode
    _worker_gamma = gamma
    _worker_gae_lambda = gae_lambda
    _worker_seed = seed


def _play_one_ppo_game(game_index: int) -> tuple[list[PPOSample], int]:
    """ワーカープロセス内で1試合分の自己対戦を行い、GAEまで計算する。

    `play_ppo_game`は座席ごとに分けたサンプル列を返す("mirror"/"generalist"では両座席分)ので、
    GAEはそれぞれの軌跡に対して独立に計算してから結合する。

    Args:
        game_index: run内で一意な試合番号。乱数seedにも使用する。

    Returns:
        tuple[list[PPOSample], int]: (advantage/return計算済みのPPOSampleのリスト
            (座席を結合済み), `state.result`)。
    """
    game_seed = _worker_seed + game_index
    random.seed(game_seed)
    torch.manual_seed(game_seed)
    fixed_deck_seat = fixed_deck_seat_for_game(_worker_mode, game_index)
    samples_by_seat, winner = play_ppo_game(
        _worker_network, _worker_deck, _worker_opponent_pool, _worker_mode, fixed_deck_seat
    )
    samples: list[PPOSample] = []
    for seat_samples in samples_by_seat:
        compute_gae(seat_samples, _worker_gamma, _worker_gae_lambda)
        samples.extend(seat_samples)
    return samples, winner


def _run_selfplay_round(
    network: PolicyValueNet,
    deck: list[int],
    opponent_deck_pool: list[list[int]],
    num_games: int,
    mode: SelfplayMode,
    round_num: int,
) -> tuple[list[PPOSample], dict[int, int]]:
    """1ラウンド分の自己対戦を、`NUM_SELFPLAY_WORKERS`個のプロセスで並列実行する。

    `mcts/train.py`の`_run_selfplay_round`と同じ構成(spawn・per-game timeout)。

    Args:
        network: 自己対戦に使う`PolicyValueNet`(eval modeにしておくこと)。
        deck: こちらの60枚のデッキリスト。
        opponent_deck_pool: 対戦相手として選ぶ実在デッキのリスト。
        num_games: このラウンドで行う試合数。
        mode: "asymmetric"、"mirror"、"generalist"。

    Returns:
        tuple[list[PPOSample], dict[int, int]]: (全試合分のPPOSample, 勝敗内訳)。
    """
    all_samples: list[PPOSample] = []
    results = {0: 0, 1: 0, 2: 0}
    processed = 0

    def on_result(_game_index: int, result) -> None:
        nonlocal processed
        samples, winner = result
        all_samples.extend(samples)
        results[winner] += 1
        processed += 1
        if processed % 20 == 0 or processed == num_games:
            print(f"  games processed={processed}/{num_games}  results_so_far={results}")

    def on_failure(game_index: int, reason: str) -> None:
        nonlocal processed
        processed += 1
        print(f"  game {game_index + 1}/{num_games} failed: {reason}")

    _completed, skipped = run_parallel_games(
        num_games=num_games,
        num_workers=NUM_SELFPLAY_WORKERS,
        initializer=_init_selfplay_worker,
        initargs=(network.state_dict(), deck, opponent_deck_pool, mode, GAMMA, GAE_LAMBDA, SEED),
        task=_play_one_ppo_game,
        game_timeout_seconds=SELFPLAY_GAME_TIMEOUT_SECONDS,
        round_timeout_seconds=SELFPLAY_ROUND_TIMEOUT_SECONDS,
        on_result=on_result,
        on_failure=on_failure,
        event_log_path=CHECKPOINT_DIR.parent / "worker_events.jsonl",
        event_context={"algorithm": "ppo", "round": round_num, "mode": mode},
    )
    if skipped:
        print(f"  skipped {skipped}/{num_games} failed or timed-out games")
    return all_samples, results


def make_ppo_eval_agent(network: PolicyValueNet, our_deck: list[int]):
    """`network`で、`match_runner`の固定matchup評価用のagent()関数を作る(貪欲方策)。

    探索を行わないPPOでは、評価時は方策のargmaxを取る(学習時のサンプリングと違い、
    最も自信のある行動を決定的に選ぶ)。

    Args:
        network: 評価対象の`PolicyValueNet`(eval modeにしておくこと)。
        our_deck: こちらの60枚のデッキリスト。

    Returns:
        Callable[[dict], list[int]]: `evaluate_fixed_matchups`に渡せるagent関数。
    """

    def agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return our_deck

        actions, scores, _value, _encoder_sv, _decoder_sv = evaluate_policy(network, obs, our_deck)
        best_index = int(torch.argmax(scores).item())
        return actions[best_index]

    return agent


def run_training_loop(
    deck: list[int],
    initial_checkpoint: Path | None = None,
    optimizer_checkpoint: Path | None = None,
    start_round: int = 1,
) -> PolicyValueNet:
    """自己対戦→PPO更新を`N_ROUNDS`回繰り返すメインループ。

    Args:
        deck: 本番でも使う、こちらの60枚のデッキリスト。
        initial_checkpoint: 指定があれば、ランダム初期化の代わりにこのチェックポイントを
            初期値として使う（共通`PolicyValueNet`のcheckpointを想定）。
        optimizer_checkpoint: 指定があれば、Adamのモーメント推定もこのチェックポイントから
            復元する(クラッシュ後に同じラウンドの続きから再開する用途。新規学習では`None`)。
        start_round: 学習を開始するラウンド番号(1始まり)。クラッシュ後の再開時は
            `optimizer_checkpoint`を保存したラウンドの次番号を指定する。

    Returns:
        PolicyValueNet: 最終ラウンド終了時点のネットワーク。
    """
    torch.manual_seed(SEED)
    # PolicyValueNet構築(AllCard/AllAttackを呼ぶ)の直後に大量のファイルI/Oを行うと
    # ネイティブ側でクラッシュするため、load_opponent_deck_pool()を先に呼ぶ
    # (mcts/train.pyのrun_training_loopと同じ理由)。
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    opponent_deck_pool = load_opponent_deck_pool()
    print(f"loaded {len(opponent_deck_pool)} distinct opponent decks for self-play")
    print(f"self-play mode: {SELFPLAY_MODE}")

    network = PolicyValueNet(
        D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER
    )
    if initial_checkpoint is not None:
        network.load_state_dict(torch.load(initial_checkpoint, weights_only=True))
        print(f"loaded initial network weights from {initial_checkpoint}")
    network.eval()
    # ラウンドをまたいで同じoptimizerを使う(毎ラウンド作り直すとAdamのモーメント推定が
    # リセットされてしまうため)。
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    if optimizer_checkpoint is not None:
        restore_optimizer_state(
            optimizer,
            optimizer_checkpoint,
            learning_rate=LEARNING_RATE,
        )
        print(
            f"restored optimizer state from {optimizer_checkpoint} "
            f"with configured learning_rate={LEARNING_RATE}"
        )

    for round_num in range(start_round, N_ROUNDS + 1):
        print(f"=== round {round_num}/{N_ROUNDS}: self-play ({NUM_SELFPLAY_WORKERS} workers) ===")
        all_samples, results = _run_selfplay_round(
            network, deck, opponent_deck_pool, GAMES_PER_ROUND, SELFPLAY_MODE, round_num
        )
        print(f"  round results: {results}  total_samples={len(all_samples)}")

        print(f"=== round {round_num}/{N_ROUNDS}: PPO update ===")
        train_one_round(network, optimizer, all_samples, EPOCHS_PER_ROUND)

        # round間のモデル差だけを比較できるよう、matchupとPython seedを固定する。
        evaluation_seed = SEED + 100_000
        matchups = build_fixed_matchups(
            SELFPLAY_MODE,
            deck,
            opponent_deck_pool,
            EVAL_GAMES_PER_ROUND,
            evaluation_seed,
        )
        print(f"=== round {round_num}/{N_ROUNDS}: fixed-matchup eval vs random ===")
        vs_random = evaluate_fixed_matchups(
            lambda selected_deck: make_ppo_eval_agent(network, selected_deck),
            random_agent_factory,
            matchups,
            seed=evaluation_seed,
        )
        print(f"  vs random: {vs_random}")

        checkpoint_path = CHECKPOINT_DIR / f"{SELFPLAY_MODE}_round{round_num}.pt"
        torch.save(network.state_dict(), checkpoint_path)
        # optimizer(Adamのモーメント推定)は別ファイルに保存する。初期重み読み込み
        # (`network.load_state_dict(torch.load(path))`)との互換性を保つため、
        # 本体のcheckpointにはネットワークのstate_dictだけを入れる。
        torch.save(
            optimizer.state_dict(),
            CHECKPOINT_DIR / f"{SELFPLAY_MODE}_round{round_num}_optimizer.pt",
        )
        print(f"  saved checkpoint to {checkpoint_path}")

    return network


def main(config_path: Path) -> int:
    """PPO trainerを1つのrun configで実行する。再起動時は最新roundから再開する。"""
    global CHECKPOINT_DIR, SELFPLAY_MODE, GAMES_PER_ROUND, N_ROUNDS
    global NUM_SELFPLAY_WORKERS, SELFPLAY_GAME_TIMEOUT_SECONDS
    global SELFPLAY_ROUND_TIMEOUT_SECONDS, EPOCHS_PER_ROUND, MINIBATCH_SIZE
    global LEARNING_RATE, TARGET_KL, GAMMA, GAE_LAMBDA, CLIP_EPS
    global VALUE_LOSS_COEF, ENTROPY_COEF, MAX_GRAD_NORM
    global EVAL_GAMES_PER_ROUND, SEED

    config = load_run_config(config_path)
    validate_algorithm(config, "ppo")
    settings = config.training
    CHECKPOINT_DIR = config.checkpoint_dir
    SELFPLAY_MODE = config.selfplay_mode
    GAMES_PER_ROUND = config.games_per_round
    N_ROUNDS = config.n_rounds
    NUM_SELFPLAY_WORKERS = config.runtime.workers
    SELFPLAY_GAME_TIMEOUT_SECONDS = config.runtime.game_timeout_seconds
    SELFPLAY_ROUND_TIMEOUT_SECONDS = config.runtime.round_timeout_seconds
    EPOCHS_PER_ROUND = int(settings["epochs_per_round"])
    MINIBATCH_SIZE = int(settings["minibatch_size"])
    LEARNING_RATE = float(settings["learning_rate"])
    TARGET_KL = float(settings["target_kl"])
    GAMMA = float(settings["gamma"])
    GAE_LAMBDA = float(settings["gae_lambda"])
    CLIP_EPS = float(settings["clip_epsilon"])
    VALUE_LOSS_COEF = float(settings["value_loss_coef"])
    ENTROPY_COEF = float(settings["entropy_coef"])
    MAX_GRAD_NORM = float(settings["max_grad_norm"])
    EVAL_GAMES_PER_ROUND = int(settings["eval_games_per_round"])
    SEED = config.runtime.seed

    configure_sampling_snapshot(config.sampling_snapshot)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config_path, config.output_dir)

    latest_round = latest_checkpoint_round(CHECKPOINT_DIR, SELFPLAY_MODE)
    if latest_round is not None:
        initial_checkpoint = CHECKPOINT_DIR / f"{SELFPLAY_MODE}_round{latest_round}.pt"
        optimizer_candidate = CHECKPOINT_DIR / f"{SELFPLAY_MODE}_round{latest_round}_optimizer.pt"
        optimizer_ckpt = optimizer_candidate if optimizer_candidate.exists() else None
        start_round = latest_round + 1
        print(f"resuming {config.name} from round {latest_round} (start_round={start_round})")
    else:
        initial_checkpoint = config.model.initial_checkpoint
        if initial_checkpoint is not None and not initial_checkpoint.exists():
            raise FileNotFoundError(f"initial checkpoint does not exist: {initial_checkpoint}")
        optimizer_ckpt = None
        start_round = 1
        print(f"starting {config.name} from initial checkpoint {initial_checkpoint}")

    network = run_training_loop(
        config.deck,
        initial_checkpoint=initial_checkpoint,
        optimizer_checkpoint=optimizer_ckpt,
        start_round=start_round,
    )
    final_path = CHECKPOINT_DIR / "final.pt"
    torch.save(network.state_dict(), final_path)
    print(f"saved final checkpoint to {final_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python ppo/train.py CONFIG_PATH")
    raise SystemExit(main(Path(sys.argv[1])))
