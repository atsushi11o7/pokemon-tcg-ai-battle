"""PPOによる自己対戦で、価値・方策ネットワークを学習する(MCTSを使わない軽量な代替経路)。

自己対戦→GAE計算→PPO更新、を繰り返す。1手あたりのネット評価が1回で済むため、
MCTS自己対戦(train_mcts.py、1手ごとにSEARCH_COUNT回評価)よりスループットが高い。
ネットワークはMCTS側と共通の`PolicyValueNet`をそのまま使うので、BC/MCTS自己対戦の
チェックポイントをwarm-startとして使うこともできる。

AlphaZero式のゲーティング(勝率が閾値を超えたら採用)は行わず、毎ラウンドそのまま
方策を更新し続ける(PPOはオンポリシー更新のため、探索で洗練した教師信号を前提とする
ゲーティングとは相性が良くない)。ラウンドごとのvs random評価は、悪化していないかの
モニタリング用に残す。

`_run_selfplay_round`は1試合単位のタイムアウト/例外を捕まえてその試合だけ諦める
(cgエンジンはまれにネイティブクラッシュすることがあるため)。プロセスごと落ちた場合は
`scripts/run_ppo_with_retry.sh`で再起動でき、`__main__`が保存済みの最新ラウンドの
チェックポイントから自動で再開する。

Usage:
    uv run python src/training/ppo/ppo_train.py
    (クラッシュしても自動で再起動・再開したい場合は scripts/run_ppo_with_retry.sh を使う)
"""

import multiprocessing
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "training" / "common"))
from model_config import (  # noqa: E402
    D_FEEDFORWARD,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS_DECODER,
    NUM_LAYERS_ENCODER,
)
from network import PolicyValueNet, collate_encoder_decoder  # noqa: E402
from opponent_pool import load_opponent_deck_pool  # noqa: E402
from ppo_selfplay import PPOSample, compute_gae, evaluate_policy, play_ppo_game  # noqa: E402
from run_config import config_path_from_argv, load_run_config, save_config_snapshot  # noqa: E402
from selfplay_modes import SelfplayMode  # noqa: E402

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import to_observation_class  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from match_runner import evaluate as match_evaluate  # noqa: E402
from match_runner import random_agent_factory  # noqa: E402

# デッキ・自己対戦モード・ウォームスタート元・回数・出力先は実験のたびに変わるため、
# `configs/*.yaml`で管理する(`__main__`参照)。ここではアーキテクチャ以外の、
# 基本的に変えないアルゴリズムのハイパーパラメータだけを定数として持つ。
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
EVAL_GAMES_PER_ROUND = 10
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
    network.to("cpu")
    network.eval()


_worker_network: PolicyValueNet | None = None
_worker_deck: list[int] | None = None
_worker_opponent_pool: list[list[int]] | None = None
_worker_mode: SelfplayMode | None = None


def _init_selfplay_worker(
    state_dict: dict,
    deck: list[int],
    opponent_deck_pool: list[list[int]],
    mode: SelfplayMode,
) -> None:
    """自己対戦ワーカープロセスの初期化(プロセスごとに1回だけ呼ばれる)。

    `cg`エンジンはプロセスグローバルな状態を持つため、対局はプロセスをまたいで並列化する
    必要がある(1プロセス内では同時に複数対局を進められない)。forkで複製された`random`の
    状態がワーカー間で重複しないよう、PIDでの再シードも行う(行動サンプリングは
    `torch.distributions.Categorical`経由でtorchの乱数を使うため、`torch.manual_seed`も
    合わせて行う)。トーチのスレッド内並列はプロセス間並列と競合するため無効化する。

    Args:
        state_dict: この自己対戦ラウンドで使う`PolicyValueNet`の`state_dict()`。
        deck: こちらの60枚のデッキリスト。
        opponent_deck_pool: 対戦相手として選ぶ実在デッキのリスト。
        mode: "asymmetric"、"mirror"、"generalist"(`ppo_selfplay.play_ppo_game`に渡す自己対戦モード)。
    """
    global _worker_network, _worker_deck, _worker_opponent_pool, _worker_mode
    torch.set_num_threads(1)
    random.seed(os.getpid())
    torch.manual_seed(os.getpid())
    network = PolicyValueNet(
        D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER
    )
    network.load_state_dict(state_dict)
    network.eval()
    _worker_network = network
    _worker_deck = deck
    _worker_opponent_pool = opponent_deck_pool
    _worker_mode = mode


def _play_one_ppo_game(_game_index: int) -> tuple[list[PPOSample], int]:
    """ワーカープロセス内で1試合分の自己対戦を行い、GAEまで計算する。

    `play_ppo_game`は座席ごとに分けたサンプル列を返す("mirror"/"generalist"では両座席分)ので、
    GAEはそれぞれの軌跡に対して独立に計算してから結合する。

    Args:
        _game_index: `pool.apply_async`が渡すインデックス(使わないが引数として必要)。

    Returns:
        tuple[list[PPOSample], int]: (advantage/return計算済みのPPOSampleのリスト
            (座席を結合済み), `state.result`)。
    """
    samples_by_seat, winner = play_ppo_game(
        _worker_network, _worker_deck, _worker_opponent_pool, _worker_mode
    )
    samples: list[PPOSample] = []
    for seat_samples in samples_by_seat:
        compute_gae(seat_samples, GAMMA, GAE_LAMBDA)
        samples.extend(seat_samples)
    return samples, winner


def _run_selfplay_round(
    network: PolicyValueNet,
    deck: list[int],
    opponent_deck_pool: list[list[int]],
    num_games: int,
    mode: SelfplayMode,
) -> tuple[list[PPOSample], dict[int, int]]:
    """1ラウンド分の自己対戦を、`NUM_SELFPLAY_WORKERS`個のプロセスで並列実行する。

    `mcts/train_mcts.py`の`_run_selfplay_round`と同じ構成(spawn・per-game timeout)。

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
    skipped = 0
    # ワーカーが軒並みハングするとPool自体が壊れ、`.get(timeout=...)`が個々の試合の
    # タイムアウト通りに機能しなくなることがある。それに備え、ラウンド全体にも
    # 締め切りを設け、超えたらPoolを強制終了して残りを諦める。
    round_deadline = time.monotonic() + SELFPLAY_ROUND_TIMEOUT_SECONDS
    mp_ctx = multiprocessing.get_context("spawn")
    with mp_ctx.Pool(
        processes=NUM_SELFPLAY_WORKERS,
        initializer=_init_selfplay_worker,
        initargs=(network.state_dict(), deck, opponent_deck_pool, mode),
    ) as pool:
        async_results = [pool.apply_async(_play_one_ppo_game, (i,)) for i in range(num_games)]
        for i, async_result in enumerate(async_results):
            remaining = round_deadline - time.monotonic()
            if remaining <= 0:
                skipped += num_games - i
                print(
                    f"  round exceeded {SELFPLAY_ROUND_TIMEOUT_SECONDS}s deadline, "
                    f"terminating pool and skipping remaining {num_games - i} games"
                )
                pool.terminate()
                break
            try:
                samples, winner = async_result.get(
                    timeout=min(SELFPLAY_GAME_TIMEOUT_SECONDS, remaining)
                )
                all_samples.extend(samples)
                results[winner] += 1
            except multiprocessing.TimeoutError:
                skipped += 1
                print(
                    f"  game {i + 1}/{num_games} timed out after "
                    f"{SELFPLAY_GAME_TIMEOUT_SECONDS}s, skipping"
                )
            except Exception as e:
                # 1試合だけの想定外の異常でラウンド全体を落とさない(タイムアウトと同じ扱い)。
                skipped += 1
                print(f"  game {i + 1}/{num_games} raised {e!r}, skipping")
            done = i + 1
            if done % 20 == 0 or done == num_games:
                print(f"  game {done}/{num_games}  results_so_far={results}  skipped={skipped}")
    return all_samples, results


def make_ppo_eval_agent(network: PolicyValueNet, our_deck: list[int]):
    """`network`で、`match_runner.evaluate`用のagent()関数を作る(貪欲方策)。

    探索を行わないPPOでは、評価時は方策のargmaxを取る(学習時のサンプリングと違い、
    最も自信のある行動を決定的に選ぶ)。

    Args:
        network: 評価対象の`PolicyValueNet`(eval modeにしておくこと)。
        our_deck: こちらの60枚のデッキリスト。

    Returns:
        Callable[[dict], list[int]]: `match_runner.evaluate`に渡せるagent関数。
    """

    def agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return our_deck

        actions, scores, _value, _encoder_sv, _decoder_sv = evaluate_policy(network, obs, our_deck)
        best_index = int(torch.argmax(scores).item())
        return actions[best_index]

    return agent


def make_random_deck_ppo_eval_agent(network: PolicyValueNet, opponent_deck_pool: list[list[int]]):
    """`network`で、対局のたびに実在デッキをランダムに選び直すagent()を作る(generalist評価用)。

    `make_ppo_eval_agent`は1つの`our_deck`に固定するが、generalistは特定のデッキを
    学習していないため、評価も自己対戦と同じ「毎回ランダムな実在デッキ」の分布で行う
    (`mcts/train_mcts.py`の`make_random_deck_mcts_eval_agent`と同じ考え方)。

    Args:
        network: 評価対象の`PolicyValueNet`(eval modeにしておくこと)。
        opponent_deck_pool: 対局のたびに選ぶ実在デッキのリスト。

    Returns:
        Callable[[dict], list[int]]: `match_runner.evaluate`に渡せるagent関数。
    """
    current_deck: list[int] = []

    def agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            current_deck[:] = random.choice(opponent_deck_pool)
            return current_deck

        actions, scores, _value, _encoder_sv, _decoder_sv = evaluate_policy(
            network, obs, current_deck
        )
        best_index = int(torch.argmax(scores).item())
        return actions[best_index]

    return agent


def run_training_loop(
    deck: list[int],
    warmstart_checkpoint: Path | None = None,
    optimizer_checkpoint: Path | None = None,
    start_round: int = 1,
) -> PolicyValueNet:
    """自己対戦→PPO更新を`N_ROUNDS`回繰り返すメインループ。

    Args:
        deck: 本番でも使う、こちらの60枚のデッキリスト。
        warmstart_checkpoint: 指定があれば、ランダム初期化の代わりにこのチェックポイントを
            初期値として使う(`bc_pretrain.py`/`train_mcts.py`が保存したものを想定)。
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
    # (train_mcts.pyのrun_training_loopと同じ理由)。
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    opponent_deck_pool = load_opponent_deck_pool()
    print(f"loaded {len(opponent_deck_pool)} distinct opponent decks for self-play")
    print(f"self-play mode: {SELFPLAY_MODE}")

    network = PolicyValueNet(
        D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER
    )
    if warmstart_checkpoint is not None:
        network.load_state_dict(torch.load(warmstart_checkpoint, weights_only=True))
        print(f"warm-started network from {warmstart_checkpoint}")
    network.eval()
    # ラウンドをまたいで同じoptimizerを使う(毎ラウンド作り直すとAdamのモーメント推定が
    # リセットされてしまうため)。
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    if optimizer_checkpoint is not None:
        optimizer.load_state_dict(torch.load(optimizer_checkpoint, weights_only=True))
        print(f"restored optimizer state from {optimizer_checkpoint}")

    for round_num in range(start_round, N_ROUNDS + 1):
        print(f"=== round {round_num}/{N_ROUNDS}: self-play ({NUM_SELFPLAY_WORKERS} workers) ===")
        all_samples, results = _run_selfplay_round(
            network, deck, opponent_deck_pool, GAMES_PER_ROUND, SELFPLAY_MODE
        )
        print(f"  round results: {results}  total_samples={len(all_samples)}")

        print(f"=== round {round_num}/{N_ROUNDS}: PPO update ===")
        train_one_round(network, optimizer, all_samples, EPOCHS_PER_ROUND)

        print(f"=== round {round_num}/{N_ROUNDS}: eval vs random (diverse deck) ===")
        # generalistは特定のデッキを学習していないため、自己対戦と同じ分布
        # (実在デッキから毎回ランダム)で評価する(mcts/train_mcts.pyのmake_eval_agentと同じ考え方)。
        eval_agent = (
            make_random_deck_ppo_eval_agent(network, opponent_deck_pool)
            if SELFPLAY_MODE == "generalist"
            else make_ppo_eval_agent(network, deck)
        )
        random_opponent_deck = random.choice(opponent_deck_pool)
        vs_random = match_evaluate(
            eval_agent, random_agent_factory(random_opponent_deck), n_episodes=EVAL_GAMES_PER_ROUND
        )
        print(f"  vs random (opponent deck size {len(set(random_opponent_deck))}): {vs_random}")

        checkpoint_path = CHECKPOINT_DIR / f"{SELFPLAY_MODE}_round{round_num}.pt"
        torch.save(network.state_dict(), checkpoint_path)
        # optimizer(Adamのモーメント推定)は別ファイルに保存する。warm-start読み込み
        # (`network.load_state_dict(torch.load(path))`)との互換性を保つため、
        # 本体のcheckpointにはネットワークのstate_dictだけを入れる。
        torch.save(
            optimizer.state_dict(),
            CHECKPOINT_DIR / f"{SELFPLAY_MODE}_round{round_num}_optimizer.pt",
        )
        print(f"  saved checkpoint to {checkpoint_path}")

    return network


def _latest_checkpoint_round() -> int | None:
    """`CHECKPOINT_DIR`から、`{SELFPLAY_MODE}_round{N}.pt`の最大ラウンド番号を探す。

    `_optimizer.pt`側は末尾が`.pt`で終わらない(`_round{N}_optimizer.pt`)ため、
    このパターンには一致しない。

    Returns:
        int | None: 見つかった最大のラウンド番号。1つも無ければNone。
    """
    pattern = re.compile(rf"{re.escape(SELFPLAY_MODE)}_round(\d+)\.pt$")
    rounds = [
        int(m.group(1))
        for p in CHECKPOINT_DIR.glob(f"{SELFPLAY_MODE}_round*.pt")
        if (m := pattern.match(p.name))
    ]
    return max(rounds) if rounds else None


if __name__ == "__main__":
    # デッキ・モード・ウォームスタート元・回数・出力先はconfigから読む
    # (`uv run python src/training/ppo/ppo_train.py [configのパス]`、省略時は既定のconfig)。
    config_path = config_path_from_argv(ROOT / "configs" / "ppo_generalist.yaml")
    config = load_run_config(config_path)
    CHECKPOINT_DIR = config.checkpoint_dir
    SELFPLAY_MODE = config.selfplay_mode
    GAMES_PER_ROUND = config.games_per_round
    N_ROUNDS = config.n_rounds
    INITIAL_WARMSTART_CHECKPOINT = config.initial_warmstart_checkpoint

    # どのconfigで作ったチェックポイント群かを後から追えるよう、使ったconfigを
    # checkpoint_dir直下にコピーしておく。
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config_path, CHECKPOINT_DIR)

    # 実行するたびに、保存済みの最新ラウンドから自動で再開する(クラッシュ後の
    # 再実行でも同じコマンドをそのまま使える)。何も無ければconfigの
    # `initial_warmstart_checkpoint`から新規に開始する。
    latest_round = _latest_checkpoint_round()
    if latest_round is not None:
        warmstart = CHECKPOINT_DIR / f"{SELFPLAY_MODE}_round{latest_round}.pt"
        optimizer_ckpt = CHECKPOINT_DIR / f"{SELFPLAY_MODE}_round{latest_round}_optimizer.pt"
        start_round = latest_round + 1
        print(f"resuming from round {latest_round} (start_round={start_round})")
    else:
        warmstart = (
            INITIAL_WARMSTART_CHECKPOINT
            if INITIAL_WARMSTART_CHECKPOINT and INITIAL_WARMSTART_CHECKPOINT.exists()
            else None
        )
        optimizer_ckpt = None
        start_round = 1
        print(f"no existing round checkpoint found; starting fresh from {warmstart}")
    run_training_loop(
        config.deck,
        warmstart_checkpoint=warmstart,
        optimizer_checkpoint=optimizer_ckpt,
        start_round=start_round,
    )
