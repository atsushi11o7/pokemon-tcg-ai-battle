"""PPOによる自己対戦で、価値・方策ネットワークを学習する(MCTSを使わない軽量な代替経路)。

自己対戦→GAE計算→PPO更新、を繰り返す。1手あたりのネット評価が1回で済むため、
MCTS自己対戦(train_mcts.py、1手ごとにSEARCH_COUNT回評価)よりスループットが高い。
ネットワークはMCTS側と共通の`PolicyValueNet`をそのまま使うので、BC/MCTS自己対戦の
チェックポイントをwarm-startとして使うこともできる。

AlphaZero式のゲーティング(勝率が閾値を超えたら採用)は行わず、毎ラウンドそのまま
方策を更新し続ける(PPOはオンポリシー更新のため、探索で洗練した教師信号を前提とする
ゲーティングとは相性が良くない)。ラウンドごとのvs random評価は、悪化していないかの
モニタリング用に残す。

Usage:
    uv run python src/training/ppo/ppo_train.py
"""

import multiprocessing
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "training" / "common"))
sys.path.insert(0, str(ROOT / "src" / "training" / "mcts"))
from network import PolicyValueNet, SparseBatch  # noqa: E402
from opponent_pool import load_opponent_deck_pool  # noqa: E402
from ppo_selfplay import PPOSample, compute_gae, evaluate_policy, play_ppo_game  # noqa: E402
from train_mcts import (  # noqa: E402
    D_FEEDFORWARD,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS_DECODER,
    NUM_LAYERS_ENCODER,
    _random_agent_factory,
    read_deck,
)

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import to_observation_class  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from match_runner import evaluate as match_evaluate  # noqa: E402

CHECKPOINT_DIR = ROOT / "outputs" / "ppo_checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMES_PER_ROUND = 200
N_ROUNDS = 100  # MCTSと違い1局が軽い(探索無し)ため、ラウンド数を多めに取れる
NUM_SELFPLAY_WORKERS = max(1, (os.cpu_count() or 4) - 2)
SELFPLAY_GAME_TIMEOUT_SECONDS = 300  # 1試合がこれを超えて終わらなければフリーズとみなして諦める
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
    """可変長の行動群を、バッチ内の最大行動数までパディングし、疎ベクトルを連結する。

    `mcts/train_mcts.py`の`_collate`と同じ考え方(デコーダ側だけパディングが必要)に、
    PPO固有のフィールド(選んだ行動のindex/behavior policyのlog確率/advantage/return)
    を加えたもの。

    Args:
        batch: `PPOSample`のリスト。

    Returns:
        tuple: (index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec,
            mask, action_indices, old_log_probs, advantages, returns)。
    """
    max_actions = max(len(s.decoder_sv.offset) for s in batch)

    encoder_batch = SparseBatch()
    decoder_batch = SparseBatch()
    mask = torch.zeros(len(batch), max_actions, dtype=torch.bool)
    action_indices = torch.zeros(len(batch), dtype=torch.int64)
    old_log_probs = torch.zeros(len(batch), dtype=torch.float32)
    advantages = torch.zeros(len(batch), dtype=torch.float32)
    returns = torch.zeros(len(batch), dtype=torch.float32)

    for i, sample in enumerate(batch):
        encoder_batch.add(sample.encoder_sv)
        decoder_batch.add(sample.decoder_sv)
        n = len(sample.decoder_sv.offset)
        mask[i, :n] = True
        action_indices[i] = sample.action_index
        old_log_probs[i] = sample.old_log_prob
        advantages[i] = sample.advantage
        returns[i] = sample.return_
        for _ in range(max_actions - n):
            decoder_batch.add_empty_word()

    index_enc, value_enc, offset_enc = encoder_batch.to_tensors()
    index_dec, value_dec, offset_dec = decoder_batch.to_tensors()
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """クリップ付きサロゲート目的関数+価値損失+エントロピー項を計算する。

    Args:
        network: 更新対象の`PolicyValueNet`。
        batch: `_collate`が返すバッチ(1ミニバッチ分)。
        device: 計算に使うデバイス。

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            (合計損失, 方策損失, 価値損失, エントロピー)。いずれもスカラー。
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

    normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * normalized_advantages
    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * normalized_advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = functional.mse_loss(values.squeeze(-1), returns)

    loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy
    return loss, policy_loss, value_loss, entropy


def train_one_round(network: PolicyValueNet, samples: list[PPOSample], epochs: int) -> None:
    """1ラウンド分のロールアウトで、`network`をPPOで数エポック学習する。

    ロールアウト収集時の方策(behavior policy)に対するクリップ付き目的関数のため、
    同じデータを複数エポック(`epochs`)再利用できる(AlphaZero式のTD学習と異なり、
    PPOはオンポリシー更新をこの範囲内で複数回行うのが特徴)。

    Args:
        network: 更新対象のネットワーク(このラウンドの自己対戦にも使われたもの)。
        samples: このラウンドの自己対戦で集めた`PPOSample`のリスト(GAE計算済み)。
        epochs: 学習エポック数。
    """
    dataset = PPODataset(samples)
    network.to(DEVICE)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

    network.train()
    for epoch in range(epochs):
        loader = DataLoader(dataset, batch_size=MINIBATCH_SIZE, shuffle=True, collate_fn=_collate)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n = 0
        for batch in loader:
            loss, policy_loss, value_loss, entropy = _ppo_loss(network, batch, DEVICE)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            batch_size = batch[6].shape[0]  # mask
            total_policy_loss += policy_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size
            total_entropy += entropy.item() * batch_size
            n += batch_size

        print(
            f"  epoch {epoch + 1}/{epochs}  policy_loss={total_policy_loss / n:.4f}  "
            f"value_loss={total_value_loss / n:.4f}  entropy={total_entropy / n:.4f}"
        )
    network.to("cpu")
    network.eval()


_worker_network: PolicyValueNet | None = None
_worker_deck: list[int] | None = None
_worker_opponent_pool: list[list[int]] | None = None


def _init_selfplay_worker(
    state_dict: dict, deck: list[int], opponent_deck_pool: list[list[int]]
) -> None:
    """自己対戦ワーカープロセスの初期化(プロセスごとに1回だけ呼ばれる)。

    `cg`エンジンはプロセスグローバルな状態を持つため、対局はプロセスをまたいで並列化する
    必要がある(1プロセス内では同時に複数対局を進められない)。forkで複製された`random`の
    状態がワーカー間で重複しないよう、PIDでの再シードも行う。トーチのスレッド内並列は
    プロセス間並列と競合するため無効化する。

    Args:
        state_dict: この自己対戦ラウンドで使う`PolicyValueNet`の`state_dict()`。
        deck: こちらの60枚のデッキリスト。
        opponent_deck_pool: 対戦相手として選ぶ実在デッキのリスト。
    """
    global _worker_network, _worker_deck, _worker_opponent_pool
    torch.set_num_threads(1)
    random.seed(os.getpid())
    network = PolicyValueNet(
        D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER
    )
    network.load_state_dict(state_dict)
    network.eval()
    _worker_network = network
    _worker_deck = deck
    _worker_opponent_pool = opponent_deck_pool


def _play_one_ppo_game(_game_index: int) -> tuple[list[PPOSample], int]:
    """ワーカープロセス内で1試合分の自己対戦を行い、GAEまで計算する。

    Args:
        _game_index: `pool.apply_async`が渡すインデックス(使わないが引数として必要)。

    Returns:
        tuple[list[PPOSample], int]: `ppo_selfplay.play_ppo_game`の戻り値
            (advantage/return計算済み)。
    """
    samples, winner = play_ppo_game(_worker_network, _worker_deck, _worker_opponent_pool)
    compute_gae(samples, GAMMA, GAE_LAMBDA)
    return samples, winner


def _run_selfplay_round(
    network: PolicyValueNet,
    deck: list[int],
    opponent_deck_pool: list[list[int]],
    num_games: int,
) -> tuple[list[PPOSample], dict[int, int]]:
    """1ラウンド分の自己対戦を、`NUM_SELFPLAY_WORKERS`個のプロセスで並列実行する。

    `mcts/train_mcts.py`の`_run_selfplay_round`と同じ構成(spawn・per-game timeout)。

    Args:
        network: 自己対戦に使う`PolicyValueNet`(eval modeにしておくこと)。
        deck: こちらの60枚のデッキリスト。
        opponent_deck_pool: 対戦相手として選ぶ実在デッキのリスト。
        num_games: このラウンドで行う試合数。

    Returns:
        tuple[list[PPOSample], dict[int, int]]: (全試合分のPPOSample, 勝敗内訳)。
    """
    all_samples: list[PPOSample] = []
    results = {0: 0, 1: 0, 2: 0}
    timed_out = 0
    mp_ctx = multiprocessing.get_context("spawn")
    with mp_ctx.Pool(
        processes=NUM_SELFPLAY_WORKERS,
        initializer=_init_selfplay_worker,
        initargs=(network.state_dict(), deck, opponent_deck_pool),
    ) as pool:
        async_results = [pool.apply_async(_play_one_ppo_game, (i,)) for i in range(num_games)]
        for i, async_result in enumerate(async_results):
            try:
                samples, winner = async_result.get(timeout=SELFPLAY_GAME_TIMEOUT_SECONDS)
                all_samples.extend(samples)
                results[winner] += 1
            except multiprocessing.TimeoutError:
                timed_out += 1
                print(
                    f"  game {i + 1}/{num_games} timed out after "
                    f"{SELFPLAY_GAME_TIMEOUT_SECONDS}s, skipping"
                )
            done = i + 1
            if done % 20 == 0 or done == num_games:
                print(f"  game {done}/{num_games}  results_so_far={results}  timed_out={timed_out}")
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


def run_training_loop(deck: list[int], warmstart_checkpoint: Path | None = None) -> PolicyValueNet:
    """自己対戦→PPO更新を`N_ROUNDS`回繰り返すメインループ。

    Args:
        deck: 本番でも使う、こちらの60枚のデッキリスト。
        warmstart_checkpoint: 指定があれば、ランダム初期化の代わりにこのチェックポイントを
            初期値として使う(`bc_pretrain.py`/`train_mcts.py`が保存したものを想定)。

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

    network = PolicyValueNet(
        D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER
    )
    if warmstart_checkpoint is not None:
        network.load_state_dict(torch.load(warmstart_checkpoint, weights_only=True))
        print(f"warm-started network from {warmstart_checkpoint}")
    network.eval()

    for round_index in range(N_ROUNDS):
        print(
            f"=== round {round_index + 1}/{N_ROUNDS}: self-play ({NUM_SELFPLAY_WORKERS} workers) ==="
        )
        all_samples, results = _run_selfplay_round(
            network, deck, opponent_deck_pool, GAMES_PER_ROUND
        )
        print(f"  round results: {results}  total_samples={len(all_samples)}")

        print(f"=== round {round_index + 1}/{N_ROUNDS}: PPO update ===")
        train_one_round(network, all_samples, EPOCHS_PER_ROUND)

        print(f"=== round {round_index + 1}/{N_ROUNDS}: eval vs random (diverse deck) ===")
        eval_agent = make_ppo_eval_agent(network, deck)
        random_opponent_deck = random.choice(opponent_deck_pool)
        vs_random = match_evaluate(
            eval_agent, _random_agent_factory(random_opponent_deck), n_episodes=EVAL_GAMES_PER_ROUND
        )
        print(f"  vs random (opponent deck size {len(set(random_opponent_deck))}): {vs_random}")

        checkpoint_path = CHECKPOINT_DIR / f"round{round_index + 1}.pt"
        torch.save(network.state_dict(), checkpoint_path)
        print(f"  saved checkpoint to {checkpoint_path}")

    return network


if __name__ == "__main__":
    from bc_pretrain import BC_CHECKPOINT_PATH

    warmstart = BC_CHECKPOINT_PATH if BC_CHECKPOINT_PATH.exists() else None
    run_training_loop(read_deck(), warmstart_checkpoint=warmstart)
