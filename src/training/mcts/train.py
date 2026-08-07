"""Determinized MCTSの自己対戦(selfplay.py)で集めたデータで、価値・方策ネットワークを学習する。

自己対戦→学習→(更新したネットワークで)自己対戦、を繰り返すAlphaZero的なループ。
`_run_selfplay_round`は1試合単位のタイムアウト/例外を捕まえてその試合だけ諦める
(cgエンジンはまれにネイティブクラッシュすることがあるため)。プロセスごと落ちた場合は統一training CLIが再起動し、保存済みの最新ラウンドの
チェックポイントから自動で再開する。

このmoduleは内部trainer。表向きの実行入口は `python -m training.cli`。
"""

import copy
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
    seed_opponent_deck_pool_cache,
)
from ..common.parallel_games import run_parallel_games  # noqa: E402
from ..common.run_config import (  # noqa: E402
    load_run_config,
    save_config_snapshot,
    validate_algorithm,
)
from ..common.selfplay_modes import SelfplayMode  # noqa: E402
from .selfplay import Sample, play_selfplay_game, run_determinized_mcts  # noqa: E402
from .training_state import (  # noqa: E402
    restore_training_state,
    save_training_state,
    saved_training_state_round,
)

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import Observation, to_observation_class  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from match_runner import evaluate_fixed_matchups, random_agent_factory  # noqa: E402

# 以下は関数単体利用時の既定値。通常実行ではmainがrun configの値で上書きする。
# ネットワーク構造だけはcommon/model_config.pyで一元管理する。
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # 学習(train_one_round)のみで使う

SEARCH_COUNT = 50  # 1手あたりのMCTSシミュレーション回数
NUM_SELFPLAY_WORKERS = max(1, (os.cpu_count() or 4) - 2)  # 自己対戦を並列実行するプロセス数
SELFPLAY_GAME_TIMEOUT_SECONDS = 300  # 1試合がこれを超えて終わらなければフリーズとみなして諦める
# ラウンド全体の上限。ワーカーが軒並みハングするとPool自体が壊れ、`.get(timeout=...)`が
# 機能しなくなることがある(1試合ごとのタイムアウトが効かず、逐次`num_games`回分の
# タイムアウトを律儀に待ち続けてしまう)。それを避けるため、ラウンド開始からこの時間を
# 超えたらPoolごと強制終了し、残りの試合は諦めて次のラウンドに進む。
SELFPLAY_ROUND_TIMEOUT_SECONDS = 1800
EPOCHS_PER_ROUND = 3
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SEED = 0
EVAL_GAMES_PER_ROUND = 32  # ラウンドごとのアリーナ戦・ランダム相手戦の試合数
GATING_WIN_RATE = 0.5  # 候補ネットワークを採用する最低勝率(全ゲーティング対戦の合算で判定)
CHECKPOINT_POOL_SIZE = 3  # ゲーティングで保持する過去に採用されたネットワークの数
GATING_POOL_SAMPLE = 2  # 1ラウンドで追加サンプリングする過去ネットワーク数
REPLAY_BUFFER_ROUNDS = 5  # 学習に使う直近ラウンド数(公式サンプルは常に1ラウンド分のみ)


def should_accept_candidate(current_best_win_rate: float, pooled_win_rate: float) -> bool:
    """現bestへの直接対戦と過去モデルを含む全体の両方で勝ち越した場合だけ採用する。"""
    return current_best_win_rate > GATING_WIN_RATE and pooled_win_rate > GATING_WIN_RATE


class SelfplayDataset(Dataset):
    """自己対戦で集めた`Sample`のリストをそのままDatasetにする。"""

    def __init__(self, samples: list[Sample]) -> None:
        """
        Args:
            samples: `selfplay.play_selfplay_game`が返す`Sample`のリスト(複数ゲーム分をまとめてよい)。
        """
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


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
    epochs: int,
) -> None:
    """1ラウンド分の自己対戦データで、`network`を数エポック学習する。

    Args:
        network: 更新対象のネットワーク(このラウンドの自己対戦にも使われたもの)。
        samples: このラウンドの自己対戦で集めた`Sample`のリスト。
        epochs: 学習エポック数。
    """
    if not samples:
        print("  no self-play samples collected; skipping MCTS update")
        return

    dataset = SelfplayDataset(samples)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate)

    network.to(DEVICE)
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(DEVICE)
    network.train()
    for epoch in range(epochs):
        total_policy_loss = 0.0
        total_value_loss = 0.0
        for (
            index_enc,
            value_enc,
            offset_enc,
            index_dec,
            value_dec,
            offset_dec,
            mask,
            policy_targets,
            value_labels,
        ) in loader:
            index_enc = index_enc.to(DEVICE)
            value_enc = value_enc.to(DEVICE)
            offset_enc = offset_enc.to(DEVICE)
            index_dec = index_dec.to(DEVICE)
            value_dec = value_dec.to(DEVICE)
            offset_dec = offset_dec.to(DEVICE)
            mask = mask.to(DEVICE)
            policy_targets = policy_targets.to(DEVICE)
            value_labels = value_labels.to(DEVICE)

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

            batch_size = value_labels.shape[0]
            total_policy_loss += policy_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size

        n = len(dataset)
        print(
            f"  epoch {epoch + 1}/{epochs}  policy_loss={total_policy_loss / n:.4f}  "
            f"value_loss={total_value_loss / n:.4f}"
        )
    network.to("cpu")
    network.eval()


def make_mcts_eval_agent(
    network: PolicyValueNet,
    our_deck: list[int],
    opponent_deck_pool: list[list[int]] | None = None,
):
    """`network`で、`match_runner.evaluate`用のagent()関数を作る。

    本番のagent()(推論エージェント)と同じ構成(自分のデッキは既知、
    相手はカード出現頻度から推測)。

    Args:
        network: 評価対象の`PolicyValueNet`(eval modeにしておくこと)。
        our_deck: こちらの60枚のデッキリスト。

    Returns:
        Callable[[dict], list[int]]: `match_runner.evaluate`に渡せるagent関数。
    """

    def agent(obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return our_deck

        select, _policy_target, _root_value, _actions = run_determinized_mcts(
            network, obs, our_deck, opponent_deck_pool, SEARCH_COUNT
        )
        return select

    return agent


_worker_network: PolicyValueNet | None = None
_worker_deck: list[int] | None = None
_worker_opponent_pool: list[list[int]] | None = None
_worker_search_count: int | None = None
_worker_mode: SelfplayMode | None = None
_worker_seed = SEED


def _init_selfplay_worker(
    state_dict: dict,
    deck: list[int],
    opponent_deck_pool: list[list[int]],
    search_count: int,
    mode: SelfplayMode,
    seed: int,
) -> None:
    """自己対戦ワーカープロセスの初期化(プロセスごとに1回だけ呼ばれる)。

    `cg`エンジンはプロセスグローバルな状態を持つため、対局はプロセスをまたいで並列化する
    必要がある(1プロセス内では同時に複数対局を進められない)。各試合は
    `run seed + game index`でPython/torchの乱数を再シードし、worker配置に依存しない。
    torchのスレッド内並列はプロセス間並列と競合するため無効化する。デッキプールは
    spawnワーカーへ注入し、snapshotをワーカーごとに再ロードしない。

    Args:
        state_dict: この自己対戦ラウンドで使う`PolicyValueNet`の`state_dict()`。
        deck: こちらの60枚のデッキリスト。
        opponent_deck_pool: 対戦相手として選ぶ実在デッキのリスト。
        search_count: 1手あたりのMCTSシミュレーション回数。
        mode: "asymmetric"、"mirror"、"generalist"。
        seed: run全体の乱数seed。
    """
    global _worker_network, _worker_deck, _worker_opponent_pool, _worker_search_count, _worker_mode
    global _worker_seed
    torch.set_num_threads(1)
    seed_opponent_deck_pool_cache(opponent_deck_pool)
    network = PolicyValueNet(
        D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER
    )
    network.load_state_dict(state_dict, assign=True)
    network.eval()
    _worker_network = network
    _worker_deck = deck
    _worker_opponent_pool = opponent_deck_pool
    _worker_search_count = search_count
    _worker_mode = mode
    _worker_seed = seed


def _play_one_selfplay_game(game_index: int) -> tuple[list[Sample], int]:
    """ワーカープロセス内で1試合分の自己対戦を行う(`_init_selfplay_worker`の初期化後に呼ばれる)。

    Args:
        game_index: run内で一意な試合番号。乱数seedにも使用する。

    Returns:
        tuple[list[Sample], int]: `selfplay.play_selfplay_game`の戻り値そのまま。
    """
    game_seed = _worker_seed + game_index
    random.seed(game_seed)
    torch.manual_seed(game_seed)
    return play_selfplay_game(
        _worker_network, _worker_deck, _worker_opponent_pool, _worker_search_count, _worker_mode
    )


def _run_selfplay_round(
    network: PolicyValueNet,
    deck: list[int],
    opponent_deck_pool: list[list[int]],
    search_count: int,
    num_games: int,
    mode: SelfplayMode,
    round_num: int,
) -> tuple[list[Sample], dict[int, int]]:
    """1ラウンド分の自己対戦を、`NUM_SELFPLAY_WORKERS`個のプロセスで並列実行する。

    Args:
        network: 自己対戦に使う`PolicyValueNet`(eval modeにしておくこと)。
        deck: こちらの60枚のデッキリスト。
        opponent_deck_pool: 対戦相手として選ぶ実在デッキのリスト。
        search_count: 1手あたりのMCTSシミュレーション回数。
        num_games: このラウンドで行う試合数。
        mode: "asymmetric"、"mirror"、"generalist"。

    Returns:
        tuple[list[Sample], dict[int, int]]: (全試合分のSample, 勝敗内訳{0: , 1: , 2(引分): })。
    """
    all_samples: list[Sample] = []
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
        initargs=(network.state_dict(), deck, opponent_deck_pool, search_count, mode, SEED),
        task=_play_one_selfplay_game,
        game_timeout_seconds=SELFPLAY_GAME_TIMEOUT_SECONDS,
        round_timeout_seconds=SELFPLAY_ROUND_TIMEOUT_SECONDS,
        on_result=on_result,
        on_failure=on_failure,
        event_log_path=CHECKPOINT_DIR.parent / "worker_events.jsonl",
        event_context={"algorithm": "mcts", "round": round_num, "mode": mode},
    )
    if skipped:
        print(f"  skipped {skipped}/{num_games} failed or timed-out games")
    return all_samples, results


def run_training_loop(
    deck: list[int],
    initial_checkpoint: Path | None = None,
    optimizer_checkpoint: Path | None = None,
    start_round: int = 1,
) -> PolicyValueNet:
    """自己対戦、replay学習、固定matchup arena評価を繰り返す。"""
    torch.manual_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    opponent_deck_pool = load_opponent_deck_pool()
    print(f"loaded {len(opponent_deck_pool)} distinct opponent decks for self-play")
    print(f"self-play mode: {SELFPLAY_MODE}")

    # replayの大きなI/Oはnative card masterを触るPolicyValueNet構築より先に済ませる。
    replay_buffer, checkpoint_pool = restore_training_state(
        CHECKPOINT_DIR,
        SELFPLAY_MODE,
        expected_round=start_round - 1,
        replay_buffer_rounds=REPLAY_BUFFER_ROUNDS,
    )

    best_network = PolicyValueNet(
        D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER
    )
    if initial_checkpoint is not None:
        best_network.load_state_dict(torch.load(initial_checkpoint, weights_only=True))
        print(f"loaded initial best_network weights from {initial_checkpoint}")
    best_network.eval()
    best_optimizer = torch.optim.Adam(best_network.parameters(), lr=LEARNING_RATE)
    if optimizer_checkpoint is not None and optimizer_checkpoint.exists():
        restore_optimizer_state(
            best_optimizer,
            optimizer_checkpoint,
            learning_rate=LEARNING_RATE,
        )
        print(
            f"restored optimizer state from {optimizer_checkpoint} "
            f"with configured learning_rate={LEARNING_RATE}"
        )

    def make_eval_factory(network: PolicyValueNet):
        return lambda selected_deck: make_mcts_eval_agent(
            network, selected_deck, opponent_deck_pool
        )

    for round_num in range(start_round, N_ROUNDS + 1):
        print(f"=== round {round_num}/{N_ROUNDS}: self-play ({NUM_SELFPLAY_WORKERS} workers) ===")
        all_samples, results = _run_selfplay_round(
            best_network,
            deck,
            opponent_deck_pool,
            SEARCH_COUNT,
            GAMES_PER_ROUND,
            SELFPLAY_MODE,
            round_num,
        )
        print(f"  round results: {results}  total_samples={len(all_samples)}")

        replay_buffer.append((round_num, all_samples))
        train_samples = [
            sample for _replay_round, round_samples in replay_buffer for sample in round_samples
        ]

        print(f"=== round {round_num}/{N_ROUNDS}: training ===")
        print(f"  training on {len(train_samples)} samples from last {len(replay_buffer)} round(s)")
        candidate_network = copy.deepcopy(best_network)
        candidate_optimizer = torch.optim.Adam(candidate_network.parameters(), lr=LEARNING_RATE)
        candidate_optimizer.load_state_dict(best_optimizer.state_dict())
        train_one_round(candidate_network, candidate_optimizer, train_samples, EPOCHS_PER_ROUND)

        # round間のモデル差だけを比較できるよう、matchupとPython seedを固定する。
        evaluation_seed = SEED + 100_000
        matchups = build_fixed_matchups(
            SELFPLAY_MODE,
            deck,
            opponent_deck_pool,
            EVAL_GAMES_PER_ROUND,
            evaluation_seed,
        )
        print(f"=== round {round_num}/{N_ROUNDS}: fixed-matchup arena ===")
        gating_opponents = [("current_best", best_network)]
        sampled_past = random.sample(checkpoint_pool, min(len(checkpoint_pool), GATING_POOL_SAMPLE))
        gating_opponents += [(f"past_{i}", net) for i, net in enumerate(sampled_past)]

        total_wins = 0
        total_games = 0
        current_best_win_rate = 0.0
        for name, opponent_network in gating_opponents:
            result = evaluate_fixed_matchups(
                make_eval_factory(candidate_network),
                make_eval_factory(opponent_network),
                matchups,
                seed=evaluation_seed,
            )
            print(f"  candidate vs {name}: {result}")
            total_wins += result["wins"]
            total_games += result["games"]
            if name == "current_best":
                current_best_win_rate = result["win_rate"]
        pooled_win_rate = total_wins / total_games
        accepted = should_accept_candidate(current_best_win_rate, pooled_win_rate)
        print(
            f"  pooled_win_rate={pooled_win_rate:.3f} ({total_wins}/{total_games})  accepted={accepted}"
        )

        if accepted:
            checkpoint_pool.append(copy.deepcopy(best_network))
            if len(checkpoint_pool) > CHECKPOINT_POOL_SIZE:
                checkpoint_pool.pop(0)
            best_network = candidate_network
            best_optimizer = candidate_optimizer

        print(f"=== round {round_num}/{N_ROUNDS}: fixed-matchup eval vs random ===")
        vs_random = evaluate_fixed_matchups(
            make_eval_factory(best_network),
            random_agent_factory,
            matchups,
            seed=evaluation_seed,
        )
        print(f"  vs random: {vs_random}")

        checkpoint_path = CHECKPOINT_DIR / f"{SELFPLAY_MODE}_round{round_num}.pt"
        torch.save(best_network.state_dict(), checkpoint_path)
        torch.save(
            best_optimizer.state_dict(),
            CHECKPOINT_DIR / f"{SELFPLAY_MODE}_round{round_num}_optimizer.pt",
        )
        save_training_state(
            CHECKPOINT_DIR,
            SELFPLAY_MODE,
            round_num,
            replay_buffer,
            checkpoint_pool,
        )
        print(f"  saved checkpoint and MCTS training state to {checkpoint_path}")

    return best_network


def main(config_path: Path) -> int:
    """MCTS trainerを1つのrun configで実行する。再起動時は最新roundから再開する。"""
    global CHECKPOINT_DIR, SELFPLAY_MODE, GAMES_PER_ROUND, N_ROUNDS
    global SEARCH_COUNT, NUM_SELFPLAY_WORKERS, SELFPLAY_GAME_TIMEOUT_SECONDS
    global SELFPLAY_ROUND_TIMEOUT_SECONDS, EPOCHS_PER_ROUND, BATCH_SIZE
    global LEARNING_RATE, SEED, EVAL_GAMES_PER_ROUND, GATING_WIN_RATE
    global CHECKPOINT_POOL_SIZE, GATING_POOL_SAMPLE, REPLAY_BUFFER_ROUNDS

    config = load_run_config(config_path)
    validate_algorithm(config, "mcts")
    settings = config.training
    CHECKPOINT_DIR = config.checkpoint_dir
    SELFPLAY_MODE = config.selfplay_mode
    GAMES_PER_ROUND = config.games_per_round
    N_ROUNDS = config.n_rounds
    SEARCH_COUNT = int(settings["search_count"])
    NUM_SELFPLAY_WORKERS = config.runtime.workers
    SELFPLAY_GAME_TIMEOUT_SECONDS = config.runtime.game_timeout_seconds
    SELFPLAY_ROUND_TIMEOUT_SECONDS = config.runtime.round_timeout_seconds
    EPOCHS_PER_ROUND = int(settings["epochs_per_round"])
    BATCH_SIZE = int(settings["batch_size"])
    LEARNING_RATE = float(settings["learning_rate"])
    SEED = config.runtime.seed
    EVAL_GAMES_PER_ROUND = int(settings["eval_games_per_round"])
    GATING_WIN_RATE = float(settings["gating_win_rate"])
    CHECKPOINT_POOL_SIZE = int(settings["checkpoint_pool_size"])
    GATING_POOL_SAMPLE = int(settings["gating_pool_sample"])
    REPLAY_BUFFER_ROUNDS = int(settings["replay_buffer_rounds"])

    configure_sampling_snapshot(config.sampling_snapshot)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config_path, config.output_dir)

    latest_round = latest_checkpoint_round(CHECKPOINT_DIR, SELFPLAY_MODE)
    state_round = saved_training_state_round(CHECKPOINT_DIR, SELFPLAY_MODE)
    if (
        latest_round is not None
        and state_round is not None
        and state_round < latest_round
        and (CHECKPOINT_DIR / f"{SELFPLAY_MODE}_round{state_round}.pt").exists()
    ):
        print(
            f"warning: round {latest_round} checkpoint has no matching training state; "
            f"rolling back resume to atomic round {state_round}"
        )
        latest_round = state_round
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
        raise SystemExit("Usage: python train.py CONFIG_PATH")
    raise SystemExit(main(Path(sys.argv[1])))
