"""Determinized MCTSの自己対戦(selfplay.py)で集めたデータで、価値・方策ネットワークを学習する。

自己対戦→学習→(更新したネットワークで)自己対戦、を繰り返すAlphaZero的なループ。
`_run_selfplay_round`は1試合単位のタイムアウト/例外を捕まえてその試合だけ諦める
(cgエンジンはまれにネイティブクラッシュすることがあるため)。プロセスごと落ちた場合は
`scripts/run_mcts_with_retry.sh`で再起動でき、`__main__`が保存済みの最新ラウンドの
チェックポイントから自動で再開する。

Usage:
    uv run python src/training/mcts/train_mcts.py
    (クラッシュしても自動で再起動・再開したい場合は scripts/run_mcts_with_retry.sh を使う)
"""

import copy
import multiprocessing
import os
import random
import re
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "training" / "common"))
from determinize import search_begin_kwargs  # noqa: E402
from model_config import (  # noqa: E402
    D_FEEDFORWARD,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS_DECODER,
    NUM_LAYERS_ENCODER,
)
from network import PolicyValueNet, collate_encoder_decoder  # noqa: E402
from opponent_pool import (  # noqa: E402
    load_opponent_card_pool,
    load_opponent_deck_pool,
    seed_opponent_pool_cache,
)
from run_config import config_path_from_argv, load_run_config, save_config_snapshot  # noqa: E402
from search import run_mcts  # noqa: E402
from selfplay import Sample, make_eval_fn, play_selfplay_game  # noqa: E402
from selfplay_modes import SelfplayMode  # noqa: E402

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import (  # noqa: E402
    Observation,
    search_begin,
    search_end,
    to_observation_class,
)

sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from match_runner import evaluate as match_evaluate  # noqa: E402
from match_runner import random_agent_factory  # noqa: E402

# デッキ・自己対戦モード・ウォームスタート元・回数・出力先は実験のたびに変わるため、
# `configs/*.yaml`で管理する(`__main__`参照)。ここではアーキテクチャ以外の、
# 基本的に変えないアルゴリズムのハイパーパラメータだけを定数として持つ。
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
EVAL_GAMES_PER_ROUND = 10  # ラウンドごとのアリーナ戦・ランダム相手戦の試合数
GATING_WIN_RATE = 0.5  # 候補ネットワークを採用する最低勝率(全ゲーティング対戦の合算で判定)
CHECKPOINT_POOL_SIZE = 3  # ゲーティングで保持する過去に採用されたネットワークの数
GATING_POOL_SAMPLE = 2  # 1ラウンドで追加サンプリングする過去ネットワーク数
REPLAY_BUFFER_ROUNDS = 5  # 学習に使う直近ラウンド数(公式サンプルは常に1ラウンド分のみ)


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


def train_one_round(network: PolicyValueNet, samples: list[Sample], epochs: int) -> None:
    """1ラウンド分の自己対戦データで、`network`を数エポック学習する。

    Args:
        network: 更新対象のネットワーク(このラウンドの自己対戦にも使われたもの)。
        samples: このラウンドの自己対戦で集めた`Sample`のリスト。
        epochs: 学習エポック数。
    """
    dataset = SelfplayDataset(samples)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate)

    network.to(DEVICE)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

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


def make_mcts_eval_agent(network: PolicyValueNet, our_deck: list[int]):
    """`network`で、`match_runner.evaluate`用のagent()関数を作る。

    本番のagent()(`trained_agent.py`)と同じ構成(自分のデッキは既知、
    相手はカード出現頻度から推測)。

    Args:
        network: 評価対象の`PolicyValueNet`(eval modeにしておくこと)。
        our_deck: こちらの60枚のデッキリスト。

    Returns:
        Callable[[dict], list[int]]: `match_runner.evaluate`に渡せるagent関数。
    """
    eval_fn = make_eval_fn(network, [our_deck, our_deck])

    def agent(obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return our_deck

        root_state = search_begin(obs, **search_begin_kwargs(obs, our_deck))
        try:
            select, _policy_target, _root_value, _actions = run_mcts(
                root_state, obs.current.yourIndex, eval_fn, SEARCH_COUNT
            )
        finally:
            search_end()
        return select

    return agent


def make_random_deck_mcts_eval_agent(network: PolicyValueNet, opponent_deck_pool: list[list[int]]):
    """`network`で、対局のたびに実在デッキをランダムに選び直すagent()を作る(generalist評価用)。

    `make_mcts_eval_agent`は1つの`our_deck`に固定するが、generalistは特定のデッキを
    学習していないため、評価も自己対戦と同じ「毎回ランダムな実在デッキ」の分布で行う。
    `match_runner.evaluate`は1試合ごとに`agent(deck_request_obs)`を1回だけ呼ぶので、
    その都度選び直せば試合ごとに異なるデッキになる。

    Args:
        network: 評価対象の`PolicyValueNet`(eval modeにしておくこと)。
        opponent_deck_pool: 対局のたびに選ぶ実在デッキのリスト。

    Returns:
        Callable[[dict], list[int]]: `match_runner.evaluate`に渡せるagent関数。
    """
    current_deck: list[int] = []

    def agent(obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            current_deck[:] = random.choice(opponent_deck_pool)
            return current_deck

        eval_fn = make_eval_fn(network, [current_deck, current_deck])
        root_state = search_begin(obs, **search_begin_kwargs(obs, current_deck))
        try:
            select, _policy_target, _root_value, _actions = run_mcts(
                root_state, obs.current.yourIndex, eval_fn, SEARCH_COUNT
            )
        finally:
            search_end()
        return select

    return agent


_worker_network: PolicyValueNet | None = None
_worker_deck: list[int] | None = None
_worker_opponent_pool: list[list[int]] | None = None
_worker_search_count: int | None = None
_worker_mode: SelfplayMode | None = None


def _init_selfplay_worker(
    state_dict: dict,
    deck: list[int],
    opponent_deck_pool: list[list[int]],
    opponent_card_pool: list[int],
    search_count: int,
    mode: SelfplayMode,
) -> None:
    """自己対戦ワーカープロセスの初期化(プロセスごとに1回だけ呼ばれる)。

    `cg`エンジンはプロセスグローバルな状態を持つため、対局はプロセスをまたいで並列化する
    必要がある(1プロセス内では同時に複数対局を進められない)。forkで複製された`random`の
    状態がワーカー間で重複しないよう、PIDでの再シードも行う。トーチのスレッド内並列は
    プロセス間並列と競合するため無効化する。`opponent_card_pool`を`seed_opponent_pool_cache`で
    注入するのは、`determinize.py`が遅延ロードすると"spawn"のワーカーごとに
    `data/episodes`(2.7GB)を再スキャンしてしまうのを避けるため。

    Args:
        state_dict: この自己対戦ラウンドで使う`PolicyValueNet`の`state_dict()`。
        deck: こちらの60枚のデッキリスト。
        opponent_deck_pool: 対戦相手として選ぶ実在デッキのリスト。
        opponent_card_pool: MCTS探索の隠れ情報推測に使うカード出現頻度プール。
        search_count: 1手あたりのMCTSシミュレーション回数。
        mode: "asymmetric"、"mirror"、"generalist"(`selfplay.play_selfplay_game`に渡す自己対戦モード)。
    """
    global _worker_network, _worker_deck, _worker_opponent_pool, _worker_search_count, _worker_mode
    torch.set_num_threads(1)
    random.seed(os.getpid())
    seed_opponent_pool_cache(opponent_deck_pool, opponent_card_pool)
    network = PolicyValueNet(
        D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER
    )
    network.load_state_dict(state_dict)
    network.eval()
    _worker_network = network
    _worker_deck = deck
    _worker_opponent_pool = opponent_deck_pool
    _worker_search_count = search_count
    _worker_mode = mode


def _play_one_selfplay_game(_game_index: int) -> tuple[list[Sample], int]:
    """ワーカープロセス内で1試合分の自己対戦を行う(`_init_selfplay_worker`の初期化後に呼ばれる)。

    Args:
        _game_index: `pool.imap_unordered`が渡すインデックス(使わないが引数として必要)。

    Returns:
        tuple[list[Sample], int]: `selfplay.play_selfplay_game`の戻り値そのまま。
    """
    return play_selfplay_game(
        _worker_network, _worker_deck, _worker_opponent_pool, _worker_search_count, _worker_mode
    )


def _run_selfplay_round(
    network: PolicyValueNet,
    deck: list[int],
    opponent_deck_pool: list[list[int]],
    opponent_card_pool: list[int],
    search_count: int,
    num_games: int,
    mode: SelfplayMode,
) -> tuple[list[Sample], dict[int, int]]:
    """1ラウンド分の自己対戦を、`NUM_SELFPLAY_WORKERS`個のプロセスで並列実行する。

    Args:
        network: 自己対戦に使う`PolicyValueNet`(eval modeにしておくこと)。
        deck: こちらの60枚のデッキリスト。
        opponent_deck_pool: 対戦相手として選ぶ実在デッキのリスト。
        opponent_card_pool: MCTS探索の隠れ情報推測に使うカード出現頻度プール。
        search_count: 1手あたりのMCTSシミュレーション回数。
        num_games: このラウンドで行う試合数。
        mode: "asymmetric"、"mirror"、"generalist"。

    Returns:
        tuple[list[Sample], dict[int, int]]: (全試合分のSample, 勝敗内訳{0: , 1: , 2(引分): })。
    """
    all_samples: list[Sample] = []
    results = {0: 0, 1: 0, 2: 0}
    skipped = 0
    # ワーカーが軒並みハングするとPool自体が壊れ、`.get(timeout=...)`が個々の試合の
    # タイムアウト通りに機能しなくなることがある。それに備え、ラウンド全体にも
    # 締め切りを設け、超えたらPoolを強制終了して残りを諦める。
    round_deadline = time.monotonic() + SELFPLAY_ROUND_TIMEOUT_SECONDS
    # "fork"(デフォルト)だと、親プロセスで既にimport済みのtorchが持つ内部スレッドが
    # forkの瞬間にロックを保持していた場合、そのロックは子プロセスに引き継がれたまま
    # 二度と解放されずデッドロックしうる(PyTorch公式が警告している既知の問題)。
    # "spawn"は子プロセスをまっさらな状態から起動するため、この問題を避けられる。
    with multiprocessing.get_context("spawn").Pool(
        processes=NUM_SELFPLAY_WORKERS,
        initializer=_init_selfplay_worker,
        initargs=(
            network.state_dict(),
            deck,
            opponent_deck_pool,
            opponent_card_pool,
            search_count,
            mode,
        ),
    ) as pool:
        # imap_unorderedだと1局でもフリーズすると全体が永久に止まるため、試合ごとに
        # タイムアウトを設け、詰まった試合は諦めて先に進む。
        async_results = [pool.apply_async(_play_one_selfplay_game, (i,)) for i in range(num_games)]
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
                # ゲーム木のエッジケース等、1試合だけの異常でラウンド全体を落とさない
                # (タイムアウトと同じ扱いでスキップする)。
                skipped += 1
                print(f"  game {i + 1}/{num_games} raised {e!r}, skipping")
            done = i + 1
            if done % 20 == 0 or done == num_games:
                print(f"  game {done}/{num_games}  results_so_far={results}  skipped={skipped}")
    return all_samples, results


def run_training_loop(
    deck: list[int], warmstart_checkpoint: Path | None = None, start_round: int = 1
) -> PolicyValueNet:
    """自己対戦→学習→アリーナ評価を`N_ROUNDS`回繰り返すメインループ。

    各ラウンド、直近`REPLAY_BUFFER_ROUNDS`ラウンド分の自己対戦データ(リプレイバッファ)で
    候補ネットワークを学習する。`best_network`と過去チェックポイントの一部(`checkpoint_pool`)
    との対戦を合算した勝率が`GATING_WIN_RATE`を超えた場合のみ採用する
    (AlphaZeroのアリーナ/ゲーティング。個別の勝ち越しではなく合算勝率で判定するのは、
    相手が増えるほど合格率が乗算的に下がるのを避けるため)。
    ゲーティングは公平な比較のため両者とも`deck`を使うミラー戦だが、それだけだと
    自己対戦で学んだはずの多様な相手への対応力を検証できないため、ランダム相手の
    デッキは毎ラウンド`opponent_deck_pool`からランダムに選び、汎化の下限の目安とする。

    Args:
        deck: 本番でも使う、こちらの60枚のデッキリスト。
        warmstart_checkpoint: 指定があれば、ランダム初期化の代わりにこのチェックポイントを
            `best_network`の初期値として使う。
        start_round: 学習を開始するラウンド番号(1始まり)。クラッシュ後の再開時は
            `warmstart_checkpoint`を保存したラウンドの次番号を指定する。ただし
            `checkpoint_pool`(ゲーティング対戦相手)と`replay_buffer`(直近ラウンドの
            自己対戦データ)は保存していないため、再開時は空の状態から作り直される
            (`best_network`の重みだけが引き継がれる簡易的な再開)。

    Returns:
        PolicyValueNet: 最終ラウンド終了時点の`best_network`。
    """
    torch.manual_seed(SEED)
    # PolicyValueNet構築(encoder_size/decoder_size経由でAllCard/AllAttackを呼ぶ)の直後に
    # 大量のファイルI/Oを行うとネイティブ側でクラッシュするため、load_opponent_deck_pool()を先に呼ぶ。
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    opponent_deck_pool = load_opponent_deck_pool()
    opponent_card_pool = load_opponent_card_pool()  # 上と同じスキャン結果を共有するのでコスト0
    print(f"loaded {len(opponent_deck_pool)} distinct opponent decks for self-play")
    print(f"self-play mode: {SELFPLAY_MODE}")

    best_network = PolicyValueNet(
        D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER
    )
    if warmstart_checkpoint is not None:
        best_network.load_state_dict(torch.load(warmstart_checkpoint, weights_only=True))
        print(f"warm-started best_network from {warmstart_checkpoint}")
    best_network.eval()

    checkpoint_pool: list[
        PolicyValueNet
    ] = []  # 採用済みの過去ネットワーク(最大CHECKPOINT_POOL_SIZE件)
    replay_buffer: deque[list[Sample]] = deque(maxlen=REPLAY_BUFFER_ROUNDS)  # 直近ラウンドのSample

    def make_eval_agent(network: PolicyValueNet):
        """ゲーティング・floor-check用のagent()を、`SELFPLAY_MODE`に応じて作る。

        "generalist"は特定のデッキを学習していないため、自己対戦と同じ分布
        (実在デッキから毎回ランダム)で評価する。それ以外は本番と同じ固定`deck`で評価する。
        """
        if SELFPLAY_MODE == "generalist":
            return make_random_deck_mcts_eval_agent(network, opponent_deck_pool)
        return make_mcts_eval_agent(network, deck)

    for round_num in range(start_round, N_ROUNDS + 1):
        print(f"=== round {round_num}/{N_ROUNDS}: self-play ({NUM_SELFPLAY_WORKERS} workers) ===")
        all_samples, results = _run_selfplay_round(
            best_network,
            deck,
            opponent_deck_pool,
            opponent_card_pool,
            SEARCH_COUNT,
            GAMES_PER_ROUND,
            SELFPLAY_MODE,
        )
        print(f"  round results: {results}  total_samples={len(all_samples)}")

        replay_buffer.append(all_samples)
        train_samples = [sample for round_samples in replay_buffer for sample in round_samples]

        print(f"=== round {round_num}/{N_ROUNDS}: training ===")
        print(f"  training on {len(train_samples)} samples from last {len(replay_buffer)} round(s)")
        candidate_network = copy.deepcopy(best_network)
        train_one_round(candidate_network, train_samples, EPOCHS_PER_ROUND)

        print(f"=== round {round_num}/{N_ROUNDS}: arena vs best_network + checkpoint pool ===")
        candidate_agent = make_eval_agent(candidate_network)
        gating_opponents = [("current_best", best_network)]
        sampled_past = random.sample(checkpoint_pool, min(len(checkpoint_pool), GATING_POOL_SAMPLE))
        gating_opponents += [(f"past_{i}", net) for i, net in enumerate(sampled_past)]

        total_wins = 0
        total_games = 0
        for name, opponent_network in gating_opponents:
            opponent_agent = make_eval_agent(opponent_network)
            result = match_evaluate(
                candidate_agent, opponent_agent, n_episodes=EVAL_GAMES_PER_ROUND
            )
            print(f"  candidate vs {name}: {result}")
            total_wins += result["wins"]
            total_games += EVAL_GAMES_PER_ROUND
        pooled_win_rate = total_wins / total_games
        accepted = pooled_win_rate > GATING_WIN_RATE
        print(
            f"  pooled_win_rate={pooled_win_rate:.3f} ({total_wins}/{total_games})  accepted={accepted}"
        )

        if accepted:
            checkpoint_pool.append(copy.deepcopy(best_network))
            if len(checkpoint_pool) > CHECKPOINT_POOL_SIZE:
                checkpoint_pool.pop(0)
            best_network = candidate_network

        print(f"=== round {round_num}/{N_ROUNDS}: eval vs random (diverse deck) ===")
        eval_agent = make_eval_agent(best_network)
        random_opponent_deck = random.choice(opponent_deck_pool)
        vs_random = match_evaluate(
            eval_agent, random_agent_factory(random_opponent_deck), n_episodes=EVAL_GAMES_PER_ROUND
        )
        print(f"  vs random (opponent deck size {len(set(random_opponent_deck))}): {vs_random}")

        checkpoint_path = CHECKPOINT_DIR / f"{SELFPLAY_MODE}_round{round_num}.pt"
        torch.save(best_network.state_dict(), checkpoint_path)
        print(f"  saved checkpoint to {checkpoint_path}")

    return best_network


def _latest_checkpoint_round() -> int | None:
    """`CHECKPOINT_DIR`から、`{SELFPLAY_MODE}_round{N}.pt`の最大ラウンド番号を探す。

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
    # (`uv run python src/training/mcts/train_mcts.py [configのパス]`、省略時は既定のconfig)。
    config_path = config_path_from_argv(ROOT / "configs" / "mcts_mirror_lucario.yaml")
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
    # 再実行でも同じコマンドをそのまま使える。ただしcheckpoint_pool/replay_bufferは
    # 引き継がれない簡易的な再開、詳細はrun_training_loopのdocstring参照)。
    # 何も無ければconfigの`initial_warmstart_checkpoint`から新規に開始する。
    latest_round = _latest_checkpoint_round()
    if latest_round is not None:
        warmstart = CHECKPOINT_DIR / f"{SELFPLAY_MODE}_round{latest_round}.pt"
        start_round = latest_round + 1
        print(f"resuming from round {latest_round} (start_round={start_round})")
    else:
        warmstart = (
            INITIAL_WARMSTART_CHECKPOINT
            if INITIAL_WARMSTART_CHECKPOINT and INITIAL_WARMSTART_CHECKPOINT.exists()
            else None
        )
        start_round = 1
        print(f"no existing round checkpoint found; starting fresh from {warmstart}")
    run_training_loop(config.deck, warmstart_checkpoint=warmstart, start_round=start_round)
