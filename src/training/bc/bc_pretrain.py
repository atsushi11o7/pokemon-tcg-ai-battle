"""実リプレイのあらゆるアーキタイプの勝ち対局を抽出し、自己対戦のウォームスタート用に
`PolicyValueNet`を事前学習する。

`train_mcts.py`はランダム初期化から自己対戦のみで学習するが、自己対戦データ量が
小さい(30ラウンド×200試合)ため、実リプレイでの模倣学習を初期値として与えることで
基礎的な立ち回りを再発見する手間を省く狙い。generalist自己対戦(あらゆる実在デッキを
乗りこなす方策)のウォームスタート用途を想定し、特定のデッキに絞らず全アーキタイプの
勝ち対局を集める(各サンプルは、そのエピソードで実際に使われていたデッキでエンコードする)。

Usage:
    uv run python src/training/bc/bc_pretrain.py
"""

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]

# MCTSの学習ループ(`Sample`/`train_one_round`)をそのまま模倣学習にも使うため、
# `mcts/`にも依存する(BCはMCTSの学習ステップを1回だけ、教師データを差し替えて回す形)。
sys.path.insert(0, str(ROOT / "src" / "training" / "mcts"))
sys.path.insert(0, str(ROOT / "src" / "training" / "common"))
from model_config import (  # noqa: E402
    D_FEEDFORWARD,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS_DECODER,
    NUM_LAYERS_ENCODER,
)
from network import PolicyValueNet  # noqa: E402
from opponent_pool import _read_episode_json  # noqa: E402
from search import _enumerate_actions  # noqa: E402
from selfplay import Sample  # noqa: E402
from sparse_features import get_decoder_input, get_encoder_input  # noqa: E402
from train_mcts import train_one_round  # noqa: E402

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import to_observation_class  # noqa: E402

EPISODES_DIR = ROOT / "data" / "episodes"
BC_CHECKPOINT_PATH = ROOT / "outputs" / "mcts_checkpoints" / "bc_pretrained.pt"
EPOCHS = 5


def _winner_seat_and_deck(episode: dict) -> tuple[int, list[int]] | None:
    """エピソード最初のデッキ公開と結果から、勝った側の座席と実際のデッキを特定する。

    Args:
        episode: リプレイJSON全体。

    Returns:
        tuple[int, list[int]] | None: (勝った側の座席, その60枚デッキ)。
            引き分け/不明な場合はNone。
    """
    rewards = episode.get("rewards")
    if rewards is None:
        return None
    viz0 = episode["steps"][0][0].get("visualize") or []
    for viz in viz0:
        action = viz.get("action")
        if (
            isinstance(action, list)
            and len(action) == 2
            and all(isinstance(d, list) and len(d) == 60 for d in action)
        ):
            for seat, deck in enumerate(action):
                if rewards[seat] == 1:
                    return seat, deck
            return None  # 引き分け(rewardsに1が無い)
    return None


def extract_samples_from_episode(episode: dict) -> list[Sample]:
    """1エピソードから、勝った側が実際に使っていたデッキでの(局面, 実際の選択)ペアを抽出する。

    記録された選択がその時点の合法手の列挙(`search._enumerate_actions`)に含まれない
    決定点は、エンジンが選択をやり直させた際の無効な提出などとみなして除外する
    (`obs.current.yourIndex`が座席と一致することも確認し、視点のずれも防ぐ)。
    負け/引き分けの対局は模倣元として使わない(下手な手も学習してしまうため)。

    Args:
        episode: リプレイJSON全体。

    Returns:
        list[Sample]: labelまで埋めた学習サンプル(値は常に+1、探索によるTD補正は無し)。
    """
    found = _winner_seat_and_deck(episode)
    if found is None:
        return []
    seat, deck = found
    value = 1.0

    samples: list[Sample] = []
    for step in episode["steps"]:
        s = step[seat]
        obs_dict = s.get("observation")
        if not obs_dict or obs_dict.get("select") is None:
            continue
        obs = to_observation_class(obs_dict)
        if obs.current.yourIndex != seat:
            continue

        actions = _enumerate_actions(obs.select)
        taken = s.get("action")
        if taken not in actions:
            continue

        policy_target = [0.0] * len(actions)
        policy_target[actions.index(taken)] = 1.0

        encoder_sv = get_encoder_input(obs, deck)
        decoder_sv = get_decoder_input(obs, actions)
        sample = Sample(encoder_sv, decoder_sv, policy_target, value)
        sample.label = value
        samples.append(sample)

    return samples


def build_dataset() -> list[Sample]:
    """`data/episodes`配下の、あらゆるアーキタイプの勝ち対局から模倣学習データを集める。

    Returns:
        list[Sample]: 全エピソード分をまとめた学習サンプル。
    """
    all_samples: list[Sample] = []
    n_episodes = 0
    for path in sorted(EPISODES_DIR.glob("*.json")):
        try:
            episode = _read_episode_json(path)
        except RuntimeError as e:
            print(f"skipping unreadable episode {path}: {e}")
            continue
        samples = extract_samples_from_episode(episode)
        if samples:
            n_episodes += 1
            all_samples.extend(samples)
    print(f"extracted {len(all_samples)} samples from {n_episodes} episodes")
    return all_samples


def main() -> None:
    samples = build_dataset()
    random.shuffle(samples)

    network = PolicyValueNet(
        D_MODEL, NUM_HEADS, D_FEEDFORWARD, NUM_LAYERS_ENCODER, NUM_LAYERS_DECODER
    )
    train_one_round(network, samples, EPOCHS)

    BC_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), BC_CHECKPOINT_PATH)
    print(f"saved BC-pretrained checkpoint to {BC_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
