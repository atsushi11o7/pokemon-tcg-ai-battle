"""実リプレイから今のアーキタイプの対局を抽出し、自己対戦のウォームスタート用に
`PolicyValueNet`を事前学習する。

`train_mcts.py`はランダム初期化から自己対戦のみで学習するが、自己対戦データ量が
小さい(30ラウンド×200試合)ため、実リプレイでの模倣学習を初期値として与えることで
基礎的な立ち回りを再発見する手間を省く狙い。

Usage:
    uv run python src/training/mcts/bc_pretrain.py
"""

import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src" / "training" / "common"))
from network import PolicyValueNet  # noqa: E402
from search import _enumerate_actions  # noqa: E402
from selfplay import Sample  # noqa: E402
from sparse_features import get_decoder_input, get_encoder_input  # noqa: E402
from train_mcts import (  # noqa: E402
    D_FEEDFORWARD,
    D_MODEL,
    NUM_HEADS,
    NUM_LAYERS_DECODER,
    NUM_LAYERS_ENCODER,
    read_deck,
    train_one_round,
)

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import to_observation_class  # noqa: E402

EPISODES_DIR = ROOT / "data" / "episodes"
ARCHETYPE_CARD_ID = 381  # Cynthia's Garchomp ex(このアーキタイプの識別に使う代表カード)
BC_CHECKPOINT_PATH = ROOT / "outputs" / "mcts_checkpoints" / "bc_pretrained.pt"
EPOCHS = 5


def _find_archetype_seat(episode: dict) -> int | None:
    """エピソード最初のデッキ公開から、このアーキタイプを使っている側の座席を探す。

    Args:
        episode: リプレイJSON全体。

    Returns:
        int | None: 見つかった座席(0か1)。見つからなければNone。
    """
    viz0 = episode["steps"][0][0].get("visualize") or []
    for viz in viz0:
        action = viz.get("action")
        if (
            isinstance(action, list)
            and len(action) == 2
            and all(isinstance(d, list) and len(d) == 60 for d in action)
        ):
            for i, d in enumerate(action):
                if ARCHETYPE_CARD_ID in d:
                    return i
            return None
    return None


def extract_samples_from_episode(episode: dict, our_deck: list[int]) -> list[Sample]:
    """1エピソードから、このアーキタイプを使っている側の(局面, 実際の選択)ペアを抽出する。

    記録された選択がその時点の合法手の列挙(`search._enumerate_actions`)に含まれない
    決定点は、エンジンが選択をやり直させた際の無効な提出などとみなして除外する
    (`obs.current.yourIndex`が座席と一致することも確認し、視点のずれも防ぐ)。

    Args:
        episode: リプレイJSON全体。
        our_deck: エンコーダに渡す「自分のデッキ」(今採用している60枚のカードID)。

    Returns:
        list[Sample]: labelまで埋めた学習サンプル(値は常に+1、探索によるTD補正は無し)。
    """
    seat = _find_archetype_seat(episode)
    if seat is None:
        return []

    rewards = episode.get("rewards")
    if rewards is None or rewards[seat] != 1:
        return []  # 負け/引き分けの対局は模倣元として使わない(下手な手も学習してしまうため)
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

        encoder_sv = get_encoder_input(obs, our_deck)
        decoder_sv = get_decoder_input(obs, actions)
        sample = Sample(encoder_sv, decoder_sv, policy_target, value)
        sample.label = value
        samples.append(sample)

    return samples


def build_dataset(our_deck: list[int]) -> list[Sample]:
    """`data/episodes`配下から、このアーキタイプの対局をすべて抽出してまとめる。

    Args:
        our_deck: エンコーダに渡す「自分のデッキ」。

    Returns:
        list[Sample]: 全エピソード分をまとめた学習サンプル。
    """
    all_samples: list[Sample] = []
    n_episodes = 0
    for path in sorted(EPISODES_DIR.glob("*.json")):
        with path.open() as f:
            episode = json.load(f)
        samples = extract_samples_from_episode(episode, our_deck)
        if samples:
            n_episodes += 1
            all_samples.extend(samples)
    print(f"extracted {len(all_samples)} samples from {n_episodes} episodes")
    return all_samples


def main() -> None:
    our_deck = read_deck()
    samples = build_dataset(our_deck)
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
