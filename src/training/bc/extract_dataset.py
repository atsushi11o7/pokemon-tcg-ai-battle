"""上位リプレイのepisode JSONから、Behavior Cloning用の(状態, 選択肢, 正解index)を抽出する。

対象はMAIN/EVOLVE(SelectType)の、単一選択(minCount==maxCount==1)の意思決定に絞る。
これがゲームを一番左右する意思決定であり、ルールベースエージェントのchoose_main_or_evolveと
役割が対応する。

Usage:
    uv run python src/training/bc/extract_dataset.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import encode_option, encode_state  # noqa: E402

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "data/sample_submission/sample_submission")
)
from cg.api import SelectType, to_observation_class  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
EPISODES_DIR = ROOT / "data" / "episodes"
OUTPUT_PATH = ROOT / "outputs" / "bc_dataset.npz"

TARGET_SELECT_TYPES = (SelectType.MAIN, SelectType.EVOLVE)


def extract_examples_from_episode(episode: dict) -> list[tuple[np.ndarray, list[np.ndarray], int]]:
    """1試合分のepisode JSONから、学習に使える意思決定だけを抽出する。

    kaggle_environmentsのepisode JSONは、`episode["steps"][t][playerIndex]`が
    そのプレイヤーの`observation`(agent()に渡るのと同じ形のobs_dict)と、
    そのとき実際に取った`action`を保持している。対象を`TARGET_SELECT_TYPES`
    (MAIN/EVOLVE)かつ単一選択(minCount==maxCount==1)の局面に絞り、
    選択肢が1つしかない（学習する意味が無い）ものは除外する。

    Args:
        episode: kaggle_environmentsのepisode JSON(`json.load`した辞書)。

    Returns:
        list[tuple[np.ndarray, list[np.ndarray], int]]: (状態ベクトル, 各選択肢の
            ベクトルのリスト, 実際に選ばれた選択肢のindex)のタプルのリスト。
    """
    examples = []
    steps = episode["steps"]
    for step in steps:
        for player_state in step:
            obs_dict = player_state.get("observation")
            action = player_state.get("action")
            if not obs_dict or not action or obs_dict.get("select") is None:
                continue
            obs = to_observation_class(obs_dict)
            sel = obs.select
            if sel.type not in TARGET_SELECT_TYPES:
                continue
            if sel.minCount != 1 or sel.maxCount != 1 or len(action) != 1:
                continue
            if len(sel.option) < 2:
                continue  # 選択肢が1つしかないなら学習する意味が無い

            state_vec = encode_state(obs)
            option_vecs = [encode_option(obs, opt) for opt in sel.option]
            label = action[0]
            if not (0 <= label < len(sel.option)):
                continue
            examples.append((state_vec, option_vecs, label))
    return examples


def main() -> None:
    """`data/episodes/`配下の全episode JSONから例を集め、npz形式でまとめて保存する。

    複数試合ぶんの状態・選択肢は、選択肢数が試合/局面ごとに異なるため単純には
    積み上げられない。そのため`options`は全試合分の選択肢ベクトルを1本の配列に
    フラット化して保存し、`option_counts`（各局面の選択肢数）を別途持たせることで、
    後から`states[i]`に対応する選択肢群を`options[offset:offset+option_counts[i]]`
    として復元できるようにしている（`train_bc.BCDataset`で実際に復元している）。
    """
    episode_files = sorted(EPISODES_DIR.glob("*.json"))
    print(f"found {len(episode_files)} episode files")

    all_states = []
    all_options = []
    all_labels = []
    all_option_counts = []

    for path in episode_files:
        with path.open() as f:
            episode = json.load(f)
        examples = extract_examples_from_episode(episode)
        for state_vec, option_vecs, label in examples:
            all_states.append(state_vec)
            all_options.extend(option_vecs)
            all_option_counts.append(len(option_vecs))
            all_labels.append(label)

    print(f"extracted {len(all_labels)} examples from {len(episode_files)} episodes")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        states=np.stack(all_states),
        options=np.stack(all_options),
        option_counts=np.array(all_option_counts, dtype=np.int64),
        labels=np.array(all_labels, dtype=np.int64),
    )
    print(f"saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
