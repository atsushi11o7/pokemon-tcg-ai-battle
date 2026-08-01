"""学習済みBehavior CloningモデルでMAIN/EVOLVE系の意思決定を行う、動作確認用のエージェント。

MAIN/EVOLVE以外の選択(セットアップ・サーチ等)は、学習データの対象外なのでランダムに選ぶ。
本番提出物として整えたものではなく、勝率を試しに測るための最小構成
（`src/evaluation/match_runner.py`での評価用）。
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import OPTION_DIM, STATE_DIM, encode_option, encode_state  # noqa: E402
from train_bc import HIDDEN_DIM, MODEL_PATH, PolicyNet  # noqa: E402

sys.path.insert(0, str(ROOT / "data" / "sample_submission" / "sample_submission"))
from cg.api import Observation, SelectData, SelectType, to_observation_class  # noqa: E402

# 現在採用中のデッキ(Issue #4で決定したもの、decks/配下で管理)を使う
DECK_PATH = ROOT / "decks" / "cynthias_garchomp_ex.csv"

_model = PolicyNet(STATE_DIM, OPTION_DIM, HIDDEN_DIM)
_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
_model.eval()


def read_deck() -> list[int]:
    """デッキ提出用に、decks/配下の共通deck.csv（60行のカードID）を読み込む。

    Returns:
        list[int]: 60枚分のカードID。
    """
    return [int(x) for x in DECK_PATH.read_text().split("\n") if x.strip()]


def choose_with_model(obs: Observation) -> list[int]:
    """学習済みモデルで、今の局面の選択肢の中から最もスコアが高いものを1つ選ぶ。

    Args:
        obs: `select.type`がMAIN/EVOLVEで、`minCount==maxCount==1`であることを
            呼び出し側で保証しておくこと（学習データがこの条件のみを対象にしているため）。

    Returns:
        list[int]: 選んだ選択肢のindexを1つだけ含むリスト。
    """
    state = torch.from_numpy(encode_state(obs)).unsqueeze(0)
    option_vecs = np.stack([encode_option(obs, opt) for opt in obs.select.option])
    options = torch.from_numpy(option_vecs).unsqueeze(0)
    with torch.no_grad():
        scores = _model(state, options)[0]
    return [int(torch.argmax(scores).item())]


def fallback_random(sel: SelectData) -> list[int]:
    """学習対象外の選択(セットアップ・サーチ等)に対して、ランダムに選ぶ。

    Args:
        sel: 今回の選択肢一式。

    Returns:
        list[int]: `minCount`以上`maxCount`以下の個数だけランダムに選んだindexのリスト。
    """
    count = random.randint(sel.minCount, sel.maxCount)
    return random.sample(range(len(sel.option)), count)


def agent(obs_dict: dict) -> list[int]:
    """MAIN/EVOLVEの単一選択は学習済みモデルで、それ以外はランダムで応答する。

    Args:
        obs_dict: kaggle_environmentsから渡される観測（生dict）。

    Returns:
        list[int]: 選択した選択肢のindexのリスト。
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck()
    if (
        obs.select.type in (SelectType.MAIN, SelectType.EVOLVE)
        and obs.select.minCount == obs.select.maxCount == 1
    ):
        return choose_with_model(obs)
    return fallback_random(obs.select)
