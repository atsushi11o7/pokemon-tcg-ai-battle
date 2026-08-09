"""trainer(MCTS/PPO)とspawnワーカーが共通で使う、学習まわりの小さなユーティリティ。"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset


def training_device() -> torch.device:
    """学習(train_one_round)に使うデバイスを返す。

    module import時ではなく呼び出し時に判定する。trainerモジュールはspawnワーカーで
    毎回まるごと再importされるため、import時に`torch.cuda.is_available()`を評価すると
    ワーカーが使いもしないCUDA初期化を1プロセスにつき1回払うことになる。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_game(seed: int) -> None:
    """1試合分の乱数を再シードする。

    Python側(デッキ抽選・隠れ情報のサンプリング)とtorch側(方策からのサンプリング)の
    両方を揃えて振り直し、どのワーカーがその試合を引いても結果が変わらないようにする。
    """
    random.seed(seed)
    torch.manual_seed(seed)


def move_optimizer_state_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """optimizerのモーメント推定を`device`へ移す。

    checkpointから復元した状態は保存時のデバイスに乗っていることがある
    (例: GPUで保存 → CPU上のnetworkでoptimizerを作ってからload_state_dict)。
    network側のパラメータとデバイスを揃えないとstep()で失敗する。
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


class ListDataset(Dataset):
    """学習サンプルのリストをそのまま`DataLoader`へ渡すためのDataset。

    バッチ化は`collate_fn`側が行うため、ここでは要素をそのまま返すだけでよい。
    """

    def __init__(self, samples: list) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
