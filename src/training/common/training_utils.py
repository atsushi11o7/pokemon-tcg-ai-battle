"""trainer(MCTS/PPO/BC)とspawnワーカーが共通で使う、学習まわりの小さなユーティリティ。"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset, Sampler

from .network import collate_encoder_decoder


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


def collate_samples(batch: list):
    """`Sample`のリストをバッチ化する。MCTS・BCで共通。

    `Sample.policy_target`は方策の教師分布。MCTSでは訪問回数の正規化、BCでは
    実際に選ばれた手のone-hotが入る。どちらも「合法手上の分布」なので同じ扱いでよい。

    Args:
        batch: `Sample`のリスト。`label`(価値の教師)が埋まっている必要がある。

    Returns:
        tuple: (index_enc, value_enc, offset_enc, index_dec, value_dec, offset_dec,
            mask, policy_targets, value_labels)。
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


def masked_policy_loss(
    scores: torch.Tensor, mask: torch.Tensor, policy_targets: torch.Tensor
) -> torch.Tensor:
    """パディングを除外した上で、方策の教師分布に対する交差エントロピーを返す。

    BCは価値だけを学ばせたいサンプル(敗者の局面)の`policy_target`を全ゼロにする。
    バッチ全体で平均すると、それらが分母にだけ入って方策の勾配が薄まるため、
    教師を持つサンプル数で割る。MCTSは全サンプルが教師を持つので挙動は変わらない。

    Args:
        scores: 形状`(batch, max_actions)`の生スコア。
        mask: 形状`(batch, max_actions)`のbool。パディング位置はFalse。
        policy_targets: 形状`(batch, max_actions)`の教師分布(パディング部分は0)。
    """
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    log_probs = functional.log_softmax(masked_scores, dim=-1)
    log_probs = log_probs.masked_fill(~mask, 0.0)  # -inf*0のnan化を防ぐ(教師も0なので影響なし)
    per_sample = -(policy_targets * log_probs).sum(dim=-1)
    supervised = policy_targets.sum(dim=-1) > 0
    n = int(supervised.sum().item())
    if n == 0:
        return per_sample.sum() * 0.0
    return per_sample.sum() / n


class LengthBucketSampler(Sampler):
    """行動数が近いサンプルを同じバッチにまとめ、デコーダのパディングを減らす。

    行動数は中央値5・平均7に対して最大64まで開く。無作為に256件集めると
    バッチ内の最大がほぼ常に64になり、実測でパディング率86%、学習1ステップは
    max_actions=8のときの約1.5倍かかっていた。

    バッチ内の順序と、バッチ自体の出現順はエポックごとにシャッフルするので、
    確率的勾配降下としての性質は保たれる。
    """

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        order = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(order)  # 同じ長さの中での並びを毎エポック変える
        order.sort(key=lambda i: self.lengths[i])
        batches = [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
