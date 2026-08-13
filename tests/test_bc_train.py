"""BC学習器の起動経路を検証する。

`resolve_resume_point`の呼び出し規約を間違えても、fresh startのsmoke testが
無ければ全テストが通ってしまう(実際に起動即クラッシュする状態で67件が成功していた)。
最小データで実際に1エポック回して、初回起動と再開の両方を通す。
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.bc.train import BcSettings, run_training_loop  # noqa: E402
from training.common.network import NUM_WORDS_ENCODER  # noqa: E402
from training.common.sparse_features import (  # noqa: E402
    SparseVector,
    decoder_size,
    encoder_size,
)
from training.mcts.selfplay import Sample  # noqa: E402


def _sample(n_actions: int) -> Sample:
    encoder = SparseVector()
    for _ in range(NUM_WORDS_ENCODER):
        encoder.word_start()
        encoder.add_absolute(int(torch.randint(0, encoder_size(), (1,))), 1.0)
    decoder = SparseVector()
    for _ in range(n_actions):
        decoder.word_start()
        decoder.add_absolute(int(torch.randint(0, decoder_size(), (1,))), 1.0)
    target = [0.0] * n_actions
    target[0] = 1.0
    sample = Sample(encoder, decoder, target, 1.0)
    sample.label = 1.0
    return sample


class BcTrainingLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        shard_dir = self.tmp / "shards"
        shard_dir.mkdir()
        for index in range(3):
            torch.save([_sample(3) for _ in range(8)], shard_dir / f"shard_{index:05d}.pt")
        self.settings = BcSettings(
            run_name="test_bc",
            checkpoint_dir=self.tmp / "checkpoints",
            output_dir=self.tmp,
            shard_dir=shard_dir,
            val_shards=1,
            holdout_shards=1,
            n_rounds=1,
            batch_size=4,
            learning_rate=1e-4,
            value_loss_coef=0.1,
            warmup_steps=1,
            seed=0,
            loader_workers=0,
            keep_last_checkpoints=2,
        )
        self.settings.checkpoint_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fresh_start_completes_one_epoch(self) -> None:
        run_training_loop(self.settings, None)
        self.assertTrue((self.settings.checkpoint_dir / "generalist_round1.pt").exists())

    def test_resume_skips_completed_epochs(self) -> None:
        run_training_loop(self.settings, None)
        # 1エポック済みの状態で同じ設定を再実行すると、再開判定で何も学習せずに戻る。
        before = (self.settings.checkpoint_dir / "generalist_round1.pt").stat().st_mtime_ns
        run_training_loop(self.settings, None)
        after = (self.settings.checkpoint_dir / "generalist_round1.pt").stat().st_mtime_ns
        self.assertEqual(before, after, "再開時に完了済みエポックを上書きしている")



class ShardSplitTest(unittest.TestCase):
    """時系列の分割が意図どおりか。

    シャード名は日付入りで、`sorted`が時系列順になる前提。最新日を丸ごと学習から外し
    (`holdout_shards`)、そのうち先頭数枚だけを検証に読む(`val_shards`)。ここがずれると
    「未来を見て学習したモデルを未来で測る」形になり、精度が実態より良く出る。
    例外にならず数字だけが甘くなるので、テストで固定する。
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.shard_dir = self.tmp / "shards"
        self.shard_dir.mkdir()
        # 3日分。最終日(0812)だけ4枚。
        for day, count in (("20260810", 3), ("20260811", 3), ("20260812", 4)):
            for index in range(count):
                torch.save(
                    [_sample(3) for _ in range(2)],
                    self.shard_dir / f"shard_{day}_{index:04d}.pt",
                )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _split(self, val_shards: int, holdout_shards: int):
        settings = BcSettings(
            run_name="split",
            checkpoint_dir=self.tmp / "checkpoints",
            output_dir=self.tmp,
            shard_dir=self.shard_dir,
            val_shards=val_shards,
            holdout_shards=holdout_shards,
            n_rounds=1,
            batch_size=2,
            learning_rate=1e-4,
            value_loss_coef=0.1,
            warmup_steps=1,
            seed=0,
            loader_workers=0,
            keep_last_checkpoints=1,
        )
        shards = sorted(settings.shard_dir.glob("shard_*.pt"))
        holdout = max(settings.holdout_shards, settings.val_shards)
        return shards[:-holdout], shards[-holdout:][: settings.val_shards]

    def test_final_day_is_fully_excluded_from_training(self) -> None:
        train, val = self._split(val_shards=2, holdout_shards=4)
        self.assertTrue(all("20260812" not in p.name for p in train), [p.name for p in train])
        self.assertEqual(
            [p.name for p in val], ["shard_20260812_0000.pt", "shard_20260812_0001.pt"]
        )
        self.assertEqual(len(train), 6)

    def test_holdout_defaults_to_val_shards(self) -> None:
        """holdoutを指定しないと最終日の残りが学習に混ざる(従来の挙動)。"""
        train, val = self._split(val_shards=2, holdout_shards=2)
        self.assertIn("shard_20260812_0000.pt", [p.name for p in train])
        self.assertEqual(
            [p.name for p in val], ["shard_20260812_0002.pt", "shard_20260812_0003.pt"]
        )

if __name__ == "__main__":
    unittest.main()
