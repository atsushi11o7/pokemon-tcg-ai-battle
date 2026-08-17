"""BC学習器の起動経路を検証する。

`resolve_resume_point`の呼び出し規約を間違えても、fresh startのsmoke testが
無ければ全テストが通ってしまう(実際に起動即クラッシュする状態で67件が成功していた)。
最小データで実際に1エポック回して、初回起動と再開の両方を通す。
"""

import json
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.bc.dataset import (  # noqa: E402
    SHARD_COUNT_CACHE,
    drop_from_page_cache,
    load_shard,
    shard_sample_counts,
)
from training.bc.train import (  # noqa: E402
    BcSettings,
    mask_loser_targets,
    run_training_loop,
)
from training.common.network import (  # noqa: E402
    NUM_WORDS_ENCODER,
    build_policy_value_net,
)
from training.common.sparse_features import (  # noqa: E402
    SparseVector,
    decoder_size,
    encoder_size,
)
from training.common.training_utils import collate_samples  # noqa: E402
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
            min_shard_day=None,
            val_day=None,
            loser_policy_weight=1.0,
            freeze_policy=False,
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
            min_shard_day=None,
            val_day=None,
            loser_policy_weight=1.0,
            freeze_policy=False,
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


class ShardCountCacheTest(unittest.TestCase):
    """件数キャッシュが、シャードを読み直さずに同じ答えを返すこと。

    学習率スケジュールの総ステップ数のためだけに全シャードを`torch.load`すると、
    実測で387シャード(68GB)に11分かかる。再起動のたびに払う固定費なので
    キャッシュするが、シャードが差し替わったのに古い件数を使うと
    スケジュールがずれる。どちらも例外にならないのでテストで固定する。
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.paths = []
        for index, size in enumerate((5, 3)):
            path = self.tmp / f"shard_2026081{index}_0000.pt"
            torch.save([_sample(3) for _ in range(size)], path)
            self.paths.append(path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_counts_are_correct_and_cached(self) -> None:
        self.assertEqual(shard_sample_counts(self.paths, self.tmp), [5, 3])
        self.assertTrue((self.tmp / SHARD_COUNT_CACHE).exists())

        # 2回目はシャードを開かないこと。開いたら落ちるようにして確かめる。
        def explode(*args, **kwargs):
            raise AssertionError("キャッシュがあるのにシャードを読み直している")

        with unittest.mock.patch("training.bc.dataset.torch.load", explode):
            self.assertEqual(shard_sample_counts(self.paths, self.tmp), [5, 3])

    def test_resized_shard_is_recounted(self) -> None:
        shard_sample_counts(self.paths, self.tmp)
        torch.save([_sample(3) for _ in range(9)], self.paths[0])
        self.assertEqual(shard_sample_counts(self.paths, self.tmp), [9, 3])

    def test_corrupt_cache_falls_back_to_counting(self) -> None:
        (self.tmp / SHARD_COUNT_CACHE).write_text("{not json", encoding="utf-8")
        self.assertEqual(shard_sample_counts(self.paths, self.tmp), [5, 3])


class HoldoutGuardTest(unittest.TestCase):
    """holdoutが全シャードを飲み込んだら止まること。

    素通りさせると`train_paths`が空になり、steps_per_epoch=0のまま
    1ステップも学習せずに「完了」する。例外にならないので気付けない。
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.shard_dir = self.tmp / "shards"
        self.shard_dir.mkdir()
        for index in range(3):
            torch.save(
                [_sample(3) for _ in range(2)], self.shard_dir / f"shard_2026081{index}_0000.pt"
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _settings(self, holdout: int) -> BcSettings:
        return BcSettings(
            run_name="guard",
            checkpoint_dir=self.tmp / "checkpoints",
            output_dir=self.tmp,
            shard_dir=self.shard_dir,
            val_shards=1,
            holdout_shards=holdout,
            min_shard_day=None,
            val_day=None,
            loser_policy_weight=1.0,
            freeze_policy=False,
            n_rounds=1,
            batch_size=2,
            learning_rate=1e-4,
            value_loss_coef=0.1,
            warmup_steps=1,
            seed=0,
            loader_workers=0,
            keep_last_checkpoints=1,
        )

    def test_holdout_covering_every_shard_is_rejected(self) -> None:
        (self.tmp / "checkpoints").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "no training shards"):
            run_training_loop(self._settings(holdout=3), None)

    def test_holdout_leaving_one_shard_is_accepted(self) -> None:
        (self.tmp / "checkpoints").mkdir(parents=True)
        run_training_loop(self._settings(holdout=2), None)
        self.assertTrue((self.tmp / "checkpoints" / "generalist_round1.pt").exists())


class PageCacheTest(unittest.TestCase):
    """シャードを読んだ後、ページキャッシュに残さないこと。

    WSL2のゲスト内ページキャッシュはWindows側の`vmmem`のメモリとして実体化する。
    1エポックで68GBを読み流すため、放置するとホストのメモリを占有し続ける。
    エポックごとにシャード順をシャッフルする以上キャッシュの再利用はほぼ無く、
    効果の無い占有でしかない。効かなくなっても例外は出ないのでテストで固定する。
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "shard_20260812_0000.pt"
        torch.save([_sample(3) for _ in range(200)], self.path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _cached_kb() -> int:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("Cached:"):
                return int(line.split()[1])
        return 0

    def test_load_shard_returns_the_samples(self) -> None:
        self.assertEqual(len(load_shard(self.path)), 200)

    def test_shard_is_evicted_after_loading(self) -> None:
        if not Path("/proc/meminfo").exists():
            self.skipTest("/proc/meminfo が無い環境")
        size_kb = self.path.stat().st_size // 1024
        if size_kb < 1024:
            self.skipTest("差分がノイズに埋もれる大きさ")
        load_shard(self.path)
        after_evict = self._cached_kb()
        with self.path.open("rb") as handle:
            handle.read()
        after_read = self._cached_kb()
        self.assertGreater(
            after_read - after_evict,
            size_kb // 2,
            "読み直してもCachedが増えない=そもそも計測できていない",
        )

    def test_missing_file_is_ignored(self) -> None:
        drop_from_page_cache(self.tmp / "does_not_exist.pt")  # 例外を出さない


class LoserPolicyWeightTest(unittest.TestCase):
    """敗者の局面の方策教師に、設定した重みだけが掛かること。

    シャードには敗者の手もone-hotで入っている。何倍で使うかを学習時に決めることで、
    抽出をやり直さずに 0.0(勝者のみ)〜1.0(同等)を比較できる。ここがずれると
    「勝者だけのはずが敗者も混ざる」といった取り違えが静かに起きる。
    """

    def setUp(self) -> None:
        # 1件目が勝者(label +1)、2件目が敗者(label -1)
        self.targets = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        self.labels = torch.tensor([1.0, -1.0])

    def test_weight_one_keeps_everything(self) -> None:
        out = mask_loser_targets(self.targets, self.labels, 1.0)
        self.assertTrue(torch.equal(out, self.targets))

    def test_weight_zero_silences_the_loser_only(self) -> None:
        out = mask_loser_targets(self.targets, self.labels, 0.0)
        self.assertTrue(torch.equal(out[0], self.targets[0]), "勝者側まで消している")
        self.assertEqual(out[1].sum().item(), 0.0)

    def test_partial_weight_scales_the_loser_only(self) -> None:
        out = mask_loser_targets(self.targets, self.labels, 0.3)
        self.assertTrue(torch.equal(out[0], self.targets[0]))
        self.assertAlmostEqual(out[1].max().item(), 0.3, places=6)

    def test_zeroed_loser_rows_produce_no_policy_gradient(self) -> None:
        """重み0の敗者行が、実際に勾配を出さないこと。"""
        from training.common.training_utils import masked_policy_loss

        scores = torch.randn(2, 3, requires_grad=True)
        mask = torch.ones(2, 3, dtype=torch.bool)
        masked_policy_loss(
            scores, mask, mask_loser_targets(self.targets, self.labels, 0.0)
        ).backward()
        self.assertGreater(scores.grad[0].abs().sum().item(), 0.0)
        self.assertEqual(scores.grad[1].abs().sum().item(), 0.0)


class BcFreezePolicyTest(unittest.TestCase):
    """`freeze_policy`のBC学習が、方策を1ビットも動かさずに価値だけ更新すること。

    ラダーで実績のある方策(1000点超)を保ったまま価値ヘッドだけ鍛えるための仕掛け。
    エンコーダごと凍結するので、方策と価値が共有する表現が価値側へ引っ張られる
    問題自体が起きない。壊れても例外にならず静かに劣化するので、テストで固定する。
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.shard_dir = self.tmp / "shards"
        self.shard_dir.mkdir()
        for day in ("20260810", "20260811", "20260812"):
            for index in range(2):
                samples = [_sample(3) for _ in range(6)]
                for i, s in enumerate(samples):
                    s.label = 1.0 if i % 2 else -1.0
                torch.save(samples, self.shard_dir / f"shard_{day}_{index:04d}.pt")
        torch.manual_seed(123)
        self.probe = collate_samples([_sample(4), _sample(3)])

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _settings(self, freeze: bool) -> BcSettings:
        return BcSettings(
            run_name="freeze",
            checkpoint_dir=self.tmp / ("ck_f" if freeze else "ck_n"),
            output_dir=self.tmp,
            shard_dir=self.shard_dir,
            val_shards=1,
            holdout_shards=2,
            min_shard_day=None,
            val_day=None,
            loser_policy_weight=0.0,
            freeze_policy=freeze,
            n_rounds=1,
            batch_size=3,
            learning_rate=1e-2,
            value_loss_coef=1.0,
            warmup_steps=1,
            seed=0,
            loader_workers=0,
            keep_last_checkpoints=1,
        )

    def _probe(self, network):
        # プローブ入力は固定する。毎回作り直すと、比べているのが重みの差なのか
        # 入力の差なのか分からなくなる。
        network.eval()
        with torch.no_grad():
            values, scores = network(*self.probe[:6], self.probe[6])
        return scores.clone(), values.clone()

    def test_policy_is_bit_identical_after_frozen_training(self) -> None:
        settings = self._settings(freeze=True)
        settings.checkpoint_dir.mkdir(parents=True)
        torch.manual_seed(0)
        before_net = build_policy_value_net()
        torch.save(before_net.state_dict(), self.tmp / "init.pt")
        before_scores, before_values = self._probe(before_net)

        after_net = run_training_loop(settings, self.tmp / "init.pt")
        after_scores, after_values = self._probe(after_net)

        # 厳密一致ではなくfloat32の丸め幅で見る。学習はGPUで行われ、凍結した重みでも
        # デバイス間の移動と演算順序の違いで最下位ビットが揺れる(実測1.2e-07)。
        drift = (before_scores - after_scores).abs().max().item()
        self.assertLess(drift, 1e-5, f"方策が動いた: 最大差 {drift}")
        self.assertGreater(
            (before_values - after_values).abs().max().item(), 1e-7, "価値が更新されていない"
        )

    def test_unfrozen_training_moves_the_policy(self) -> None:
        """対照。凍結しなければ方策も動く(テストが空振りしていないことの確認)。"""
        settings = self._settings(freeze=False)
        settings.checkpoint_dir.mkdir(parents=True)
        torch.manual_seed(0)
        net = build_policy_value_net()
        torch.save(net.state_dict(), self.tmp / "init2.pt")
        before_scores, _ = self._probe(net)
        after = run_training_loop(settings, self.tmp / "init2.pt")
        after_scores, _ = self._probe(after)
        self.assertFalse(torch.equal(before_scores, after_scores))


if __name__ == "__main__":
    unittest.main()


class ValDayTest(unittest.TestCase):
    """検証日を明示したとき、その日だけが学習から外れること。

    仕上げ工程では最新日まで学習に使いたいので、末尾ではなく日付で検証を選ぶ。
    ここがずれると、検証に使ったはずの日で学習してしまい指標が無意味になる。
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.shard_dir = self.tmp / "shards"
        self.shard_dir.mkdir()
        for day, count in (("20260812", 2), ("20260813", 1), ("20260814", 2)):
            for index in range(count):
                torch.save(
                    [_sample(3) for _ in range(4)],
                    self.shard_dir / f"shard_{day}_{index:04d}.pt",
                )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _settings(self, val_day):
        return BcSettings(
            run_name="valday",
            checkpoint_dir=self.tmp / "checkpoints",
            output_dir=self.tmp,
            shard_dir=self.shard_dir,
            val_shards=1,
            holdout_shards=1,
            min_shard_day=None,
            val_day=val_day,
            loser_policy_weight=0.0,
            freeze_policy=False,
            n_rounds=1,
            batch_size=2,
            learning_rate=1e-4,
            value_loss_coef=0.1,
            warmup_steps=1,
            seed=0,
            loader_workers=0,
            keep_last_checkpoints=1,
        )

    def test_the_named_day_is_held_out_and_the_newest_is_trained_on(self) -> None:
        settings = self._settings("20260813")
        settings.checkpoint_dir.mkdir(parents=True)
        run_training_loop(settings, None)
        counts = json.loads((self.tmp / "metrics.jsonl").read_text().splitlines()[0])
        # 8/13の1枚(4件)だけが検証に回り、残り4枚(16件)が学習に使われる
        self.assertEqual(counts["val_samples"], 4)
        self.assertEqual(counts["train_samples"], 16)

    def test_unknown_day_is_rejected(self) -> None:
        settings = self._settings("20990101")
        settings.checkpoint_dir.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "val_day"):
            run_training_loop(settings, None)


if __name__ == "__main__":
    unittest.main()
