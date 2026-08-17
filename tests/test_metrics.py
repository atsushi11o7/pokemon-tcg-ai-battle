import json
import tempfile
import unittest
from pathlib import Path

from training.common.metrics import append_round_metrics


class MetricsTest(unittest.TestCase):
    def test_appends_one_line_per_round(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sub" / "metrics.jsonl"
            append_round_metrics(p, 1, accepted=False, results={0: 5, 1: 3})
            append_round_metrics(p, 2, accepted=True, results={0: 4, 1: 4})
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            self.assertEqual([r["round"] for r in rows], [1, 2])
            self.assertEqual(rows[1]["accepted"], True)
            self.assertIn("timestamp", rows[0])

    def test_write_failure_does_not_raise(self):
        # 書き込めない場所でも学習は止めない
        append_round_metrics(Path("/proc/cannot/write/metrics.jsonl"), 1, x=1)
