import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from meta_build_registry import build_snapshot  # noqa: E402


def replay(episode_id: int, left: int, right: int) -> dict:
    return {
        "id": "uuid-not-the-numeric-episode-id",
        "info": {"EpisodeId": episode_id, "TeamNames": ["left", "right"]},
        "rewards": [1, -1],
        "steps": [[{"visualize": [{"action": [[left] * 60, [right] * 60]}]}]],
    }


class DeckRegistryTest(unittest.TestCase):
    def test_only_manifest_episodes_enter_sampling_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            replays = raw / "replays"
            replays.mkdir(parents=True)
            (raw / "episodes.json").write_text(
                json.dumps([{"id": 1, "createTime": "2026-08-06T00:00:00Z"}])
            )
            (replays / "episode-1-replay.json").write_text(json.dumps(replay(1, 1, 2)))
            (replays / "episode-2-replay.json").write_text(json.dumps(replay(2, 3, 4)))

            snapshot = build_snapshot(raw, Path(directory) / "snapshot.json", 0.60, 7.0)

            self.assertEqual(snapshot["replayCount"], 1)
            self.assertEqual(snapshot["deckCount"], 2)
            self.assertEqual({record["cards"][0] for record in snapshot["decks"]}, {1, 2})


if __name__ == "__main__":
    unittest.main()
