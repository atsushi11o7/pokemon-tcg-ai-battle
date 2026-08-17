"""sampling snapshotから重み付きデッキプールを読み込み、プロセス内で使い回す。"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SAMPLING_SNAPSHOT_PATH = ROOT / "data" / "meta" / "derived" / "sampling_snapshot.json"


class WeightedDeckPool(list[list[int]]):
    """list互換性と、learner/opponent別の抽選分布を両立する。"""

    def __init__(self, snapshot: dict) -> None:
        records = snapshot.get("decks") or []
        super().__init__([record["cards"] for record in records])
        if not self:
            raise ValueError("sampling snapshot contains no decks")
        self.weights = {
            component: [float(record.get("weights", {}).get(component, 0.0)) for record in records]
            for component in ("meta", "coverage", "exploration", "hard")
        }
        self.mixtures = {
            "learner": snapshot.get("learnerMixture") or {"coverage": 1.0},
            "opponent": snapshot.get("opponentMixture") or {"meta": 1.0},
        }

    def sample(self, role: str) -> list[int]:
        """`role`("learner"/"opponent")の混合比に従ってデッキを1つ抽選する。"""
        mixture = self.mixtures.get(role)
        if mixture is None:
            raise ValueError(f"unknown deck sampling role: {role!r}")
        available = [
            component
            for component, mixture_weight in mixture.items()
            if mixture_weight > 0 and sum(self.weights.get(component, [])) > 0
        ]
        if not available:
            return random.choice(self)
        component = random.choices(
            available,
            weights=[float(mixture[component]) for component in available],
            k=1,
        )[0]
        return random.choices(self, weights=self.weights[component], k=1)[0]


def load_weighted_deck_pool(snapshot_path: Path = SAMPLING_SNAPSHOT_PATH) -> WeightedDeckPool:
    if not snapshot_path.exists():
        raise RuntimeError(
            f"sampling snapshot not found: {snapshot_path}; run meta_collect.py and "
            "meta_build_registry.py first"
        )
    with snapshot_path.open(encoding="utf-8") as file:
        snapshot = json.load(file)
    return WeightedDeckPool(snapshot)


_opponent_deck_pool: WeightedDeckPool | list[list[int]] | None = None
_sampling_snapshot_path = SAMPLING_SNAPSHOT_PATH


def configure_sampling_snapshot(path: Path) -> None:
    """次回loadで使用するsampling snapshotを指定し、既存cacheを破棄する。"""
    global _sampling_snapshot_path, _opponent_deck_pool
    _sampling_snapshot_path = path
    _opponent_deck_pool = None


def load_opponent_deck_pool() -> WeightedDeckPool | list[list[int]]:
    global _opponent_deck_pool
    if _opponent_deck_pool is None:
        _opponent_deck_pool = load_weighted_deck_pool(_sampling_snapshot_path)
    return _opponent_deck_pool


def seed_opponent_deck_pool_cache(decks: list[list[int]]) -> None:
    """MCTS spawnワーカーへ計算済みデッキプールを注入する。"""
    global _opponent_deck_pool
    _opponent_deck_pool = decks
