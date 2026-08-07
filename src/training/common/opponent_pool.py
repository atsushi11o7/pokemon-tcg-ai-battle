"""最新Kaggleメタのsampling snapshotから重み付きデッキプールを提供する。"""

from pathlib import Path

from .meta_deck_pool import (
    SAMPLING_SNAPSHOT_PATH,
    WeightedDeckPool,
    load_weighted_deck_pool,
)

_opponent_deck_pool: WeightedDeckPool | list[list[int]] | None = None
_sampling_snapshot_path = SAMPLING_SNAPSHOT_PATH


def configure_sampling_snapshot(path: Path) -> None:
    """次回loadで使用するsampling snapshotを指定し、既存cacheを破棄する。"""
    global _sampling_snapshot_path, _opponent_deck_pool
    _sampling_snapshot_path = path
    _opponent_deck_pool = None


def _ensure_loaded() -> None:
    global _opponent_deck_pool
    if _opponent_deck_pool is None:
        _opponent_deck_pool = load_weighted_deck_pool(_sampling_snapshot_path)


def load_opponent_deck_pool() -> WeightedDeckPool | list[list[int]]:
    _ensure_loaded()
    assert _opponent_deck_pool is not None
    return _opponent_deck_pool


def seed_opponent_deck_pool_cache(decks: list[list[int]]) -> None:
    """MCTS spawnワーカーへ計算済みデッキプールを注入する。"""
    global _opponent_deck_pool
    _opponent_deck_pool = decks
