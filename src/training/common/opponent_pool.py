"""過去リプレイに含まれる実在デッキの一覧を集める。

自己対戦(MCTS/PPOいずれも)で、対戦相手のデッキをランダムに選ぶために使う。
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EPISODES_DIR = ROOT / "data" / "episodes"
_READ_RETRIES = 3  # まれに読み込み時にI/Oが化ける(ファイル自体は壊れていない)ことがあるため

_opponent_deck_pool: list[list[int]] | None = None


def _read_episode_json(path: Path) -> dict:
    """1エピソードのJSONを読む。まれな読み込みエラーは数回リトライする。

    Args:
        path: エピソードJSONファイルのパス。

    Returns:
        dict: パース済みのエピソードJSON。
    """
    last_error: Exception | None = None
    for _ in range(_READ_RETRIES):
        try:
            with path.open() as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            last_error = e
    raise RuntimeError(f"failed to read {path} after {_READ_RETRIES} retries") from last_error


def load_opponent_deck_pool() -> list[list[int]]:
    """リプレイ中の実在デッキから、重複を除いた60枚デッキの一覧を集める。

    Returns:
        list[list[int]]: 重複を除いた60枚デッキのリスト。
    """
    global _opponent_deck_pool
    if _opponent_deck_pool is not None:
        return _opponent_deck_pool

    seen: set[tuple[int, ...]] = set()
    decks: list[list[int]] = []
    for path in EPISODES_DIR.glob("*.json"):
        episode = _read_episode_json(path)
        for viz in episode["steps"][0][0].get("visualize") or []:
            action = viz.get("action")
            if (
                isinstance(action, list)
                and len(action) == 2
                and all(isinstance(deck, list) and len(deck) == 60 for deck in action)
            ):
                for deck in action:
                    key = tuple(sorted(deck))
                    if key not in seen:
                        seen.add(key)
                        decks.append(deck)
    if not decks:
        raise RuntimeError(f"No deck data found under {EPISODES_DIR}")
    _opponent_deck_pool = decks
    return _opponent_deck_pool
