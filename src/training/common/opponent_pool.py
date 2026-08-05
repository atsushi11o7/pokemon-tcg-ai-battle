"""過去リプレイ・手動追加デッキ・外部データセットから、実在デッキの一覧とカード出現頻度を集める。

デッキ一覧は自己対戦の相手デッキ選択に、カード出現頻度はMCTS探索での隠れ情報推測
(`determinize.py`)に使う。`seed_opponent_pool_cache`はワーカープロセスへの再スキャン
回避用(下記関数のdocstring参照)。
"""

import json
from collections import Counter
from pathlib import Path

from deck import parse_deck_csv

ROOT = Path(__file__).resolve().parents[3]
EPISODES_DIR = ROOT / "data" / "episodes"
EXTERNAL_DECKS_PATH = ROOT / "data" / "external_decks" / "citerne_decks.json"
CURATED_DECKS_DIR = ROOT / "decks"
_READ_RETRIES = 3  # まれに読み込み時にI/Oが化ける(ファイル自体は壊れていない)ことがあるため

_opponent_deck_pool: list[list[int]] | None = None
_opponent_card_pool: list[int] | None = None


def _read_episode_json(path: Path) -> dict:
    """1エピソードのJSONを読む。まれな読み込みエラーは数回リトライする。

    リトライしても読めない場合、そのファイル自体が壊れている(一過性のI/O揺らぎではない)
    ため、`RuntimeError`を送出する。呼び出し側でファイル単位にスキップすること。

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


def _load_external_decks() -> list[list[int]]:
    """`EXTERNAL_DECKS_PATH`から、外部の実在デッキデータセットを読み込む。

    Kaggleで公開されている実在デッキリスト(citerneさんの
    "Ultimate Pokémon TCG Deck and Card Dataset")のうち、カード名がこちらの
    カードカタログ(`cg.api.all_card_data`)と完全一致し、合計60枚になる
    デッキだけを事前に抽出・変換したもの(抽出スクリプトはこのファイルには含まない)。
    リプレイ由来のプール(50件)と重複しない70件。

    Returns:
        list[list[int]]: 60枚デッキのリスト。ファイルが無ければ空リスト。
    """
    if not EXTERNAL_DECKS_PATH.exists():
        return []
    with EXTERNAL_DECKS_PATH.open() as f:
        return json.load(f)


def _load_curated_decks() -> list[list[int]]:
    """`decks/`配下の全csvファイルから、手動で追加した実在デッキを読み込む。

    特定のデッキ(自己対戦のプールに確実に含めたいもの等)を手動で補いたいときに使う。
    ファイル名は自由(中身の60枚で重複判定するため、ファイル名自体には意味を持たせない)。

    Returns:
        list[list[int]]: 60枚デッキのリスト。
    """
    decks: list[list[int]] = []
    for path in sorted(CURATED_DECKS_DIR.glob("*.csv")):
        deck = parse_deck_csv(path)
        if len(deck) == 60:
            decks.append(deck)
    return decks


def _scan_episodes() -> tuple[list[list[int]], list[int]]:
    """重複なしデッキ一覧とカード出現頻度プールを作る。

    デッキ一覧は`data/episodes`のリプレイ・`decks/`の手動追加デッキ・外部データセットの
    3つを合わせたもの(自己対戦の相手デッキ選択はここから一様ランダムに選ぶため、
    出現頻度は関係ない)。カード出現頻度プールは`data/episodes`由来のものだけを対象にする
    (手動追加・外部データセットのデッキは実際の使用頻度が分からないため、頻度プールに
    混ぜると「1回だけ出現した」という誤った重みになってしまう)。

    Returns:
        tuple[list[list[int]], list[int]]: (重複を除いた60枚デッキのリスト,
            出現頻度に応じて重複を含むカードIDのリスト)。
    """
    seen: set[tuple[int, ...]] = set()
    decks: list[list[int]] = []
    counter: Counter[int] = Counter()
    for path in EPISODES_DIR.glob("*.json"):
        try:
            episode = _read_episode_json(path)
        except RuntimeError as e:
            print(f"skipping unreadable episode {path}: {e}")
            continue
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
                    counter.update(deck)
    for deck in _load_curated_decks() + _load_external_decks():
        key = tuple(sorted(deck))
        if key not in seen:
            seen.add(key)
            decks.append(deck)
    if not decks:
        raise RuntimeError(f"No deck data found under {EPISODES_DIR}")
    return decks, list(counter.elements())


def load_opponent_deck_pool() -> list[list[int]]:
    """リプレイ・`decks/`の手動追加・外部データセットから、重複を除いた60枚デッキの一覧を集める。

    Returns:
        list[list[int]]: 重複を除いた60枚デッキのリスト。
    """
    global _opponent_deck_pool, _opponent_card_pool
    if _opponent_deck_pool is None:
        _opponent_deck_pool, _opponent_card_pool = _scan_episodes()
    return _opponent_deck_pool


def load_opponent_card_pool() -> list[int]:
    """リプレイ中の実在デッキから、カード出現頻度プールを集める(`determinize.py`用)。

    Returns:
        list[int]: 出現頻度に応じて重複を含むカードIDのリスト(`random.choices`の母集団)。
    """
    global _opponent_deck_pool, _opponent_card_pool
    if _opponent_card_pool is None:
        _opponent_deck_pool, _opponent_card_pool = _scan_episodes()
    return _opponent_card_pool


def seed_opponent_pool_cache(decks: list[list[int]], card_pool: list[int]) -> None:
    """ワーカープロセス初期化時に、メインプロセスで計算済みのプールをキャッシュへ直接セットする。

    "spawn"のワーカーはモジュールをまっさらな状態から再importするため、何もしないと
    ワーカーごとに`data/episodes`を再スキャンしてしまう。それを避けるための注入用。

    Args:
        decks: `load_opponent_deck_pool`が返すのと同じ形式のデッキ一覧。
        card_pool: `load_opponent_card_pool`が返すのと同じ形式のカード出現頻度プール。
    """
    global _opponent_deck_pool, _opponent_card_pool
    _opponent_deck_pool = decks
    _opponent_card_pool = card_pool
