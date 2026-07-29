"""相手の隠れ情報（手札・山札・サイド）を、Determinized MCTSの探索用にサンプリングする。

自分の情報は既知のデッキリスト(deck.csv)から、相手の情報は過去リプレイに含まれる
実際のデッキ一覧から作った「カード出現頻度」の重み付きプールからサンプリングする。
相手の本当のデッキは分からないので、この頻度は「メタ全体でよく使われるカード」の
近似でしかないが、固定ダミーカードで埋めるよりは現実的な仮定になる
（公式サンプルコードは相手の手札・山札を単一のダミーカードIDで埋めていた）。
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EPISODES_DIR = ROOT / "data" / "episodes"
SAMPLE_SUBMISSION_DIR = ROOT / "data" / "sample_submission" / "sample_submission"
sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

from cg.api import CardType, all_card_data  # noqa: E402

_opponent_pool: list[int] | None = None
_pokemon_card_ids: set[int] | None = None


def _load_opponent_pool() -> list[int]:
    """過去のリプレイepisodeから、両プレイヤーの実際の60枚デッキを集めて、
    カードIDの出現頻度に応じた重み付きプールを作る。

    各episodeの`steps[0][0]["visualize"]`には対戦開始時点の可視化ログがあり、その中の
    デッキ提示イベントの`action`が`[player0の60枚, player1の60枚]`という形になっている。
    これを全episode分集計することで、「メタ全体でどのカードがよく使われているか」の
    頻度分布を作れる。

    Returns:
        list[int]: 出現頻度に応じて重複を含むカードIDのリスト。
            `random.choices`の母集団としてそのまま使う。
    """
    counter: Counter[int] = Counter()
    for path in EPISODES_DIR.glob("*.json"):
        with path.open() as f:
            episode = json.load(f)
        for viz in episode["steps"][0][0].get("visualize") or []:
            action = viz.get("action")
            if (
                isinstance(action, list)
                and len(action) == 2
                and all(isinstance(deck, list) and len(deck) == 60 for deck in action)
            ):
                counter.update(action[0])
                counter.update(action[1])
    if not counter:
        raise RuntimeError(f"No deck data found under {EPISODES_DIR}")
    return list(counter.elements())


def _opponent_card_pool() -> list[int]:
    """出現頻度プールを遅延ロードしてキャッシュする（初回のみ全episodeを走査する）。

    Returns:
        list[int]: `_load_opponent_pool`が返すプール。
    """
    global _opponent_pool
    if _opponent_pool is None:
        _opponent_pool = _load_opponent_pool()
    return _opponent_pool


def sample_opponent_hidden(
    deck_count: int, hand_count: int, prize_count: int
) -> tuple[list[int], list[int], list[int]]:
    """相手の山札・手札・サイドを、過去リプレイのカード出現頻度からサンプリングする。

    実際の相手のデッキは分からないため、頻度プールから必要枚数の合計だけ
    重複を許してサンプリングし、シャッフルしてから山札/手札/サイドに割り振る。
    「どのカードがどの領域にあるか」を区別する情報は無いので、割り振りは完全にランダム。

    Args:
        deck_count: 相手の残り山札枚数(`PlayerState.deckCount`)。
        hand_count: 相手の手札枚数(`PlayerState.handCount`)。
        prize_count: 相手の残りサイド枚数(`len(PlayerState.prize)`)。

    Returns:
        tuple[list[int], list[int], list[int]]: (opponent_deck, opponent_hand, opponent_prize)。
            `search_begin`にそのまま渡せる形式。
    """
    pool = _opponent_card_pool()
    total = deck_count + hand_count + prize_count
    sampled = random.choices(pool, k=total)
    random.shuffle(sampled)
    return (
        sampled[:deck_count],
        sampled[deck_count : deck_count + hand_count],
        sampled[deck_count + hand_count :],
    )


def sample_own_hidden(
    own_deck: list[int], deck_count: int, prize_count: int
) -> tuple[list[int], list[int]]:
    """自分の山札・サイドを、既知のデッキリストからランダムに割り当てる。

    自分のデッキ構成(`own_deck`)は分かっているが、山札の並び順やサイドに
    埋まっているカードの具体的な内訳までは分からない。そのため、既知の60枚から
    ランダムに抜き出して割り当てる。手札・場・捨札に既にあるカードを厳密には
    除外していない簡易版（公式サンプルコードと同じ割り切り）。

    Args:
        own_deck: 自分の60枚のデッキリスト(カードID)。
        deck_count: 自分の残り山札枚数。
        prize_count: 自分の残りサイド枚数。

    Returns:
        tuple[list[int], list[int]]: (your_deck, your_prize)。`search_begin`にそのまま渡せる形式。
    """
    sampled = random.sample(own_deck, deck_count + prize_count)
    return sampled[:deck_count], sampled[deck_count:]


def _pokemon_ids() -> set[int]:
    """全カードマスタから、ポケモンカードのIDだけを遅延ロードしてキャッシュする。

    Returns:
        set[int]: `CardType.POKEMON`であるカードIDの集合。
    """
    global _pokemon_card_ids
    if _pokemon_card_ids is None:
        _pokemon_card_ids = {c.cardId for c in all_card_data() if c.cardType == CardType.POKEMON}
    return _pokemon_card_ids


def sample_opponent_active_guess() -> int:
    """相手のアクティブポケモンが伏せられている場合に、その正体として仮定するカードIDを選ぶ。

    `search_begin`はここにポケモンカード以外のIDを渡すとエラーになるため、
    出現頻度プールをポケモンカードのみに絞ってサンプリングする。

    Returns:
        int: 仮定するポケモンカードのID。
    """
    pool = _opponent_card_pool()
    pokemon_ids = _pokemon_ids()
    candidates = [card_id for card_id in pool if card_id in pokemon_ids]
    if not candidates:
        candidates = list(pokemon_ids)
    return random.choice(candidates)


def sample_full_hidden(
    known_deck: list[int], deck_count: int, hand_count: int, prize_count: int
) -> tuple[list[int], list[int], list[int]]:
    """既知のデッキリストから、山札・手札・サイドをまとめてランダムに割り当てる。

    自己対戦のように「相手のデッキ構成も自分と同じで既知」だが、実際にどのカードが
    手札のどこにあるかまでは分からない場合に使う。`sample_opponent_hidden`はデッキ構成
    自体が不明な、本番の対戦相手向けの版。

    Args:
        known_deck: 既知の60枚のデッキリスト(カードID)。
        deck_count: 残り山札枚数。
        hand_count: 手札枚数。
        prize_count: 残りサイド枚数。

    Returns:
        tuple[list[int], list[int], list[int]]: (deck, hand, prize)。
            `search_begin`にそのまま渡せる形式。
    """
    sampled = random.sample(known_deck, deck_count + hand_count + prize_count)
    return (
        sampled[:deck_count],
        sampled[deck_count : deck_count + hand_count],
        sampled[deck_count + hand_count :],
    )


def sample_active_guess_from_known_deck(known_deck: list[int]) -> int:
    """既知のデッキリストの中から、伏せられたアクティブポケモンの正体として仮定するIDを選ぶ。

    Args:
        known_deck: 既知の60枚のデッキリスト(カードID)。

    Returns:
        int: 仮定するポケモンカードのID。
    """
    pokemon_ids = _pokemon_ids()
    candidates = [card_id for card_id in known_deck if card_id in pokemon_ids]
    if not candidates:
        candidates = list(pokemon_ids)
    return random.choice(candidates)
