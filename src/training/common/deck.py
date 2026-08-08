"""`decks/`配下のデッキcsvを読み込む。"""

from pathlib import Path


def parse_deck_csv(path: Path) -> list[int]:
    """`decks/`配下のcsv(1行1カードID)からカードIDのリストを読み込む。

    Args:
        path: デッキcsvのパス。

    Returns:
        list[int]: ファイルに書かれていたカードID。
    """
    return [int(x) for x in path.read_text().split() if x.strip()]
