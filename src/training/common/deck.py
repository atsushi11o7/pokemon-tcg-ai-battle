"""本番で使う、こちらの60枚デッキ(decks/配下のcsv)を読み込む。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DECK_PATH = ROOT / "decks" / "mega_lucario_ex_pixiux_v62.csv"


def parse_deck_csv(path: Path) -> list[int]:
    """`decks/`配下のcsv(1行1カードID)からカードIDのリストを読み込む。

    Args:
        path: デッキcsvのパス。

    Returns:
        list[int]: ファイルに書かれていたカードID。
    """
    return [int(x) for x in path.read_text().split() if x.strip()]


def read_deck() -> list[int]:
    """`DECK_PATH`から、60行のカードIDを読み込む。

    Returns:
        list[int]: 60枚分のカードID。
    """
    return parse_deck_csv(DECK_PATH)
