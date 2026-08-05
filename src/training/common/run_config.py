"""実験のたびに変わる設定(デッキ・自己対戦モード・ウォームスタート元・回数・出力先)をyamlから読む。

アーキテクチャ定数(`model_config.py`)や、学習率のようにほぼ変えないアルゴリズムの
ハイパーパラメータは対象外(各学習スクリプトにPythonの定数として残す)。`configs/`配下の
yamlを追加・複製するだけで、コードを編集せずに実験(デッキ・モード・ウォームスタート元の
組み合わせ)を切り替えられるようにするのが狙い。
"""

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from deck import parse_deck_csv
from selfplay_modes import SelfplayMode

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class RunConfig:
    algorithm: Literal["mcts", "ppo"]
    deck_path: Path
    selfplay_mode: SelfplayMode
    initial_warmstart_checkpoint: Path | None
    games_per_round: int
    n_rounds: int
    checkpoint_dir: Path

    @property
    def deck(self) -> list[int]:
        return parse_deck_csv(self.deck_path)


def load_run_config(path: Path) -> RunConfig:
    """`path`のyamlを読み、`RunConfig`にする(yaml中のパスは`ROOT`基準の相対パスとして書く)。

    Args:
        path: 読み込むyamlファイルのパス。

    Returns:
        RunConfig: 実験設定。
    """
    with path.open() as f:
        raw = yaml.safe_load(f)
    warmstart = raw.get("initial_warmstart_checkpoint")
    return RunConfig(
        algorithm=raw["algorithm"],
        deck_path=ROOT / raw["deck_path"],
        selfplay_mode=raw["selfplay_mode"],
        initial_warmstart_checkpoint=(ROOT / warmstart) if warmstart else None,
        games_per_round=raw["games_per_round"],
        n_rounds=raw["n_rounds"],
        checkpoint_dir=ROOT / raw["checkpoint_dir"],
    )


def config_path_from_argv(default: Path) -> Path:
    """コマンドライン引数でconfigパスが指定されていればそれを、無ければ`default`を使う。

    Args:
        default: 引数が無いときに使うconfigパス。

    Returns:
        Path: 使用するconfigファイルのパス。
    """
    return Path(sys.argv[1]) if len(sys.argv) > 1 else default


def save_config_snapshot(config_path: Path, checkpoint_dir: Path) -> None:
    """使用したconfigのコピーを`checkpoint_dir`直下に残す。

    どの設定(デッキ・モード・ウォームスタート元)で作ったチェックポイント群かを、
    後からコードを読まずに`checkpoint_dir`だけ見て分かるようにするための記録用。

    Args:
        config_path: 実際に使ったconfigファイルのパス。
        checkpoint_dir: チェックポイントの保存先ディレクトリ(既に存在すること)。
    """
    shutil.copy(config_path, checkpoint_dir / "run_config.yaml")
