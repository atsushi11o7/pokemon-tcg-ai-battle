"""Run a single random self-play battle locally and dump each turn's Observation.

Usage:
    uv run python src/exploration/observe_battle.py
"""

import json
import random
import sys
from pathlib import Path

# submission/ 配下のcgはコミット対象外のため、ダウンロード済みのdata/側から直接importする
ROOT = Path(__file__).resolve().parents[2]
SAMPLE_SUBMISSION_DIR = ROOT / "data" / "sample_submission" / "sample_submission"
sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

# Observation/SelectData: obs_dict(生dict)の中身を表す型定義（dataclass）。
#   obs_dict = {"select": {...} | None, "logs": [...], "current": {...} | None}
#   - select: 今回選ぶ必要がある選択肢(SelectData)。Noneなら選択不要（本番のデッキ登録時のみ）
#   - logs:   前回の選択から今回までに起きたイベントの差分（全履歴ではない）
#   - current: 現在の盤面（自分視点。相手の手札等は隠された状態で入っている）
# to_observation_class: その生dictを上記dataclass(Observation)に変換するだけの関数。
#   データの中身は変わらず、obs_dict["select"]["type"] が obs.select.type のように
#   属性アクセスできるようになる（本番main.pyのagent()内で使われているのと同じ関数）
from cg.api import Observation, SelectData, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

OUTPUT_PATH = ROOT / "outputs" / "battle_log.jsonl"
MAX_STEPS = 2000


def read_deck(path: Path) -> list[int]:
    """deck.csv（60行のカードID）を読み込む。"""
    deck = [int(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(deck) == 60, f"deck must have 60 cards, got {len(deck)}"
    return deck


def choose_random(select: SelectData) -> list[int]:
    """obs.select の選択肢からランダムにindexを選ぶ（本番main.pyのagent()に相当する部分）。"""
    count = random.randint(select.minCount, select.maxCount)
    return random.sample(range(len(select.option)), count)


def main() -> None:
    deck = read_deck(SAMPLE_SUBMISSION_DIR / "deck.csv")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 対戦を開始する。両プレイヤーとも同じデッキを渡している（自己対戦）
    obs_dict, start_data = battle_start(deck, deck)
    if obs_dict is None:
        # デッキがルール違反（同名カード4枚超・ACE SPEC重複・たね ポケモン無し等）の場合ここに来る
        raise RuntimeError(
            f"battle_start failed: player={start_data.errorPlayer} type={start_data.errorType}"
        )

    step = 0
    try:
        with OUTPUT_PATH.open("w") as f:
            while step < MAX_STEPS:
                # このターンの生obs_dictをそのまま1行として保存する
                f.write(json.dumps(obs_dict, ensure_ascii=False) + "\n")
                obs: Observation = to_observation_class(obs_dict)

                # current.result は未決着なら-1、決着すると勝者側のplayerIndex（引き分けは2）になる
                if obs.current is not None and obs.current.result != -1:
                    print(f"Battle finished after {step} steps. result={obs.current.result}")
                    break
                if obs.select is None:
                    # battle_startで既にデッキを渡しているため、対戦中にNoneになることは想定していない
                    raise RuntimeError("Unexpected obs.select=None mid-battle.")

                # コイントス・セットアップ・通常ターンの行動など、全ての場面で同じくbattle_selectを呼ぶ
                select = choose_random(obs.select)
                obs_dict = battle_select(select)
                step += 1
            else:
                # MAX_STEPSに達してもresultが-1のまま = 無限ループ等の異常を疑う
                print(f"Reached MAX_STEPS={MAX_STEPS} without finishing.")
    finally:
        # 例外発生時も含め、C++側で確保されたメモリを必ず解放する
        battle_finish()

    print(f"Dumped {step + 1} observations to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
