"""公式の日次エピソードzipから、模倣学習用の`Sample`を作る。

zipは展開しない(11日分で236GBになる)。メンバーを1つずつ読んでシャードへ書き出す。

行動の列挙は`mcts.search.enumerate_actions`をそのまま使う。学習・推論と同じ関数を
通すことで、教師信号のindexが食い違わない(別実装にすると、複数選択の組み合わせ順が
ずれても気付けない)。
"""

from __future__ import annotations

import json
import os
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch

from ..common.sparse_features import get_decoder_input, get_encoder_input
from ..mcts.search import enumerate_actions
from ..mcts.selfplay import Sample

DECK_SIZE = 60


def episode_decks(steps: list) -> list[list[int]] | None:
    """最初のstepのactionから、席ごとの60枚デッキを取り出す。"""
    decks: dict[int, list[int]] = {}
    for step in steps[:2]:
        for seat, agent_step in enumerate(step):
            action = agent_step.get("action")
            if isinstance(action, list) and len(action) == DECK_SIZE:
                decks.setdefault(seat, action)
    if len(decks) != 2:
        return None
    return [decks[0], decks[1]]


def winning_seat(replay: dict) -> int | None:
    """勝った席を返す。引き分けと不明はNone。"""
    rewards = replay.get("rewards") or []
    if len(rewards) != 2 or rewards[0] is None or rewards[1] is None:
        return None
    if rewards[0] == rewards[1]:
        return None
    return 0 if rewards[0] > rewards[1] else 1


def iter_decisions(
    steps: list, winner: int, stats: Counter, *, winner_only: bool = True
) -> Iterator[tuple[int, dict, list[int]]]:
    """リプレイのstep列から、(席, その席が見たobservation, 実際に返したaction)を取り出す。

    リプレイの記録形式には、実データで踏んだ罠が2つある。どちらも間違えても例外にならず、
    「教師がずれたまま学習が進む」ので、ここだけを純粋関数として切り出してテストする。

    1. `kaggle_environments`は、step iのobservationへの応答をstep i+1のactionへ記録する。
       同じstepで対応づけると、選択肢数を超えるindexや個数制限違反が3割前後混ざる。
    2. 待機中(INACTIVE)の席には前回のactionが残っている。応答したのはACTIVEの席だけ。
    """
    for index in range(2, len(steps) - 1):
        for seat, agent_step in enumerate(steps[index]):
            if winner_only and seat != winner:
                continue
            if agent_step.get("status") != "ACTIVE":
                stats["skip_inactive"] += 1
                continue
            observation = agent_step.get("observation")
            action = steps[index + 1][seat].get("action")
            if not isinstance(action, list) or not observation:
                continue
            if len(action) == DECK_SIZE:
                continue
            if not (observation.get("select") or {}).get("option"):
                continue
            yield seat, observation, action


def extract_episode(
    replay: dict, stats: Counter, *, winner_only: bool = True, imitate_loser: bool = False
) -> list[Sample]:
    """1エピソードのリプレイから学習サンプルを作る。

    `winner_only=False`にすると両側の局面を集め、模倣は勝った側の手だけにする
    (抽出CLIの既定はこちら)。方策の損失は
    `-(policy_target * log_prob).sum()`なので、`policy_target`を全ゼロにすると方策の
    勾配は出ず、価値の教師(`label`)だけが効く。方策は勝者だけを真似つつ、価値ヘッドは
    勝ち負け両方の教師で学習できる。

    勝者だけを集めると価値の教師が+1に偏り、「常に+1」を出すだけで損失が0になる。
    MCTSラップ推論は葉の評価に価値を使うため、これは探索を壊す。

    Args:
        replay: 日次データセットのエピソードJSON。
        stats: 除外理由の内訳を積む集計用カウンタ。
        winner_only: Trueなら敗者の局面自体を集めない(価値の教師も+1だけになる)。
        imitate_loser: Trueなら敗者の手もone-hotで模倣する。

    Returns:
        list[Sample]: 方策の教師は勝者one-hot/敗者ゼロ、価値の教師は勝敗(+1/-1)。
    """
    from cg.api import to_observation_class

    steps = replay.get("steps") or []
    decks = episode_decks(steps)
    if decks is None:
        stats["skip_no_deck"] += 1
        return []
    winner = winning_seat(replay)
    if winner is None:
        stats["skip_draw"] += 1
        return []

    samples: list[Sample] = []
    for seat, observation, action in iter_decisions(steps, winner, stats, winner_only=winner_only):
        try:
            obs = to_observation_class(observation)
        except Exception:
            stats["skip_parse"] += 1
            continue
        if obs.select is None or obs.current is None:
            continue

        actions = enumerate_actions(obs.select)
        if len(actions) < 2:
            # 選択の余地が無い局面は勾配を持たない。
            stats["skip_single_action"] += 1
            continue
        try:
            target_index = actions.index(sorted(action))
        except ValueError:
            # 複数選択の組み合わせが上限64件の抽出から漏れた場合。one-hotを作れない。
            stats["skip_action_not_enumerated"] += 1
            continue

        won = seat == winner
        policy_target = [0.0] * len(actions)
        if won or imitate_loser:
            policy_target[target_index] = 1.0
            stats["policy_samples"] += 1
        else:
            stats["value_only_samples"] += 1
        outcome = 1.0 if won else -1.0
        sample = Sample(
            get_encoder_input(obs, decks[seat]),
            get_decoder_input(obs, actions),
            policy_target,
            outcome,
        )
        # 価値の教師は`label`から読まれる。MCTSは終局後に探索値とブレンドして埋めるが、
        # BCには探索値が無いので勝敗をそのまま入れる。
        sample.label = outcome
        samples.append(sample)
        stats["samples"] += 1
    stats["episodes"] += 1
    return samples


def iter_replays(zip_paths: list[Path], stats: Counter) -> Iterator[dict]:
    """zipを展開せずにエピソードJSONを1件ずつ読む。エピソードIDで重複を除く。"""
    seen: set[str] = set()
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if not name.endswith(".json"):
                    continue
                key = Path(name).name
                if key in seen:
                    stats["skip_duplicate"] += 1
                    continue
                seen.add(key)
                try:
                    yield json.loads(archive.read(name))
                except (json.JSONDecodeError, OSError):
                    stats["skip_unreadable"] += 1


def drop_from_page_cache(path: Path) -> None:
    """読み終えたシャードをページキャッシュから追い出す。

    WSL2のゲスト内ページキャッシュは、Windows側の`vmmem`のメモリとして実体化する。
    1エポックは401シャード(68GB)を読み流すので、放っておくとキャッシュが空きメモリを
    埋め尽くし、ホストのメモリを占有し続ける。一方エポックごとにシャード順を
    シャッフルするため、13GBのキャッシュに68GBの作業集合が載るはずもなく、
    再利用はほぼ起きない。効果の無いキャッシュのために圧迫するのは損でしかない。

    `/proc/sys/vm/drop_caches`はコンテナから書けない(read-only)ので、ファイル単位で
    指示するこの経路を使う。失敗しても学習は続けられるので握り潰す。
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (OSError, AttributeError):
        pass  # 対応していない環境では何もしない
    finally:
        os.close(fd)


def load_shard(path: Path) -> list[Sample]:
    """シャードを1枚読み、読み終えたらページキャッシュから外す。"""
    samples = torch.load(path, weights_only=False)
    drop_from_page_cache(path)
    return samples


def load_shard_paths(paths: Iterable[Path]) -> list[Sample]:
    """指定したシャードだけを読む。"""
    samples: list[Sample] = []
    for path in paths:
        samples.extend(load_shard(path))
    return samples


SHARD_COUNT_CACHE = "shard_counts.json"


def shard_sample_counts(paths: list[Path], cache_dir: Path) -> list[int]:
    """各シャードのサンプル件数を返す。結果は`cache_dir`へ残して再利用する。

    学習率スケジュールの総ステップ数を出すために件数が要るが、数えるためだけに
    `torch.load`で全シャードを展開すると、実測で387シャード(68GB)に11分かかる。
    しかもこれは学習開始前の固定費で、ネイティブクラッシュから再起動するたびに
    払い直すことになる(`max_restarts`は20)。

    キャッシュはファイル名とバイト数で照合する。抽出し直してシャードが差し替われば
    サイズが変わるので、古い件数を使い続けることはない。
    """
    cache_path = cache_dir / SHARD_COUNT_CACHE
    cached: dict[str, list[int]] = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = {}  # 壊れていたら数え直す。ここで失敗させる価値はない

    counts: list[int] = []
    updated = False
    for path in paths:
        size = path.stat().st_size
        entry = cached.get(path.name)
        if entry is not None and entry[0] == size:
            counts.append(entry[1])
            continue
        count = len(load_shard(path))
        cached[path.name] = [size, count]
        counts.append(count)
        updated = True

    if updated:
        try:
            cache_path.write_text(json.dumps(cached), encoding="utf-8")
        except OSError:
            pass  # 読み取り専用でも学習は続けられる
    return counts


def load_shards(shard_dir: Path, limit: int = 0) -> list[Sample]:
    """`shard_*.pt`をまとめて読む。limitが正ならそのシャード数で打ち切る。"""
    paths = sorted(shard_dir.glob("shard_*.pt"))
    if limit:
        paths = paths[:limit]
    samples: list[Sample] = []
    for path in paths:
        samples.extend(torch.load(path, weights_only=False))
    return samples
