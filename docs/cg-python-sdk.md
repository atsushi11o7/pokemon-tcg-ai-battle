# `cg` パッケージ（Python SDK）の中身と使い方

`sample_submission/sample_submission/cg/` に入っている4つの `.py` ファイルについてのまとめ。
公式ドキュメント（https://matsuoinstitute.github.io/cabt/ 、未読）に同等の説明がある可能性が高いが、
現時点でリポジトリ内には整理された記述がなかったため、ソースコードを読んで作成した。

## 全体像

`ptcg_engine/`（C++ソース）をビルドすると `libcg.so` などのバイナリになる。
この4ファイルは**そのバイナリとは別に、運営が手書きしたPythonラッパー**で、依存関係は次の通り。

```
sim.py   … バイナリを ctypes でロードし、関数の引数/戻り値の型を定義する最下層
  ↑
game.py  … 実対戦を1つ動かすための高レベル関数（battle_start / battle_select / battle_finish）
api.py   … Observation 等のデータクラス定義 + 先読み探索用の関数（search_begin 等）
  ↑
utils.py … dict → dataclass 変換のヘルパー（api.py の内部実装用）
```

`game.py` と `api.py` はどちらも `sim.py` を使うが、互いには依存していない（並列の関係）。

---

## `sim.py` — 最下層: バイナリのロードとC関数の型定義

- OS判定して `cg.dll` / `libcg.so` / `libcg.dylib` / `libcg-arm64.so` のいずれかを
  `ctypes.cdll.LoadLibrary()` でロードする（[sim.py:20-29](../data/sample_submission/sample_submission/cg/sim.py#L20-L29)）
- **import された時点で即ロードが実行される**（遅延しない）。ロードに失敗するとこのモジュールを
  importした瞬間にエラーになる
- 各C関数（`Export.cpp` で `extern "C" { GAME_API ... }` として宣言されているもの）について、
  `argtypes` / `restype` を手動で対応づけている。例:
  - `Export.cpp` の `BattleStart(int* cards)` ↔ `lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]`
- `class Battle` … `battle_ptr`（現在の対戦インスタンスへのポインタ）と `obs`（直近の観測）を
  保持するだけの、状態を入れておくための入れ物

普段このファイルを直接呼ぶことはなく、`game.py` / `api.py` 経由で使う。

---

## `game.py` — 実際の対戦を1つ動かす（ローカルでの自己対戦・検証用）

Kaggle本番では、対戦の進行自体は `kaggle_environments` 側がやってくれるため、
提出した `main.py` がこれらの関数を直接呼ぶことはない。
**ローカルで自己対戦をシミュレートして検証・学習データ生成をしたいとき**に使う。

| 関数 | 役割 |
|---|---|
| `battle_start(deck0, deck1)` | 2人分のデッキ（各60枚のカードID）を渡して対戦を開始。戻り値は `(最初の観測dict, StartData)` |
| `battle_select(select_list)` | 選択したオプションのindexのリストを渡して1手進める。戻り値は次の観測dict |
| `battle_finish()` | 対戦を終了し、C++側で確保されたメモリを解放する |
| `visualize_data()` | ビジュアライザ用のデータを取得する |

使用イメージ:

```python
from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class

obs_dict, start_data = battle_start(deck0, deck1)
while True:
    obs = to_observation_class(obs_dict)
    if obs.current is not None and obs.current.result != -1:
        break  # 決着
    select = my_agent_logic(obs)  # 自作ロジックでoption indexのリストを決める
    obs_dict = battle_select(select)
battle_finish()
```

内部的には `Battle.battle_ptr` というグローバル変数1つで現在の対戦を表しているため、
**同時に2つ以上の対戦を並行して回すことはできない**（1プロセス内で1対戦ずつ）。

---

## `api.py` — 観測データの型定義 + 先読み探索

### 1. データクラス群（Observation とその中身）

`main.py` の `agent(obs_dict)` に渡ってくる `dict` を型付きで扱うための定義。
`to_observation_class(obs_dict)` に通すと `Observation` インスタンスになる。

- `Observation` … `select`（今回何を選ぶ必要があるか）、`logs`（前回選択からのイベント履歴）、
  `current`（現在の盤面 `State`）を持つ。最初のデッキ提出時だけ `select`/`current` は `None`
- `State` … ターン数、手番、両プレイヤーの `PlayerState`（場・手札・トラッシュ・特殊状態等）
- `SelectData` / `Option` … 今回選べる選択肢の一覧。`OptionType`（PLAY/ATTACK/RETREATなど）ごとに
  必要な追加情報（`area`, `index` 等）が変わる
- `Log` / `LogType` … 攻撃・ドロー・進化などのイベントログ

対応するenum（`AreaType`, `EnergyType`, `CardType`, `SelectType`, `SelectContext`, `OptionType`,
`LogType`）もすべてここに定義されている。

### 2. 先読み探索（`search_begin` / `search_step` / `search_end` / `search_release`）

不完全情報ゲームなので、相手の手札やデッキの並びを**仮定した上で**数手先を読みたい場合に使う関数群。
実際の対戦インスタンス（`apiDataType=1`）とは別に、探索専用のインスタンス（`apiDataType=2`）を
内部で保持しており、`agent_ptr` というモジュールグローバル変数に格納される
（[api.py:543-544](../data/sample_submission/sample_submission/cg/api.py#L543-L544)、
初回 `search_begin` 呼び出し時に遅延生成される点に注意）。

使い方の流れ:

1. `agent()` に渡された `Observation`（`obs.search_begin_input` に、探索を始めるためのシリアライズ済み状態が
   自動で入っている）と、相手の手札・デッキ等について**自分で予測した**カードID列を用意する
2. `search_begin(obs, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active)`
   を呼ぶと `SearchState`（新しい観測 + `searchId`）が返る
3. その観測を見て手を決め、`search_step(search_id, select)` で1手進める。以降は探索木を好きなだけ辿れる
4. 使い終わったら `search_release(search_id)` でそのノードを解放、探索全体をやめる時は `search_end()`

これはMCTSのようなシミュレーションベースの先読みを実装する際の土台になる部分で、
実際の対戦を1手も進めずに「もしこうだったら」を試せるのがポイント。

### 3. カード・攻撃メタデータ

- `all_card_data()` … 全カードの情報（HP・弱点・進化元など）を `CardData` のリストで取得
- `all_attack()` … 全ワザの情報（ダメージ・必要エネルギーなど）を `Attack` のリストで取得

`EN_Card_Data.csv` / `JP_Card_Data.csv` と重複する情報だが、こちらはエンジン内部のIDと
直接対応した生データが取れる。

---

## `utils.py` — dict → dataclass 変換ヘルパー

`api.py` が内部的に使っているだけのユーティリティ。単体で直接呼ぶ機会はほぼない。

- `to_dataclass(dic, cls)` … dictのキーをdataclassのフィールド名に合わせて再帰的に変換する
- `json_to_dataclass(bs, cls)` … JSONバイト列をデコードして `to_dataclass` にかける

`Observation` のようなネストしたdataclass（`State` の中に `PlayerState` の配列、その中に
`Pokemon` の配列…）を手作業でパースせずに済むようにするための共通処理。

---

## まとめ: どれをいつ使うか

| やりたいこと | 使うもの |
|---|---|
| 本番の `agent(obs_dict)` を実装する | `api.py` の `Observation` / `to_observation_class` （必要なら `search_begin` 系） |
| ローカルで自己対戦を回して検証・学習データ生成する | `game.py` の `battle_start` / `battle_select` / `battle_finish` |
| カード・ワザの一覧を取得したい | `api.py` の `all_card_data` / `all_attack` |
| バイナリの読み込み方や関数シグネチャを確認したい | `sim.py` |
| （直接使うことはほぼない）dict変換の内部実装を見たい | `utils.py` |
