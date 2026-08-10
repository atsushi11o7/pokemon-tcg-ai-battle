# PTCG AI Battle Challenge Simulation — コンペ仕様まとめ

Kaggle コンペ **The Pokémon Company - PTCG AI Battle Challenge Simulation** の仕様・制約を
開発着手前に整理したもの。

- コンペページ: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- 対戦エンジン: **cabt Engine**（`kaggle-environments` 上で動作）
- API ドキュメント: https://matsuoinstitute.github.io/cabt/
- 主催: The Pokémon Company / HEROZ / 松尾研究所

---

## 1. コンペの構成

ポケモンカードゲームの対戦 AI エージェントを開発し、Kaggle のラダーに提出して
Skill Rating を競う Simulation 形式のコンペ。

TCG AI Battle Challenge には 2 つのトラックがある。

| トラック | 内容 | 賞金 |
|---|---|---|
| **Simulation**（本コンペ） | エージェントを提出し自動対戦。勝率でランキング | なし |
| Hackathon (Strategy) | 戦略ロジックを説明したレポートを提出 | あり |

Hackathon の最終順位は Simulation のリーダーボード成績も加味して決まる。
本リポジトリは Simulation を主対象とするが、レポート提出も視野に入るため
**設計判断の理由は逐次記録しておく**こと。

Hackathon への参加は Simulation の参加要件ではない。

---

## 2. タイムライン

| 日付 | 内容 |
|---|---|
| 2026-06-16 11:00 UTC | 開始 |
| 2026-08-09 | エントリー締切（この日までにルール承諾が必要） |
| 2026-08-09 | チームマージ締切 |
| **2026-08-16** | **最終提出締切** |
| 2026-08-17 〜 08-31頃 | 追加対戦期間（提出不可）。収束後にリーダーボード確定 |

締切は特記なき限り各日 23:59 UTC。

---

## 3. 提出仕様

| 項目 | 値 |
|---|---|
| 形式 | `.tar.gz`（`tar -czvf submission.tar.gz *`） |
| 必須ファイル | `main.py`（**トップレベル**、ネスト不可）、`deck.csv` |
| サイズ上限 | 197.7 MiB |
| 実行時パス | `/kaggle_simulations/agent/` |
| 提出上限 | 1日 5 件、有効なのは最新 2 件 |

### 実行リソース（本番）

| 項目 | 値 |
|---|---|
| vCPU | 2 |
| RAM | 12.2 GiB |
| HDD | 11.8 GiB |
| GPU | **なし** |
| ネットワーク | **なし（オフライン）** |

本番イメージは `gcr.io/kaggle-images/python:v163` ベース。

### 派生する設計上の制約

- **本番では `pip install` できない**。Kaggle イメージにプリインストール済みのライブラリのみ使用可
- `stable-baselines3` / `sb3-contrib` は本番に存在しない前提で設計する
- RL モデルを提出する場合:
  - `model.zip`（sb3 形式）をそのまま置かない
  - `policy.state_dict()` のみ保存し、**推論用ネットワーク定義を `main.py` 側に持たせて素の PyTorch で動かす**
  - torch のバージョン差でロードが失敗しうるため、本番イメージの torch バージョンを確認して合わせる（**未確認**）
- ファイル読み込みは `/kaggle_simulations/agent/` を基準にする:
  ```python
  import os
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))
  deck_path = os.path.join(BASE_DIR, "deck.csv")
  ```
- 提出時はまず自分自身との Validation Episode が走る。ここで落ちると Error 扱い（ログはダウンロード可能）

---

## 4. 評価の仕組み

- Skill Rating は正規分布 `N(μ, σ²)` でモデル化。初期値 `μ₀ = 600`
- σ は不確実性を表し、対戦を重ねると減少する
- 勝てば μ 上昇、負ければ下降、引き分けは両者の平均へ寄る
- 更新幅は「期待結果からの乖離」と「各提出の σ」に比例する
- **勝敗の点差はレーティングに影響しない**（勝ち方の派手さは無意味、勝率のみが効く）
- 近いレーティング同士でマッチングされる
- 新しい提出ほど頻繁に対戦が組まれる（フィードバックが早い）
- リーダーボードには自分のベストスコアのみ表示。全提出の推移は Submissions タブで追える

---

## 5. データセット

Data ページから取得。全 60 ファイル / 327.59 MB。

```bash
kaggle competitions download -c pokemon-tcg-ai-battle -p data/
```

| ファイル / ディレクトリ | 内容 |
|---|---|
| `ptcg_engine/` | シミュレータ本体（SDK）。ローカルデバッグ・RL 用 |
| `sample_submission/` | 提出物の雛形。**最初に必ず読む** |
| `EN_Card_Data.csv` | カードメタデータ（英語） |
| `JP_Card_Data.csv` | 同（日本語） |
| `Card_ID List_EN.pdf` | カード一覧（画像付き、137.65 MB） |
| `Card_ID List_JP.pdf` | 同（日本語） |

EN / JP は言語以外の内容は同一。

### CSV スキーマ（EN/JP 共通・35 列）

| 列 | 内容 |
|---|---|
| `Card ID` | シミュレータが使う一意識別子 |
| `Card Name` | カード名 |
| `Expansion` | 収録拡張パック |
| `Collection No.` | 拡張内のコレクション番号 |
| `Stage (Pokémon) / Type (Energy and Trainer)` | 進化段階（Basic / Stage 1 / Stage 2）またはカード種別 |
| `Rule` | 特殊ルールテキスト |
| `Category` | カテゴリ（Pokémon / Trainer / Energy） |
| `Previous stage` | 進化元 |
| `HP` | HP |
| `Type` | タイプ（Grass / Fire / Water 等） |
| `Weakness` | 弱点 |
| `Resistance (Type)` | 抵抗力 |
| `Retreat` | にげるコスト |
| `Move Name` | ワザ名 |
| `Cost` | ワザの必要エネルギー |
| `Damage` | ダメージ |
| `Effect Explanation` | ワザ効果・追加ルールの説明 |

`Card ID` がシミュレータ内での識別子。`deck.csv` はこの ID で構成する。
1 枚のカードに複数のワザがある場合の行の持ち方は要確認。

---

## 6. エピソードリプレイ

- 自分の提出のリプレイは Submissions タブ、または CLI / MCP から取得可能
  - https://github.com/Kaggle/kaggle-cli/blob/main/docs/simulation_competitions.md
- **他チームのリプレイも Leaderboard からダウンロード可能**
- 上位エピソードの日次エクスポートがフォーラムで公開される（BC / IL / RL 用と明記）

→ **模倣学習（Behavior Cloning）の教師データとして最有力**。序盤の戦略として検討価値が高い。

---

## 7. エージェントのインターフェース

毎ターン、エージェントは observation を受け取る。

- ゲームログ
- 現在の盤面状態
- **合法手のリスト（legal options）**

返すのは**選択したオプションのインデックス**。
エンジンは常に合法手のみを提示するため、非合法手のバリデーションは不要。

正確なデータ構造は https://matsuoinstitute.github.io/cabt/ を参照（**未読**）。

公式ルールとシミュレータ挙動には差異があり、コンペページに一覧がある（**未確認**）。

---

## 8. ゲームの性質と難所

- **不完全情報ゲーム** — 相手の手札が見えない。これがコア課題としてコンペ側でも明示されている
- **確率要素** — ドロー、コイントス
- **広大な組み合わせ空間** — スタンダードレギュレーション約 2,000 種のカードプール
- **持ち時間は 1 試合およそ 10 分** — 推論は CPU 2 コアで完結する軽さに保つ必要がある

Overview には「ルールベース単独では上位は難しい」と明記されている。
先読み、リアルタイム適応、最適な意思決定が求められる、という位置づけ。

---

## 9. 開発の進め方（案）

1. **サンプル提出をそのまま通す** — パイプラインの疎通確認を最優先
2. **ルールベースのベースライン** — 「倒せるなら攻撃 / エネルギー付与 / ベンチ展開」程度
3. **デッキ構築の最適化** — `deck.csv` の中身は勝率に直結する。エージェントのロジックと同等に重要
4. **探索系（MCTS 等）または模倣学習** — 不完全情報ゲームなので determinization + MCTS が定石
5. **RL** — 必要に応じて

段階 1 と 2 は必ず通しておく。提出パイプラインの疎通を早期に確認しておかないと、
終盤に「強いが提出できない」事態になる。提出上限は 1 日 5 件なので無駄撃ちしない。

---

## 10. 未確認事項（着手時に最優先で潰す）

1. **`kaggle-environments` の正しいバージョン**
   Overview の「as of kaggle-environments 1.14.10」は PyPI のバージョン体系と一致しない
   （PyPI 最新は 1.32.0 系、1.14.x は 2024 年頃の古い系列）。
   `sample_submission/` や `ptcg_engine/` の中身を見て実際に必要な版を確定させる
2. **`make()` に渡す環境名** — `"ptcg"` などの正確な文字列
3. **本番イメージのプリインストール済みライブラリとバージョン**（特に torch）
4. **observation / action の正確なデータ構造** — cabt ドキュメント参照
5. **公式ルールとシミュレータの差異一覧**
6. **`deck.csv` のフォーマット** — 列名、枚数制約、レギュレーション
7. **`ptcg_engine` の使い方** — ローカルで自己対戦を回す手順

---

## 参考リンク

- コンペページ: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- cabt API ドキュメント: https://matsuoinstitute.github.io/cabt/
- kaggle-environments: https://github.com/Kaggle/kaggle-environments
- Simulation コンペの CLI 操作: https://github.com/Kaggle/kaggle-cli/blob/main/docs/simulation_competitions.md
