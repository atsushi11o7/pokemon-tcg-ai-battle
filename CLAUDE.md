# CLAUDE.md

## プロジェクト概要
Kaggle コンペ **The Pokémon Company - PTCG AI Battle Challenge Simulation** 用のリポジトリ。
ポケモンカードゲームの対戦 AI エージェントを開発し、Kaggle のラダーに提出して勝率（Skill Rating）を競う。

- コンペページ: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- 対戦エンジン: cabt Engine（`kaggle-environments` 上で動作）
- API ドキュメント: https://matsuoinstitute.github.io/cabt/
- 最終提出締切: 2026-08-16

## 実行環境
- devcontainer（`nvidia/cuda:12.9.2-cudnn-devel-ubuntu24.04` ベース、GPU 有効）
- GPU は RTX 5070 Ti（Blackwell / sm_120）。torch は CUDA 12.9 以降のビルドが必須で、
  cu126 ビルドは sm_120 のカーネルを含まない（`torch.cuda.is_available()` は True を返すが実行時に落ちる）
- Python 3.12 / パッケージ管理は **uv**（venv は `/opt/venv`、`PATH` に登録済み）
- 依存を追加するときは `uv add`（RL 用は `uv add --group rl`）。`pip install` は使わない。
- 依存を変更したら `uv lock` を実行し、`uv.lock` をコミットする。

## 提出に関する制約（重要）
提出物は Kaggle 側のオフライン環境（`gcr.io/kaggle-images/python:v163` 相当）で実行される。

- 形式: `.tar.gz`。`main.py` を**トップレベル**に置き、`deck.csv` を同梱する
  - `tar -czvf submission.tar.gz *`
- サイズ上限: **197.7 MiB**
- 実行リソース: vCPU 2 / RAM 12.2 GiB / **GPU なし**
- 実行時のファイルパスは `/kaggle_simulations/agent/`。相対パス読み込みはこれを基準にする
- **本番では `pip install` できない**。Kaggle イメージにプリインストール済みのライブラリのみ使用可
- `stable-baselines3` / `sb3-contrib` は本番に存在しない前提。RL モデルを提出する場合は
  `state_dict()` のみ保存し、`main.py` 側に推論用ネットワーク定義を持たせて素の PyTorch で動かす
- 1日の提出上限 5 件、有効なのは最新 2 件

## ディレクトリ方針
- `submission/` … 提出物一式（`main.py`, `deck.csv`, 重みファイル）
- `src/` … 学習・分析コード（提出物には含めない）
- `data/` … Kaggle から取得したデータ（gitignore 対象）
- `notebooks/` … 探索的分析
- `logs/` … 学習ログ・対戦ログ（gitignore 対象）

## コーディング・Lint
- Lint/format は **ruff**（`line-length=100`, `target=py312`, ルール: `E,F,I,B,UP`、`E501` は無視）。
- コミット前に `ruff check --fix .` と `ruff format .` を通す。
- データ列名やマジックナンバーなど、後で読んで非自明な箇所のみコメントを残す。
- `main.py` は本番で動く唯一のエントリポイント。依存を増やさず、推論は CPU で軽量に保つ。

## Git・コミット
- コミットメッセージ: **タイトル1行のみ・英語の命令形・〜72文字・本文なし・`Co-Authored-By` フッターなし**。例: `Add rule-based agent baseline`, `Fix deck validation`。
- 1コミット = 1論理変更。`git add -A` ではなく対象ファイルを明示的に stage する。
- ブランチ運用: `feature/<topic>` を切って作業し、PR 経由でレビュー後 `main` へマージする。`main` への直接コミットはしない。
- `.env` や `data/`・学習済みモデル・提出用 tarball など gitignore 対象は絶対にコミットしない。

## GitHub 操作（Issue / PR）
- Issue / PR の作成・更新は `gh` から行える（リモート `origin` = `atsushi11o7/pokemon-tcg-ai-battle`、`gh` 認証済み）。
  - Issue: `gh issue create --title "..." --body "..."`
  - PR: ブランチを push 後 `gh pr create --base main --title "..." --body "..."`
- トークンは絶対にコミットしない（環境変数 / `gh auth login` で管理）。

## 秘密情報の扱い
- `.env` は秘密情報。**明示的な許可がない限り閲覧・編集しない**。トークンや API キーの値はユーザー自身が設定する。
- 新しい環境変数を足すときは雛形の `.env.example` 側を更新する。
- Kaggle API トークン（`KAGGLE_API_TOKEN`）はログや出力に出さない。