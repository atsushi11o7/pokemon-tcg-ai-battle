# 最新メタデータとgeneralist学習パイプライン

更新日: 2026-08-06

## 結論

学習順は **generalist PPO → generalist MCTS → 必要なデッキだけspecialist** とする。
PPOは保存済み`outputs/saved_checkpoints/generalist_round250.pt`のネットワーク重みから開始するが、
旧データ分布に適応したAdamのモーメントは引き継がずoptimizerを初期化する。

`decks/`は削除しない。ここは学習メタの母集団ではなく、提出用・specialist候補の固定デッキ
置き場として扱う。旧`data/episodes`等は新snapshotの検証後に削除した。

## 取得済みデータ

- Leaderboard: 上位10チームと中位帯から層化抽出した5チーム
- Submission: 各チーム最大2件と自分の提出
- 期間: 直近14日
- episode: 各Submission最大10件
- 現行manifest: 441 episode、441 valid replay、失敗0
- 抽出結果: 159 exact deck、73 archetype

`data/meta/raw/replays`は差分更新用のappend-only領域である。現在の学習snapshotへ入るのは
`data/meta/raw/episodes.json`に列挙されたepisodeだけで、過去のraw replayが残っていても頻度計算には
混ざらない。

## 再取得とregistry更新

```bash
uv run python -u scripts/collect_kaggle_meta.py \
  --top-teams 10 \
  --mid-teams 5 \
  --submissions-per-team 2 \
  --episodes-per-submission 10 \
  --recent-days 14 \
  --replay-timeout-seconds 45

uv run python scripts/build_deck_registry.py
```

収集はepisode IDで重複排除し、検証済みreplayを再利用する。Kaggle SDKのreadが停止した場合は
1件45秒で打ち切り、最大2回試して失敗一覧へ送る。同じコマンドを再実行すれば失敗分だけ再取得できる。

## デッキ抽選

完全一致デッキはカードIDをsortしたhashでまとめる。アーキタイプはIDF重み付きJaccard類似度
0.60以上のデッキを連結成分としてまとめる。頻出する汎用札の影響はIDFで弱める。

使用頻度はraw対戦数ではなく、`team × day × deck`を最大1票として数え、7日半減期を掛ける。
これにより、新しいSubmissionほど対戦数が多いKaggleラダーの露出バイアスを抑える。

初期混合分布は次のとおり。

| 役割 | Meta | Coverage | Exploration | Hard |
|---|---:|---:|---:|---:|
| learner側 | 0% | 90% | 10% | 0% |
| opponent側 | 55% | 35% | 10% | 0% |

generalistでは両側の行動を学習するため、実際の学習サンプルの周辺分布はおよそ
Meta 27.5%、Coverage 62.5%、Exploration 10%となる。Hardは固定評価からmatchup別の
信頼区間が作れるまでは0%とし、未検証の疑似hard samplingを行わない。

## 学習

PPO:

```bash
uv run python -m training.cli train --config configs/ppo_generalist.yaml
```

PPO完了後のMCTS:

```bash
uv run python -m training.cli train --config configs/mcts_generalist.yaml
```

MCTS設定はPPOの`outputs/runs/ppo_generalist/checkpoints/final.pt`を初期重みとして明示する。
specialistはMCTS generalistの最終重みを初期値にし、提出候補デッキとMeta/Hard相手へ対象を
絞った別configとして作る。設定と状態確認の詳細は`docs/training-execution.md`を参照する。

## 検証済み事項

- 保存済みgeneralist checkpointは現行の共通networkへ全parameter key一致でロードできる
- sampling snapshotは441 replay、159 exact deck、73 archetypeを含む
- weighted deck sampling、PPO、MCTS、ワーカー障害回復にunit testがある
