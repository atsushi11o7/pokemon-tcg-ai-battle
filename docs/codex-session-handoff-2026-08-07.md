# Codexセッション引継ぎ: generalist PPO完了からgeneralist MCTS開始まで

更新日時: 2026-08-07 16:57 JST

## 1. この文書の目的

Tailscale経由で接続先を切り替え、Dev Containerを開き直した後でも、別のCodexセッションが
これまでの判断と作業状態を復元できるようにする。現在の最優先目標は、KaggleのSkill Rating
1000を目指し、完了済みgeneralist PPOを初期値としてgeneralist MCTSを学習することである。

この文書には認証情報、Kaggle token、Tailscaleアドレスを記載しない。

## 2. 最重要の現在状態

- generalist PPO: **完了**、200/200 round
- generalist MCTS: **未開始**、0/40 round
- 実行中のPPO/MCTSプロセス: なし
- MCTS初期重み: `outputs/runs/ppo_generalist/checkpoints/final.pt`
- 最新メタsnapshot: 441 replay、159 exact deck、73 archetype
- 提出中の最新2 agent: 14番Crustle team3、15番Crustle recent-meta
- `/workspace`の空き容量: 約168 GB
- `outputs/runs/ppo_generalist`: 約63 GB。削除しないこと

状態確認結果:

```text
Run: ppo_generalist
Algorithm: PPO
Status: COMPLETE
Progress: 200 / 200
Final checkpoint: /workspace/outputs/runs/ppo_generalist/checkpoints/final.pt

Run: mcts_generalist
Algorithm: MCTS
Status: NOT_STARTED
Progress: 0 / 40
Initial checkpoint: /workspace/outputs/runs/ppo_generalist/checkpoints/final.pt
```

## 3. リビルド後の最初の確認

`/workspace`はDev Container設定でホストのworkspaceへbind mountされている。通常の
`Dev Containers: Rebuild Container`では、ソース、data、outputs、submissionは残る。
ただし、念のため次を確認してから学習を開始する。

```bash
cd /workspace

nvidia-smi
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

test -f outputs/runs/ppo_generalist/checkpoints/final.pt
test -f data/meta/derived/sampling_snapshot.json

PYTHONPATH=src python -m training.cli validate \
  --config configs/mcts_generalist.yaml

PYTHONPATH=src python -m training.cli status \
  --config configs/mcts_generalist.yaml
```

`training.cli`という独立したconsole commandは現在の`pyproject.toml`には登録されていない。
確実な起動方法は`PYTHONPATH=src python -m training.cli ...`である。環境がeditable install済みなら
`PYTHONPATH=src`は不要だが、付けても問題ない。

## 4. 次に行うgeneralist MCTS

設定ファイルは`configs/mcts_generalist.yaml`である。PPOとMCTSは別runであり、1 YAMLにつき
1 algorithmだけを実行する。`--stage`、`enabled`、`warmstart_from`は使用しない。

現行設定:

```yaml
run:
  name: mcts_generalist
  output_dir: outputs/runs/mcts_generalist

algorithm: mcts

model:
  initial_checkpoint: outputs/runs/ppo_generalist/checkpoints/final.pt

data:
  sampling_snapshot: data/meta/derived/sampling_snapshot.json

runtime:
  seed: 0
  workers: 8
  game_timeout_seconds: 300
  round_timeout_seconds: 7200
  max_restarts: 50
  retry_delay_seconds: 5

training:
  selfplay_mode: generalist
  games_per_round: 200
  rounds: 40
  search_count: 50
  learning_rate: 0.0003
  batch_size: 64
  epochs_per_round: 2
  eval_games_per_round: 32
  gating_win_rate: 0.5
  checkpoint_pool_size: 3
  gating_pool_sample: 2
  replay_buffer_rounds: 5
```

nohupで開始する場合:

```bash
cd /workspace
mkdir -p outputs/runs/mcts_generalist

nohup env PYTHONPATH=src python -m training.cli train \
  --config configs/mcts_generalist.yaml \
  > outputs/runs/mcts_generalist/nohup.log 2>&1 &

echo $!
```

監視:

```bash
tail -f outputs/runs/mcts_generalist/nohup.log
tail -f outputs/runs/mcts_generalist/training.log
tail -f outputs/runs/mcts_generalist/worker_events.jsonl

PYTHONPATH=src python -m training.cli status \
  --config configs/mcts_generalist.yaml
```

MCTSは未開始なので、現時点では`outputs/runs/mcts_generalist`に学習成果物はない。途中再開では
最新の整合ラウンドからnetwork、Adam optimizer、replay buffer、checkpoint poolを復元する。
設定のlearning rateはoptimizer復元後に上書きされ、Adamのモーメントは維持される。

## 5. generalistのデッキ抽選

snapshotは`data/meta/derived/sampling_snapshot.json`で、2026-08-06に構築された。

- 441 replay
- 159 exact deck
- 73 archetype
- learner: Coverage 90%、Exploration 10%
- opponent: Meta 55%、Coverage 35%、Exploration 10%

generalistでは両側の行動を同じポリシーの学習対象にする。learner/opponentとして抽選した2デッキは
50%でplayer 0/1へ交換し、特定のデッキ分布と座席が固定されないようにしている。

`meta`重みは単純な対戦数ではない。同一`team × day × deck`を最大1票とし、7日半減期の
recencyを掛けて正規化する。Kaggleで新しいsubmissionほど多く試合を割り当てられる露出バイアスを
弱める目的である。

データ更新手順は`docs/meta-data-and-generalist-training-pipeline.md`を参照する。

## 6. 完了したgeneralist PPO

設定は`configs/ppo_generalist.yaml`、出力は`outputs/runs/ppo_generalist`。

主要設定:

- initial checkpoint: `outputs/saved_checkpoints/generalist_round250.pt`
- 500 games/round
- 200 rounds
- 4 workers
- learning rate: 0.0001
- minibatch: 64
- 4 epochs/round
- target KL: 0.02
- entropy coefficient: 0.001
- evaluation: 12 games/round

round 150で一度停止し、再開時はround 150のAdamモーメントを復元した。learning rateだけを
現行configの0.0001へ上書きした。そのままround 200まで完走した。

round 200の最終指標:

```text
policy_loss: -0.0085
approx_kl: 0.0103
clip_fraction: 0.0858
value_loss: 0.0844
entropy: 0.6565
vs random: 11勝1敗
```

entropyはround 140付近の約1.05からround 200で約0.66まで下がった。ただしentropy低下だけで
実戦性能向上とは判断していない。

checkpoint比較:

| 比較 | final側の結果 | 解釈 |
|---|---:|---|
| final vs initial保存済み重み | 128勝72敗、64.0% | 新PPO全体では改善 |
| final vs round 150 | 98勝98敗4分、49.0% | round 150→200の明確な改善は検出できず |

したがって、同じPPO設定をさらに延長するより、当初の計画どおりMCTSへ進む。

## 7. MCTS/PPOで修正済みの精度ロジック

詳細は`docs/training-inference-quality-review.md`を参照する。重要な修正は以下。

- PPO/MCTSは同じ`PolicyValueNet`とarchitectureを使用
- architectureは`src/training/common/model_config.py`へ一元化
- MCTS探索で真の相手デッキをnetwork特徴量へ渡していた情報漏洩を修正
- 相手determinizationをカード単独頻度ではなく実在60枚デッキ候補から生成
- 1局面5 determinizationへ探索予算を配分して単一仮説への過適合を緩和
- MCTS自己対戦にDirichlet noiseと序盤の温度付き訪問分布samplingを導入
- 複数選択64件制限の先頭偏重を、rank空間の等間隔抽出へ変更
- MCTS gatingはcurrent bestへの直接勝率とpool全体勝率の両方を要求
- MCTS/PPOとも再開時にAdam状態を復元し、configのlearning rateだけ上書き
- PPOはepoch平均KLがtarget KLを超えたら残りepochを停止
- validationは全roundで固定matchup、同一Python seed、席交換、デッキ交換を使用
- CG native engine内部shuffleはseed固定できないため完全決定的ではない

`src/training/common/opponent_pool.py`は現在未使用ではない。MCTS determinizationとPPO/MCTSの
spawn workerが、最新snapshotから読み込んだ重み付き実在デッキプールを共有するために使う。

## 8. マルチプロセスとクラッシュ

詳細は`docs/multiprocessing-crash-root-cause-report.md`を参照する。

確定・修正済み:

- PyTorch初期化後の`fork`によるfutex deadlock: `spawn`へ変更
- MCTS `SearchState`解放漏れ: `search_release`、`search_end`、`battle_finish`をfinallyで保証
- 旧Poolの投入順待ち・timeout後も処理継続する問題: 監視可能な常駐workerへ置換

現在のworker基盤:

- MCTSは自己対戦と固定matchup評価の両方をspawn workerで並列実行
- workerごとのPID、game index、RSS、elapsed、exit codeをJSONLへflush
- timeout workerをterminate/killして交換
- SIGSEGV/SIGABRT等で落ちたworkerだけ再生成
- trainer全体がnative crashした場合、CLIが最新checkpointから最大50回再試行

未解決:

- 低頻度SIGSEGV/SIGABRTの全根本原因は未特定
- OOMかnative libraryかを確定するには`memory.events`、core dump、gdb/ASanが必要

実対局smokeではPPO/MCTSとも2 spawn worker・2試合を完走し、worker RSSは1 workerあたり
約550 MBだった。ただし少数試合の成功は長時間実行でのnative crashゼロを保証しない。

## 9. デッキ調査とローカル評価

評価スクリプト: `scripts/evaluate_ppo_candidates.py`

評価JSON:

- `outputs/runs/ppo_generalist/evaluation/final_candidates.json`
- `outputs/runs/ppo_generalist/evaluation/pairwise_candidates.json`
- `outputs/runs/ppo_generalist/evaluation/lucario_variants.json`
- `outputs/runs/ppo_generalist/evaluation/lucario_crustle_alakazam.json`
- `outputs/runs/ppo_generalist/evaluation/crustle_variants.json`

最初の候補3種を同じ固定メタデッキ集合へ当てた結果:

| デッキ | 勝率 |
|---|---:|
| Marnie's Grimmsnarl ex | 52.0% |
| Mega Lucario ex | 61.0% |
| Mega Lopunny ex | 58.5% |

その後、Kaggleメタで多いCrustleとAlakazamも追加した。最終PPO同士の直接対戦では、
Mega LucarioはCrustleへ11.0%、CrustleはAlakazamへ63.0%だった。

Crustle完全一致構成3種の固定メタ相手評価:

| 内部名 | 出所 | 勝率 |
|---|---|---:|
| crustle recent-meta | 4 teams、4 episodes、最大meta weight | 79.5% |
| crustle episode14 | 2 teams、14 episodes | 81.0% |
| crustle team3 | 3 teams、10 episodes | 79.0% |

`meta`と`team3`は公式デッキ名ではなく、exact deck variantを区別する内部名である。

重要な制約: 上記デッキ評価は両agentに同じfinal PPOを使い、デッキだけを変えた比較である。
上位Kaggle agentに対する勝率ではない。提出結果がローカル79%より低いことと矛盾しない。

## 10. 現在のdecks構成

```text
decks/
├── candidates/
│   ├── crustle_meta.csv
│   ├── crustle_team3.csv
│   └── mega_lucario_ex_meta.csv
└── reference/
    └── 旧デッキ9ファイル
```

旧9デッキは削除せず`decks/reference`へ移動し、移動前のGit blobとbyte-for-byte一致を確認済み。
`src/training/common/deck.py`の既定提出デッキは`decks/candidates/crustle_meta.csv`を指す。
generalist学習では固定deck pathを使用しないため、この既定値はMCTS generalistの抽選に影響しない。

## 11. Kaggle提出14・15

提出物:

- `submission/14_ppo_generalist_crustle_team3.tar.gz`
  - submission ref: 55318072
  - SHA-256: `5ae21be84becdbca4b9809a5191ac782d5c88516b88a75603f51039899171414`
- `submission/15_ppo_generalist_crustle_meta.tar.gz`
  - submission ref: 55318342
  - SHA-256: `5dac33cd0a46b3fab23187ce4e25b56a6d86b00b2fc49603c9f66d8e1768f04d`

どちらも同じPPO round 200 final重みをgreedy inferenceで使用し、deck.csvだけが異なる。
モデル読込、60枚応答、公式CGで席交換込み4試合をローカル検証し、それぞれ4勝0敗だった。

2026-08-07 16:57 JST時点の暫定値:

| submission | games | W-L-D | Skill Rating |
|---|---:|---:|---:|
| 14 team3 | 11 | 6-5-0 | 604.1 |
| 15 recent-meta | 9 | 4-5-0 | 539.6 |

試合数が少なくscoreは大きく変動している。14番は一時696.4、15番は一時609.9だった。
この段階で600が性能上限、または片方のCrustle構成が優位とは判断しない。
Kaggleでは最新2 submissionだけがactiveなので、新しい提出を増やすと14番がactiveから外れる。

## 12. Git未コミット状態

リビルド前時点で変更はコミットしていない。主な差分:

- `.devcontainer/devcontainer.json`
  - `codex-config` Docker volumeを`/root/.codex`へmount
  - `CODEX_HOME=/root/.codex`
- `decks/`旧9ファイルを`decks/reference/`へ移動
- `decks/candidates/`へCrustle 2構成とMega Lucarioを追加
- `src/training/common/deck.py`の既定pathをCrustle recent-metaへ変更
- `scripts/evaluate_ppo_candidates.py`を追加
- `.claude/`は既存のuntracked内容。Codexは変更していない
- この引継ぎ文書を追加

`docs/`は`.gitignore`対象なので、この引継ぎ文書はGit statusやcommitには入らない。
同じホスト上でのDev Container rebuildでは`/workspace` bind mountに残るが、別PCへ移動する場合は
このファイルを別途コピーすること。

Gitは移動元を`D`、移動先を`??`と表示する。stage時にrenameとして認識される見込みだが、
意図せず削除と判断しないこと。

## 13. 検証済みコマンド

最後のコード検証:

```bash
ruff format --check src scripts tests
ruff check src scripts tests
python -m unittest discover -s tests -p 'test_*.py'
```

結果:

- 36 Python files formatted
- Ruff errors 0
- unittest 23件すべて成功

## 14. 参照すべき文書

- `docs/ptcg-ai-battle-overview.md`: コンペ全体
- `docs/training-execution.md`: 統一CLI、設定、再開方法
- `docs/meta-data-and-generalist-training-pipeline.md`: 最新メタ収集と抽選
- `docs/training-inference-quality-review.md`: PPO/MCTS精度ロジック修正
- `docs/multiprocessing-crash-root-cause-report.md`: crash調査と耐障害化
- `docs/mcts-selfplay-multiprocessing.md`: MCTS並列自己対戦

## 15. 次セッションへの依頼文例

新しいCodex会話では、次のように依頼すれば続行しやすい。

```text
/workspace/docs/codex-session-handoff-2026-08-07.md を読んでください。
generalist PPOは完了、generalist MCTSは未開始です。
configs/mcts_generalist.yamlをvalidate/status確認し、環境と空き容量に問題がなければ、
nohupでgeneralist MCTSを開始してください。worker_events.jsonlも監視対象にしてください。
```
