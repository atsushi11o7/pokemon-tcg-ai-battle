# 学習設定と実行方法

更新日: 2026-08-08

## 結論

学習設定は「1 YAML = 1 run = 1 algorithm」とする。PPOとMCTSを同じYAMLに書かず、
`algorithm: ppo`または`algorithm: mcts`が、統一CLIから起動される学習器を一意に決める。

使用する設定は次の3つである。

- `configs/ppo_generalist.yaml`: ランダム初期化からPPOを開始する
- `configs/mcts_generalist.yaml`: PPOが生成した`final.pt`からMCTSを開始する

`enabled`、`stages`、`--stage`、`warmstart_from`は使用しない。

## ソースファイル名

algorithm名はディレクトリで表し、その内側は同じ役割名に統一する。

```text
src/training/
├── common/
├── mcts/
│   ├── train.py
│   └── selfplay.py
└── ppo/
    ├── train.py
    └── selfplay.py
```

CLIは`training.mcts.train`または`training.ppo.train`をmoduleとして起動する。内部importも
package相対importなので、同名の`selfplay.py`が同一プロセス内で衝突しない。旧BC scriptと
旧MCTS専用agentは参照先・出力先が現構成と不整合で、利用箇所も無かったため削除した。


PPOが完了すると、MCTSの初期重みとして指定した
`outputs/runs/ppo_generalist/checkpoints/final.pt`が作られる。その後にMCTSを実行する。

```bash
uv run python -m training.cli validate --config configs/mcts_generalist.yaml
uv run python -m training.cli status --config configs/mcts_generalist.yaml
uv run python -m training.cli train --config configs/mcts_generalist.yaml
```

`status`は設定に書かれたalgorithmだけを調べる。PPO設定を渡せばPPO、MCTS設定を渡せば
MCTSの状態が表示される。両方をまとめて実行・表示する暗黙のpipelineはない。

## 初期重みと再開

`model.initial_checkpoint`は、そのrunにラウンドcheckpointがまだないときだけ使う
ネットワーク重みである。

- PPO: `outputs/saved_checkpoints/generalist_round250.pt`
- MCTS: `outputs/runs/ppo_generalist/checkpoints/final.pt`

現行pipelineは、旧BC checkpointを作り直す構成ではない。保存済みgeneralist round250を
PPOの初期重みにし、新しいsampling snapshotに対してfresh optimizerでfine-tuningした後、
その`final.pt`をMCTSへ渡す。これはround251として旧runを継続するのではなく、新runの
round 1から再学習する意図的な方針である。BCという手法自体を否定するものではなく、
再導入する場合は新しいKaggle replayに対応した独立stageとして設計する。

新しいrunではoptimizerを新規作成する。旧データ分布で蓄積されたAdamのモーメントを
持ち込まないためである。同じrunを再開するときは、`output_dir/checkpoints`内の最新ラウンドから
ネットワーク重みとoptimizer状態を復元する。

各runの出力は次の場所にまとまる。

```text
outputs/runs/<run-name>/
├── run_config.yaml
├── training.log
├── worker_events.jsonl
├── replay/                      # MCTSのみ。直近roundの自己対戦Sample
│   └── generalist_roundN.pt
└── checkpoints/
    ├── generalist_roundN.pt
    ├── generalist_roundN_optimizer.pt
    ├── training_state.pt         # MCTSのみ。replay索引とcheckpoint pool
    └── final.pt
```

ネイティブ異常終了（SIGSEGV、SIGABRT、OOM kill相当）だけをCLIが再試行する。設定エラーや
通常のPython例外は再試行しても直らないため、そのまま停止する。試合ワーカー単体の異常終了や
タイムアウトは親プロセスが検出し、そのワーカーだけを交換する。

## `architecture`とは何か

ここでいうarchitectureは、ニューラルネットワークの構造（カードIDの持ち方、埋め込み次元、
attention head数、feed-forward幅、encoder/decoder層数）を指す。値は
`src/training/common/model_config.py`に一度だけ定義されており、ここには転記しない
（転記すると変更のたびに食い違うため）。

run configでは指定できない。学習を実行すると、そのrunの`output_dir`へ
`architecture.yaml`として実際の値が記録されるので、どの重みがどの構造で学習されたかは
そちらで確認する。構造を変えると既存チェックポイントは形状不一致で読めなくなる。

PPOとMCTSは同じ`PolicyValueNet`とこの定義を使うため、ネットワーク重みを相互利用できる。
したがって、runごとのYAMLに`architecture`を書く必要はなく、設定項目から除外した。

checkpointはネットワーク構造に依存する。上記の値を変更すると既存checkpointを通常は
`load_state_dict`できなくなるため、既存重みを初期値に使う今回の学習では変更しない。

## generalistと固定デッキ

設定ファイルの形式を統一するため、全runで`training.deck_path`を記載する。

- `generalist`: `deck_path: null`。両席をsampling snapshotから抽選する
- `asymmetric`: `deck_path`のデッキ対sampling snapshotからのランダムデッキ
- `mirror`: `deck_path`の同一デッキ同士

`asymmetric`は固定デッキ側だけのspecialist学習ではない。固定デッキ側とランダムデッキ側の
両方を同じ方策の学習に使い、デッキ分布の半分を指定デッキへ傾けた汎用自己対戦である。
固定デッキは試合番号の偶奇でplayer 0/player 1へ交互に配置する。そのため
`games_per_round`は偶数必須である。デッキ固有の処理はなく、`deck_path`を変更すれば
Crustle以外にも同じ形式・実装を使用できる。

`asymmetric`の固定評価はミラーではなく、固定デッキ対sampling snapshotから抽選した
ランダムデッキで行う。

## 比較評価の用語

- 固定matchup: 比較する2モデルに同じデッキの組を使う
- 同一seed: 席交換する2試合でPython側の乱数系列を揃え、偶然の差を減らす
- 席交換: 同じ組合せをplayer 0/player 1を入れ替えて行い、先後・座席の偏りを相殺する

1つの固定matchup `(deck A, deck B)`は、席交換2戦に加えてモデルへ渡すデッキも交換した2戦、合計4戦で評価する。このため`eval_games_per_round`は4の倍数だけを受け付ける。generalistでは各ラウンドのmatchup集合を先にsampling snapshotから固定し、MCTSの全ゲーティング相手とrandom評価で再利用する。

MCTSでは自己対戦だけでなく、current best・checkpoint pool・randomとの固定matchup評価も
`runtime.workers`個のspawn workerで試合単位に並列実行する。各試合は席交換ペアと同じ
Python seedで個別に再シードされるため、workerへの割当順によって比較条件は変わらない。
評価workerの異常終了・timeoutも自己対戦と同じくworker単位で隔離してJSONLへ記録し、
ゲーティング試合が1局でも欠けたラウンドでは候補を採用しない。

同一seedで揃えられるのはPython側（MCTS determinization、random agent等）である。CG native engineはseed APIを公開していないため、内部shuffleまで同一になる保証はない。これは学習設定の切替方法ではなく、モデルA/Bの性能差を低分散で測るための評価条件である。
