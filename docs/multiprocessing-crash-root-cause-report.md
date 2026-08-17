# マルチプロセス・ネイティブクラッシュ調査報告

## 1. 結論

観測された異常は1つの原因ではなく、少なくとも次の4系統に分かれる。

1. PyTorch初期化後の`fork`によるfutexデッドロック
2. MCTSの`SearchState`解放漏れによるネイティブ資源の累積
3. 親プロセス終了後に子が出す`BrokenPipeError`
4. 根本原因未特定の低頻度なSIGSEGV/SIGABRT、および外部SIGTERM

1と2は原因が特定され修正済みである。3は原因ではなく二次症状である。
4については、現存ログだけでは`libcg.so`、ctypes境界、メモリ圧迫、外部停止のどれかに断定できない。

調査日: 2026-08-05

## 2. 確定した原因

### 2.1 forkとPyTorch

Linuxの既定`multiprocessing`開始方式はforkである。PyTorchが内部スレッドのロックを
保持した状態をforkすると、子にはロック状態だけが複製され、解放するスレッドが存在しない場合がある。
全ワーカーが`futex_wait_queue`で停止した症状と一致する。

対策として全ワーカーを`spawn`で起動する。現在の共通実行基盤もspawnを明示している。

### 2.2 search_stepで生成したSearchStateの解放漏れ

`search_step()`は呼び出すたびにネイティブ側へ新しい探索状態を作る。
MCTSは各手で多数回呼ぶため、`search_release()`を呼ばない旧実装では短時間に状態が蓄積した。

修正前は並列度増加に伴ってSIGSEGV/SIGABRT率が悪化したが、解放追加後は
18並列で54/54試合が正常終了し、その後200試合規模でも199/200が正常だった。
したがって、過去の頻発クラッシュの主要因はこの解放漏れだった可能性が高い。

現在は`run_mcts`の`finally`で、生成した全search IDを解放し、その外側で`search_end()`を保証する。
対局全体も`battle_finish()`を`finally`で保証する。

## 3. 原因ではないログ

### BrokenPipeError

ログでは親プロセスが先に消えた後、複数ワーカーが結果送信時に`BrokenPipeError`を出していた。
これはパイプの受信側が存在しないことを示すだけで、親が消えた理由は示さない。

### resource_trackerのsemaphore警告

強制終了または親の異常終了後にIPC資源が通常の終了処理を通らなかった結果であり、
これ自体を最初の原因とはみなせない。

## 4. 終了コードの読み方

確認された終了コードには異なる意味がある。

- 139または子の-11: SIGSEGV
- 子の-6: SIGABRT
- 143: SIGTERM
- 1: Python例外やラッパー内の通常エラー

SIGTERMまで`cg`のネイティブクラッシュとして扱うのは誤りである。

## 5. 旧Pool実装の問題

旧実装は全試合を`apply_async`で一括投入し、投入順に`get(timeout=...)`していた。

- timeoutは待機を止めるだけで、実行中の試合を停止しない
- ハングした仕事がワーカーを占有し続ける
- Poolは異常終了ワーカーを自動補充するため、終了コードが見えにくい
- 完了済みの後続結果があっても投入順待機で回収が遅れる
- 親の結果キャッシュとIPCキューへ大きな学習サンプルが滞留しうる

## 6. 実施した耐障害化

`src/training/common/parallel_games.py`へ、MCTS/PPO共通の監視可能な常駐ワーカーを実装した。

- spawnで起動
- モデルはワーカー初期化時に一度だけロードし、`assign=True`で受信重みを再利用して二重保持を避ける
- PPOのgamma/GAE lambdaと両algorithmのrun seedをspawnワーカーへ明示的に渡す
- 各試合を`run seed + game index`で再シードし、担当ワーカーによる乱数差を避ける
- 1ワーカーへ同時に1試合だけ割り当て
- 専用Pipeで完了順に結果回収
- PID、担当試合、開始時刻、定期RSSを親が管理
- `output_dir/worker_events.jsonl`へ開始・割当・完了・例外・timeout・終了をJSON Linesで逐次flush
- `worker_stopped`へexit code、各試合eventへgame index・PID・RSS・経過時間を保存
- Python例外は試合単位で報告してワーカーを継続利用
- timeout時は対象ワーカーを実際にterminate/killして再生成
- SIGSEGV等では`process.exitcode`を記録して再生成
- 親からの試合割当時にPipeが切れる競合もtrainer全体へ伝播させず、ワーカー異常として回収
- ラウンド期限では全ワーカーを回収

この変更は根本原因未特定のSIGSEGVを直すものではない。異常を試合単位へ封じ込め、
原因別の観測値を残し、他の正常試合を継続させるための修正である。

## 7. 未解決の根本原因

解放漏れ修正後にも、PPOログでexit 139、exit 143、exit 1が低頻度で確認されている。
現在のログにはネイティブスタックがなく、同一原因とは断定できない。

根本原因の確定には次が必要である。

1. ~~親と各ワーカーのPID、RSS、終了コード、担当試合を時系列保存~~（`worker_events.jsonl`として実装済み）
2. 同じcgroupの`memory.events`を実行開始から終了まで保存
3. core dumpを有効化し、SIGSEGV時にgdbでネイティブスタックを取得
4. 配布C++ソースをASan/UBSan付きでビルドできる場合、同条件を再現
5. `battle_start/select/finish`と`search_begin/step/release/end`をPID・ID付きで記録
6. PPOのみ、MCTSのみ、探索APIだけの最小再現で比較

サンドボックスでptraceやcore dumpが禁止される場合は、ASanビルドの標準エラー出力が
最も有力な代替手段になる。

## 8. テスト

`tests/test_parallel_games.py`は人工的に次を発生させ、残りの試合が完走することを確認する。

- 正常終了
- Python例外
- timeout
- SIGSEGV
- 異常ワーカー交換後の後続タスク実行

`tests/test_worker_event_log.py`では正常試合について、game index、PID、RSS、exit codeとrun contextがJSONLへ残ることも確認する。異常時は`reason`が`python_exception`、`game_timeout`、`round_timeout`等になり、SIGSEGV等は`worker_exit.exit_code`で区別できる。

このテストは耐障害化の正しさを検証するが、`libcg.so`内の未解決SIGSEGV原因を証明するものではない。

2026-08-06に現行CLIから追加の実対局スモークを実施した。

- PPO: 2 spawn worker、2試合、2/2完走、274 sample、PPO更新・4試合固定評価・final保存成功。worker RSSは約552〜555 MB、exit codeはいずれも0
- MCTS: 2 spawn worker、2試合、search count 2、2/2完走、347 sample、更新・4試合固定arena・replay/training state/final保存成功。worker RSSは約550〜553 MB、exit codeはいずれも0
- MCTS再開: round 1から1 replay roundとoptimizerを復元し、finalを再生成できた

したがってPPO/MCTSのspawn並列経路は実対局で動作確認済みである。ただし2試合の成功は
数百試合で低頻度のネイティブクラッシュがゼロになる保証ではなく、その場合はワーカー交換と
run再開で学習を継続する設計である。
