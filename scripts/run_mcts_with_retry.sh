#!/bin/bash
# train_mcts.pyを実行し、原因不明の低確率なネイティブクラッシュで落ちても
# 自動的に(最新チェックポイントから)再起動し続けるラッパー。
# train_mcts.py の __main__ が最新の {SELFPLAY_MODE}_round{N}.pt を自動検出して
# 再開するため、そのまま再実行するだけで続きから走る
# (ただしcheckpoint_pool/replay_bufferは引き継がれない簡易的な再開)。
#
# Usage: scripts/run_mcts_with_retry.sh [configのパス]
#   (省略時は train_mcts.py の既定config。ログファイル名はconfig名から決まる)
set -u

CONFIG="${1:-configs/mcts_mirror_lucario.yaml}"
LOG_FILE="logs/train_mcts_$(basename "$CONFIG" .yaml).log"
MAX_ATTEMPTS=50

for i in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== attempt $i/$MAX_ATTEMPTS: $(date) ===" >> "$LOG_FILE"
  uv run python src/training/mcts/train_mcts.py "$CONFIG" >> "$LOG_FILE" 2>&1
  exit_code=$?
  if [ "$exit_code" -eq 0 ]; then
    echo "=== training finished successfully (exit 0) ===" >> "$LOG_FILE"
    exit 0
  fi
  echo "=== crashed with exit $exit_code, retrying in 5s ===" >> "$LOG_FILE"
  sleep 5
done

echo "=== gave up after $MAX_ATTEMPTS attempts ===" >> "$LOG_FILE"
exit 1
