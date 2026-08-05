#!/bin/bash
# ppo_train.pyを実行し、原因不明の低確率なネイティブクラッシュで落ちても
# 自動的に(最新チェックポイントから)再起動し続けるラッパー。
# ppo_train.py の __main__ が最新の {SELFPLAY_MODE}_round{N}.pt を自動検出して
# 再開するため、そのまま再実行するだけで続きから走る。
#
# Usage: scripts/run_ppo_with_retry.sh [configのパス]
#   (省略時は ppo_train.py の既定config。ログファイル名はconfig名から決まる)
set -u

CONFIG="${1:-configs/ppo_generalist.yaml}"
LOG_FILE="logs/train_ppo_$(basename "$CONFIG" .yaml).log"
MAX_ATTEMPTS=50

for i in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== attempt $i/$MAX_ATTEMPTS: $(date) ===" >> "$LOG_FILE"
  uv run python src/training/ppo/ppo_train.py "$CONFIG" >> "$LOG_FILE" 2>&1
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
