#!/usr/bin/env bash
# 実行中の学習の完了を待ってから、次の設定を順に流す。
#
# GPUが空く時間を作らないための仕掛け。1本目を中断して2本目に移ると、
# 学習率を絞り切る最後のエポック(伸びが出やすい局面)を捨てることになるので、
# 完走させたうえで次を継ぐ。
#
# Usage: ./scripts/run_chain.sh <config> [config...]
set -u
log() { echo "[$(date +%F' '%T)] $*"; }

# 走っている学習があれば終わるまで待つ。CLIはネイティブクラッシュ時に自力で
# 再起動するので、プロセスが消えた=そのrunが終わった(または諦めた)と見なす。
while pgrep -f "training.cli train" > /dev/null; do
  sleep 60
done
log "no training running; starting chain"

for config in "$@"; do
  name=$(basename "$config" .yaml)
  out="outputs/runs/$(grep -m1 'output_dir:' "$config" | sed 's|.*outputs/runs/||' | tr -d ' ')"
  mkdir -p "$out"
  log "=== $name ==="
  if ! uv run python -m training.cli validate --config "$config" >> "$out/nohup.log" 2>&1; then
    log "SKIP $name (validate failed; see $out/nohup.log)"
    continue
  fi
  uv run python -u -m training.cli train --config "$config" >> "$out/nohup.log" 2>&1
  log "finished $name (exit $?)"
done
log "chain complete"
