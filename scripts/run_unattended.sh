#!/bin/bash
# 収集 -> スナップショット構築 -> PPO を順に実行する。無人実行用。
set -u
cd /workspace

log() { echo "[$(date +%F' '%T)] $*"; }

log "step 1/3: collect kaggle meta"
uv run python -u scripts/collect_kaggle_meta.py \
  --top-teams 30 --mid-teams 20 --submissions-per-team 2 \
  --episodes-per-submission 20 --recent-days 14 \
  --replay-timeout-seconds 45 --replay-attempts 8 \
  --replay-cooldown-seconds 60 --replay-throttle-seconds 0.5 \
  >> data/meta/collect.log 2>&1
log "collect exited with $?; replays=$(ls data/meta/raw/replays | wc -l)"

log "step 2/3: build deck registry"
if ! uv run python -u scripts/build_deck_registry.py >> data/meta/registry.log 2>&1; then
  log "registry build FAILED; aborting before training"
  exit 1
fi
tail -5 data/meta/registry.log

log "step 3/3: train ppo_generalist"
uv run python -u -m training.cli validate --config configs/ppo_generalist.yaml || exit 1
uv run python -u -m training.cli train --config configs/ppo_generalist.yaml \
  >> outputs/runs/ppo_generalist/nohup.log 2>&1
log "training exited with $?"
