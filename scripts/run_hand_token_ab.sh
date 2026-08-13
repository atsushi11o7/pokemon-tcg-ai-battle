#!/bin/bash
# 手札トークン数のA/B。各設定で同量を抽出し、同条件でBCを学習して方策精度を比べる。
set -u
cd /workspace
EPISODES=${EPISODES:-7000}
EPOCHS=${EPOCHS:-3}

log() { echo "[$(date +%T)] $*"; }

for H in 1 8; do
  log "=== HAND_TOKENS=$H ==="
  sed -i "s/^HAND_TOKENS = .*/HAND_TOKENS = $H/" src/training/common/model_config.py

  rm -rf "data/bc/ab_h$H"
  log "extract (episodes=$EPISODES)"
  uv run python -u scripts/extract_bc_samples.py \
    --zips data/bc/raw/pokemon-tcg-ai-battle-episodes-2026-08-11.zip \
    --output "data/bc/ab_h$H" --shard-size 50000 --max-episodes "$EPISODES" \
    > "data/bc/ab_h$H.extract.log" 2>&1
  log "extracted: $(ls data/bc/ab_h$H/shard_*.pt 2>/dev/null | wc -l) shards"

  rm -rf "outputs/runs/bc_ab_h$H"
  mkdir -p "outputs/runs/bc_ab_h$H"
  sed -e "s|shard_dir: data/bc/shards|shard_dir: data/bc/ab_h$H|" \
      -e "s|output_dir: outputs/runs/bc_toplayers|output_dir: outputs/runs/bc_ab_h$H|" \
      -e "s|name: bc_toplayers|name: bc_ab_h$H|" \
      -e "s|rounds: 3|rounds: $EPOCHS|" \
      -e "s|initial_checkpoint: .*|initial_checkpoint: null|" \
      configs/bc_toplayers.yaml > "/tmp/bc_ab_h$H.yaml"

  log "train"
  uv run python -u -m training.bc.train "/tmp/bc_ab_h$H.yaml" \
    > "outputs/runs/bc_ab_h$H/train.log" 2>&1 || log "train FAILED for H=$H"
  grep -E "^  val|^shards" "outputs/runs/bc_ab_h$H/train.log" | tail -6
done

log "=== 結果 ==="
for H in 1 8; do
  acc=$(grep -oE "val accuracy=[0-9.]+%" "outputs/runs/bc_ab_h$H/train.log" 2>/dev/null | tail -1)
  echo "  HAND_TOKENS=$H  $acc"
done
