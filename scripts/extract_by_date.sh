#!/usr/bin/env bash
# 日次zipを1日=1ジョブで抽出し、シャード名に日付を埋める。
# 名前が日付順=時系列順になるので、`sorted()`の末尾がそのまま最新日になる。
# 学習側はこの並びを前提に「末尾N枚を検証」する(train.py)。
set -u
OUT=${1:-data/bc/shards_dated}
JOBS=${2:-4}
mkdir -p "$OUT"
run_one() {
  local zip="$1" out="$2"
  local date; date=$(basename "$zip" | sed 's|.*episodes-||;s|\.zip$||')
  local stamp=${date//-/}
  local tmp="$out/.tmp_$stamp"
  # 既に完了しているものは飛ばす(再実行で作り直さない)
  if compgen -G "$out/shard_${stamp}_*.pt" > /dev/null; then
    echo "$(date +%T) skip $date (already extracted)"; return
  fi
  rm -rf "$tmp"; mkdir -p "$tmp"
  echo "$(date +%T) start $date"
  if uv run python -u scripts/extract_bc_samples.py --zips "$zip" --output "$tmp" --shard-size 50000 \
      > "$out/.log_$stamp" 2>&1; then
    local i=0
    for f in "$tmp"/shard_*.pt; do
      mv "$f" "$(printf '%s/shard_%s_%04d.pt' "$out" "$stamp" "$i")"; i=$((i+1))
    done
    mv "$tmp/extract_summary.json" "$out/summary_$stamp.json" 2>/dev/null
    rm -rf "$tmp"
    echo "$(date +%T) done $date ($i shards)"
  else
    echo "$(date +%T) FAILED $date"; rm -rf "$tmp"
  fi
}
export -f run_one
ls data/bc/raw/*.zip | sort | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} "$OUT"
echo "ALL DONE $(date +%T)  shards=$(ls "$OUT"/shard_*.pt 2>/dev/null | wc -l)"
