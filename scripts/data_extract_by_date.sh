#!/usr/bin/env bash
# 日次zipを1日=1ジョブで抽出し、シャード名に日付を埋める。
# 名前が日付順=時系列順になるので、`sorted()`の末尾がそのまま最新日になる。
# 学習側はこの並びを前提に「末尾N枚を検証」する(train.py)。
set -u
OUT=${1:-data/bc/shards_dated}
JOBS=${2:-4}
# 3番目以降は data_extract.py へそのまま渡す(例: --imitate-loser)。
shift 2 2>/dev/null || true
EXTRA=("$@")
# 環境変数で日付範囲を絞れる。質の低い期間(6月は勝者レート中央値637〜835で、
# 現在の自分より弱い手本)を混ぜたくない場合に使う。
FROM_DAY=${FROM_DAY:-00000000}
TO_DAY=${TO_DAY:-99999999}
mkdir -p "$OUT"
run_one() {
  local zip="$1" out="$2"
  shift 2
  local date; date=$(basename "$zip" | sed 's|.*episodes-||;s|\.zip$||')
  local stamp=${date//-/}
  local tmp="$out/.tmp_$stamp"
  # 既に完了しているものは飛ばす(再実行で作り直さない)
  if compgen -G "$out/shard_${stamp}_*.pt" > /dev/null; then
    echo "$(date +%T) skip $date (already extracted)"; return
  fi
  rm -rf "$tmp"; mkdir -p "$tmp"
  echo "$(date +%T) start $date"
  if uv run python -u scripts/data_extract.py --zips "$zip" --output "$tmp" --shard-size 50000 "$@" \
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
ls data/bc/raw/*.zip | sort |
  while read -r z; do d=$(basename "$z" | sed "s|.*episodes-||;s|\.zip$||;s|-||g");
    [ "$d" \< "$FROM_DAY" ] || [ "$d" \> "$TO_DAY" ] || echo "$z"; done | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} "$OUT" "${EXTRA[@]}"
echo "ALL DONE $(date +%T)  shards=$(ls "$OUT"/shard_*.pt 2>/dev/null | wc -l)"
