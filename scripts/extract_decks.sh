#!/usr/bin/env bash
# 候補デッキごとの学習データを、1日=1ジョブで並列抽出する。
#
# デッキの数だけ走査を繰り返すとJSONの解析が重複する(候補5件で8時間)。
# extract_bc_samples.py の --deck-spec で1回の走査から全デッキへ振り分けるので、
# ここは日付方向の並列だけを担う。
#
# 出力は <OUT>/<デッキ名>/shard_YYYYMMDD_NNNN.pt。学習側は日付順に並ぶ前提で
# 末尾(最新日)を検証に使うため、シャード名に日付を入れておく必要がある。
#
# Usage: ./scripts/extract_decks.sh <出力先> <並列数> <リーダーボードCSV> <最低スコア> <デッキ名...>
set -u
OUT=${1:?出力先}
JOBS=${2:-3}
RATINGS=${3:?リーダーボードCSV}
MIN_RATING=${4:-0}
shift 4
DECKS=("$@")
FROM_DAY=${FROM_DAY:-20260706}
TO_DAY=${TO_DAY:-99999999}

SPECS=()
for d in "${DECKS[@]}"; do SPECS+=(--deck-spec "$d=decks/archetypes/$d.csv"); done

run_one() {
  local zip="$1" out="$2" ratings="$3" min_rating="$4"; shift 4
  local date stamp
  date=$(basename "$zip" | sed 's|.*episodes-||;s|\.zip$||')
  stamp=${date//-/}
  # 既にこの日の出力があるデッキは飛ばせないので(振り分けは一括)、1デッキでも
  # 残っていれば済んだものとみなす。再実行で作り直さないための判定。
  if compgen -G "$out/*/shard_${stamp}_*.pt" > /dev/null; then
    echo "$(date +%T) skip $date"; return
  fi
  echo "$(date +%T) start $date"
  if uv run python -u scripts/extract_bc_samples.py \
      --zips "$zip" --output "$out" --shard-size 50000 --shard-prefix "$stamp" \
      --min-jaccard 0.5 --ratings "$ratings" --min-winner-rating "$min_rating" \
      "$@" > "$out/.log_$stamp" 2>&1; then
    echo "$(date +%T) done $date"
  else
    echo "$(date +%T) FAILED $date (see $out/.log_$stamp)"
  fi
}
export -f run_one

mkdir -p "$OUT"
ls data/bc/raw/*.zip | sort |
  while read -r z; do
    d=$(basename "$z" | sed 's|.*episodes-||;s|\.zip$||;s|-||g')
    [ "$d" \< "$FROM_DAY" ] || [ "$d" \> "$TO_DAY" ] || echo "$z"
  done |
  xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} "$OUT" "$RATINGS" "$MIN_RATING" "${SPECS[@]}"

echo "ALL DONE $(date +%T)"
for d in "${DECKS[@]}"; do
  printf "%-14s %s枚\n" "$d" "$(ls "$OUT/$d"/shard_*.pt 2>/dev/null | wc -l)"
done
