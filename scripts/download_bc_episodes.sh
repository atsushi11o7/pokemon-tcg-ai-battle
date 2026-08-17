#!/usr/bin/env bash
# 日次エピソードデータセットを新しい順に取得する。
# 途中で止めても取れたぶんだけで抽出・学習できるよう、新しい日付から並べている。
# 公開範囲は 2026-06-16 〜 2026-08-12。既取得は data/bc/raw の中身で判定する。
set -u
RAW=data/bc/raw
FROM=${1:-2026-07-16}
TO=${2:-2026-08-12}
mkdir -p "$RAW"
DATES=$(python3 - "$FROM" "$TO" <<'PY'
import datetime, sys
a, b = (datetime.date.fromisoformat(x) for x in sys.argv[1:3])
days = [a + datetime.timedelta(days=i) for i in range((b - a).days + 1)]
print("\n".join(d.isoformat() for d in sorted(days, reverse=True)))
PY
)
TOTAL=$(echo "$DATES" | wc -l)
i=0
for D in $DATES; do
  i=$((i+1))
  F="$RAW/pokemon-tcg-ai-battle-episodes-$D.zip"
  if [ -s "$F" ]; then echo "[$i/$TOTAL] $D already present"; continue; fi
  echo "[$i/$TOTAL] $(date +%T) downloading $D"
  if kaggle datasets download "kaggle/pokemon-tcg-ai-battle-episodes-$D" -p "$RAW" --force >/dev/null 2>&1; then
    echo "[$i/$TOTAL] $(date +%T) done $D ($(du -h "$F" 2>/dev/null | cut -f1))"
  else
    echo "[$i/$TOTAL] FAILED $D"
  fi
done
echo "ALL DONE $(date +%T)  zips=$(ls "$RAW"/*.zip | wc -l)  size=$(du -sh "$RAW" | cut -f1)"
