#!/bin/sh
# 틱 다운로드를 한 달씩 새 프로세스로 (메모리를 한 달치만 씀). 사용법: sh download_monthly.sh XETUSD US30 ...
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
cur=$(date +%Y-%m)
for s in "$@"; do
  while true; do
    out=$(python -m backtests.download_symbol --slow --one --quiet "$s" 2>&1)
    echo "$out"
    echo "$out" | grep -q "한 달 받음" || break
    echo "$out" | grep -q "$cur" && break
  done
done
# 끝나면 종목별로 받은 기간만 출력 (텔레그램 알림 없음)
for s in "$@"; do echo "$s 완료: $(ls data/ticks_z/$s | head -1) ~ $(ls data/ticks_z/$s | tail -1), $(du -sh data/ticks_z/$s | cut -f1)"; done
