"""CalendarLive 서비스 기록 요약: 실제치가 발표 시각보다 몇 초 늦게 MT5에 들어오는지 (읽기만 함)

기록: Common\\Files\\calendar_live.csv (mql5/CalendarLive.mq5 서비스가 씀)
  delay_ms = 받은 시각 - 발표 시각. 받은 시각은 PC 시계로 계산한 서버시간이라, clock_ms(받은 시각 - 그때 금 마지막 틱 시각)도 같이 봄
  (장중 금 틱은 1초에 여러 번 와서 clock_ms가 크게 +면 PC 시계가 빠른 것)
사용법: python scripts/calendar_latency.py           (USD·EUR·GBP moderate 이상)
        python scripts/calendar_latency.py --all     (전부)
"""
import os
import sys
from pathlib import Path

import pandas as pd

COMMON = Path(os.environ["APPDATA"]) / "MetaQuotes" / "Terminal" / "Common" / "Files"


def main():
    f = COMMON / "calendar_live.csv"
    if not f.exists():
        raise SystemExit(f"기록 없음: {f} (MT5 내비게이터 > 서비스에서 CalendarLive를 켰는지 확인)")
    st = COMMON / "calendar_live_status.txt"
    if st.exists():
        print("서비스 상태:", st.read_text(encoding="cp949").strip())
    d = pd.read_csv(f, encoding="cp949")
    for c in ("recv_server", "tick_ms"):
        d[c] = pd.to_datetime(d[c], format="%Y.%m.%d %H:%M:%S.%f", errors="coerce")
    d["event_time"] = pd.to_datetime(d.event_time, format="%Y.%m.%d %H:%M:%S")
    d["clock_ms"] = (d.recv_server - d.tick_ms).dt.total_seconds() * 1000
    if "--all" not in sys.argv:
        d = d[d.currency.isin(["USD", "EUR", "GBP"]) & d.importance.isin(["moderate", "high"])]
    first = d.sort_values("recv_server").drop_duplicates("value_id")      # 처음 들어온 것만 (뒤의 수정치 제외)
    first = first[first.delay_ms.between(-60_000, 3_600_000)]
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\n실제치 {len(first)}건 (지표별 첫 수신)")
    print(first.groupby(["currency", "importance"]).delay_ms.describe(percentiles=[0.5, 0.9])[["count", "min", "50%", "90%", "max"]]
          .div([1, 1000, 1000, 1000, 1000]).round(2).rename(columns=lambda c: c if c == "count" else f"{c} 초").to_string())
    cols = ["event_time", "currency", "importance", "event", "actual", "forecast", "impact", "delay_ms", "clock_ms"]
    print("\n최근 20건")
    print(first.tail(20)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
