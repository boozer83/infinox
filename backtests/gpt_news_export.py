"""GPT 봇용 뉴스 데이터: ../gpt_data/news/
- calendar.csv: MT5 경제 캘린더 전체 (UTF-8), time = 미국 서머타임 보정한 서버시간, time_raw = 캘린더 원래 시각
- XAUUSD_news_ticks.csv.gz: 미국 중요도 high 발표 시각마다 앞 5분 ~ 뒤 35분 금 틱 (Bid/Ask)

사용법: python -m backtests.gpt_news_export
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.long_period as lb

OUT = Path(__file__).resolve().parents[2] / "gpt_data" / "news"
CAL = os.path.join(os.environ["APPDATA"], "MetaQuotes/Terminal/Common/Files/calendar_all.csv")
BEFORE_MS, AFTER_MS = 5 * 60_000, 35 * 60_000


def calendar():
    c = pd.read_csv(CAL, encoding="cp949")
    c["event"] = c.event.str.strip()
    raw = pd.to_datetime(c.time, format="%Y.%m.%d %H:%M")
    # 캘린더는 과거 일정도 지금 시차(GMT+3)로 보여줌 -> 미국 서머타임이 아닌 날은 1시간 당김
    dst = raw.dt.tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward").map(lambda x: bool(x.dst()))
    c.insert(0, "time_raw", raw)
    c["time"] = raw - pd.to_timedelta((~dst.values).astype(int), unit="h")
    c["all_day"] = (raw.dt.hour == 0) & (raw.dt.minute == 0)       # 시각 없는 하루 종일 일정
    cols = ["time", "time_raw", "all_day", "importance", "country", "currency", "event", "actual", "forecast", "previous", "event_id"]
    return c[cols].sort_values("time").reset_index(drop=True)


def news_ticks(cal):
    ev = cal[(cal.currency == "USD") & (cal.importance == "high") & ~cal.all_day & (cal.time.dt.weekday < 5)]
    times = np.sort(ev.time.unique())
    ms_all = times.astype("datetime64[ms]").astype(np.int64)
    out = []
    for (y, m), grp in pd.Series(ms_all, index=pd.DatetimeIndex(times)).groupby(lambda d: (d.year, d.month)):
        try:
            t, bid, ask = lb.load_month(y, m)
        except FileNotFoundError:
            continue
        for ms in grp.values:
            a, b = np.searchsorted(t, [ms - BEFORE_MS, ms + AFTER_MS])
            if b - a < 10:
                continue
            out.append(pd.DataFrame({"event_time": pd.to_datetime(ms, unit="ms"),
                                     "time": pd.to_datetime(t[a:b], unit="ms"),
                                     "bid": bid[a:b], "ask": ask[a:b]}))
    return pd.concat(out, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cal = calendar()
    cal.to_csv(OUT / "calendar.csv", index=False, encoding="utf-8")
    print("calendar", len(cal), cal.time.min(), cal.time.max())
    tk = news_ticks(cal)
    tk.to_csv(OUT / "XAUUSD_news_ticks.csv.gz", index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
    print("ticks", len(tk), "events", tk.event_time.nunique())


if __name__ == "__main__":
    main()
