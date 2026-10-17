"""뉴스 필터 비교: 마틴 전략(RSI50 + H1 SAR, SL 7.5 / TP 3)을 뉴스 시간에 쉬면 나아지는가 (2022-01 ~ 현재, 실제 틱)

뉴스 일정: mql5/ExportCalendar.mq5로 MT5 경제 캘린더를 내보낸 calendar_usd.csv (USD, 중요도 high만 사용)
  - MT5 캘린더는 과거 일정도 지금 서버 시차(GMT+3)로 표시함. 실제 서버시간은 미국 서머타임이 아닐 때 GMT+2라서
    그 기간 일정은 1시간 당겨서 보정 (NFP 발표 때 금값이 튄 5분봉으로 확인: 겨울에는 캘린더 16:30, 실제 15:30)

비교하는 필터 (신호를 미리 거른 뒤 시뮬레이션 -> 쉬는 동안 잡을 수 있게 된 다음 신호도 반영):
  없음              : 현재 규칙 그대로
  미국장 제외       : 서버시간 15:00~21:00 신호 무시 (demo_trader.py의 NO_ENTRY_FROM/UNTIL)
  뉴스 ±30분        : 중요 뉴스 발표 30분 전 ~ 30분 후 신호 무시
  뉴스 ±60분        : 중요 뉴스 발표 60분 전 ~ 60분 후 신호 무시
  뉴스일만 미국장 제외: 15:00~21:00 사이에 중요 뉴스가 있는 날만 15:00~21:00 신호 무시

결과: 연도별 PF, 2025-01부터 10,000$ / 시작 0.30 lot 고정 일일 마틴 (martin_scaled.py와 같은 규칙)
사용법: python -m backtests.news_filter   (long_period를 먼저 한 번 실행해 틱이 저장돼 있어야 함)
        캘린더 파일 위치가 다르면 환경변수 CALENDAR_CSV로 지정
"""
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.martin_scaled import NO_ENTRY_HOURS, martin_money, simulate
from backtests.scalp_ideas import m5_bars
from backtests.strategies import pf, resample

CALENDAR_CSV = Path(os.environ.get("CALENDAR_CSV") or
                    Path(os.environ["APPDATA"]) / "MetaQuotes" / "Terminal" / "Common" / "Files" / "calendar_usd.csv")
CACHE_DIR = lb.TICK_DIR.parent / "news_filter"
YEARS = [2022, 2023, 2024, 2025, 2026]
MONEY_START = pd.Timestamp("2025-01-01")
NY = ZoneInfo("America/New_York")


def load_news():
    c = pd.read_csv(CALENDAR_CSV, encoding="cp949")
    c = c[c.importance == "high"].copy()
    t = pd.to_datetime(c.time, format="%Y.%m.%d %H:%M")
    # 미국 서머타임이 아닌 날은 1시간 당김 (캘린더 GMT+3 고정 -> 서버 GMT+2)
    winter = np.array([ts.to_pydatetime().replace(tzinfo=NY).dst().total_seconds() == 0 for ts in t])
    c["time"] = t - pd.to_timedelta(winter.astype(int), unit="h")
    c = c[(c.time.dt.hour > 0) | (c.time.dt.minute > 0)]     # 시각 없는 하루 종일 일정(00:00) 제외
    return c


def news_mask(sig_ms, news_ms, before, after):
    """각 신호가 [뉴스 - before, 뉴스 + after] 안에 있으면 True"""
    i = np.searchsorted(news_ms, sig_ms + before, side="right")   # 신호 + before 이하인 마지막 뉴스
    last = np.where(i > 0, news_ms[np.maximum(i - 1, 0)], np.iinfo(np.int64).min // 2)
    return sig_ms - last <= after


def main():
    news = load_news()
    print(f"중요 뉴스 {len(news):,}건 ({news.time.min():%Y-%m-%d} ~ {news.time.max():%Y-%m-%d})", flush=True)
    m5 = m5_bars()
    h1 = resample(m5, "1h")
    sigs = lb.signals(m5, h1)
    del m5, h1
    st = np.array([s[0] for s in sigs], dtype=np.int64)
    hour = pd.to_datetime(st, unit="ms").hour
    news_ms = np.sort(news.time.values.astype("datetime64[ms]").astype(np.int64))
    us = np.isin(hour, list(NO_ENTRY_HOURS))
    us_news_days = set(news[news.time.dt.hour.isin(list(NO_ENTRY_HOURS))].time.dt.date)
    day = pd.to_datetime(st, unit="ms").date

    m30, m60 = 30 * 60_000, 60 * 60_000
    filters = {
        "없음": np.zeros(len(st), bool),
        "미국장 제외 (15~21시)": us,
        "뉴스 ±30분": news_mask(st, news_ms, m30, m30),
        "뉴스 ±60분": news_mask(st, news_ms, m60, m60),
        "뉴스일만 미국장 제외": us & np.array([d in us_news_days for d in day]),
    }
    CACHE_DIR.mkdir(exist_ok=True)
    rows = []
    for i, (name, skip) in enumerate(filters.items()):
        cache = CACHE_DIR / f"trades_{i}.csv"
        if cache.exists():
            tr = pd.read_csv(cache)
            tr["open"] = pd.to_datetime(tr.open_ms, unit="ms")
        else:
            tr = simulate([s for s, k in zip(sigs, skip) if not k])
            tr.drop(columns=["open"]).to_csv(cache, index=False)
        y = tr.open.dt.year
        row = {"필터": name, "무시한 신호": int(skip.sum()), "거래수": len(tr), "PF 전체": pf(tr.pnl_oz)}
        for yr in YEARS:
            row[f"PF {yr}"] = pf(tr.pnl_oz[y == yr])
        d = martin_money(tr[tr.open >= MONEY_START], scaled=False)
        eq = np.r_[10_000.0, d.balance.values]
        peak = np.maximum.accumulate(eq)
        daily = d.groupby("day").profit.sum()
        row.update({"2025~ 최종잔고": round(d.balance.iat[-1]), "최대낙폭%": round(((peak - eq) / peak).max() * 100, 1),
                    "최악의날": round(daily.min()), "손실일": f"{(daily < 0).sum()}/{len(daily)}"})
        rows.append(row)
        print(f"완료: {name}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv("news_filter.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
