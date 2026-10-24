"""뉴스 일정 = MT5 캘린더 내보내기(calendar_all.csv) + ForexFactory 주간 일정

ForexFactory: nfs.faireconomy.media/ff_calendar_thisweek.json (HK 프로그램 뉴스 목록과 같은 곳).
  MT5 캘린더에 빠지는 영국·유럽 지표(예: 2026-10-21 영국 CPI, 10-23 영국 소매판매·독일 속보 PMI)가 있음.
  이번 주 일정만 주고 실제치는 없음 -> 받을 때마다 data/ff_calendar.csv에 쌓아 둠 (FF_EVERY마다 한 번만 받음, 자주 받으면 막힘)
시각은 모두 서버시간 (= 뉴욕 + 7시간, 서머타임 따라 같이 바뀜)
"""
import json
import logging
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FF_CSV = Path(__file__).resolve().parent.parent / "data" / "ff_calendar.csv"
FF_EVERY = 6 * 3600
MT5_CSV = Path(os.environ["APPDATA"]) / "MetaQuotes" / "Terminal" / "Common" / "Files" / "calendar_all.csv"
IMPACT = {"High": "high", "Medium": "moderate", "Low": "low", "Holiday": "none"}
log = logging.getLogger("calendar")
_last_try = 0.0


def ff_update(force=False):
    """ForexFactory 이번 주 일정을 받아 FF_CSV에 합침. 실패하면 이유 문구, 아니면 ''"""
    global _last_try
    if not force and (time.time() - _last_try < 600 or
                      (FF_CSV.exists() and time.time() - FF_CSV.stat().st_mtime < FF_EVERY)):
        return ""
    _last_try = time.time()
    try:
        req = urllib.request.Request(FF_URL, headers={"User-Agent": "Mozilla/5.0"})
        rows = json.loads(urllib.request.urlopen(req, timeout=20).read().decode("utf-8"))
    except Exception as e:
        log.warning(f"ForexFactory 일정 받기 실패: {e}")
        return f"ForexFactory 일정 받기 실패 ({e})"
    d = pd.DataFrame(rows)
    t = pd.to_datetime(d.date, utc=True).dt.tz_convert("America/New_York")
    d = pd.DataFrame({"time": (t + pd.Timedelta(hours=7)).dt.tz_localize(None), "currency": d.country,
                      "importance": d.impact.map(IMPACT).fillna("low"), "event": d.title,
                      "forecast": d.forecast, "previous": d.previous})
    if FF_CSV.exists():
        d = pd.concat([pd.read_csv(FF_CSV, parse_dates=["time"]), d])
    d = d.drop_duplicates(["time", "currency", "event"], keep="last").sort_values("time")
    FF_CSV.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(FF_CSV, index=False, encoding="utf-8")
    log.info(f"ForexFactory 일정 {len(rows)}건 받음 ({FF_CSV.name} 누적 {len(d)}건)")
    return ""


def ff_events():
    if not FF_CSV.exists():
        return pd.DataFrame(columns=["time", "currency", "importance", "event"])
    return pd.read_csv(FF_CSV, parse_dates=["time"])


def mt5_events():
    if not MT5_CSV.exists():
        return pd.DataFrame(columns=["time", "currency", "importance", "event"])
    c = pd.read_csv(MT5_CSV, encoding="cp949")
    c["time"] = pd.to_datetime(c.time, format="%Y.%m.%d %H:%M")   # 앞으로의 일정은 지금 시차 그대로라 보정 없음
    return c


def news_times(currencies=("USD", "EUR", "GBP"), importances=("moderate", "high")):
    """두 일정을 합친 발표 시각 (서버시간 ms, 정렬, 중복 제거)"""
    ff_update()
    out = []
    for c in (mt5_events(), ff_events()):
        c = c[c.currency.isin(currencies) & c.importance.isin(importances)]
        out.append(c.time.values.astype("datetime64[ms]").astype(np.int64))
    return np.unique(np.concatenate(out))


def near_news(t, minutes, **kw):
    """서버 시각 t가 발표 앞뒤 minutes분 안이면 True"""
    nt = news_times(**kw)
    ms = pd.Timestamp(t).to_datetime64().astype("datetime64[ms]").astype(np.int64)
    j = int(np.searchsorted(nt, ms))
    w = minutes * 60_000
    return (j < len(nt) and nt[j] - ms <= w) or (j > 0 and ms - nt[j - 1] <= w)
