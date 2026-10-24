"""뉴스 발표 뒤 움직임 표 (실제 틱): 발표마다 발표 직전 중간가 대비 움직임 + 발표 순간 알 수 있는 조건
대상: USD·EUR·GBP high + moderate, 시각 없는 일정 제외, 2022-01 ~ 2026-10 (같은 시각 발표는 한 줄)

움직임 ($, 중간가): d1s d3s d5s d10s d30s d60s d5m d10m d15m d30m, up15/dn15 = 15분 안 최고 / 최저 (기준 대비)
조건: pre5m pre30m pre2h = 발표 전 움직임, ret1d ret5d = 금 하루 / 5일 수익률%, atr = 일봉 ATR14 (전날까지)
      spr0 = 발표 직전 스프레드, spr10s = 발표 뒤 10초 최대 스프레드
      서프라이즈: 같은 시각 지표마다 (실제 - 예측) / 그 지표의 2022~2024 (실제 - 예측) 표준편차 = z
      sur_hi = high 지표 z 목록 "이름:z" (분석 스크립트가 지표별 방향을 2022~2024로 정함)

사용법: python -m backtests.news_moves2      (results/news_moves2.csv, ../gpt_data/news/news_moves.csv)
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.gpt_news_export import calendar

ROOT = Path(__file__).resolve().parents[1]
OFFS = {"d1s": 1, "d3s": 3, "d5s": 5, "d10s": 10, "d30s": 30, "d60s": 60, "d5m": 300, "d10m": 600, "d15m": 900, "d30m": 1800}
PRES = {"pre5m": 300, "pre30m": 1800, "pre2h": 7200}


def releases():
    c = calendar()
    c = c[c.currency.isin(["USD", "EUR", "GBP"]) & c.importance.isin(["high", "moderate"]) & ~c.all_day]
    c = c[(c.time >= "2022-01-03") & (c.time < "2026-10-24") & (c.time.dt.weekday < 5)].copy()
    c["key"] = c.currency + " " + c.event.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    c["err"] = c.actual - c.forecast
    sd = c[c.time.dt.year <= 2024].groupby("key").err.std()
    c["z"] = (c.err / c.key.map(sd)).round(2)
    rows = []
    for t, g in c.groupby("time"):
        hi = g[g.importance == "high"]
        main = (hi if len(hi) else g).iloc[0]
        rows.append({"time": t, "importance": "high" if len(hi) else "moderate", "currency": main.currency, "key": main.key,
                     "n_events": len(g), "keys": "|".join(g.key),
                     "sur": "|".join(f"{k}:{z}" for k, z in zip(g.key, g.z) if pd.notna(z))})
    return pd.DataFrame(rows)


def month_job(args):
    y, m, items = args
    try:
        t, bid, ask = lb.load_month(y, m)
    except FileNotFoundError:
        return []
    mid = (bid + ask) / 2
    out = []
    for i, ms in items:
        r = int(np.searchsorted(t, ms))
        if r == 0 or r >= len(t) or ms - t[r - 1] > 60_000:
            continue
        ref = mid[r - 1]
        row = {"i": i, "ref": ref, "spr0": round(ask[r - 1] - bid[r - 1], 2)}
        e15 = int(np.searchsorted(t, ms + 900_000))
        if e15 - r < 5:
            continue
        for k, s in OFFS.items():
            j = int(np.searchsorted(t, ms + s * 1000, side="right")) - 1
            row[k] = round(mid[j] - ref, 2) if j >= r else 0.0
        row["up15"] = round(mid[r:e15].max() - ref, 2)
        row["dn15"] = round(mid[r:e15].min() - ref, 2)
        row["t_up15"] = round((t[r + int(np.argmax(mid[r:e15]))] - ms) / 1000, 1)
        row["t_dn15"] = round((t[r + int(np.argmin(mid[r:e15]))] - ms) / 1000, 1)
        e10 = int(np.searchsorted(t, ms + 10_000))
        row["spr10s"] = round((ask[r:e10] - bid[r:e10]).max(), 2) if e10 > r else np.nan
        for k, s in PRES.items():
            j = int(np.searchsorted(t, ms - s * 1000)) - 1
            row[k] = round(ref - mid[j], 2) if j >= 0 and ms - s * 1000 - t[j] < 600_000 else np.nan
        out.append(row)
    return out


def main():
    ev = releases()
    ms = ev.time.values.astype("datetime64[ms]").astype(np.int64)
    jobs = []
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        idx = np.flatnonzero((ms >= int(s.timestamp() * 1000)) & (ms < int(e.timestamp() * 1000)))
        if len(idx):
            jobs.append((y, m, [(int(i), int(ms[i])) for i in idx]))
    rows = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 4))) as ex:
        for (y, m, _), res in zip(jobs, ex.map(month_job, jobs)):
            rows += res
            print(f"  {y}-{m:02d} 완료", flush=True)
    mv = pd.DataFrame(rows).set_index("i")
    d = ev.join(mv, how="inner")
    # 금 일봉 맥락 (전날까지): 하루 / 5일 수익률, ATR14
    h4 = pd.read_csv(ROOT.parent / "gpt_data" / "XAUUSD" / "H4.csv.gz", index_col="time", parse_dates=True)
    day = h4.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    tr = np.maximum(day.high - day.low, np.maximum((day.high - day.close.shift()).abs(), (day.low - day.close.shift()).abs()))
    ctx = pd.DataFrame({"ret1d": day.close.pct_change() * 100, "ret5d": day.close.pct_change(5) * 100,
                        "atr": tr.rolling(14).mean()}).shift(1)        # 그날 이전 값만
    d = d.join(ctx.reindex(d.time.dt.normalize()).set_index(d.index).round(3))
    d.to_csv(ROOT / "results" / "news_moves2.csv", index=False, encoding="utf-8-sig")
    d.to_csv(ROOT.parent / "gpt_data" / "news" / "news_moves.csv", index=False, encoding="utf-8")
    print(len(d), "발표")


if __name__ == "__main__":
    main()
