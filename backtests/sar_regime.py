"""SAR 단타: 상승장이면 매수만, 하락장이면 매도만, 횡보면 지금처럼 (2026-10-28 사용자 아이디어, 실제 틱)

장 판단 (신호 시점에 완성된 봉만 사용):
  h4sar   : 4시간봉 SAR 방향과 같은 신호만 (항상)
  d1sar   : 일봉 SAR 방향과 같은 신호만 (항상)
  h4adx   : 4시간봉 ADX(14) >= 25면 추세장 -> 종가가 EMA50 위면 매수만 / 아래면 매도만, 아니면 횡보 (둘 다)
  d1adx   : 일봉으로 같은 방식
  today   : 오늘(서버 날짜) 시가 대비 지금 가격이 일봉 ATR(14, 어제까지) x 0.5 넘게 올랐으면 매수만 / 내렸으면 매도만
  yday    : 어제 일봉이 ATR x 0.5 넘게 양봉이면 매수만 / 음봉이면 매도만
나머지는 지금 봇 그대로 (backtests.sar_martin.trades, 2022~2024 ATR 기준 가격비례). 기준은 results/sar_us_now_YYYY.csv
결과 results/sar_regime_<방식>_YYYY.csv

사용법: python -m backtests.sar_regime 2025 h4sar d1sar h4adx d1adx today yday
        python -m backtests.sar_regime sum
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.sar_martin as sm
from backtests.gold_rest_others import resample
from indicators import atr, ema, psar

RES = Path(__file__).resolve().parent.parent / "results"
MODE = "h4sar"
NAMES = {"h4sar": "4시간봉 SAR 방향만", "d1sar": "일봉 SAR 방향만", "h4adx": "4시간봉 ADX 25+면 EMA50 방향만",
         "d1adx": "일봉 ADX 25+면 EMA50 방향만", "today": "오늘 0.5 ATR 넘게 움직이면 그 방향만",
         "yday": "어제 0.5 ATR 넘는 봉이면 그 방향만"}
_orig = fe.bar_close_signals


def adx(b, n=14):
    up, dn = b.high.diff(), -b.low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(b, n)
    pdi = 100 * pd.Series(pdm).ewm(alpha=1 / n, adjust=False).mean().values / a.values
    ndi = 100 * pd.Series(ndm).ewm(alpha=1 / n, adjust=False).mean().values / a.values
    dx = 100 * np.abs(pdi - ndi) / (pdi + ndi)
    return pd.Series(dx).ewm(alpha=1 / n, adjust=False).mean().values


def daily(m5):
    d = m5.set_index("time").resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    pc = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    d["datr"] = tr.rolling(14).mean().shift(1)
    return d


def regime(m5, sig):
    """신호마다 허용 방향: 1 매수만, -1 매도만, 0 둘 다"""
    t = sig[["t"]]
    if MODE in ("h4sar", "d1sar", "h4adx", "d1adx"):
        mins = 240 if MODE.startswith("h4") else 1440
        b = resample(m5, mins)
        if MODE.endswith("sar"):
            d = np.where(psar(b.high.values, b.low.values), 1, -1)
        else:
            d = np.where(adx(b) >= 25, np.where(b.close.values > ema(b.close, 50).values, 1, -1), 0)
        q = pd.DataFrame({"end": b.time + pd.Timedelta(minutes=mins), "d": d})
        return pd.merge_asof(t, q, left_on="t", right_on="end", direction="backward").d.fillna(0).values
    d = daily(m5)
    day = t.t.dt.normalize() - pd.Timedelta(minutes=5) * (t.t.dt.hour.eq(0) & t.t.dt.minute.eq(0))
    x = pd.DataFrame({"time": day.dt.normalize()}).merge(d, on="time", how="left")
    if MODE == "today":
        px = pd.merge_asof(t, pd.DataFrame({"t": m5.time + pd.Timedelta(minutes=5), "c": m5.close}), on="t").c.values
        mv = (px - x.open.values) / x.datr.values
    else:
        prev = d.assign(time=d.time.shift(-1))[["time", "open", "close"]].rename(columns={"open": "po", "close": "pcl"})
        y = x[["time", "datr"]].merge(prev, on="time", how="left")
        mv = (y.pcl.values - y.po.values) / y.datr.values
    return np.where(mv >= 0.5, 1, np.where(mv <= -0.5, -1, 0))


def signals(m5, *a, **k):
    sig = _orig(m5, *a, **k)
    allow = regime(m5, sig)
    bad = (allow != 0) & (allow != sig.side.values)
    print(f"  거름 {bad.sum()}건 / 신호 {len(sig)}건", flush=True)
    return sig[~bad].reset_index(drop=True)


def run(year, modes):
    global MODE
    fe.bar_close_signals = signals
    for m in modes:
        MODE = m
        sm.MAE.clear()
        scaled = year < 2025
        sm.trades(year, scaled)
        os.replace(f"sar_martin_{'scaled' if scaled else 'trades'}_{year}.csv", RES / f"sar_regime_{m}_{year}.csv")


def summary():
    rows = []
    names = {"지금 봇": "sar_us_now_{y}.csv"}
    names.update({v: f"sar_regime_{k}_{{y}}.csv" for k, v in NAMES.items() if (RES / f"sar_regime_{k}_2025.csv").exists()})
    for name, pat in names.items():
        parts = [pd.read_csv(RES / pat.format(y=y)).assign(y=y) for y in range(2022, 2027) if (RES / pat.format(y=y)).exists()]
        tr = pd.concat(parts)
        row = {"규칙": name, "건수": len(tr), "승률%": round((tr.oz > 0).mean() * 100, 1), "oz합": round(tr.oz.sum(), 1)}
        row.update({f"승률{y % 100}": round((tr.oz[tr.y == y] > 0).mean() * 100, 1) for y in range(2022, 2027)})
        row.update({f"oz{y % 100}": round(tr.oz[tr.y == y].sum(), 1) for y in range(2022, 2027)})
        eq = tr.oz.cumsum()
        row["낙폭oz"] = round((eq.cummax() - eq).max(), 1)
        rows.append(row)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.to_string(index=False))
    res.to_csv(RES / "sar_regime.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    if sys.argv[1] == "sum":
        summary()
    else:
        run(int(sys.argv[1]), sys.argv[2:] or [MODE])
