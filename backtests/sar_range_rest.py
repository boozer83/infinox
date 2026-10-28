"""SAR 단타: 최근 2시간에 이미 크게 움직였으면 쉬기 (2026-10-27, 다른 AI 분석을 우리 봇에 확인, 실제 틱)

쉬기: 신호봉까지 완성된 5분봉 24개(2시간)의 최고가 - 최저가가 일봉 ATR(14, 전날까지)의 TH배 이상이면 그 신호 버림
나머지는 지금 봇 그대로 (backtests.sar_martin.trades, 2022~2024 ATR 기준 가격비례). 버린 신호 뒤의 다음 신호는 그대로 잡음
결과 results/sar_rest_<TH>_YYYY.csv (open, day, mins, oz, mae), 기준은 results/sar_us_now_YYYY.csv

사용법: python -m backtests.sar_range_rest 2025 0.35 0.4 0.45
        python -m backtests.sar_range_rest sum
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.sar_martin as sm

RES = Path(__file__).resolve().parent.parent / "results"
TH = 0.4
_orig = fe.bar_close_signals


def range_ratio(m5):
    d = m5.set_index("time").resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    pc = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    datr = m5.time.dt.normalize().map(tr.rolling(14).mean().shift(1)).values
    r2 = (m5.high.rolling(24).max() - m5.low.rolling(24).min()).values / datr
    return pd.DataFrame({"t": m5.time + pd.Timedelta(minutes=5), "r2": r2})


def signals(m5, *a, **k):
    sig = _orig(m5, *a, **k)
    r = sig[["t"]].merge(range_ratio(m5), on="t", how="left").r2.values
    keep = ~(r >= TH)                                          # ATR 없는 첫 2주는 그대로 둠
    print(f"  쉬기 {(~keep).sum()}건 / 신호 {len(sig)}건", flush=True)
    return sig[keep].reset_index(drop=True)


def run(year, ths):
    global TH
    fe.bar_close_signals = signals
    for th in ths:
        TH = th
        sm.MAE.clear()
        scaled = year < 2025
        sm.trades(year, scaled)
        os.replace(f"sar_martin_{'scaled' if scaled else 'trades'}_{year}.csv", RES / f"sar_rest_{th:g}_{year}.csv")


def summary():
    rows = []
    names = {"지금 봇": "sar_us_now_{y}.csv"}
    for p in sorted(RES.glob("sar_rest_*_2022.csv")):
        th = p.stem.split("_")[2]
        names[f"2시간 범위 ATRx{th} 이상 쉬기"] = f"sar_rest_{th}_{{y}}.csv"
    for name, pat in names.items():
        parts = [pd.read_csv(RES / pat.format(y=y)).assign(y=y) for y in range(2022, 2027) if (RES / pat.format(y=y)).exists()]
        tr = pd.concat(parts)
        row = {"규칙": name, "건수": len(tr), "승률%": round((tr.oz > 0).mean() * 100, 1), "oz합": round(tr.oz.sum(), 1)}
        row.update({f"oz {y}": round(tr.oz[tr.y == y].sum(), 1) for y in range(2022, 2027)})
        eq = tr.oz.cumsum()
        row["낙폭oz"] = round((eq.cummax() - eq).max(), 1)
        rows.append(row)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.to_string(index=False))
    res.to_csv(RES / "sar_range_rest.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    if sys.argv[1] == "sum":
        summary()
    else:
        run(int(sys.argv[1]), [float(x) for x in sys.argv[2:]] or [TH])
