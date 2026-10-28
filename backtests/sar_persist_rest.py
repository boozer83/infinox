"""SAR 단타: 몇 시간 동안 한 방향으로 계속 갔으면 쉬기 (2026-10-28 사용자 아이디어, 실제 틱)

쉬기 (신호 시점에 완성된 봉만, 쉬면 매수·매도 둘 다 안 함):
  move_<H>_<K> : 최근 H시간 순변동 |종가 - H시간 전 종가| >= 일봉 ATR(14, 어제까지) x K
  bars_<N>     : 1시간봉 N개 연속 같은 색 (양봉 N개 또는 음봉 N개)
급등락 쉬기(sar_range_rest, 2시간 고저 범위)와 달리 "방향이 이어졌는지"를 봄
나머지는 지금 봇 그대로 (backtests.sar_martin.trades, 2022~2024 ATR 기준 가격비례). 기준은 results/sar_us_now_YYYY.csv
결과 results/sar_persist_<방식>_YYYY.csv

사용법: python -m backtests.sar_persist_rest 2025 move_3_0.5 move_6_0.8 bars_4
        python -m backtests.sar_persist_rest sum
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.sar_martin as sm
from backtests.gold_rest_others import resample

RES = Path(__file__).resolve().parent.parent / "results"
MODE = "move_3_0.5"
_orig = fe.bar_close_signals


def daily_atr(m5):
    d = m5.set_index("time").resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    pc = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    return m5.time.dt.normalize().map(tr.rolling(14).mean().shift(1)).values


def rest_flags(m5):
    kind, *p = MODE.split("_")
    end = m5.time + pd.Timedelta(minutes=5)
    if kind == "move":
        h, k = int(p[0]), float(p[1])
        mv = (m5.close - m5.close.shift(h * 12)).abs().values
        return pd.DataFrame({"t": end, "rest": mv >= k * daily_atr(m5)})
    n = int(p[0])
    h1 = resample(m5, 60)
    c = np.sign(h1.close - h1.open)
    same = (c.rolling(n).sum().abs() == n).values
    return pd.DataFrame({"t": h1.time + pd.Timedelta(hours=1), "rest": same})


def signals(m5, *a, **k):
    sig = _orig(m5, *a, **k)
    f = rest_flags(m5).sort_values("t")
    rest = pd.merge_asof(sig[["t"]], f, on="t", direction="backward").rest.fillna(False).values.astype(bool)
    print(f"  쉼 {rest.sum()}건 / 신호 {len(sig)}건", flush=True)
    return sig[~rest].reset_index(drop=True)


def run(year, modes):
    global MODE
    fe.bar_close_signals = signals
    for m in modes:
        MODE = m
        sm.MAE.clear()
        scaled = year < 2025
        sm.trades(year, scaled)
        os.replace(f"sar_martin_{'scaled' if scaled else 'trades'}_{year}.csv", RES / f"sar_persist_{m}_{year}.csv")


def label(m):
    kind, *p = m.split("_")
    return f"{p[0]}시간 순변동 ATR x{p[1]} 이상 쉬기" if kind == "move" else f"1시간봉 {p[0]}개 연속 같은 색 쉬기"


def summary():
    rows = []
    names = {"지금 봇": "sar_us_now_{y}.csv"}
    for p in sorted(RES.glob("sar_persist_*_2025.csv")):
        m = p.stem[len("sar_persist_"):-5]
        names[label(m)] = f"sar_persist_{m}_{{y}}.csv"
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
    res.to_csv(RES / "sar_persist_rest.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    if sys.argv[1] == "sum":
        summary()
    else:
        run(int(sys.argv[1]), sys.argv[2:] or [MODE])
