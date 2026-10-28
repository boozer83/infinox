"""SAR 단타: 최근 1~2시간 고점/저점 근처에서 나온 신호 거르기 (2026-10-28 사용자: "내릴대로 내렸을 때 매도, 오를대로 올랐을 때 매수")

위치 = (신호봉 종가 - 최근 W봉 최저가) / (최근 W봉 최고가 - 최저가), 신호봉까지 완성된 5분봉 W개
매수는 위치 >= P (고점 근처), 매도는 위치 <= 1-P (저점 근처)면 그 신호 버림
나머지는 지금 봇 그대로 (backtests.sar_martin.trades, 2022~2024 ATR 기준 가격비례). 기준은 results/sar_us_now_YYYY.csv
결과 results/sar_edge_<W>_<P>_YYYY.csv

사용법: python -m backtests.sar_edge_skip 2025 12_0.9 24_0.9
        python -m backtests.sar_edge_skip sum
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.sar_martin as sm

RES = Path(__file__).resolve().parent.parent / "results"
W, P = 12, 0.9
_orig = fe.bar_close_signals


def position(m5):
    hi, lo = m5.high.rolling(W).max(), m5.low.rolling(W).min()
    pos = ((m5.close - lo) / (hi - lo)).values
    return pd.DataFrame({"t": m5.time + pd.Timedelta(minutes=5), "pos": pos})


def signals(m5, *a, **k):
    sig = _orig(m5, *a, **k)
    pos = sig[["t"]].merge(position(m5), on="t", how="left").pos.values
    edge = np.where(sig.side.values == 1, pos >= P, pos <= 1 - P)
    print(f"  거름 {edge.sum()}건 / 신호 {len(sig)}건", flush=True)
    return sig[~edge].reset_index(drop=True)


def run(year, variants):
    global W, P
    fe.bar_close_signals = signals
    for v in variants:
        w, p = v.split("_")
        W, P = int(w), float(p)
        sm.MAE.clear()
        scaled = year < 2025
        sm.trades(year, scaled)
        os.replace(f"sar_martin_{'scaled' if scaled else 'trades'}_{year}.csv", RES / f"sar_edge_{W}_{P:g}_{year}.csv")


def summary():
    rows = []
    names = {"지금 봇": "sar_us_now_{y}.csv"}
    for p in sorted(RES.glob("sar_edge_*_2025.csv")):
        _, _, w, pp, _ = p.stem.split("_")
        names[f"최근 {int(w) * 5}분 {float(pp):.0%} 끝자리 거르기"] = f"sar_edge_{w}_{pp}_{{y}}.csv"
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
    res.to_csv(RES / "sar_edge_skip.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    if sys.argv[1] == "sum":
        summary()
    else:
        run(int(sys.argv[1]), sys.argv[2:] or [f"{W}_{P:g}"])
