"""SAR 단타: 익절 짧게 / 진입 뒤 N초에 마이너스면 바로 청산 (2026-10-27 사용자 아이디어, 실제 틱)

진입·반대 콩 청산·필터는 지금 봇 그대로 (backtests.sar_martin.trades), 바꾸는 것만:
  익절 X       : 손절 7.5 / 익절 X
  N초 컷 (기준) : 진입 N초 뒤 첫 틱의 손익이 기준보다 나쁘면 그 틱에 청산 (그 전에 끝난 거래는 그대로)
2022~24는 ATR 쉬기 기준을 금값 비례 (sar_martin scaled와 같음)

사용법: python -m backtests.sar_tp_cut 2024 2025 2026     (결과 results/sar_tp_cut.csv)
"""
import sys

import numpy as np
import pandas as pd

import backtests.sar_martin as sm
import backtests.trail_exit as te
from backtests.strategies import pf

VARIANTS = {f"익절 {x}": ("tp", x) for x in (2.0, 2.5, 3.0, 3.5)}
VARIANTS.update({f"{n}초 뒤 마이너스면 컷": ("cut", n, 0.0) for n in (5, 10, 30, 60, 120, 300)})
VARIANTS.update({f"{n}초 뒤 -1$ 밑이면 컷": ("cut", n, -1.0) for n in (10, 30, 60, 120)})


def run(year, v):
    orig = sm.TRADE

    def trade(t, bid, ask, k, side, fc, against, o, sig_px):
        te.SL, te.TP = 7.5, (v[1] if v[0] == "tp" else 3.0)
        res = orig(t, bid, ask, k, side, fc, against, o, sig_px)
        if res is None or v[0] != "cut":
            return res
        k0, j, oz = res
        c = int(np.searchsorted(t, t[k0] + v[1] * 1000))
        if c >= j or c >= len(t):
            return res
        e = ask[k0] if side == 1 else bid[k0]
        px = bid[c] if side == 1 else ask[c]
        pnl = (px - e) * side
        return (k0, c, pnl) if pnl < v[2] else res

    sm.TRADE = trade
    sm.MAE.clear()
    scaled = year < 2025
    try:
        sm.trades(year, scaled)
    finally:
        sm.TRADE = orig
        te.SL, te.TP = 7.5, 3.0
    return pd.read_csv(f"sar_martin_{'scaled' if scaled else 'trades'}_{year}.csv")


def main():
    years = [int(y) for y in sys.argv[1:]]
    rows = []
    for name, v in VARIANTS.items():
        tr = pd.concat([run(y, v).assign(year=y) for y in years], ignore_index=True)
        eq = tr.oz.cumsum()
        q = pd.to_datetime(tr.open).dt.to_period("Q")
        row = {"규칙": name, "건수": len(tr), "승률%": round((tr.oz > 0).mean() * 100, 1), "oz합": round(tr.oz.sum(), 1),
               "PF": pf(tr.oz), "낙폭": round((eq.cummax() - eq).max(), 1),
               "분기 플러스": f"{(tr.groupby(q).oz.sum() > 0).sum()}/{q.nunique()}"}
        row.update({f"oz {y}": round(tr.oz[tr.year == y].sum(), 1) for y in years})
        rows.append(row)
        print("완료", name, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("results/sar_tp_cut.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
