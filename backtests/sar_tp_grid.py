"""SAR 단타 익절·손절 크기 바꾸기 (2026-10-28 사용자: 손익비를 늘려서 익절 10$쯤?)

나머지는 지금 봇 그대로 (반대 콩 청산 + 과열 거르기, backtests.sar_martin.trades, 2022~2024 ATR 기준 가격비례)
기준(익절 3 / 손절 7.5)은 results/sar_us_now_YYYY.csv. 결과 results/sar_tp_<TP>_<SL>_YYYY.csv

사용법: python -m backtests.sar_tp_grid 2025 5_7.5 10_7.5 10_10
        python -m backtests.sar_tp_grid sum
"""
import os
import sys
from pathlib import Path

import pandas as pd

import backtests.sar_martin as sm
import backtests.trail_exit as te

RES = Path(__file__).resolve().parent.parent / "results"


def run(year, variants):
    for v in variants:
        tp, sl = (float(x) for x in v.split("_"))
        te.TP, te.SL = tp, sl
        sm.MAE.clear()
        scaled = year < 2025
        sm.trades(year, scaled)
        os.replace(f"sar_martin_{'scaled' if scaled else 'trades'}_{year}.csv", RES / f"sar_tp_{tp:g}_{sl:g}_{year}.csv")


def summary():
    rows = []
    names = {"지금 봇 (익절 3 / 손절 7.5)": "sar_us_now_{y}.csv"}
    for p in sorted(RES.glob("sar_tp_*_2025.csv"), key=lambda p: [float(x) for x in p.stem.split("_")[2:4]]):
        _, _, tp, sl, _ = p.stem.split("_")
        names[f"익절 {tp} / 손절 {sl}"] = f"sar_tp_{tp}_{sl}_{{y}}.csv"
    for name, pat in names.items():
        parts = [pd.read_csv(RES / pat.format(y=y)).assign(y=y) for y in range(2022, 2027) if (RES / pat.format(y=y)).exists()]
        tr = pd.concat(parts)
        row = {"규칙": name, "건수": len(tr), "승률%": round((tr.oz > 0).mean() * 100, 1), "oz합": round(tr.oz.sum(), 1),
               "평균보유분": round(tr.mins.mean(), 1)}
        row.update({f"승률{y % 100}": round((tr.oz[tr.y == y] > 0).mean() * 100, 1) for y in range(2022, 2027)})
        row.update({f"oz{y % 100}": round(tr.oz[tr.y == y].sum(), 1) for y in range(2022, 2027)})
        eq = tr.oz.cumsum()
        row["낙폭oz"] = round((eq.cummax() - eq).max(), 1)
        rows.append(row)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.to_string(index=False))
    res.to_csv(RES / "sar_tp_grid.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    if sys.argv[1] == "sum":
        summary()
    else:
        run(int(sys.argv[1]), sys.argv[2:])
