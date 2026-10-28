"""1분봉 틱 수 (트레이딩뷰 차트의 '거래량'과 같은 개념) 를 틱 파일에서 세서 저장
금: data/ticks -> data/m1_tickvol.csv, 다른 종목: data/ticks_z/<종목> -> data/m1_tickvol_<종목>.csv

사용법: python -m backtests.tickvol_m1 [종목 ...]   (없으면 금)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.download_symbol as ds
import backtests.long_period as lb

OUT = Path(__file__).resolve().parent.parent / "data" / "m1_tickvol.csv"


def load(symbol="XAUUSD"):
    out = OUT if symbol == "XAUUSD" else OUT.with_name(f"m1_tickvol_{symbol}.csv")
    if out.exists():
        v = pd.read_csv(out, parse_dates=["time"])
        return v.set_index("time").vol
    parts = []
    for y, m in lb.months():
        if symbol == "XAUUSD":
            t, _, _ = lb.load_month(y, m)
        else:
            f = ds.ROOT / symbol / f"{y}-{m:02d}.npz"
            if not f.exists():
                continue
            with np.load(f) as z:
                t = z["t0"][0] + np.cumsum(z["dt"], dtype=np.int64)
        u, cnt = np.unique(t // 60_000, return_counts=True)
        parts.append(pd.Series(cnt, index=pd.to_datetime(u * 60_000, unit="ms")))
        print(f"  틱 수 {symbol} {y}-{m:02d}", flush=True)
    v = pd.concat(parts).groupby(level=0).sum()
    v.index.name = "time"
    v.rename("vol").reset_index().to_csv(out, index=False)
    return v


if __name__ == "__main__":
    for s in sys.argv[1:] or ["XAUUSD"]:
        print(s, load(s).describe().to_dict())
