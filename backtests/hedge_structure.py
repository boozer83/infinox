"""헤지 마틴 꼬리 위험 줄이기: 최대 단계 x 사이클 손절 (2026-10-27 사용자 요청, 실제 틱, 해마다 1만$)

지금 봇: 익절 5$ / 목표 +1% / 손절 -30% / 최대 4단계 / 이어가기 / 시작은 SAR 단타 신호 (backtests.hedge_martin_ticks)
바꾸는 것: 최대 단계 2 / 3 / 4, 사이클 손절 5 / 10 / 15 / 20 / 30%

사용법: python -m backtests.hedge_structure 2025   (결과 results/hedge_structure_YYYY.csv)
"""
import gc
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.hedge_martin_ticks as hm
import backtests.long_period as lb

RES = Path(__file__).resolve().parent.parent / "results"
GRID = list(itertools.product((2, 3, 4), (0.05, 0.10, 0.15, 0.20, 0.30)))


def main():
    year = int(sys.argv[1])
    hm.START_SIG, hm.CONTINUE = True, True
    fl = hm.flips(year)
    gc.collect()
    sims = {g: hm.Sim(5.0, 0.01, g[1], g[0]) for g in GRID}
    for m in range(1, 13):
        if (year, m) > (hm.END.year, hm.END.month):
            break
        s, e = lb.month_range(year, m)
        sm_, em = int(s.timestamp() * 1000), min(int(e.timestamp() * 1000), int(hm.END.value // 1_000_000))
        t, bid, ask = lb.load_month(year, m)
        f = fl[(fl.ms >= sm_) & (fl.ms < em)]
        ks = np.searchsorted(t, f.ms.values)
        for sim in sims.values():
            prev = 0
            for side, k, st in zip(f.side.values, ks, f.start.values):
                sim.run(prev, k, bid, ask)
                if k < len(t):
                    sim.on_flip(int(side), k, bid, ask, bool(st))
                prev = k
            sim.run(prev, len(t), bid, ask)
        del t, bid, ask
        gc.collect()
        print(f"  {year}-{m:02d}", flush=True)
    rows = [{"최대단계": ml, "손절%": lim * 100, "사이클": s.cycles, "사이클승": s.wins, "손절걸림": s.limits,
             "연말$": round(s.bal), "수익%": round((s.bal / hm.BAL0 - 1) * 100, 1), "최대낙폭%": round(s.mdd * 100, 1)}
            for (ml, lim), s in sims.items()]
    res = pd.DataFrame(rows)
    res.to_csv(RES / f"hedge_structure_{year}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
