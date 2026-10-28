"""헤지 마틴 + 반대 콩 손절 (2026-10-28 사용자 아이디어, 실제 틱, 해마다 1만$)

지금 봇(backtests.hedge_martin_ticks, 시작 = SAR 단타 신호, 이어가기) + 5분봉 SAR이 전환되면 반대 방향 포지션을 그 틱에 청산
  2배     : 새 방향 진입 lot은 지금처럼 단계마다 2배 (최대 단계 4)
  같은 lot: 단계가 올라가도 시작 lot 그대로 (최대 단계 4 / 10)
반대 쪽을 다 닫았는데 최대 단계면 사이클 끝. 확정 손익만으로 목표·한도에 닿아도 사이클 끝
나머지 그대로: 익절 5$, 사이클 +1% 청산 / -30% 손절

사용법: python -m backtests.hedge_sar_stop 2025   (결과 results/hedge_sar_stop_YYYY.csv)
"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.hedge_martin_ticks as hm
import backtests.long_period as lb

RES = Path(__file__).resolve().parent.parent / "results"


class StopSim(hm.Sim):
    def __init__(self, maxl, flat, stop=True):
        super().__init__(5.0, 0.01, 0.30, maxl)
        self.flat, self.stop = flat, stop

    def open(self, side, lot, px):
        super().open(side, self.lot0() if self.flat else lot, px)

    def on_flip(self, side, k, bid, ask, start_ok=True):
        if self.stop and self.pos and not self.dead:
            keep = []
            for p in self.pos:
                if p[0] == side:
                    keep.append(p)
                else:
                    px = bid[k] if p[0] == 1 else ask[k]
                    self.real += p[1] * hm.CONTRACT * (px - p[2]) * p[0]
            self.pos = keep
            if self.real <= -self.limit * self.start_bal:
                return self.end_cycle(self.real, "limit")
            if self.real >= self.target * self.start_bal and not self.pos:
                return self.end_cycle(self.real, "target")
            if not self.pos and self.level >= self.maxl:
                return self.end_cycle(self.real, "stop")
        super().on_flip(side, k, bid, ask, start_ok)


VARIANTS = {"지금 봇": (4, False, False), "반대콩 손절 + 2배 (4단계)": (4, False, True),
            "반대콩 손절 + 같은 lot (4단계)": (4, True, True), "반대콩 손절 + 같은 lot (10단계)": (10, True, True)}


def main():
    year = int(sys.argv[1])
    hm.START_SIG, hm.CONTINUE = True, True
    fl = hm.flips(year)
    gc.collect()
    sims = {k: StopSim(ml, flat, stop) for k, (ml, flat, stop) in VARIANTS.items()}
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
    rows = [{"규칙": k, "사이클": s.cycles, "사이클승": s.wins, "-30%": s.limits, "연말$": round(s.bal),
             "수익%": round((s.bal / hm.BAL0 - 1) * 100, 1), "최대낙폭%": round(s.mdd * 100, 1), "최대lot": round(s.max_lot, 2)}
            for k, s in sims.items()]
    res = pd.DataFrame(rows)
    res.to_csv(RES / f"hedge_sar_stop_{year}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
