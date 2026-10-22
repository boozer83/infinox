"""헤지마틴 분할 진입 조건 비교 (실제 틱, 연도별 1만 달러)

HK 패널 "Max Pos N / Entry Gap $1.0": 이기는 쪽이 마지막 진입가에서 간격만큼 가면 같은 lot 추가 (묶음당 최대 N개)
자동주행모드는 장세로 N을 정함 (추세 1 / 일반 2 / 횡보 3, hedge_martin_v2.SPLIT)
비교:
  간격  : 고정 0.5 / 1 / 2 $, 사이클 시작 때 5분봉 ATR x 0.5 / x 1
  분할 수: 자동(장세) / 1 / 3 / 5
나머지는 hk_apply 마틴과 같음: 우리 SAR 신호로 시작, 이긴 묶음만 익절 후 2배, 목표 +1%, 구조대 + 한도 50%, 5마틴, 뉴스 없음

2배 단계 조건 (up 모드, python -m backtests.martin_split 2025 up):
  전환+이익 : 봉 안 SAR 전환 때 그 방향 묶음이 이익이면 익절 + 2배 (기본)
  전환+다참 : 위와 같은데 묶음이 다 찼을 때만 (243$ 계좌: 3개 다 찬 뒤 익절 -> 2배)
  다참+1칸  : 묶음이 다 차고 간격만큼 더 가면 SAR과 상관없이 바로 익절 + 2배

사용법: python -m backtests.martin_split 2025      (results/martin_split_2025.csv)
"""
import gc
import itertools
import sys

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.hedge_martin_v2 as hm2
import backtests.hk_apply as ha
import backtests.long_period as lb
from indicators import atr

GAPS = (0.5, 1.0, 2.0, "ATRx0.5", "ATRx1")
POS = ("자동", 1, 3, 5)
UPS = ("전환+이익", "전환+다참", "다참+1칸")
OUT = ha.OUT


class SplitSim(ha.HKSim):
    def __init__(self, gap, pos, up="전환+이익", limit=0.50, rescue=True):
        super().__init__(0.01, limit, 4, 0.01, rescue)
        self.gap_mode, self.pos_mode, self.gap, self.up = gap, pos, 1.0, up

    def flip(self, side, k, bid, ask):
        if self.up == "다참+1칸":
            return                      # 2배는 run 안에서 (SAR 전환 무관)
        if self.up == "전환+다참" and (side not in self.g or len(self.g[side][1]) < self.maxpos):
            return
        super().flip(side, k, bid, ask)

    def full_step(self, k0, k1, bid, ask):
        """다 찬 묶음이 마지막 진입가에서 간격만큼 더 간 첫 틱 -> 그 묶음 익절 + 같은 방향 2배 (5마틴까지)"""
        j, sd = k1, None
        for s, (lot, es) in self.g.items():
            if len(es) >= self.maxpos and self.level < self.maxl:
                h = np.flatnonzero(ask[k0:k1] >= es[-1] + self.gap) if s == 1 else \
                    np.flatnonzero(bid[k0:k1] <= es[-1] - self.gap)
                if h.size and k0 + h[0] < j:
                    j, sd = k0 + int(h[0]), s
        return j, sd

    def start_with(self, k, bid, ask, regime, a5):
        was = bool(self.g)
        self.start(k, bid, ask, regime)
        if was or not self.g:
            return
        if self.pos_mode != "자동":
            self.maxpos = self.pos_mode
        self.gap = self.gap_mode if not isinstance(self.gap_mode, str) else \
            max(0.2, a5 * float(self.gap_mode.split("x")[1]))

    def run(self, k0, k1, bid, ask):
        hm2.GAP = self.gap
        while self.up == "다참+1칸" and self.g and k0 < k1:
            j, sd = self.full_step(k0, k1, bid, ask)
            if sd is None:
                break
            super().run(k0, j, bid, ask)
            if not self.g:
                return
            lot, es = self.g[sd]
            if len(es) >= self.maxpos and (ask[j] >= es[-1] + self.gap if sd == 1 else bid[j] <= es[-1] - self.gap):
                ha.HKSim.flip(self, sd, j, bid, ask)
            k0 = j
        super().run(k0, k1, bid, ask)


def main():
    year = int(sys.argv[1])
    fe.START, fe.END = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
    hm2.REGIME, hm2.FLIP_ONLY_WIN = True, True
    m5, h1 = ha.load_year(year)
    fl, _ = ha.hk_homerun(m5)
    our = ha.our_signals(m5, h1)
    our["regime"] = hm2.regime_at(m5, h1, pd.to_datetime(our.ms, unit="ms").astype(m5.time.dtype))
    a = pd.DataFrame({"end": m5.time + pd.Timedelta(minutes=5), "a5": atr(m5).values})
    our["a5"] = pd.merge_asof(pd.DataFrame({"t": pd.to_datetime(our.ms, unit="ms").astype(m5.time.dtype)}), a,
                              left_on="t", right_on="end", direction="backward").a5.values
    del m5, h1
    gc.collect()
    ev = pd.concat([our.assign(kind=0), fl[~fl.ms.isin(our.ms)].assign(kind=1, regime=1, a5=np.nan)])
    ev = ev.sort_values(["ms", "kind"]).reset_index(drop=True)
    up = len(sys.argv) > 2 and sys.argv[2] == "up"
    cyc = len(sys.argv) > 2 and sys.argv[2] == "cycles"     # 장세 판단 검사용: 장세와 무관하게 분할 3 고정, 사이클 기록
    if cyc:
        hm2.REGIME = False
        grid = [(1.0, 3, "전환+다참"), (1.0, 3, "전환+이익")]
    else:
        grid = itertools.product((1.0, "ATRx1"), ("자동", 3), UPS) if up else itertools.product(GAPS, POS, ("전환+이익",))
    sims = {(g, p, u): SplitSim(g, p, u) for g, p, u in grid}
    last = None
    for y, m in [(y, m) for y, m in lb.months() if y == year]:
        s, e = lb.month_range(y, m)
        f = ev[(ev.ms >= int(s.timestamp() * 1000)) & (ev.ms < int(e.timestamp() * 1000))]
        t, bid, ask = lb.load_month(y, m)
        ks = np.searchsorted(t, f.ms.values)
        for sim in sims.values():
            sim.t = t
            prev = 0
            for kind, side, k, rg, a5 in zip(f.kind.values, f.side.values, ks, f.regime.values, f.a5.values):
                if k >= len(t):
                    break
                sim.run(prev, k, bid, ask)
                if kind == 0:
                    sim.start_with(k, bid, ask, rg, a5)
                elif sim.g:
                    sim.flip(int(side), k, bid, ask)
                prev = k
            sim.run(prev, len(t), bid, ask)
        last = (bid[-1:], ask[-1:])
        del t, bid, ask
        gc.collect()
        print(f"  {y}-{m:02d} 완료", flush=True)
    rows = []
    for (g, p, u), sim in sims.items():
        final = sim.bal if not sim.g else sim.start_bal + sim.real + sum(sim.pnl_group(s, 0, *last) for s in sim.g)
        lv = np.bincount(np.asarray(sim.levels, int), minlength=6)[1:6]
        rows.append({"연도": year, "간격": g, "분할": p, "2배 조건": u, "수익%": round((final / hm2.BAL0 - 1) * 100, 1),
                     "최대낙폭%": round(sim.mdd * 100, 1), "사이클": sim.cycles,
                     "승률%": round(sim.wins / sim.cycles * 100, 1) if sim.cycles else np.nan,
                     "한도+동결": sim.limits, "동결": sim.freezes, "마틴분포 1~5": "/".join(map(str, lv)),
                     "사이클일수 중간": round(float(np.median(sim.days)), 2) if sim.days else np.nan,
                     "최대lot": round(sim.max_lot, 2), "깡통": sim.dead})
    res = pd.DataFrame(rows)
    if cyc:
        pd.DataFrame([(u, *x) for (g, p, u), sim in sims.items() for x in sim.log],
                     columns=["2배 조건", "t0", "t1", "마틴", "이유", "손익%"]).to_csv(
            OUT / f"martin_cycles_{year}.csv", index=False, encoding="utf-8-sig")
    res.to_csv(OUT / f"martin_split_{year}{'_up' if up else '_cycles' if cyc else ''}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
