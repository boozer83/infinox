"""헤지마틴 + 구조대(동결 후 단타로 회복) (실제 틱, 연도별)

HK 구조대 (3-3): 사이클이 크게 지면 매수·매도 lot을 맞춰 손익 동결 -> 홍콩 신호 매매로 +3%, +5% 쌓아 손실 상쇄 -> 본전 탈출
여기서는:
  마틴: 우리 SAR 신호로 시작, 분할 3 고정 (장세 무관), 1$ 간격, 2배 조건 "전환+다참", 목표 +1%, 5마틴
        사이클 손실이 -FREEZE%에 닿으면 동결 (= 그 자리 손실 확정과 같음, 스왑·증거금 비용은 뺌)
  회복: 동결 뒤에는 마틴을 쉬고 단타만. 동결 이후 단타 손익 합이 동결 손실 이상이 되면 본전 탈출 -> 마틴 다시 시작
        단타 = HK 홈런 또는 우리 SAR (hk_apply와 같은 진입·청산), lot 1,000달러당 0.1 (1$/oz = 잔고 1%)
  % 는 단리 합 (사이클 손익 % + 단타 %)
비교: 마틴만 / 마틴 + 구조대(HK 홈런) / 마틴 + 구조대(우리 SAR) / 단타만 (같은 lot, 쉬지 않고)

사용법: python -m backtests.martin_rescue 2025      (results/martin_rescue_2025.csv)
"""
import gc
import sys

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.hedge_martin_v2 as hm2
import backtests.hk_apply as ha
import backtests.long_period as lb
import backtests.martin_split as ms_
from indicators import psar

FREEZES = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)


def run_martin(year, our, fl):
    hm2.REGIME, hm2.FLIP_ONLY_WIN = False, True
    sims = {f: ms_.SplitSim(1.0, 3, "전환+다참", limit=f, rescue=False) for f in FREEZES}
    ev = pd.concat([our.assign(kind=0), fl[~fl.ms.isin(our.ms)].assign(kind=1, regime=1, a5=np.nan)])
    ev = ev.sort_values(["ms", "kind"]).reset_index(drop=True)
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
        del t, bid, ask
        gc.collect()
        print(f"  마틴 {y}-{m:02d} 완료", flush=True)
    return {f: pd.DataFrame(sim.log, columns=["t0", "t1", "마틴", "이유", "손익%"]) for f, sim in sims.items()}


def combine(cyc, scalp):
    """마틴 사이클 + (지면) 동결 뒤 단타로 회복. scalp: [(ms, oz)] -> (합계%, 마틴%, 단타%, 동결 수, 회복 못 한 손실%, 쉰 날)"""
    sc = np.array(scalp, dtype=float).reshape(-1, 2)
    mart = rec = 0.0
    frozen, n_freeze, paused, until = None, 0, 0.0, -1
    curve, low = [0.0], 0.0
    for t0, t1, why, pnl in zip(cyc.t0.values, cyc.t1.values, cyc.이유.values, cyc["손익%"].values):
        if t0 < until:
            continue                                    # 회복 중이라 마틴 쉼
        mart += pnl
        curve.append(mart + rec)
        if why == "limit":
            n_freeze += 1
            after = sc[sc[:, 0] >= t1]
            cum = np.cumsum(after[:, 1])
            k = np.flatnonzero(cum >= -pnl)
            if k.size:
                until = int(after[k[0], 0]) + 1
                rec += cum[k[0]]
            else:
                until = np.iinfo(np.int64).max
                rec += cum[-1] if len(cum) else 0.0
            paused += ((until if k.size else (after[-1, 0] if len(after) else t1)) - t1) / 86_400_000
            curve.append(mart + rec)
    curve = np.array(curve)
    dd = (np.maximum.accumulate(curve) - curve).max()
    return round(mart + rec, 1), round(mart, 1), round(rec, 1), n_freeze, round(dd, 1), round(paused, 1), until == np.iinfo(np.int64).max


def main():
    year = int(sys.argv[1])
    fe.START, fe.END = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
    m5, h1 = ha.load_year(year)
    fl, hr = ha.hk_homerun(m5)
    our = ha.our_signals(m5, h1)
    our["regime"], our["a5"] = 1, np.nan
    up = psar(m5.high.values, m5.low.values)
    end5 = m5.time + pd.Timedelta(minutes=5)
    against = {1: ha.to_ms(end5[~up]), -1: ha.to_ms(end5[up])}
    del m5, h1
    gc.collect()
    sc = ha.run_scalp(year, {"HK 홈런": (hr, "hk"), "우리 SAR": (our, "our")}, against)
    cycs = run_martin(year, our, fl)
    rows = []
    for f, cyc in cycs.items():
        for name, rec in (("마틴만", None), ("마틴 + 구조대(HK 홈런)", sc["HK 홈런"]), ("마틴 + 구조대(우리 SAR)", sc["우리 SAR"])):
            if rec is None:
                c = np.r_[0, np.cumsum(cyc["손익%"].values)]
                rows.append({"연도": year, "동결%": int(f * 100), "방법": name, "합계%": round(c[-1], 1), "마틴%": round(c[-1], 1),
                             "단타%": 0.0, "동결": int((cyc.이유 == "limit").sum()),
                             "최대낙폭%p": round((np.maximum.accumulate(c) - c).max(), 1), "쉰 날": 0.0, "회복 못 함": False,
                             "사이클": len(cyc), "승률%": round((cyc["손익%"] > 0).mean() * 100, 1)})
            else:
                tot, mart, r, nf, dd, paused, stuck = combine(cyc, rec)
                rows.append({"연도": year, "동결%": int(f * 100), "방법": name, "합계%": tot, "마틴%": mart, "단타%": r,
                             "동결": nf, "최대낙폭%p": dd, "쉰 날": paused, "회복 못 함": stuck})
    for name in ("HK 홈런", "우리 SAR"):
        oz = np.array([x[1] for x in sc[name]])
        c = np.r_[0, np.cumsum(oz)]
        rows.append({"연도": year, "동결%": None, "방법": f"단타만 ({name})", "합계%": round(c[-1], 1),
                     "최대낙폭%p": round((np.maximum.accumulate(c) - c).max(), 1)})
    res = pd.DataFrame(rows)
    res.to_csv(ha.OUT / f"martin_rescue_{year}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
