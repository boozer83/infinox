"""추세 연장 청산 비교: +3달러 도달 후에만 수익을 늘리는 방식들 (2022-01 ~ 현재, 실제 틱)

진입과 첫 단계는 모두 같음: RSI50 + H1 SAR 신호, 손절 7.5달러, +3달러 도달 여부.
+3달러에 닿은 뒤의 처리만 다름:
  고정 TP 3        : 전량 익절 (현재 방식)
  반익절+SAR       : 절반 익절, 나머지는 손절을 본전으로 옮기고 5분봉 SAR 반대 전환까지 보유
  반익절+트레일링3 : 절반 익절, 나머지는 본전 손절 + 최고점에서 3달러 되돌리면 청산
  반익절+TP8       : 절반 익절, 나머지는 본전 손절 + 진입가 +8달러에서 익절
  잠금2+SAR        : 전량 유지, 손절을 +2달러로 올리고 SAR 반대 전환까지 보유
  잠금2+트레일링3  : 전량 유지, 손절을 +2달러로 올리고 최고점에서 3달러 되돌리면 청산 (+2 아래로는 안 내려감)

손익은 전체 수량 기준 달러/온스 (절반 청산이면 두 부분의 평균). 마틴의 '이김'은 거래 전체 손익 > 0.
사용법: python -m backtests.exits2   (long_period를 먼저 한 번 실행해 틱이 저장돼 있어야 함)
"""
import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.exits import CHUNK, SL, money, pf, simulate
from backtests.m5_ticks import first_hit
from indicators import psar

TP1 = 3.0


def runner(mode, frac, lock, dist=None, tp2=None):
    """+TP1 도달 후: frac만큼 익절, 나머지는 손절을 진입가+lock으로 올리고 mode('sar'|'trail'|'tp')로 청산"""
    def f(side, k, t, bid, ask, e, ctx):
        price = bid if side == 1 else ask
        sgn = 1 if side == 1 else -1
        if side == 1:
            j1 = first_hit(bid, k + 1, e - SL, e + TP1)
        else:
            j1 = first_hit(ask, k + 1, e - TP1, e + SL)
        if j1 < 0:
            return None
        p1 = (price[j1] - e) * sgn
        if p1 < 0:                         # 손절
            return j1, p1
        stop = e + lock * sgn               # 나머지의 손절선
        if mode == "tp":
            if side == 1:
                j2 = first_hit(bid, j1 + 1, stop, e + tp2)
            else:
                j2 = first_hit(ask, j1 + 1, e - tp2, stop)
            if j2 < 0:
                return None
        elif mode == "sar":
            t5, nxt_dn, nxt_up = ctx
            b = np.searchsorted(t5, t[j1], side="right") - 1
            i = (nxt_dn if side == 1 else nxt_up)[b]
            if i < 0 or i + 1 >= len(t5):
                return None
            x = max(np.searchsorted(t, t5[i + 1]), j1 + 1)
            if x >= len(t):
                return None
            seg = price[j1 + 1:x]
            hit = np.flatnonzero(seg <= stop) if side == 1 else np.flatnonzero(seg >= stop)
            j2 = j1 + 1 + hit[0] if hit.size else x
        else:                               # trail
            best, s, j2 = price[j1], j1 + 1, -1
            while s < len(price):
                seg = price[s:s + CHUNK]
                if side == 1:
                    run = np.maximum.accumulate(np.r_[best, seg])[1:]
                    lvl = np.maximum(stop, run - dist)
                    hit = np.flatnonzero(seg <= lvl)
                else:
                    run = np.minimum.accumulate(np.r_[best, seg])[1:]
                    lvl = np.minimum(stop, run + dist)
                    hit = np.flatnonzero(seg >= lvl)
                if hit.size:
                    j2 = s + hit[0]
                    break
                best, s = run[-1], s + CHUNK
            if j2 < 0:
                return None
        p2 = (price[j2] - e) * sgn
        return j2, frac * p1 + (1 - frac) * p2
    return f


def fixed3(side, k, t, bid, ask, e, ctx):
    if side == 1:
        j = first_hit(bid, k + 1, e - SL, e + TP1)
        return (j, bid[j] - e) if j >= 0 else None
    j = first_hit(ask, k + 1, e - TP1, e + SL)
    return (j, e - ask[j]) if j >= 0 else None


METHODS = [
    ("고정 TP 3", fixed3),
    ("반익절+SAR", runner("sar", 0.5, 0.0)),
    ("반익절+트레일링3", runner("trail", 0.5, 0.0, dist=3.0)),
    ("반익절+TP8", runner("tp", 0.5, 0.0, tp2=8.0)),
    ("잠금2+SAR", runner("sar", 0.0, 2.0)),
    ("잠금2+트레일링3", runner("trail", 0.0, 2.0, dist=3.0)),
]


def main():
    m5 = lb.bars_from_ticks(5)
    h1 = lb.bars_from_ticks(60)
    sigs = lb.signals(m5, h1)
    up = psar(m5.high.values, m5.low.values)
    t5 = m5.time.values.astype("datetime64[ms]").astype(np.int64)
    n = len(up)
    nxt_dn, nxt_up = np.full(n, -1), np.full(n, -1)
    nd = nu = -1
    for i in range(n - 1, -1, -1):
        if not up[i]:
            nd = i
        else:
            nu = i
        nxt_dn[i], nxt_up[i] = nd, nu
    ctx = (t5, nxt_dn, nxt_up)

    start = pd.Timestamp("2025-01-01")
    rows = []
    for name, fn in METHODS:
        tr = simulate(sigs, fn, ctx)
        y = tr.groupby(tr.close.dt.year).pnl_oz.apply(pf)
        mins = (tr.close_ms - tr.open_ms) / 60000
        f0, d0 = money(tr, start)
        f5, d5 = money(tr, start, max_level=5)
        rows.append({
            "청산 방식": name, "거래수": len(tr), "승률%": round((tr.pnl_oz > 0).mean() * 100, 1),
            "평균이익": round(tr.pnl_oz[tr.pnl_oz > 0].mean(), 2), "평균손실": round(tr.pnl_oz[tr.pnl_oz < 0].mean(), 2),
            "최대이익": round(tr.pnl_oz.max(), 1), "보유(분)": round(mins.median()), "전체PF": pf(tr.pnl_oz),
            **{f"PF {k}": v for k, v in y.items()},
            "25~26 고정0.03": f"{f0} ({d0:.0%})", "25~26 마틴5": f"{f5} ({d5:.0%})",
        })
        print(f"완료: {name}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("exits2_results.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print()
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
