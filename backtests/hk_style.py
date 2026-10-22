"""HK(홍콩 SIGNAL) 방식 vs 우리 SAR 단타 진입 비교 (실제 틱)

MT4 계좌(86056871, HK_HR 등 118건)에서 본 HK 방식:
  - 5분봉 SAR 전환 방향으로, 봉 마감을 기다리지 않고 Bid가 SAR 값을 넘는 첫 틱에 진입 (86%가 전환 방향, 41건은 전환 봉 안)
  - 처음 손절 약 7.5$, 익절 주문 없음. 이긴 거래는 대부분 +2~3.5$에서 올려 둔 손절선에 걸려 청산 (중간값 +2.6$)
진입 3가지 x 청산 2가지:
  진입  우리 (봉 마감)   : 지금 봇 신호 (RSI50 + 1시간 SAR + 시간/ATR 필터)
        봉 안 + 필터     : 같은 필터, 진입만 Bid가 SAR 넘는 첫 틱
        HK (봉 안, 필터 없음): 전환 봉마다 첫 틱 진입, RSI/1시간/시간/ATR 필터 없음
  청산  +3$ 익절         : 손절 7.5 / 익절 3
        잠금 2 트레일 1  : +3$ 닿으면 손절선을 +2$로, 이후 최고점에서 1$ 되돌리면 청산 (trail_exit와 같음)
한 번에 1포지션, 금요일 23:30 청산. 결과는 $/oz (lot 무관)

사용법: python -m backtests.hk_style 2025-01-01 2025-12-31
        python -m backtests.hk_style 2026-01-01
"""
import gc
import sys

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.long_period as lb
import backtests.trail_exit as te
from backtests.add_positions_ticks import load_m1_recent
from backtests.gold_rest_others import (FRIDAY_NO_ENTRY, MAX_H1_FLIPS, MIN_ATR, NO_ENTRY_FROM, NO_ENTRY_UNTIL,
                                        resample)

EXITS = {"+3$ 익절": None, "잠금 2 트레일 1": (2.0, 1.0, None)}


def our_filter(sig):
    mins = sig.t.dt.hour * 60 + sig.t.dt.minute
    bad = ((sig.t.dt.weekday == 4) & (mins >= FRIDAY_NO_ENTRY)) | ((mins >= NO_ENTRY_FROM) & (mins < NO_ENTRY_UNTIL)) \
        | (sig.flips >= MAX_H1_FLIPS) | (sig.atr < MIN_ATR) | ((sig.ref == sig.ref) & (sig.atr > 2.0 * sig.ref))
    return sig[~bad].reset_index(drop=True)


def hk_signals(cand):
    """필터 없이 전환 봉마다 Bid가 SAR 값을 넘는 첫 틱"""
    rows = []
    cand = cand[(cand.s >= int(fe.START.value // 1_000_000)) & (cand.s < int(fe.END.value // 1_000_000))]
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        c = cand[(cand.s >= int(s.timestamp() * 1000)) & (cand.s < int(e.timestamp() * 1000))]
        if c.empty:
            continue
        t, bid, _ = lb.load_month(y, m)
        for r in c.itertuples(index=False):
            k0, k1 = np.searchsorted(t, r.s), np.searchsorted(t, r.e)
            k = np.flatnonzero(bid[k0:k1] > r.trig if r.side == 1 else bid[k0:k1] < r.trig)
            if k.size:
                rows.append((pd.Timestamp(int(t[k0 + k[0]]), unit="ms"), r.side))
        del t, bid
        gc.collect()
    return pd.DataFrame(rows, columns=["t", "side"])


def simulate(sigs, exits):
    """sigs: {이름: DataFrame(t, side)} -> {(진입, 청산): [$/oz ...]}"""
    res = {(n, x): [] for n in sigs for x in exits}
    busy = {k: 0 for k in res}
    ms = [(y, m) for y, m in lb.months() if (fe.START.year, fe.START.month) <= (y, m) <= (fe.END.year, fe.END.month)]
    for i, (y, m) in enumerate(ms):
        s, e = lb.month_range(y, m)
        sm, em = int(s.timestamp() * 1000), int(e.timestamp() * 1000)
        t, bid, ask = lb.load_month(y, m)
        if i + 1 < len(ms):
            nt, nb, na = lb.load_month(*ms[i + 1])
            t, bid, ask = np.concatenate([t, nt]), np.concatenate([bid, nb]), np.concatenate([ask, na])
            del nt, nb, na
        for n, sig in sigs.items():
            ts = sig.t.values.astype("datetime64[ms]").astype(np.int64)
            for r_t, r_side, r_ms in zip(sig.t, sig.side, ts):
                if not sm <= r_ms < em:
                    continue
                k = int(np.searchsorted(t, r_ms))
                if k >= len(t) - 1:
                    continue
                fri = r_t.normalize() + pd.Timedelta(days=(4 - r_t.weekday()) % 7, hours=23, minutes=30)
                fri_ms = int(np.datetime64(fri, "ms").astype(np.int64))
                for x, case in exits.items():
                    if r_ms < busy[(n, x)]:
                        continue
                    out = te.trade(t, bid, ask, k, int(r_side), case, fri_ms, None)
                    if out is None:
                        continue
                    j, oz = out
                    busy[(n, x)] = int(t[j])
                    res[(n, x)].append(oz)
        del t, bid, ask
        gc.collect()
        print(f"  {y}-{m:02d} 완료", flush=True)
    return res


def stats(name, oz):
    oz = np.asarray(oz)
    if not len(oz):
        return {"방식": name, "거래": 0}
    eq = np.cumsum(oz)
    dd = (np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max()
    w, l = oz[oz > 0], oz[oz <= 0]
    return {"방식": name, "거래": len(oz), "승률%": round(len(w) / len(oz) * 100, 1), "평균이익": round(w.mean(), 2) if len(w) else 0,
            "평균손실": round(l.mean(), 2) if len(l) else 0, "PF": round(w.sum() / -l.sum(), 2) if l.sum() else np.inf,
            "합계$/oz": round(oz.sum(), 1), "최대낙폭$/oz": round(dd, 1)}


def main():
    fe.START = pd.Timestamp(sys.argv[1])
    if len(sys.argv) > 2:
        fe.END = pd.Timestamp(sys.argv[2])
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = load_m1_recent()
    m5, h1 = resample(m1, 5), resample(m1, 60)
    del m1
    gc.collect()
    ctx = fe.context(m5, h1)
    cand = fe.tick_candidates(m5)
    sigs = {"우리 (봉 마감)": our_filter(fe.bar_close_signals(m5, 5, 0.02, ctx, True)),
            "봉 안 + 필터": our_filter(fe.tick_signals(cand, ctx)),
            "HK (봉 안, 필터 없음)": hk_signals(cand)}
    del m5, h1
    gc.collect()
    for n, s in sigs.items():
        print(f"  신호 {n}: {len(s)}개", flush=True)
    res = simulate({n: s[["t", "side"]] for n, s in sigs.items()}, EXITS)
    out = pd.DataFrame([stats(f"{n} / {x}", oz) for (n, x), oz in res.items()])
    print(out.to_string(index=False))
    out.to_csv(f"hk_style_{fe.START.year}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
