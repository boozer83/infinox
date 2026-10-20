"""반대 콩 청산 때 반대 방향으로 바로 진입 (실제 틱). 지금 봇(반대 콩 청산 + 승률 쉬기 끔) 기준
  지금           : 반대 콩이면 청산만
  반대 진입 SL 7.5: 반대 콩 청산 틱에 반대 방향 진입, TP 3 / SL 7.5
  반대 진입 SL 본전: 반대 진입의 손절 = 처음 포지션 진입가 (처음 거래가 0이 되는 가격). 처음 거래가 이익으로
                   청산돼서 그 가격이 반대쪽이면 SL 7.5
반대 진입은 한 번만 (반대 진입 포지션도 반대 콩이면 청산, 그때는 다시 뒤집지 않음). 반대 진입은 시간/ATR 쉬기 안 봄

사용법: python -m backtests.sar_reverse [시작 YYYY-MM-DD] [끝 YYYY-MM-DD]
"""
import sys

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.long_period as lb
import backtests.trail_exit as te
from backtests.add_positions_ticks import hit, load_m1_recent
from backtests.gold_rest_others import (FRIDAY_NO_ENTRY, MAX_H1_FLIPS, MIN_ATR, NO_ENTRY_FROM, NO_ENTRY_UNTIL,
                                        resample)
from backtests.sar_exit import against_times

CASES = {"지금 (반대 콩 청산만)": None, "반대 진입 SL 7.5": "fixed", "반대 진입 SL 본전": "be"}


def one(t, bid, ask, k, side, sl_px, fc, against):
    """k틱 진입 -> (청산 idx, $/oz, 이유, 진입가) 또는 None"""
    X = bid if side == 1 else ask
    e = ask[k] if side == 1 else bid[k]
    if sl_px is None:
        sl_px = e - side * te.SL
    tp_px = e + side * te.TP
    j = hit(X, k + 1, sl_px, tp_px) if side == 1 else hit(X, k + 1, tp_px, sl_px)
    a = against[side]
    i = np.searchsorted(a, t[k], side="right")
    js = int(np.searchsorted(t, a[i])) if i < len(a) else len(t)
    j, why = min((j, "TPSL"), (js, "SAR"), (fc, "FRI"))
    if j >= len(t):
        return None
    return j, (X[j] - e) * side, why, e


class RAcc(te.Acc):
    def __init__(self, case, against):
        super().__init__(case)
        self.against = against
        self.n_rev = 0

    def on_signal(self, s, t, bid, ask, _):
        ms = int(np.datetime64(s.t, "ms").astype(np.int64))
        if ms < self.busy:
            return
        if s.t.date() != self.day:
            self.day, self.base = s.t.date(), te.floor_lot(self.bal / 1000 * te.LOT_PER_1000)
        mins = s.t.hour * 60 + s.t.minute
        if (s.t.weekday() == 4 and mins >= FRIDAY_NO_ENTRY) or NO_ENTRY_FROM <= mins < NO_ENTRY_UNTIL \
                or s.flips >= MAX_H1_FLIPS or s.atr < MIN_ATR or (s.ref == s.ref and s.atr > 2.0 * s.ref):
            return
        if self.base > te.MAX_LOT:
            return
        k = int(np.searchsorted(t, ms))
        if k >= len(t) - 1 or t[k] - ms > 600_000:
            return
        d = s.t.normalize() + pd.Timedelta(days=(4 - s.t.weekday()) % 7, hours=23, minutes=30)
        fc = int(np.searchsorted(t, int(np.datetime64(d, "ms").astype(np.int64))))
        res = one(t, bid, ask, k, s.side, None, fc, self.against)
        if res is None:
            return
        self.book(t, k, res)
        j, oz, why, e = res
        if why == "SAR" and self.case is not None:
            side = -s.side
            x = bid[j] if s.side == 1 else ask[j]          # 처음 포지션 청산가
            sl_px = None
            if self.case == "be" and (e - x) * s.side > 0:  # 손실 청산이면 처음 진입가가 반대 포지션의 손실 쪽
                sl_px = e
            r2 = one(t, bid, ask, j, side, sl_px, fc, self.against)
            if r2 is not None:
                self.n_rev += 1
                self.book(t, j, r2)

    def book(self, t, k, res):
        j, oz, _, _ = res
        self.busy = int(t[j])
        self.results.append(oz > 0)
        profit = oz * self.base * te.CONTRACT
        self.bal += profit
        self.peak = max(self.peak, self.bal)
        self.mdd = max(self.mdd, (self.peak - self.bal) / self.peak)
        self.rows.append((pd.Timestamp(int(t[k]), unit="ms"), self.day, (t[j] - t[k]) / 60000, oz, profit, self.bal))


def main():
    if len(sys.argv) > 1:
        fe.START = pd.Timestamp(sys.argv[1])
    if len(sys.argv) > 2:
        fe.END = pd.Timestamp(sys.argv[2])
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = load_m1_recent()
    m5, h1 = resample(m1, 5), resample(m1, 60)
    del m1
    against = against_times(m5, 5)
    sig = fe.bar_close_signals(m5, 5, 0.02, fe.context(m5, h1), True)
    accs = {n: RAcc(c, against) for n, c in CASES.items()}
    sig_ms = sig.t.values.astype("datetime64[ms]").astype(np.int64)
    ms = [(y, m) for y, m in lb.months() if (fe.START.year, fe.START.month) <= (y, m) <= (fe.END.year, fe.END.month)]
    for i, (y, m) in enumerate(ms):
        s, e = lb.month_range(y, m)
        lo, hi = np.searchsorted(sig_ms, int(s.timestamp() * 1000)), np.searchsorted(sig_ms, int(e.timestamp() * 1000))
        if lo == hi:
            continue
        t, bid, ask = lb.load_month(y, m)
        if i + 1 < len(ms):
            nt, nb, na = lb.load_month(*ms[i + 1])
            t, bid, ask = np.concatenate([t, nt]), np.concatenate([bid, nb]), np.concatenate([ask, na])
            del nt, nb, na
        for r in sig.iloc[lo:hi].itertuples(index=False):
            for a in accs.values():
                a.on_signal(r, t, bid, ask, None)
        del t, bid, ask
    rows = [{"방식": n, **te.summary(a), "반대 진입": a.n_rev} for n, a in accs.items()]
    res = pd.DataFrame(rows).drop(columns=["평균 이익$/oz", "3$넘게 번 비율%", "최대 이익$/oz"])
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
