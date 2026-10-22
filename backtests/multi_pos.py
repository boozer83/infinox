"""SAR 단타(지금 봇: 반대 콩 청산 + 과열 거르기)를 '보유 중에도 새 신호마다 추가 진입'으로 바꾸면? (실제 틱)

  1포지션 (지금 봇) : 들고 있는 동안 새 신호 무시
  겹쳐 진입         : 새 신호마다 같은 lot으로 새 포지션 (각자 손절 7.5 / 익절 3 / 반대 콩 청산)
  겹쳐 진입 최대 2개 / 3개 : 동시에 열린 포지션 수 제한
잔고는 청산 순서대로가 아니라 신호 순서대로 반영 (근사). lot = 그날 시작 잔고 1,000달러당 0.05

사용법: python -m backtests.multi_pos 2025-01-01 2025-12-31
        python -m backtests.multi_pos 2026-01-01
"""
import gc
import sys

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.long_period as lb
import backtests.trail_exit as te
from backtests.add_positions_ticks import load_m1_recent
from backtests.ext_filter import features
from backtests.gold_rest_others import resample
from backtests.sar_exit import against_times
from backtests.winrate_ideas import Acc

BASE = [("ext", 0.8), ("risk", 2.8)]
VARIANTS = {"1포지션 (지금 봇)": 1, "겹쳐 진입 최대 2개": 2, "겹쳐 진입 최대 3개": 3, "겹쳐 진입 (제한 없음)": 99}


class Multi(Acc):
    def __init__(self, maxn, against):
        super().__init__({"feats": BASE}, against, set())
        self.maxn, self.ends, self.max_open = maxn, [], 0

    def on_signal(self, s, t, bid, ask, _):
        ms = int(np.datetime64(s.t, "ms").astype(np.int64))
        self.ends = [e for e in self.ends if e > ms]
        if len(self.ends) >= self.maxn:
            return
        self.busy = 0
        n0 = len(self.rows)
        super().on_signal(s, t, bid, ask, None)
        if len(self.rows) > n0:
            self.ends.append(self.busy)
            self.max_open = max(self.max_open, len(self.ends))


def main():
    fe.START = pd.Timestamp(sys.argv[1])
    if len(sys.argv) > 2:
        fe.END = pd.Timestamp(sys.argv[2])
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = load_m1_recent()
    m5, h1 = resample(m1, 5), resample(m1, 60)
    del m1
    gc.collect()
    against = against_times(m5, 5)
    sig = fe.bar_close_signals(m5, 5, 0.02, fe.context(m5, h1), True).merge(features(m5), on="t", how="left")
    del m5, h1
    gc.collect()
    accs = {n: Multi(k, against) for n, k in VARIANTS.items()}
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
        gc.collect()
        print(f"  {y}-{m:02d} 완료", flush=True)
    rows = [{"방식": n, **te.summary(a), "동시 최대": a.max_open} for n, a in accs.items()]
    res = pd.DataFrame(rows).drop(columns=["평균 이익$/oz", "3$넘게 번 비율%", "최대 이익$/oz", "이긴 거래 보유(분, 중간값)"])
    print(res.to_string(index=False))
    res.to_csv(f"multi_pos_{fe.START.year}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
