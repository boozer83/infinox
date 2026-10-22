"""HK 홈런식 역추세 진입 시험: 5분봉 SAR 전환 때 5분봉 RSI가 반대쪽이면 진입 (실제 틱)

MT4 HK_HR 118건: 매수 때 5분 RSI 평균 37.6 (35~41), 매도 때 60.8 (57~65)
변형 (청산은 지금 봇: 손절 7.5 / 익절 3 / 5분봉 반대 콩 청산, 시간·ATR 필터 같음):
  기준 (지금 봇)        : RSI가 신호 방향 쪽(매수 > 50) + 1시간 SAR 일치 + 과열 거르기
  역RSI 40/60 + 1시간   : 매수 RSI < 40 / 매도 > 60, 1시간 SAR 일치
  역RSI 40/60           : 1시간 SAR 상관없이
  역RSI 45/55 + 1시간
  역RSI 35/65 + 1시간
  기준 + 역RSI 40/60 + 1시간 : 둘 다 진입

사용법: python -m backtests.rsi_counter 2025-01-01 2025-12-31
        python -m backtests.rsi_counter 2026-01-01
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
from indicators import atr, rsi

BASE = [("ext", 0.8), ("risk", 2.8)]
VARIANTS = {
    "기준 (지금 봇)": {"src": "base", "feats": BASE},
    "역RSI 40/60 + 1시간": {"src": ("ctr", 10, True)},
    "역RSI 40/60": {"src": ("ctr", 10, False)},
    "역RSI 45/55 + 1시간": {"src": ("ctr", 5, True)},
    "역RSI 35/65 + 1시간": {"src": ("ctr", 15, True)},
    "기준 + 역RSI 40/60 + 1시간": {"src": "both"},
}


class Acc2(Acc):
    def on_signal(self, s, t, bid, ask, _):
        src = self.o["src"]
        d = (s.r5 - 50) * s.side
        h1 = s.hdir == s.side
        base = d > 0 and h1 and s.ext <= 0.8 and s.risk <= 2.8
        if src == "base":
            ok = base
        elif src == "both":
            ok = base or (d < -10 and h1)
        else:
            _, gap, need_h1 = src
            ok = d < -gap and (h1 or not need_h1)
        if ok:
            super().on_signal(s, t, bid, ask, None)


def signals(m5, h1):
    ctx = fe.context(m5, h1)
    up, _ = fe.psar_trigger(m5.high.values, m5.low.values)
    r = rsi(m5.close).values
    end = m5.time + pd.Timedelta(minutes=5)
    i = np.nonzero(up[1:] != up[:-1])[0] + 1
    side = np.where(up[i], 1, -1)
    ctx5, h, fl = ctx
    sig = pd.DataFrame({"t": end.iloc[i].values, "side": side, "atr": atr(m5).values[i], "r5": r[i]})
    m = pd.merge_asof(sig[["t"]], ctx5, left_on="t", right_on="end", direction="backward")
    sig["ref"] = m.ref.values
    sig["hdir"] = pd.merge_asof(sig[["t"]], h, left_on="t", right_on="end", direction="backward").hdir.values
    f = pd.merge_asof(sig[["t"]], fl, left_on="t", right_on="end", direction="backward")
    sig["flips"] = np.where(f.day.values == sig.t.dt.date.values, f.flips.fillna(0).values, 0)
    sig = sig.merge(features(m5), on="t", how="left")
    sig["rsi"] = sig.r5
    return sig[(sig.t >= fe.START) & (sig.t < fe.END)].reset_index(drop=True)


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
    sig = signals(m5, h1)
    del m5, h1
    gc.collect()
    accs = {n: Acc2(o, against, set()) for n, o in VARIANTS.items()}
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
    rows = [{"방식": n, **te.summary(a)} for n, a in accs.items()]
    res = pd.DataFrame(rows).drop(columns=["평균 이익$/oz", "3$넘게 번 비율%", "최대 이익$/oz", "이긴 거래 보유(분, 중간값)"])
    print(res.to_string(index=False))
    res.to_csv(f"rsi_counter_{fe.START.year}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
