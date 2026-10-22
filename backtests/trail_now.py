"""지금 봇 진입(반대 콩 청산 + 과열 거르기)에 '익절 고정 대신 크게 먹기' 청산을 붙여 보기 (실제 틱)

HK 방송 백테스트: 이긴 거래가 +9$/oz까지 감 (3$ 익절 없음) -> 같은 진입에서 청산만 바꿔 비교
청산 (모두 손절 7.5$, 5분봉 반대 콩 청산, 금요일 23:30 청산은 같음):
  익절 3$ (지금 봇)
  +3$부터 추적 2$   : +3$에 닿으면 손절선을 +1$로 올리고, 이후 최고점에서 2$ 되돌리면 청산
  +3$부터 추적 3$   : 같은데 3$ 되돌림
  +5$부터 추적 3$
  익절 없음         : 손절 7.5 또는 반대 콩에서만 청산
  익절 6$
lot = 그날 시작 잔고 1,000달러당 0.05 (지금 봇)

사용법: python -m backtests.trail_now 2025-01-01 2025-12-31
        python -m backtests.trail_now 2026-01-01
        python -m backtests.trail_now 2026-03-01 2026-03-07
"""
import gc
import sys

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.long_period as lb
import backtests.trail_exit as te
import backtests.winrate_ideas as wi
from backtests.add_positions_ticks import hit, load_m1_recent
from backtests.ext_filter import features
from backtests.gold_rest_others import resample
from backtests.sar_exit import against_times

BASE = [("ext", 0.8), ("risk", 2.8)]
INF = np.inf
# (익절, 추적 시작, 추적 폭, 잠금)
EXITS = {
    "익절 3$ (지금 봇)": (3.0, None, None, None),
    "+3$부터 추적 2$": (None, 3.0, 2.0, 1.0),
    "+3$부터 추적 3$": (None, 3.0, 3.0, 1.0),
    "+5$부터 추적 3$": (None, 5.0, 3.0, 2.0),
    "익절 6$": (6.0, None, None, None),
    "익절 없음 (손절·반대콩만)": (None, None, None, None),
}


def trade(t, bid, ask, k, side, fc, against, o, _):
    tp, start, trail, lock = o["exit"]
    X = bid if side == 1 else ask
    e = ask[k] if side == 1 else bid[k]
    a = against[side]
    i = np.searchsorted(a, t[k], side="right")
    js = int(np.searchsorted(t, a[i])) if i < len(a) else len(t)
    end = min(js, fc)
    sl_px = e - side * te.SL
    if tp is not None:
        tp_px = e + side * tp
        j = hit(X, k + 1, sl_px, tp_px) if side == 1 else hit(X, k + 1, tp_px, sl_px)
    elif start is not None:
        st_px = e + side * start
        j0 = hit(X, k + 1, sl_px, st_px) if side == 1 else hit(X, k + 1, st_px, sl_px)
        if j0 < end and (X[j0] - e) * side > 0:
            j = te.trail_hit(X, j0 + 1, side, e + side * lock, trail, end)
        else:
            j = j0
    else:
        j = hit(X, k + 1, sl_px, INF) if side == 1 else hit(X, k + 1, -INF, sl_px)
    j = min(j, end)
    if j >= len(t):
        return None
    return k, j, (X[j] - e) * side


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
    wi.trade = trade
    accs = {n: wi.Acc({"feats": BASE, "exit": x}, against, set()) for n, x in EXITS.items()}
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
    rows = []
    for n, a in accs.items():
        tr = pd.DataFrame(a.rows, columns=["open", "day", "mins", "oz", "profit", "bal"])
        rows.append({"청산": n, **te.summary(a), "평균$/oz": round(tr.oz.mean(), 2) if len(tr) else 0,
                     "이긴 평균$/oz": round(tr.oz[tr.oz > 0].mean(), 2) if len(tr) else 0})
    res = pd.DataFrame(rows).drop(columns=["평균 이익$/oz", "3$넘게 번 비율%", "최대 이익$/oz", "이긴 거래 보유(분, 중간값)"])
    print(res.to_string(index=False))
    res.to_csv(f"trail_now_{fe.START:%Y%m%d}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
