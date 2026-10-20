"""SAR 단타에 '반대 콩' 청산 추가: 진입 후 SAR 점이 반대로 찍히면 (반대 전환 봉이 끝나면) 그 틱에 시장가 청산 (실제 틱)
  지금        : TP 3 / SL 7.5만
  5분봉 반대콩 : TP 3 / SL 7.5 + 5분봉 SAR 반대 전환
  1분봉 반대콩 : TP 3 / SL 7.5 + 1분봉 SAR 반대 전환
  ... SL 20   : 손절을 20$로 넓히고 반대콩을 실제 손절처럼 사용
승률 쉬기는 끔 (청산 방식이 바뀌면 승률이 달라져서 공정 비교가 안 됨). 나머지는 trail_exit 기준 방식과 같음

사용법: python -m backtests.sar_exit [시작 YYYY-MM-DD] [끝 YYYY-MM-DD]
"""
import sys

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.long_period as lb
import backtests.trail_exit as te
from backtests.add_positions_ticks import hit, load_m1_recent
from backtests.gold_rest_others import resample
from indicators import psar

CASES = {"지금 (TP 3 / SL 7.5)": None, "5분봉 반대콩": ("m5", 7.5), "1분봉 반대콩": ("m1", 7.5),
         "5분봉 반대콩, SL 20": ("m5", 20.0), "1분봉 반대콩, SL 20": ("m1", 20.0)}


def against_times(bars, minutes):
    """방향별로 SAR이 반대인 봉의 끝 시각(ms) 목록"""
    end = (bars.time + pd.Timedelta(minutes=minutes)).values.astype("datetime64[ms]").astype(np.int64)
    up = psar(bars.high.values, bars.low.values)
    return {1: end[~up], -1: end[up]}


def make_trade(against):
    def trade(t, bid, ask, k, side, case, fri_ms, _):
        X = bid if side == 1 else ask
        e = ask[k] if side == 1 else bid[k]
        fc = int(np.searchsorted(t, fri_ms))
        sl = te.SL if case is None else case[1]
        sl_px, tp_px = e - side * sl, e + side * te.TP
        j = hit(X, k + 1, sl_px, tp_px) if side == 1 else hit(X, k + 1, tp_px, sl_px)
        ex = [j, fc]
        if case is not None:
            a = against[case[0]][side]
            i = np.searchsorted(a, t[k], side="right")
            if i < len(a):
                ex.append(int(np.searchsorted(t, a[i])))
        j = min(ex)
        if j >= len(t):
            return None
        return j, (X[j] - e) * side
    return trade


def main():
    sys.argv = [a for a in sys.argv if a != "--pause"] + (["--pause"] if "--pause" in sys.argv else [])
    if len(sys.argv) > 1 and sys.argv[1] != "--pause":
        fe.START = pd.Timestamp(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] != "--pause":
        fe.END = pd.Timestamp(sys.argv[2])
    if "--pause" not in sys.argv:
        te.WINRATE_MIN = 0
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = load_m1_recent()
    m5, h1 = resample(m1, 5), resample(m1, 60)
    te.trade = make_trade({"m5": against_times(m5, 5), "m1": against_times(m1, 1)})
    del m1
    sig = fe.bar_close_signals(m5, 5, 0.02, fe.context(m5, h1), True)
    accs = {n: te.Acc(c) for n, c in CASES.items()}
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
    rows = []
    for n, a in accs.items():
        tr = pd.DataFrame(a.rows, columns=["open", "day", "mins", "oz", "profit", "bal"])
        sl = te.SL if CASES[n] is None else CASES[n][1]
        tx = tr[(tr.oz < te.TP - 0.01) & (tr.oz > -sl + 0.01)]
        rows.append({"방식": n, **te.summary(a), "반대콩 청산": len(tx),
                     "평균$/oz": round(tx.oz.mean(), 2) if len(tx) else 0,
                     "그중 이익%": round((tx.oz > 0).mean() * 100, 1) if len(tx) else 0})
    res = pd.DataFrame(rows).drop(columns=["평균 이익$/oz", "3$넘게 번 비율%", "최대 이익$/oz"])
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
