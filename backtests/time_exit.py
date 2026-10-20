"""SAR 단타에 시간 청산 추가: 진입 후 N분 안에 TP 3 / SL 7.5 둘 다 안 닿으면 그때 시장가로 정리 (실제 틱)
승률 쉬기(최근 200건 71.4%)는 끔: 시간 청산은 승률을 낮춰서 켜 두면 200건 뒤로 계속 쉬게 됨 (공정 비교 위해 전부 끔)
나머지(쉬기 규칙, 금요일 청산, 1,000달러 시작, 1,000달러당 0.05 lot)는 trail_exit 기준 방식과 같음

사용법: python -m backtests.time_exit [시작 YYYY-MM-DD] [끝 YYYY-MM-DD]
"""
import sys

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.long_period as lb
import backtests.trail_exit as te
from backtests.add_positions_ticks import hit, load_m1_recent
from backtests.gold_rest_others import resample

LIMITS = (None, 10, 15, 20, 30)     # 분 (None = 지금 봇)


def trade(t, bid, ask, k, side, limit, fri_ms, against):
    X = bid if side == 1 else ask
    e = ask[k] if side == 1 else bid[k]
    end_ms = fri_ms if limit is None else min(fri_ms, int(t[k]) + limit * 60_000)
    fc = int(np.searchsorted(t, end_ms))
    sl_px, tp_px = e - side * te.SL, e + side * te.TP
    j = hit(X, k + 1, sl_px, tp_px) if side == 1 else hit(X, k + 1, tp_px, sl_px)
    j = min(j, fc)
    if j >= len(t):
        return None
    return j, (X[j] - e) * side


def main():
    if len(sys.argv) > 1:
        fe.START = pd.Timestamp(sys.argv[1])
    if len(sys.argv) > 2:
        fe.END = pd.Timestamp(sys.argv[2])
    te.trade = trade
    te.WINRATE_MIN = 0
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = load_m1_recent()
    m5, h1 = resample(m1, 5), resample(m1, 60)
    del m1
    sig = fe.bar_close_signals(m5, 5, 0.02, fe.context(m5, h1), True)
    accs = {n: te.Acc(n) for n in LIMITS}
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
        tx = tr[(tr.oz < te.TP - 0.01) & (tr.oz > -te.SL + 0.01)]       # TP/SL 아닌 청산 (시간·금요일)
        rows.append({"시간 청산": f"{n}분" if n else "없음 (지금)", **te.summary(a), "시간청산 건수": len(tx),
                     "시간청산 평균$/oz": round(tx.oz.mean(), 2) if len(tx) else 0,
                     "그중 이익%": round((tx.oz > 0).mean() * 100, 1) if len(tx) else 0})
    res = pd.DataFrame(rows).drop(columns=["평균 이익$/oz", "3$넘게 번 비율%", "최대 이익$/oz"])
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
