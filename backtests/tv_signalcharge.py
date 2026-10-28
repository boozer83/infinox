"""Signal Charge: Momentum Buy Sell Signals (트레이딩뷰 지표) 를 금에 (2022-01 ~ 2026-10, 1분봉 zip)

원본 Pine 그대로 (봉 마감 확정 신호만):
  d = (종가 - 전 종가) / 전 봉 ATR(14), charge = EMA(d,L) / norm x (0.5 + 0.5 x |EMA(d,L)| / EMA(|d|,L)) / 1.0
  norm = sqrt(a / (2 - a)) x max(전 봉 stdev(d,100), 0.2), a = 2/(L+1). L = Fast 8 / Balanced 13 / Smooth 21 (기본)
  charge >= 1 이면 매수 확정, <= -1 이면 매도 확정. 매수 중 charge < 0.15 면 중립 (fade), 매도는 반대
매매 두 가지 (봉 마감 신호 -> 다음 봉 시가):
  보유      : 지표 상태(매수/매도/중립) 그대로 들고 있기 (tv_single.trades)
  손절·익절 : 확정 신호에 진입, 손절 신호 봉 종가 ∓ 1.5 x ATR(14), 익절 1R / 2R / 3R (tv_batch2.rr_trades)
비용: tv_single과 같음 (실제 스프레드, 미끄러짐 0.02, 스왑)

사용법: python -m backtests.tv_signalcharge
"""
import numpy as np
import pandas as pd

import backtests.tv_single as ts
from backtests.tv_batch2 import rr_trades
from indicators import atr, ema

TFS = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240}
SENS = {"Smooth": 21, "Balanced": 13, "Fast": 8}
FADE, ATR_K = 0.15, 1.5


def charge(d, L):
    c = d.close
    a = atr(d, 14)
    dd = (c - c.shift()) / a.shift().clip(lower=0.01)
    ed, ea = ema(dd, L), ema(dd.abs(), L)
    al = 2.0 / (L + 1)
    norm = np.sqrt(al / (2 - al)) * dd.rolling(100).std(ddof=0).shift().fillna(0.8).clip(lower=0.2)
    eff = np.where(ea > 1e-9, ed.abs() / ea, 0.0)
    ch = (ed / norm * (0.5 + 0.5 * eff)).to_numpy(copy=True)
    ch[:120] = 0.0
    return np.nan_to_num(ch), a.values


def stance(ch):
    """(지표 상태 배열, 확정 신호 배열)"""
    st, lock = np.zeros(len(ch), int), np.zeros(len(ch), int)
    cur = 0
    for i, x in enumerate(ch):
        buy, sell = x >= 1.0 and cur != 1, x <= -1.0 and cur != -1
        if buy:
            cur, lock[i] = 1, 1
        elif sell:
            cur, lock[i] = -1, -1
        elif (cur == 1 and x < FADE) or (cur == -1 and x > -FADE):
            cur = 0
        st[i] = cur
    return st, lock


def main():
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = ts.load_m1("XAUUSD")
    out = []
    for tf, mins in TFS.items():
        d = m1[["time", "open", "high", "low", "close", "sp"]].reset_index(drop=True) if mins == 1 else ts.bars(m1, mins)
        for sn, L in SENS.items():
            ch, a = charge(d, L)
            st, lock = stance(ch)
            tr = ts.trades(d, st)
            out.append({"봉": tf, "민감도": sn, "매매": "보유", **ts.summary(tr)})
            if sn == "Smooth":
                stop = np.where(lock == 1, d.close.values - ATR_K * a, d.close.values + ATR_K * a)
                for rr in (1.0, 2.0, 3.0):
                    tr = rr_trades(d, lock, stop, rr)
                    out.append({"봉": tf, "민감도": sn, "매매": f"손절 1.5ATR, {rr:.0f}R", **ts.summary(tr)})
        print(tf, "끝", flush=True)
    res = pd.DataFrame(out)
    res.to_csv("results/tv_signalcharge.csv", index=False, encoding="utf-8-sig")
    print(res.drop(columns=["평균%"]).to_string(index=False))


if __name__ == "__main__":
    main()
