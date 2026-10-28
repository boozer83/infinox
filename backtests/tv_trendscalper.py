"""Trend Scalper (트레이딩뷰 지표, EMA 9/21/89 눌림 삼각형) 를 금에 (2022-01 ~ 2026-10, 1분봉 zip)

원본 Pine 그대로 (기본값):
  매수 삼각형: EMA9 > EMA21 > EMA89 이고 저가가 EMA21 ~ EMA9 사이 (EMA9까지 눌렸지만 EMA21 위). 매도는 반대 (고가가 EMA9 ~ EMA21)
  "첫 신호만": 50봉 안에 반대 정렬이 있었고 같은 방향 첫 신호일 때만
지표에 청산이 없어서: 봉 마감 신호 -> 다음 봉 시가 진입, 손절 = 신호 봉 종가 ∓ 1.5 x ATR(14), 익절 1R / 2R / 3R. 한 번에 한 거래
  (tv_batch2.rr_trades: 실제 스프레드, 미끄러짐 0.02, 스왑, 한 봉에 둘 다 닿으면 손절)

사용법: python -m backtests.tv_trendscalper
"""
import numpy as np
import pandas as pd

import backtests.tv_single as ts
from backtests.tv_batch2 import rr_trades
from indicators import atr, ema

TFS = {"M1": 1, "M5": 5, "M15": 15, "H1": 60}
ATR_K, LOOKBACK = 1.5, 50


def signals(d, first_only):
    c, h, l = d.close, d.high, d.low
    e1, e2, e3 = ema(c, 9), ema(c, 21), ema(c, 89)
    bull, bear = (e1 > e2) & (e2 > e3), (e1 < e2) & (e2 < e3)
    lc = bull & (l <= e1) & (l >= e2)
    sc = bear & (h >= e1) & (h <= e2)
    if not first_only:
        side = np.where(lc, 1, np.where(sc, -1, 0))
    else:
        idx = np.arange(len(d))
        since = lambda m: idx - pd.Series(np.where(m, idx, np.nan)).ffill().fillna(-10 ** 9).values
        rb, rs = lc & (since(bear.values) <= LOOKBACK), sc & (since(bull.values) <= LOOKBACK)
        side, last = np.zeros(len(d), int), 0
        for i in np.flatnonzero(rb.values | rs.values):
            if rb.iat[i] and last != 1:
                side[i], last = 1, 1
            elif rs.iat[i] and last != -1:
                side[i], last = -1, -1
    a = atr(d, 14)
    stop = np.where(side == 1, c - ATR_K * a, c + ATR_K * a)
    return side, stop


def main():
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = ts.load_m1("XAUUSD")
    out = []
    for tf, mins in TFS.items():
        d = m1[["time", "open", "high", "low", "close", "sp"]].reset_index(drop=True) if mins == 1 else ts.bars(m1, mins)
        for mode, first in (("모든 삼각형", False), ("첫 신호만", True)):
            side, stop = signals(d, first)
            for rr in (1.0, 2.0, 3.0):
                tr = rr_trades(d, side, stop, rr)
                s = ts.summary(tr)
                out.append({"봉": tf, "신호": mode, "익절": f"{rr:.0f}R", **s, "거래당R": round(tr.r.mean(), 3) if len(tr) else np.nan})
            print(tf, mode, "끝", flush=True)
    res = pd.DataFrame(out)
    res.to_csv("results/tv_trendscalper.csv", index=False, encoding="utf-8-sig")
    print(res.drop(columns=["평균%"]).to_string(index=False))


if __name__ == "__main__":
    main()
