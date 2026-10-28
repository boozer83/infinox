"""사용자가 준 트레이딩뷰 지표 3개를 금에 (2022-01 ~ 2026-10, 기본 설정, 원본 Pine 규칙대로)

1) DTC EMA Trend System 1.3.6: EMA 30/35/40/45/50/60 이 정렬되기 시작한 봉 + ATR14 > ATR14의 SMA50 -> BUY / SELL
   a. 지표의 손절(1 ATR)과 TP1~TP3 (1R/2R/3R) / b. 반대 신호까지 보유
2) Reversal Z-Score Bands (Black-Chart): EMA 200, 지수 가중 표준편차, k = 2
   종가가 ±2σ 밖으로 처음 나간 봉의 밴드 가격(첫 터치 가격)을 고정 -> 종가가 그 가격 안으로 돌아오거나 밴드가 돌아오면 삼각형
   위에서 돌아오면 매도, 아래에서 돌아오면 매수, 종가가 EMA에 닿으면 청산
3) Smart Money Concepts (LuxAlgo): 스윙 구조(50) / 내부 구조(5)의 BOS·CHoCH 로 정해지는 추세 방향대로 항상 보유
비용·체결: tv_single과 같음

사용법: python -m backtests.tv_batch3
"""
import numpy as np
import pandas as pd

import backtests.tv_single as ts
from backtests.tv_batch2 import rr_trades
from indicators import atr, ema

TFS = {"M5": 5, "M15": 15, "H1": 60, "H4": 240}


def dtc(d):
    e = np.column_stack([ema(d.close, n).values for n in (30, 35, 40, 45, 50, 60)])
    bull = (np.diff(e, axis=1) < 0).all(axis=1)
    bear = (np.diff(e, axis=1) > 0).all(axis=1)
    tr = np.where(bull, 1, np.where(bear, -1, 0))
    prev = np.r_[0, tr[:-1]]
    a = atr(d, 14)
    act = (a > a.rolling(50).mean()).values
    buy = (tr == 1) & (prev != 1) & act
    sell = (tr == -1) & (prev != -1) & act
    side = np.where(buy, 1, np.where(sell, -1, 0))
    stop = d.close.values - side * a.values
    return side, stop, buy, sell


def zscore_rev(d, n=200, k=2.0):
    c = d.close
    m = ema(c, n)
    sd = np.sqrt(np.maximum(ema(c * c, n) - m * m, 0)).values
    mv, cv = m.values, c.values
    z = np.where(sd > 0, (cv - mv) / np.where(sd > 0, sd, 1), np.nan)
    side_t, lvl = 0, np.nan
    long_on, short_on = np.zeros(len(d), bool), np.zeros(len(d), bool)
    for i in range(1, len(d)):
        s0 = side_t
        if s0 != 0:
            p_back = cv[i] <= lvl if s0 > 0 else cv[i] >= lvl
            e_back = cv[i] <= mv[i] if s0 > 0 else cv[i] >= mv[i]
            z_re = (z[i - 1] > k and z[i] <= k) if s0 > 0 else (z[i - 1] < -k and z[i] >= -k)
            b_re = (not p_back) and z_re
            if p_back or b_re:
                if s0 > 0:
                    short_on[i] = True
                else:
                    long_on[i] = True
            if p_back or e_back:
                side_t = 0
        elif not np.isnan(z[i]) and abs(z[i]) > k:
            side_t = 1 if z[i] > 0 else -1
            lvl = mv[i] + side_t * k * sd[i]
    return ts.hold(long_on, short_on, cv >= mv, cv <= mv)


def smc_bias(d, size):
    """LuxAlgo SMC: size봉 다리(leg)로 스윙 고저를 잡고, 종가가 그 고저를 넘으면 추세 전환(BOS/CHoCH)"""
    h, l, c = d.high.values, d.low.values, d.close.values
    n = len(d)
    hh = pd.Series(h).rolling(size).max().values      # ta.highest(size): 지금 봉 포함 최근 size봉
    ll = pd.Series(l).rolling(size).min().values
    leg, prev_leg = 0, 0
    top = bot = np.nan
    top_x = bot_x = False
    prev_top = prev_bot = np.nan
    bias = 0
    out = np.zeros(n)
    for i in range(n):
        if i >= size:
            if h[i - size] > hh[i]:
                leg = 0
            elif l[i - size] < ll[i]:
                leg = 1
        if leg != prev_leg:
            if leg == 1:                      # 상승 다리 시작 = 저점 확정
                bot, bot_x = l[i - size], False
            else:
                top, top_x = h[i - size], False
        prev_leg = leg
        if i > 0:
            if not np.isnan(top) and not top_x and c[i] > top and not (c[i - 1] > prev_top):
                top_x, bias = True, 1
            if not np.isnan(bot) and not bot_x and c[i] < bot and not (c[i - 1] < prev_bot):
                bot_x, bias = True, -1
        prev_top, prev_bot = top, bot
        out[i] = bias
    return out


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = ts.load_m1("XAUUSD")
    out = []

    def add(ind, tf, how, tr):
        if len(tr) >= 10:
            out.append({"지표": ind, "봉": tf, "방식": how, **ts.summary(tr)})

    for tf, mins in TFS.items():
        d = ts.bars(m1, mins)
        side, stop, buy, sell = dtc(d)
        for rr in (1.0, 2.0, 3.0):
            add("DTC EMA Trend", tf, f"손절 1ATR, 익절 {rr:.0f}R", rr_trades(d, side, stop, rr))
        add("DTC EMA Trend", tf, "반대 신호까지 보유", ts.trades(d, ts.hold(buy, sell)))
        add("Z-Score Bands", tf, "삼각형 진입, EMA 청산", ts.trades(d, zscore_rev(d)))
        add("SMC LuxAlgo", tf, "스윙 구조(50) 방향", ts.trades(d, smc_bias(d, 50)))
        add("SMC LuxAlgo", tf, "내부 구조(5) 방향", ts.trades(d, smc_bias(d, 5)))
        print(f"{tf} 끝", flush=True)
    res = pd.DataFrame(out)
    res.to_csv("results/tv_batch3.csv", index=False, encoding="utf-8-sig")
    print(res.drop(columns=["평균%"]).to_string(index=False))


if __name__ == "__main__":
    main()
