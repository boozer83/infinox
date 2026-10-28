"""Trend Fair Value Gap Delta [ChartPrime] 를 금에 (2022-01 ~ 2026-10)

지표 (원본 Pine 그대로):
  밴드 = SMA(종가, 20) +- ATR(50) x 2. 종가가 위 밴드를 아래에서 위로 넘으면 추세 +1, 아래 밴드를 위에서 아래로 깨면 -1
  상승 추세에서 저가 > 2봉 전 고가 -> 상승 FVG (위 = 이번 저가, 아래 = 2봉 전 고가). 하락 추세는 반대
  소멸: 저가가 FVG 아래선 밑으로 (상승). 잠김: 아래 밴드가 FVG 아래선 이상으로 올라오면 더 이상 안 봄
  (박스의 매수/매도 % 는 표시용이라 안 씀)
지표에 매매 신호가 없어서 흔히 쓰는 방식 두 가지:
  A. 추세 밴드 전환: 추세 +1 매수 / -1 매도, 항상 보유
  B. FVG 되돌림: 추세 방향의 살아 있는 FVG 위선(매도는 아래선)까지 가격이 돌아오면 지정가 진입,
     손절 = FVG 반대선 (소멸 가격), 익절 = 1R / 2R / 3R / 추세가 바뀔 때까지. 한 번에 한 거래
     지정가 매수는 Ask(저가+스프레드)가 위선에 닿으면 위선 가격 (시가가 이미 아래면 시가)
     진입한 봉에서는 손절만 확인 (보수적), 익절은 다음 봉부터. 같은 봉에 손절·익절이 다 닿으면 손절
비용: tv_single과 같음

사용법: python -m backtests.tv_fvg
"""
import numpy as np
import pandas as pd

import backtests.tv_single as ts
from indicators import atr

TFS = {"M5": 5, "M15": 15, "H1": 60, "H4": 240}
TPS = {"익절 1R": 1.0, "익절 2R": 2.0, "익절 3R": 3.0, "추세 바뀔 때까지": None}


def trend_band(d):
    c = d.close.values
    basis = d.close.rolling(20).mean().values
    a = atr(d, 50).values
    up, lo = basis + 2 * a, basis - 2 * a
    tr = np.ones(len(c))
    for i in range(1, len(c)):
        if c[i] > up[i] and c[i - 1] < up[i - 1]:      # 원본은 close[1] < upperBand (지금 봉 밴드) - 거의 같음
            tr[i] = 1
        elif c[i] < lo[i] and c[i - 1] > lo[i - 1]:
            tr[i] = -1
        else:
            tr[i] = tr[i - 1]
    return tr, up, lo


def fvg_trades(d, rr):
    o, h, l, c, sp, t = (d[k].values for k in ("open", "high", "low", "close", "sp", "time"))
    tr, up, lo = trend_band(d)
    bull, bear = [], []                                 # [위, 아래]
    rows = []
    cur, ent, et, sl, tp = 0, 0.0, None, 0.0, np.nan
    for i in range(51, len(d)):
        # 1) 포지션 관리
        if cur:
            just = t[i] == et
            if cur == 1:
                stop_hit = l[i] <= sl
                tp_hit = (not just) and tp == tp and h[i] >= tp
            else:
                stop_hit = h[i] + sp[i] >= sl
                tp_hit = (not just) and tp == tp and l[i] + sp[i] <= tp
            px = None
            if stop_hit:
                op = o[i] if cur == 1 else o[i] + sp[i]
                px = (min(sl, op) - ts.SLIP) if cur == 1 else (max(sl, op) + ts.SLIP)
            elif tp_hit:
                op = o[i] if cur == 1 else o[i] + sp[i]
                px = max(tp, op) if cur == 1 else min(tp, op)
            elif rr is None and tr[i - 1] != cur:       # 직전 봉에서 추세가 바뀜 -> 이번 봉 시가 청산
                px = o[i] - ts.SLIP if cur == 1 else o[i] + sp[i] + ts.SLIP
            if px is not None:
                pnl = (px - ent) * cur + ts.swap_nights(et, t[i]) * (ts.SWAP_LONG if cur == 1 else ts.SWAP_SHORT)
                rows.append((et, t[i], cur, ent, pnl / ent * 100))
                cur = 0
        # 2) 지정가 진입 (직전 봉까지 생긴 살아 있는 FVG, 지난 봉 추세 방향만)
        if not cur and tr[i - 1] == 1 and bull:
            top, bot = max(bull, key=lambda z: z[0])      # 가장 위에 있는 FVG에 먼저 닿음
            if l[i] + sp[i] <= top:
                ent = min(top, o[i] + sp[i])
                cur, et, sl = 1, t[i], bot
                tp = ent + rr * (ent - sl) if rr else np.nan
                bull.remove([top, bot])
                if l[i] <= sl:                            # 같은 봉에 손절
                    px = min(sl, o[i]) - ts.SLIP
                    rows.append((et, t[i], 1, ent, (px - ent) / ent * 100))
                    cur = 0
        elif not cur and tr[i - 1] == -1 and bear:
            top, bot = min(bear, key=lambda z: z[1])
            if h[i] >= bot:
                ent = max(bot, o[i])
                cur, et, sl = -1, t[i], top
                tp = ent - rr * (sl - ent) if rr else np.nan
                bear.remove([top, bot])
                if h[i] + sp[i] >= sl:
                    px = max(sl, o[i] + sp[i]) + ts.SLIP
                    rows.append((et, t[i], -1, ent, (ent - px) / ent * 100))
                    cur = 0
        # 3) FVG 소멸·잠김 정리, 새 FVG (봉 i 마감)
        bull = [z for z in bull if not (l[i] < z[1] or lo[i] >= z[1])]
        bear = [z for z in bear if not (h[i] > z[0] or up[i] <= z[0])]
        if tr[i] == 1 and l[i] > h[i - 2]:
            bull.append([l[i], h[i - 2]])
        if tr[i] == -1 and h[i] < l[i - 2]:
            bear.append([l[i - 2], h[i]])
    return pd.DataFrame(rows, columns=["open", "close", "side", "entry", "pct"])


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = ts.load_m1("XAUUSD")
    out = []
    for tf, mins in TFS.items():
        d = ts.bars(m1, mins)
        cases = {"A. 추세 밴드 전환": ts.trades(d, trend_band(d)[0])}
        for n, rr in TPS.items():
            cases[f"B. FVG 되돌림, {n}"] = fvg_trades(d, rr)
        for n, tr in cases.items():
            if len(tr) >= 10:
                out.append({"봉": tf, "방식": n, **ts.summary(tr)})
        print(f"{tf} 끝", flush=True)
    res = pd.DataFrame(out)
    res.to_csv("results/tv_fvg.csv", index=False, encoding="utf-8-sig")
    print(res.drop(columns=["평균%"]).to_string(index=False))


if __name__ == "__main__":
    main()
