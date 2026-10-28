"""ICT Power Of Three | Flux Charts (트레이딩뷰) 를 금에 (2022-01 ~ 2026-10, 1분봉 zip으로 만든 봉)

원본 Pine 그대로 옮김 (한 번에 한 PO3, 봉 마감 기준):
  매집   : 최근 L봉 고가-저가 폭 <= ATR(50) x A  ->  박스 = 최고가 + 0.5 ATR / 최저가 - 0.5 ATR
  이탈   : 고가(꼬리)가 박스 위 또는 저가가 박스 아래
  조작   : 박스 위 + ATR x M 넘음 -> 매도 / 박스 아래 - ATR x M 깨짐 -> 매수 (돌파 반대로)
  진입   : 그 봉 마감 (원본은 종가, 여기서는 다음 봉 시가 + 스프레드)
  청산   : 손절 ATR x 5 (Risk "Normal"), 익절 = 손절폭 x 0.86. 한 봉에 둘 다 닿으면 손절 (원본도 손절이 덮어씀)
  모드   : Small Manipulation (L40, A5, M0.6, 기본) / Short Accumulation (L11, A2, M1) / Big Manipulation (L11, A5, M1)
비교용 "지표 표 방식": 원본 대시보드처럼 신호 봉 종가 체결, 스프레드·스왑 없음
비용: tv_single과 같음 (실제 스프레드, 미끄러짐 0.02, 스왑)

사용법: python -m backtests.tv_po3
"""
import numpy as np
import pandas as pd

import backtests.tv_single as ts
from indicators import atr

MODES = {"Small Manipulation": (40, 5.0, 0.6), "Short Accumulation": (11, 2.0, 1.0), "Big Manipulation": (11, 5.0, 1.0)}
TFS = {"M5": 5, "M15": 15, "H1": 60, "H4": 240}
SL_ATR, RR, EXPAND = 5.0, 0.86, 0.5


def po3(d, acc_len, acc_mult, man_mult, real=True):
    o, h, l, c, sp, t = (d[k].values for k in ("open", "high", "low", "close", "sp", "time"))
    a = atr(d, 50).values
    hh = d.high.rolling(acc_len).max().values
    ll = d.low.rolling(acc_len).min().values
    rows = []
    state, top, bot = "acc", 0.0, 0.0
    side, ent, et, sl, tp = 0, 0.0, None, 0.0, 0.0
    i, n = 50, len(d)
    while i < n:
        if state == "trade":
            if real:
                lo_px, hi_px = (l[i], h[i]) if side == 1 else (l[i] + sp[i], h[i] + sp[i])
            else:
                lo_px, hi_px = l[i], h[i]
            hit_sl = lo_px <= sl if side == 1 else hi_px >= sl
            hit_tp = hi_px >= tp if side == 1 else lo_px <= tp
            if hit_sl or hit_tp:
                px = sl if hit_sl else tp
                if real:                                  # 갭이면 시가, 손절은 미끄러짐
                    op = o[i] if side == 1 else o[i] + sp[i]
                    if hit_sl:
                        px = min(sl, op) - ts.SLIP if side == 1 else max(sl, op) + ts.SLIP
                    else:
                        px = max(tp, op) if side == 1 else min(tp, op)
                pnl = (px - ent) * side
                if real:
                    pnl += ts.swap_nights(et, t[i]) * (ts.SWAP_LONG if side == 1 else ts.SWAP_SHORT)
                rows.append((et, t[i], side, ent, pnl / ent * 100, "손절" if hit_sl else "익절"))
                state = "new"
            i += 1
            continue
        if state == "new":                                # 청산 다음 봉에 새 PO3
            state = "acc"
        if state == "acc" and hh[i] - ll[i] <= a[i] * acc_mult:
            state, top, bot = "acc_end", hh[i] + a[i] * EXPAND, ll[i] - a[i] * EXPAND
        elif state == "acc_end":
            if h[i] > top or l[i] < bot:
                state = "man"
        if state == "man":
            sig = -1 if h[i] > top + a[i] * man_mult else (1 if l[i] < bot - a[i] * man_mult else 0)
            if sig and i + 1 < n:
                side = sig
                if real:
                    k = i + 1
                    ent = o[k] + sp[k] + ts.SLIP if side == 1 else o[k] - ts.SLIP
                    et = t[k]
                else:
                    k, ent, et = i + 1, c[i], t[i]
                sl = ent - side * a[i] * SL_ATR
                tp = ent + side * a[i] * SL_ATR * RR
                state = "trade"
                i = k
                continue
        i += 1
    return pd.DataFrame(rows, columns=["open", "close", "side", "entry", "pct", "why"])


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = ts.load_m1("XAUUSD")
    out = []
    for tf, mins in TFS.items():
        d = ts.bars(m1, mins)
        for mode, (L, A, M) in MODES.items():
            for real in (False, True):
                tr = po3(d, L, A, M, real)
                if len(tr) < 5:
                    continue
                r = ts.summary(tr)
                hold_h = (pd.to_datetime(tr.close) - pd.to_datetime(tr.open)).dt.total_seconds().median() / 3600
                out.append({"봉": tf, "모드": mode, "계산": "실제 비용" if real else "지표 표 방식", **r,
                            "보유(시간,중간값)": round(hold_h, 1)})
        print(f"{tf} 끝", flush=True)
    res = pd.DataFrame(out)
    res.to_csv("results/tv_po3.csv", index=False, encoding="utf-8-sig")
    print(res.drop(columns=["평균%"]).to_string(index=False))


if __name__ == "__main__":
    main()
