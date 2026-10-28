"""Macketings 1min Scalping V3 [Smart Exit + Time] (트레이딩뷰 전략) 를 금에 (2022-01 ~ 2026-10, 1분봉 zip)

원본 Pine 그대로 옮김 (기본값):
  추세   : EMA340 > EMA500 이고 EMA80/90 띠가 EMA500 위 -> 매수만 (매도는 반대), 종가 > SMA325 (매도는 <)
  진입   : 종가가 EMA80~90 띠 안 (되돌림) 또는 띠 위로 돌파 (매도는 아래), 종가 > 전 봉 종가, EMA80·90 둘 다 3봉 전보다 위
           청산 뒤 100봉 쉬기, 스톡홀름 00:00~22:00 (= 서버 01:00~23:00, 서버 = 스톡홀름 + 1시간으로 고정)
           봉 마감 신호 -> 다음 봉 시가 진입
  손절   : 진입가 -0.2%. 종가가 +0.3% 넘으면 손절을 +0.2%로. 고점이 +0.5% 넘으면 고점 -0.3% 따라가기
  익절   : +1.5% 지정가
  청산   : 종가가 띠 아래로 이탈 (수익 0.3% 이상일 때만), EMA340/500 교차, EMA80 또는 90이 EMA340을 교차 -> 다음 봉 시가
  손절·익절 주문은 봉 마감에 갱신돼 다음 봉부터 적용 (진입한 봉에는 없음, 트레이딩뷰와 같음). 한 봉에 둘 다 닿으면 손절
비교용 "트레이딩뷰 방식": 원본 설정처럼 수수료 0.05% x 2, 미끄러짐 1틱, 스프레드·스왑 없음
비용: tv_single과 같음 (실제 스프레드, 미끄러짐 0.02, 스왑)

사용법: python -m backtests.tv_macketing
"""
import numpy as np
import pandas as pd

import backtests.tv_single as ts
from indicators import ema

SL, BE_TRIG, BE_PROFIT, TP = 0.002, 0.003, 0.002, 0.015
TRAIL_TRIG, TRAIL = 0.005, 0.003
SMART_MIN = 0.003
COOLDOWN, SLOPE = 100, 3
TV_COMM, TV_SLIP = 0.0005, 0.01


def signals(d):
    c = d.close
    e80, e90, e340, e500 = (ema(c, n) for n in (80, 90, 340, 500))
    safe = c.rolling(325).mean()
    lo, hi = np.minimum(e80, e90), np.maximum(e80, e90)
    in_band = (c >= lo) & (c <= hi)
    brk_up = (c > hi) & (c.shift() <= hi.shift())
    brk_dn = (c < lo) & (c.shift() >= lo.shift())
    up = (e80 > e80.shift(SLOPE)) & (e90 > e90.shift(SLOPE))
    dn = (e80 < e80.shift(SLOPE)) & (e90 < e90.shift(SLOPE))
    sthm = ((d.time - pd.Timedelta(hours=1)).dt.hour * 60 + d.time.dt.minute)
    time_ok = sthm < 22 * 60
    long_sig = (e340 > e500) & (lo > e500) & (in_band | brk_up) & (c > c.shift()) & (c > safe) & up & time_ok
    short_sig = (e340 < e500) & (hi < e500) & (in_band | brk_dn) & (c < c.shift()) & (c < safe) & dn & time_ok
    cross_dn = lambda a, b: (a < b) & (a.shift() >= b.shift())
    x = dict(
        side=np.where(long_sig, 1, np.where(short_sig, -1, 0)),
        band_dn=cross_dn(c, lo).values, band_up=cross_dn(-c, -hi).values,
        trend_dn=cross_dn(e340, e500).values, trend_up=cross_dn(e500, e340).values,
        fast_dn=(cross_dn(e80, e340) | cross_dn(e90, e340)).values,
        fast_up=(cross_dn(e340, e80) | cross_dn(e340, e90)).values)
    return x


def run(d, x, real):
    o, h, l, c, sp, t = (d[k].values for k in ("open", "high", "low", "close", "sp", "time"))
    if not real:
        sp = np.zeros_like(sp)
    slip = ts.SLIP if real else TV_SLIP
    side = x["side"]
    rows = []
    pos, ep, et, stop, tp, peak = 0, 0.0, None, np.nan, np.nan, np.nan
    pend_entry, pend_close, last_exit, prev_pos = 0, False, -10 ** 9, 0

    def close_at(j, px):
        nonlocal pos
        pnl = (px - ep) * pos
        if real:
            pnl += ts.swap_nights(et, t[j]) * (ts.SWAP_LONG if pos == 1 else ts.SWAP_SHORT)
        else:
            pnl -= TV_COMM * (ep + px)
        rows.append((et, t[j], pos, ep, pnl / ep * 100))
        pos = 0

    for j in range(len(o)):
        armed = pos != 0 and not np.isnan(tp)          # 지난 봉 마감에 걸어 둔 손절·익절
        if pend_close and pos:
            close_at(j, o[j] - slip if pos == 1 else o[j] + sp[j] + slip)
            armed = False
        pend_close = False
        if pend_entry and pos == 0:
            pos = pend_entry
            ep = o[j] + sp[j] + slip if pos == 1 else o[j] - slip
            et, stop, tp, peak, armed = t[j], np.nan, np.nan, np.nan, False
        pend_entry = 0
        if armed:
            if pos == 1:
                if not np.isnan(stop) and l[j] <= stop:
                    close_at(j, min(stop, o[j]) - slip)
                elif h[j] >= tp:
                    close_at(j, max(tp, o[j]))
            else:
                if not np.isnan(stop) and h[j] + sp[j] >= stop:
                    close_at(j, max(stop, o[j] + sp[j]) + slip)
                elif l[j] + sp[j] <= tp:
                    close_at(j, min(tp, o[j] + sp[j]))
        # 봉 마감
        if prev_pos != 0 and pos == 0:
            last_exit = j
        if pos == 0:
            stop = tp = peak = np.nan
            if side[j] and j - last_exit >= COOLDOWN:
                pend_entry = side[j]
        else:
            s = pos
            gain = (c[j] - ep) / ep * s
            band = x["band_dn"][j] if s == 1 else x["band_up"][j]
            trend = x["trend_dn"][j] if s == 1 else x["trend_up"][j]
            fast = x["fast_dn"][j] if s == 1 else x["fast_up"][j]
            if (band and gain >= SMART_MIN) or trend or fast:
                pend_close = True
            else:
                ext = h[j] if s == 1 else l[j]
                peak = ext if np.isnan(peak) else (max(peak, ext) if s == 1 else min(peak, ext))
                if np.isnan(stop):
                    stop = ep * (1 - s * SL)
                if gain >= BE_TRIG:
                    nb = ep * (1 + s * BE_PROFIT)
                    stop = max(stop, nb) if s == 1 else min(stop, nb)
                if (peak - ep) / ep * s >= TRAIL_TRIG:
                    tr = peak * (1 - s * TRAIL)
                    stop = max(stop, tr) if s == 1 else min(stop, tr)
            tp = ep * (1 + s * TP)
        prev_pos = pos
    return pd.DataFrame(rows, columns=["open", "close", "side", "entry", "pct"])


def main():
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = ts.load_m1("XAUUSD")
    out = []
    for tf, mins in (("M1", 1), ("M5", 5)):
        d = m1[["time", "open", "high", "low", "close", "sp"]] if mins == 1 else ts.bars(m1, mins)
        x = signals(d)
        for how, real in (("트레이딩뷰 방식", False), ("실제 비용", True)):
            tr = run(d, x, real)
            hold = (pd.to_datetime(tr.close) - pd.to_datetime(tr.open)).dt.total_seconds().median() / 60
            out.append({"봉": tf, "계산": how, **ts.summary(tr), "보유(분,중간값)": round(hold, 1)})
            print(tf, how, "끝", flush=True)
    res = pd.DataFrame(out)
    res.to_csv("results/tv_macketing.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
