"""4시간봉 SuperTrend(10/3) vs 지금 H4 추세 봇(trend_trader.py)이 얼마나 겹치나 (금, 2022-01 ~ 2026-10)

같은 조건으로 둘 다 4시간봉에서 계산 (tv_single과 같은 체결·비용: 다음 봉 시가, 실제 스프레드, 미끄러짐 0.02, 스왑)
  SuperTrend : 방향이 바뀌면 반대로 갈아탐 (항상 보유, 손절 없음)
  추세 봇     : 종가 > 직전 25봉 최고가 매수 / < 최저가 매도, 12봉 반대 돌파 청산, 손절 3 ATR(14)
                (손절은 4시간봉 고가·저가로 확인, 갭이면 시가), 청산한 봉에서는 새로 진입 안 함
  추세 봇+구름: 위 + 신호 봉 종가가 일목 구름(9/26/52) 같은 쪽일 때만 진입, 아니면 그림자 포지션 (봇과 같음)
겹침: 4시간봉마다 두 전략의 포지션(+1/-1/0)을 비교, 손익은 거래마다 진입가 대비 % (크기 같다고 봄)

사용법: python -m backtests.tv_overlap
"""
import numpy as np
import pandas as pd

import backtests.tv_single as ts
from indicators import atr

N_IN, N_OUT, SL_ATR = 25, 12, 3.0


def cloud(d):
    mid = lambda n: (d.high.rolling(n).max() + d.low.rolling(n).min()) / 2
    a = ((mid(9) + mid(26)) / 2).shift(26).values
    b = mid(52).shift(26).values
    return np.maximum(a, b), np.minimum(a, b)


def trend_bot(d, use_cloud):
    """봉마다 포지션 배열 + 거래 목록 (봇 규칙)"""
    o, h, l, c, sp, t = (d[k].values for k in ("open", "high", "low", "close", "sp", "time"))
    hh = d.high.rolling(N_IN).max().shift().values
    ll = d.low.rolling(N_IN).min().shift().values
    xh = d.high.rolling(N_OUT).max().shift().values
    xl = d.low.rolling(N_OUT).min().shift().values
    a = atr(d).values
    top, bot = cloud(d)
    pos = np.zeros(len(d))
    rows = []
    cur, ent, et, sl = 0, 0.0, None, 0.0
    shadow = None                                   # (방향, 손절가)
    pending = 0                                     # 다음 봉 시가에 할 일: 진입 방향
    for i in range(len(d)):
        if pending:                                 # 직전 봉 신호 -> 이번 봉 시가 진입
            cur = pending
            ent = o[i] + sp[i] + ts.SLIP if cur == 1 else o[i] - ts.SLIP
            et, sl = t[i], ent - cur * SL_ATR * a[i - 1]
            pending = 0
        exited = False
        if cur:                                     # 손절 (봉 안)
            hit = l[i] <= sl if cur == 1 else h[i] + sp[i] >= sl
            if hit:
                px = min(sl, o[i]) - ts.SLIP if cur == 1 else max(sl, o[i] + sp[i]) + ts.SLIP
                rows.append(close_row(et, t[i], cur, ent, px))
                cur, exited = 0, True
        if shadow:
            s, ssl = shadow
            if (l[i] <= ssl) if s == 1 else (h[i] >= ssl):
                shadow = None
        # 봉 마감 판단
        if cur and ((cur == 1 and c[i] < xl[i]) or (cur == -1 and c[i] > xh[i])) and i + 1 < len(d):
            k = i + 1
            px = o[k] - ts.SLIP if cur == 1 else o[k] + sp[k] + ts.SLIP
            rows.append(close_row(et, t[k], cur, ent, px))
            cur, exited = 0, True
            pos[i] = 0
            continue
        if shadow and ((shadow[0] == 1 and c[i] < xl[i]) or (shadow[0] == -1 and c[i] > xh[i])):
            shadow, exited = None, True
        sig = 1 if c[i] > hh[i] else (-1 if c[i] < ll[i] else 0)
        if sig and not cur and not exited and not shadow and i + 1 < len(d):
            if use_cloud and not (c[i] > top[i] if sig == 1 else c[i] < bot[i]):
                shadow = (sig, c[i] - sig * SL_ATR * a[i])
            else:
                pending = sig
        pos[i] = pending or cur
    return pos, pd.DataFrame(rows, columns=["open", "close", "side", "entry", "pct"])


def close_row(et, tx, side, ent, px):
    pnl = (px - ent) * side + ts.swap_nights(et, tx) * (ts.SWAP_LONG if side == 1 else ts.SWAP_SHORT)
    return (et, tx, side, ent, pnl / ent * 100)


def monthly(tr):
    return tr.groupby(pd.to_datetime(tr.close).dt.to_period("M")).pct.sum()


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    d = ts.bars(ts.load_m1("XAUUSD"), 240)
    st_pos = ts.supertrend_dir(d)
    strat = {"SuperTrend 10/3": (st_pos, ts.trades(d, st_pos))}
    for name, cl in (("추세 봇 (구름 없이)", False), ("추세 봇 + 구름 (지금)", True)):
        strat[name] = trend_bot(d, cl)

    print("===== 성적 (같은 조건, 4시간봉 체결) =====")
    print(pd.DataFrame([{"전략": n, **ts.summary(tr)} for n, (_, tr) in strat.items()]).to_string(index=False))

    st = strat["SuperTrend 10/3"][0]
    print("\n===== 4시간봉마다 포지션 겹침 (전체 봉 중 %) =====")
    for name in list(strat)[1:]:
        tb = strat[name][0]
        on = tb != 0
        print(f"\n[SuperTrend vs {name}]")
        print(f"  추세 봇이 포지션 들고 있는 시간     {on.mean()*100:5.1f}%   (SuperTrend는 항상 100%)")
        print(f"  둘 다 같은 방향                     {((tb == st) & on).mean()*100:5.1f}%"
              f"   (매수 {((tb == 1) & (st == 1)).mean()*100:.1f}% / 매도 {((tb == -1) & (st == -1)).mean()*100:.1f}%)")
        print(f"  서로 반대 방향                       {((tb == -st) & on).mean()*100:5.1f}%")
        print(f"  추세 봇은 쉬고 SuperTrend만 보유     {(~on).mean()*100:5.1f}%")
        print(f"  추세 봇이 들고 있을 때 같은 방향 비율 {((tb == st) & on).sum() / on.sum()*100:5.1f}%")
        m = pd.concat([monthly(strat['SuperTrend 10/3'][1]), monthly(strat[name][1])], axis=1).fillna(0)
        print(f"  월 손익 상관계수                     {m.corr().iat[0, 1]:5.2f}")
        both = m.iloc[:, 0] + m.iloc[:, 1]
        print(f"  같은 달 둘 다 손실                   {((m.iloc[:, 0] < 0) & (m.iloc[:, 1] < 0)).sum()}달 / "
              f"SuperTrend 손실 달 {(m.iloc[:, 0] < 0).sum()} / 추세 봇 손실 달 {(m.iloc[:, 1] < 0).sum()} (전체 {len(m)}달)")
        eq = both.cumsum()
        print(f"  둘을 같은 크기로 합치면: 합계 {both.sum():.1f}%, 월 기준 최대낙폭 {(eq.cummax() - eq).max():.1f}%"
              f" (SuperTrend만 {dd(m.iloc[:, 0]):.1f}%, 추세 봇만 {dd(m.iloc[:, 1]):.1f}%)")


def dd(x):
    eq = x.cumsum()
    return (eq.cummax() - eq).max()


if __name__ == "__main__":
    main()
