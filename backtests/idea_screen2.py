"""다른 AI가 준 새 전략 후보 스크리닝 2 (2026-10-31, 사용자가 받은 xauusd_new_strategy_candidates.md, 실제 틱 체결)

검증 방식은 idea_screen.py와 같음: 설정 8개 중 2022~23 PF 1등 (학습 30건 이상) -> 같은 설정 2024/25/26 모두 PF 1.1 이상이면 통과.
추가로 거래당 0.2$ 비용을 더 뺀 2024~26 PF도 표시. 시간은 서버시간 (뉴욕 + 7시간)

 3. 뉴스 후 2차 돌파 : 미국 CPI·고용(비농장 급여)·FOMC 금리 결정. 발표 뒤 R분 고가/저가 = 뉴스 범위,
                       발표 30~120분 사이 5분봉 종가가 범위를 처음 넘으면 그 방향 (발표마다 1번)
                       손절 = 범위 반대편 또는 1.5 x 1시간봉 ATR, 익절 2R 또는 없음, 발표 4시간 뒤 시간 청산
 4. 변동성 압축 돌파 : 직전 봉이 압축 (ATR20/ATR100 <= 0.8, 또는 볼린저 폭이 최근 100봉 중 하위 20%)일 때
                       종가가 직전 20봉 고가/저가를 넘으면 그 방향. 손절 2 ATR, 청산 = 3R 익절(최대 60봉) 또는 10봉 반대 돌파
 5. 뉴욕 VWAP 과도 이탈 회귀 : 뉴욕장(서버 16:30~23:00) VWAP(세션 OHLC4 평균, CFD라 거래량 없음),
                       거리 = (종가 - VWAP) / ATR14. 최근 3봉 중 거리 >= k 이고 이번 봉 종가가 직전 봉보다 낮으면 매도 (매수는 반대)
                       손절 = 최근 3봉 극값 + 0.5 ATR, 익절 = 종가가 VWAP 닿으면 청산 또는 1.5R, 23:00 시간 청산
                       미국 중요 지표 앞뒤 30분은 진입 안 함

사용법: python -m backtests.idea_screen2 [3 4 5]
"""
import itertools
import sys

import numpy as np
import pandas as pd

from backtests.gpt_news_export import calendar
from backtests.idea_screen import LightStore, TRAIN_END, next_true
from backtests.scalp_ideas import m5_bars
from backtests.strategies import PASS_PF, YEARS, bar_exit_specs, evaluate, execute, pf, resample, to_ms
from indicators import atr

NEWS = ("CPI m/m", "비농장 급여", "연방 금리 결정")
NY_OPEN, NY_END = 16 * 60 + 30, 23 * 60


def usd_high_times():
    c = calendar()
    return c[(c.currency == "USD") & (c.importance == "high") & ~c.all_day].time


# ---------------------------------------------------------------- 3. 뉴스 후 2차 돌파
def news_second(m5, h1, rng_min, sl_mode, tp_r):
    c = calendar()
    rel = np.sort(c[(c.currency == "USD") & (c.importance == "high") & c.event.isin(NEWS)].time.unique())
    t = m5.time.values
    tm = to_ms(m5.time)
    h1_atr = atr(h1).values
    h1_t = h1.time.values
    specs = []
    for r in pd.to_datetime(rel):
        a, b = np.searchsorted(t, np.datetime64(r)), np.searchsorted(t, np.datetime64(r + pd.Timedelta(minutes=rng_min)))
        if b - a < rng_min // 5:
            continue
        hi, lo = m5.high.values[a:b].max(), m5.low.values[a:b].min()
        w0 = np.searchsorted(t, np.datetime64(r + pd.Timedelta(minutes=25)))     # 봉 마감 = 발표 30분 뒤부터
        w1 = np.searchsorted(t, np.datetime64(r + pd.Timedelta(minutes=115)))
        w0 = max(w0, b)
        cl = m5.close.values[w0:w1 + 1]
        hit = np.flatnonzero((cl > hi) | (cl < lo))
        if not hit.size or w0 + hit[0] + 1 >= len(m5):
            continue
        i = w0 + hit[0]
        side = 1 if m5.close.values[i] > hi else -1
        if sl_mode == "범위":
            sl = {"sl_abs": lo if side == 1 else hi}
        else:
            k = np.searchsorted(h1_t, t[i], side="right") - 2            # 진입 전에 끝난 1시간봉
            if k < 14:
                continue
            sl = {"sl_dist": 1.5 * h1_atr[k]}
        specs.append({"entry": tm[i + 1], "side": side, "exit": to_ms(pd.Series([r + pd.Timedelta(hours=4)]))[0],
                      "tp_r": tp_r, **sl})
    return specs


# ---------------------------------------------------------------- 4. 변동성 압축 돌파
def compression(b, comp, exit_mode):
    a = atr(b)
    if comp == "ATR":
        squeezed = (atr(b, 20) / atr(b, 100) <= 0.8).values
    else:
        mid, sd = b.close.rolling(20).mean(), b.close.rolling(20).std()
        bbw = 4 * sd / mid
        squeezed = (bbw.rolling(100).rank(pct=True) <= 0.2).values
    hh, ll = b.high.rolling(20).max().shift(1).values, b.low.rolling(20).min().shift(1).values
    c = b.close.values
    prev = np.roll(squeezed, 1)
    prev[0] = False
    ent = sorted([(i, 1) for i in np.flatnonzero(prev & (c > hh))] + [(i, -1) for i in np.flatnonzero(prev & (c < ll))])
    if exit_mode == "3R":
        never = {1: np.array([], int), -1: np.array([], int)}
        specs = bar_exit_specs(b, ent, next_true(never), (2 * a).values, max_hold=60)
        for s in specs:
            s["tp_r"] = 3.0
        return specs
    xh, xl = b.high.rolling(10).max().shift(1).values, b.low.rolling(10).min().shift(1).values
    ex = {1: np.flatnonzero(c < xl), -1: np.flatnonzero(c > xh)}
    return bar_exit_specs(b, ent, next_true(ex), (2 * a).values)


# ---------------------------------------------------------------- 5. 뉴욕 VWAP 과도 이탈 회귀
def vwap_dev(b, minutes, k, tp_mode, news_ms):
    hm = b.time.dt.hour * 60 + b.time.dt.minute
    live = ((hm >= NY_OPEN) & (hm + minutes <= NY_END)).values
    day = b.time.dt.normalize()
    key = pd.Series(np.where(live, day.astype("int64"), -1), index=b.index)
    src = (b.open + b.high + b.low + b.close) / 4
    g = src.where(live)
    vwap = (g.groupby(key).cumsum() / g.groupby(key).cumcount().add(1)).where(live).values
    a = atr(b).values
    c = b.close.values
    dist = (c - vwap) / a
    same = (key == key.shift(1)) & (key == key.shift(2))
    dmax = pd.Series(dist).rolling(3).max().values
    dmin = pd.Series(dist).rolling(3).min().values
    shorts = np.flatnonzero(same.values & (dmax >= k) & (dist > 0) & (c < np.roll(c, 1)))
    longs = np.flatnonzero(same.values & (dmin <= -k) & (dist < 0) & (c > np.roll(c, 1)))
    hi3, lo3 = b.high.rolling(3).max().values, b.low.rolling(3).min().values
    t = to_ms(b.time)
    end_t = to_ms(day + pd.Timedelta(minutes=NY_END))
    cross = {1: np.flatnonzero(c >= vwap), -1: np.flatnonzero(c <= vwap)}
    specs = []
    for i, side in sorted([(i, 1) for i in longs] + [(i, -1) for i in shorts]):
        if i + 1 >= len(b):
            continue
        j = np.searchsorted(news_ms, t[i])
        if (j < len(news_ms) and news_ms[j] - t[i] <= 1_800_000) or (j > 0 and t[i] - news_ms[j - 1] <= 1_800_000):
            continue
        ex = end_t[i]
        if tp_mode == "VWAP":
            p = np.searchsorted(cross[side], i + 1)
            if p < len(cross[side]) and cross[side][p] + 1 < len(b):
                ex = min(ex, t[cross[side][p] + 1])
        sl = lo3[i] - 0.5 * a[i] if side == 1 else hi3[i] + 0.5 * a[i]
        specs.append({"entry": t[i + 1], "side": side, "exit": int(ex), "sl_abs": sl,
                      "tp_r": 1.5 if tp_mode == "1.5R" else None})
    return specs


def main():
    pick = set(sys.argv[1:])
    m5 = m5_bars()
    bars = {"M5": m5, "M15": resample(m5, "15min"), "H1": resample(m5, "1h"), "H4": resample(m5, "4h"), "D1": resample(m5, "1D")}
    news_ms = to_ms(pd.Series(np.sort(usd_high_times().unique())))
    store = LightStore()
    P = itertools.product
    grids = {
        "3. 뉴스 후 2차 돌파": [(f"범위 {r}분 손절 {s} TP {tp or '없음'}",
                              lambda r=r, s=s, tp=tp: news_second(m5, bars["H1"], r, s, tp))
                             for r, s, tp in P([15, 30], ["범위", "1.5ATR"], [2.0, None])],
        "4. 변동성 압축 돌파": [(f"{tf} {cm} 청산 {ex}", lambda tf=tf, cm=cm, ex=ex: compression(bars[tf], cm, ex))
                             for tf, cm, ex in P(["H4", "D1"], ["ATR", "볼린저폭"], ["3R", "10봉 반대"])],
        "5. VWAP 과도 이탈": [(f"{tf} 거리{k} 청산 {tp}", lambda tf=tf, k=k, tp=tp: vwap_dev(bars[tf], int(tf[1:]), k, tp, news_ms))
                            for tf, k, tp in P(["M5", "M15"], [2.0, 3.0], ["VWAP", "1.5R"])],
    }
    rows, picks = [], []
    for fam, grid in grids.items():
        if pick and fam.split(".")[0] not in pick:
            continue
        best = None
        for name, make in grid:
            tr = execute(make(), store)
            if tr.empty:
                print(f"  {fam} {name}: 거래 없음", flush=True)
                continue
            y = tr.close.dt.year
            row = {"전략": fam, "설정": name, **evaluate(tr), "oz합": round(tr.pnl_oz.sum(), 1),
                   "PF 24-26 비용0.2": pf(tr.pnl_oz[y >= TRAIN_END] - 0.2)}
            rows.append(row)
            if (y < TRAIN_END).sum() >= 30 and (best is None or row["PF 22-23"] > best["PF 22-23"]):
                best = row
            print(f"  {fam} {name}: {len(tr)}건 승률 {row['승률%']} PF22-23 {row['PF 22-23']} PF24-26 {row['PF 24-26']}", flush=True)
        if best:
            best["통과"] = all(best[f"PF {yr}"] >= PASS_PF for yr in YEARS if yr >= TRAIN_END)
            picks.append(best)
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n고른 설정")
    print(pd.DataFrame(picks).to_string(index=False))


if __name__ == "__main__":
    main()
