"""전략 카테고리별 아이디어 중 아직 안 해본 것 스크리닝 (2026-10-29 사용자가 다른 곳에서 받은 목록, 실제 틱 체결)

검증 방식은 strategies.py와 같음: 전략마다 설정 후보 8개 중 2022~2023년 PF가 가장 좋은 것 하나를 고르고
(학습 거래 30건 이상), 그 설정 그대로 2024 / 2025 / 2026 PF를 확인. 통과 기준: 모든 연도 PF 1.1 이상.
진입은 신호 봉이 끝난 뒤 다음 봉 첫 틱 (매수 Ask / 매도 Bid), 손절/익절은 처음 닿는 틱. 전략마다 포지션 1개.
시간은 서버시간 (= 한국시간 - 6시간)

이미 해본 것 (여기서 안 함): 런던 돌파·뉴욕 오픈 15/30분 돌파 (strategies.py), 아시아 박스 돌파 (asia_breakout, 운영 중),
  돈치안 추세 (trend_trader), RSI(2) 되돌림·스퀴즈 (strategies / scalp_ideas), SuperTrend를 SAR 대신 (indicator_test),
  뉴스 발표 뒤 추종·되돌림·양방향 (news_*), 미국장 시간대 (sar_us_hours), ADX 장세 필터 (regime_*)
여기서 하는 것:
 1. 볼린저+RSI 되돌림 : 종가 < 볼린저(20,2) 아래 & RSI14 < X -> 매수 (매도는 반대), 종가가 SMA20 넘으면 청산, 최대 30봉
 2. z점수 되돌림      : z = (종가 - 50봉 평균) / 표준편차, z < -k 매수 / > k 매도, z가 0을 넘으면 청산, 최대 48봉
 3. RSI 다이버전스    : 피벗 저점이 직전 피벗 저점보다 낮은데 RSI14는 높고 RSI < 40 -> 피벗 확정 봉에 매수 (고점은 반대)
                        손절 = 피벗 저점 - 0.2 ATR, 익절 R배, 최대 48봉
 4. ATR 돌파          : 그날 시가 ± k x 전날까지 일봉 ATR(14)을 5분봉 종가로 처음 넘으면 그 방향 (하루 1번), 22:00 청산
                        손절 = 그날 시가 또는 k/2 지점, 익절 2R 또는 없음
 5. EMA 교차 + ADX    : EMA 빠른/느린 교차, ADX(14) > 25일 때만(선택), 반대 교차에 청산, 손절 2.5 ATR
 6. MACD + EMA 정렬   : MACD(12,26,9) 히스토그램 부호가 바뀐 봉에 EMA 5>15>50(매도는 반대) 정렬이면 진입,
                        청산 = 히스토그램 반대 전환 또는 정렬 깨짐, 손절 k ATR
 7. SuperTrend 추종   : SuperTrend 전환마다 그 방향, 반대 전환에 청산 (안전 손절 3 ATR)
 8. 금/은 비율        : GSR = 금 / 은, z(N봉) > k 면 금 매도 (금이 비쌈), < -k 면 금 매수, z가 0을 넘으면 청산, 손절 3 ATR
 9. 달러 역상관       : 달러 인덱스 근사 (-0.576 ln EURUSD + 0.136 ln USDJPY - 0.119 ln GBPUSD)의 EMA 교차 반대로 금 매매
                        (달러 하락 교차 -> 금 매수), 반대 교차에 청산, 손절 k ATR
10. 시간대 통계       : 2022~2023 1시간봉에서 시간(서버)별 평균 수익이 가장 큰 N개 시간은 매수, 가장 작은 N개는 매도,
                        그 시간 시가에 진입 hold시간 뒤 청산, 손절 2 ATR (시간 고르기는 학습 기간에서만)
11. 런던-뉴욕 겹침    : 15:00(한국 21:00)에 10:00~15:00 움직임 방향으로 따라가기 / 거꾸로, 18:00(한국 24:00) 청산,
                        움직임이 일봉 ATR x m 이상일 때만, 손절 k x 1시간 ATR
은·환율은 gpt_data/<종목>/M5.csv.gz (Bid, 서버시간)를 같은 봉으로 묶어 금 봉 시각에 맞춤 (그 봉까지 완성된 값만)
실계좌 조건 (금리 TIPS)은 데이터가 없어서 안 함

사용법: python -m backtests.idea_screen [번호 ...]   -> results/idea_screen_all[_번호].csv, idea_screen_picked[_번호].csv
        python -m backtests.idea_screen sum            -> 번호별 결과를 합쳐 results/idea_screen_picked.csv
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.indicator_test import supertrend
from backtests.scalp_ideas import m5_bars
from backtests.strategies import TickStore, bar_exit_specs, evaluate, execute, resample, to_ms, YEARS, PASS_PF
from indicators import adx, atr, ema, rsi

RES = Path(__file__).resolve().parent.parent / "results"
GPT = Path(__file__).resolve().parents[2] / "gpt_data"
TRAIN_END = 2024


class LightStore(TickStore):
    """TickStore와 같지만 필요한 달만 (최소 2달로 묶지 않음), 캐시 1달 -> 메모리 8GB PC에서 봇과 같이 돌리려고"""

    def __init__(self):
        super().__init__(keep=1)

    def window(self, start_ms, end_ms):
        i0, i1 = self.index_of(start_ms), min(self.index_of(end_ms), len(self.months) - 1)
        if self.last[0] is not None and self.last[0][0] <= i0 and i1 <= self.last[0][1]:
            return self.last[1]
        self.last = (None, None)
        if i0 == i1:
            arrs = self._month(i0)
        else:
            parts = [lb.load_month(*self.months[i]) for i in range(i0, i1 + 1)]
            arrs = tuple(np.concatenate([p[j] for p in parts]) for j in range(3))
            del parts
        self.last = ((i0, i1), arrs)
        return arrs


def next_true(arr_idx):
    def f(i, side):
        arr = arr_idx[side]
        p = np.searchsorted(arr, i + 1)
        return arr[p] if p < len(arr) else None
    return f


# ---------------------------------------------------------------- 1. 볼린저 + RSI
def bb_rsi(b, x, k):
    c = b.close
    m, sd = c.rolling(20).mean(), c.rolling(20).std()
    r = rsi(c).values
    longs = np.flatnonzero((c < m - 2 * sd).values & (r < x))
    shorts = np.flatnonzero((c > m + 2 * sd).values & (r > 100 - x))
    ex = {1: np.flatnonzero((c > m).values), -1: np.flatnonzero((c < m).values)}
    ent = sorted([(i, 1) for i in longs] + [(i, -1) for i in shorts])
    return bar_exit_specs(b, ent, next_true(ex), (k * atr(b)).values, max_hold=30)


# ---------------------------------------------------------------- 2. z점수
def zscore(b, k, s):
    c = b.close
    z = ((c - c.rolling(50).mean()) / c.rolling(50).std()).values
    ent = sorted([(i, 1) for i in np.flatnonzero(z < -k)] + [(i, -1) for i in np.flatnonzero(z > k)])
    ex = {1: np.flatnonzero(z > 0), -1: np.flatnonzero(z < 0)}
    return bar_exit_specs(b, ent, next_true(ex), (s * atr(b)).values, max_hold=48)


# ---------------------------------------------------------------- 3. RSI 다이버전스
def pivots(x, p, low):
    s = pd.Series(x)
    w = s.rolling(2 * p + 1, center=True)
    ext = (s == (w.min() if low else w.max())).values
    return np.flatnonzero(ext)


def divergence(b, p, tp_r):
    r = rsi(b.close).values
    a = atr(b).values
    t = to_ms(b.time)
    specs = []
    for low, side in ((True, 1), (False, -1)):
        px = b.low.values if low else b.high.values
        pv = pivots(px, p, low)
        for j0, j in zip(pv[:-1], pv[1:]):
            if j - j0 > 50 or j - j0 < p:
                continue
            i = j + p                                  # 피벗이 확정되는 봉
            if i + 49 >= len(b):
                continue
            ok = (px[j] < px[j0] and r[j] > r[j0] and r[j] < 40) if side == 1 else \
                 (px[j] > px[j0] and r[j] < r[j0] and r[j] > 60)
            if ok:
                sl = px[j] - side * 0.2 * a[i]
                specs.append({"entry": t[i + 1], "side": side, "exit": t[i + 49], "sl_abs": sl, "tp_r": tp_r})
    specs.sort(key=lambda s: s["entry"])
    return specs


# ---------------------------------------------------------------- 4. ATR 돌파
def daily(m5):
    d = m5.set_index("time").resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    d["atr_prev"] = atr(d).shift(1)
    return d


def atr_breakout(m5, k, sl_mode, tp_r):
    d = daily(m5)
    t = to_ms(m5.time)
    day = m5.time.dt.normalize()
    hm = m5.time.dt.hour * 60 + m5.time.dt.minute
    specs = []
    for dt, g in m5[(hm >= 60) & (hm < 21 * 60)].groupby(day[(hm >= 60) & (hm < 21 * 60)]):
        if dt not in d.index or not d.at[dt, "atr_prev"] > 0:
            continue
        o, a = d.at[dt, "open"], d.at[dt, "atr_prev"]
        up, dn = o + k * a, o - k * a
        hit = g[(g.close > up) | (g.close < dn)]
        if hit.empty:
            continue
        i = hit.index[0]
        if i + 1 >= len(m5):
            continue
        side = 1 if m5.close.iat[i] > up else -1
        sl = o if sl_mode == "시가" else o + side * k * a / 2
        exit_ms = to_ms(pd.Timestamp(dt) + pd.Timedelta(hours=22))
        specs.append({"entry": t[i + 1], "side": side, "exit": int(exit_ms), "sl_abs": sl, "tp_r": tp_r})
    return specs


# ---------------------------------------------------------------- 5. EMA 교차 + ADX
def ema_cross(b, f, s, use_adx):
    d = np.sign(ema(b.close, f) - ema(b.close, s)).values
    cr = np.flatnonzero(d[1:] != d[:-1]) + 1
    ad = adx(b).values
    ent = [(i, int(d[i])) for i in cr if d[i] != 0 and (not use_adx or ad[i] > 25)]
    ex = {1: np.flatnonzero(d < 0), -1: np.flatnonzero(d > 0)}
    return bar_exit_specs(b, ent, next_true(ex), (2.5 * atr(b)).values)


# ---------------------------------------------------------------- 6. MACD + EMA 정렬
def macd_align(b, k, exit_mode):
    c = b.close
    m = ema(c, 12) - ema(c, 26)
    h = (m - ema(m, 9)).values
    e5, e15, e50 = ema(c, 5).values, ema(c, 15).values, ema(c, 50).values
    al = np.where((e5 > e15) & (e15 > e50), 1, np.where((e5 < e15) & (e15 < e50), -1, 0))
    hs = np.sign(h)
    flip = np.flatnonzero(hs[1:] != hs[:-1]) + 1
    ent = [(i, int(hs[i])) for i in flip if hs[i] != 0 and al[i] == hs[i]]
    if exit_mode == "히스토그램":
        ex = {1: np.flatnonzero(hs < 0), -1: np.flatnonzero(hs > 0)}
    else:
        ex = {1: np.flatnonzero(al != 1), -1: np.flatnonzero(al != -1)}
    return bar_exit_specs(b, ent, next_true(ex), (k * atr(b)).values)


# ---------------------------------------------------------------- 7. SuperTrend
def st_follow(b, n, mult):
    d = supertrend(b, n, mult)
    cr = np.flatnonzero(d[1:] != d[:-1]) + 1
    ent = [(i, int(d[i])) for i in cr]
    ex = {1: np.flatnonzero(d < 0), -1: np.flatnonzero(d > 0)}
    return bar_exit_specs(b, ent, next_true(ex), (3 * atr(b)).values)


# ---------------------------------------------------------------- 8, 9. 다른 종목
_other = {}


def other_close(sym, b, rule):
    """다른 종목 종가를 금 봉 시각에 맞춤 (같은 봉 규칙으로 묶어서, 금 봉과 같은 시각의 완성 봉)"""
    key = (sym, rule)
    if key not in _other:
        m5 = pd.read_csv(GPT / sym / "M5.csv.gz", parse_dates=["time"])[["time", "open", "high", "low", "close"]]
        _other[key] = resample(m5, rule).set_index("time").close
    return _other[key].reindex(b.time).ffill().values


def gsr(b, rule, n, k):
    ratio = pd.Series(b.close.values / other_close("XAGUSD", b, rule))
    z = ((ratio - ratio.rolling(n).mean()) / ratio.rolling(n).std()).values
    ent = sorted([(i, -1) for i in np.flatnonzero(z > k)] + [(i, 1) for i in np.flatnonzero(z < -k)])
    ex = {1: np.flatnonzero(z > 0), -1: np.flatnonzero(z < 0)}
    return bar_exit_specs(b, ent, next_true(ex), (3 * atr(b)).values)


def dxy(b, rule, f, s, k):
    lx = (-0.576 * np.log(other_close("EURUSD", b, rule)) + 0.136 * np.log(other_close("USDJPY", b, rule))
          - 0.119 * np.log(other_close("GBPUSD", b, rule)))
    x = pd.Series(lx)
    d = -np.sign(ema(x, f) - ema(x, s)).values                     # 달러 반대 방향 = 금 방향
    d = np.nan_to_num(d)
    cr = np.flatnonzero(d[1:] != d[:-1]) + 1
    ent = [(i, int(d[i])) for i in cr if d[i] != 0]
    ex = {1: np.flatnonzero(d < 0), -1: np.flatnonzero(d > 0)}
    return bar_exit_specs(b, ent, next_true(ex), (k * atr(b)).values)


# ---------------------------------------------------------------- 10. 시간대 통계
def hour_stats(h1, nh, hold, long_only=False):
    tr = h1[h1.time.dt.year < TRAIN_END]
    ret = (tr.close - tr.open).groupby(tr.time.dt.hour).mean()
    longs = list(ret.sort_values(ascending=False).index[:nh])
    shorts = [] if long_only else list(ret.sort_values().index[:nh])
    hr = h1.time.dt.hour.values
    a = (2 * atr(h1)).values
    t = to_ms(h1.time)
    specs = []
    for i in range(1, len(h1) - hold):
        side = 1 if hr[i] in longs else -1 if hr[i] in shorts else 0
        if side and (t[i + hold] - t[i]) <= hold * 3_600_000 and a[i - 1] > 0:   # 주말·장 쉬는 시간을 넘으면 뺌
            specs.append({"entry": t[i], "side": side, "exit": t[i + hold], "sl_dist": a[i - 1]})
    return specs


# ---------------------------------------------------------------- 11. 런던-뉴욕 겹침
def overlap(m5, h1, mode, m, k):
    d = daily(m5)
    a1 = atr(h1).values
    h1t = h1.time.values
    specs = []
    by_t = m5.set_index("time")
    for dt in d.index:
        t10, t15, t18 = dt + pd.Timedelta(hours=10), dt + pd.Timedelta(hours=15), dt + pd.Timedelta(hours=18)
        if t10 not in by_t.index or t15 not in by_t.index or not d.at[dt, "atr_prev"] > 0:
            continue
        mv = by_t.at[t15, "open"] - by_t.at[t10, "open"]
        if abs(mv) < m * d.at[dt, "atr_prev"] or mv == 0:
            continue
        side = int(np.sign(mv)) * (1 if mode == "따라가기" else -1)
        p = np.searchsorted(h1t, np.datetime64(t15)) - 1               # 15:00 전 마지막 완성 1시간봉
        if p < 0 or not a1[p] > 0:
            continue
        specs.append({"entry": int(to_ms(t15)), "side": side, "exit": int(to_ms(t18)), "sl_dist": k * a1[p]})
    return specs


def summary():
    pk = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("idea_screen_picked_*.csv"))], ignore_index=True)
    al = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("idea_screen_all_*.csv"))], ignore_index=True)
    key = lambda s: s.str.split(".").str[0].astype(int)
    pk, al = pk.sort_values("전략", key=key), al.sort_values("전략", key=key, kind="stable")
    pk.to_csv(RES / "idea_screen_picked.csv", index=False, encoding="utf-8-sig")
    al.to_csv(RES / "idea_screen_all.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.max_rows", 200)
    print(al.to_string(index=False))
    print(pk.to_string(index=False))


def main():
    if sys.argv[1:] == ["sum"]:
        return summary()
    pick = set(sys.argv[1:])
    print("봉 만드는 중...", flush=True)
    m5 = m5_bars()
    bars = {"M5": m5, "M15": resample(m5, "15min"), "H1": resample(m5, "1h"), "H4": resample(m5, "4h")}
    rule = {"M5": "5min", "M15": "15min", "H1": "1h", "H4": "4h"}
    store = LightStore()
    P = itertools.product
    grids = {
        "1. 볼린저+RSI": [(f"{tf} RSI<{x} 손절 {k}ATR", lambda tf=tf, x=x, k=k: bb_rsi(bars[tf], x, k))
                         for tf, x, k in P(["M15", "H1"], [30, 25], [2.0, 3.0])],
        "2. z점수": [(f"{tf} z{k} 손절 {s}ATR", lambda tf=tf, k=k, s=s: zscore(bars[tf], k, s))
                    for tf, k, s in P(["M5", "M15"], [2.5, 3.0], [1.5, 2.5])],
        "3. RSI 다이버전스": [(f"{tf} 피벗{p} {r}R", lambda tf=tf, p=p, r=r: divergence(bars[tf], p, r))
                             for tf, p, r in P(["M15", "H1"], [3, 5], [1.5, 2.5])],
        "4. ATR 돌파": [(f"k{k} 손절 {s} TP {r or '없음'}", lambda k=k, s=s, r=r: atr_breakout(m5, k, s, r))
                       for k, s, r in P([0.3, 0.6], ["시가", "절반"], [2.0, None])],
        "5. EMA교차+ADX": [(f"{tf} {f}/{s}{' ADX25' if a else ''}", lambda tf=tf, f=f, s=s, a=a: ema_cross(bars[tf], f, s, a))
                          for tf, (f, s), a in P(["M15", "H1"], [(5, 15), (9, 21)], [False, True])],
        "6. MACD+EMA정렬": [(f"{tf} 손절 {k}ATR 청산 {e}", lambda tf=tf, k=k, e=e: macd_align(bars[tf], k, e))
                           for tf, k, e in P(["M15", "H1"], [2.0, 3.0], ["히스토그램", "정렬"])],
        "7. SuperTrend": [(f"{tf} ({n},{mu})", lambda tf=tf, n=n, mu=mu: st_follow(bars[tf], n, mu))
                         for tf, (n, mu) in [("M15", (10, 3)), ("M15", (10, 2)), ("H1", (10, 3)), ("H1", (10, 2)),
                                             ("H1", (14, 3)), ("H4", (10, 3)), ("H4", (10, 2)), ("H4", (14, 3))]],
        "8. 금/은 비율": [(f"{tf} N{n} z{k}", lambda tf=tf, n=n, k=k: gsr(bars[tf], rule[tf], n, k))
                       for tf, n, k in P(["H1", "H4"], [50, 100], [2.0, 2.5])],
        "9. 달러 역상관": [(f"{tf} EMA{f}/{s} 손절 {k}ATR", lambda tf=tf, f=f, s=s, k=k: dxy(bars[tf], rule[tf], f, s, k))
                        for tf, (f, s), k in P(["H1", "H4"], [(9, 21), (20, 50)], [2.0, 3.0])],
        "10. 시간대 통계": [(f"상위 {n}시간 {h}시간 보유{' 매수만' if lo else ''}",
                          lambda n=n, h=h, lo=lo: hour_stats(bars["H1"], n, h, lo))
                         for n, h, lo in [(1, 1, False), (2, 1, False), (3, 1, False), (1, 2, False), (2, 2, False),
                                          (1, 1, True), (2, 1, True), (3, 1, True)]],
        "11. 런던-뉴욕 겹침": [(f"{mo} 움직임≥{mm}ATR 손절 {k}ATR", lambda mo=mo, mm=mm, k=k: overlap(m5, bars["H1"], mo, mm, k))
                            for mo, mm, k in P(["따라가기", "거꾸로"], [0.0, 0.5], [1.0, 2.0])],
    }
    all_rows, picks = [], []
    for fam, grid in grids.items():
        if pick and fam.split(".")[0] not in pick:
            continue
        best = None
        for name, make in grid:
            tr = execute(make(), store)
            if tr.empty:
                continue
            row = {"전략": fam, "설정": name, **evaluate(tr), "oz합": round(tr.pnl_oz.sum(), 1)}
            n_train = int((tr.close.dt.year < TRAIN_END).sum())
            all_rows.append(row)
            if n_train >= 30 and row["PF 22-23"] != float("inf") and (best is None or row["PF 22-23"] > best["PF 22-23"]):
                best = row
            print(f"  {fam} {name}: {len(tr)}건 PF22-23 {row['PF 22-23']} PF24-26 {row['PF 24-26']}", flush=True)
        if best:
            picks.append(best)
    res = pd.DataFrame(all_rows)
    pk = pd.DataFrame(picks)
    pk["통과"] = pk[[f"PF {y}" for y in YEARS]].min(axis=1).ge(PASS_PF).map({True: "통과", False: "-"})
    suffix = "" if not pick else "_" + "_".join(sorted(pick))
    res.to_csv(RES / f"idea_screen_all{suffix}.csv", index=False, encoding="utf-8-sig")
    pk.to_csv(RES / f"idea_screen_picked{suffix}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.max_rows", 200)
    print("\n=== 전체 설정 ===")
    print(res.to_string(index=False))
    print("\n=== 2022~2023 PF로 고른 설정 -> 2024~2026 검증 ===")
    print(pk.to_string(index=False))


if __name__ == "__main__":
    main()
