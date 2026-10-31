"""새 전략 4종 스크리닝 (2022-01 ~ 현재, 실제 틱 체결)

검증 방식: 전략마다 설정 후보 8개 중 2022~2023년 PF가 가장 좋은 것 하나를 고르고,
그 설정 그대로 2024 / 2025 / 2026 PF를 확인. 통과 기준: 모든 연도 PF 1.1 이상.

시간은 모두 서버시간 (미국 서머타임을 따름: 15:30 미국 지표, 16:30 뉴욕 장 시작, 10:00 런던 장)
진입은 신호 봉이 끝난 뒤 다음 봉 첫 틱 (매수 Ask / 매도 Bid), 손절/익절은 처음 닿는 틱 가격으로 체결.
전략마다 포지션은 한 번에 1개.

1. 런던 돌파   : 01:00~10:00 고가/저가 범위를 10:00~14:00 사이 5분봉 종가로 돌파하면 진입 (하루 1번)
                 손절 = 범위 반대편 또는 범위의 절반, 익절 = 위험의 R배 또는 없음, 22:00 시간 청산
2. 뉴욕 오픈   : 16:30부터 15/30분 범위를 19:30 전에 5분봉 종가로 돌파하면 진입 (하루 1번)
                 손절 = 범위 반대편, 익절 = 위험의 R배 또는 없음, 23:00 시간 청산
3. 추세추종    : 1시간봉/4시간봉 종가가 직전 N봉 최고가 위(최저가 아래)면 진입,
                 종가가 직전 N/2봉 최저가 아래(최고가 위)면 청산, 손절 = 진입가 ± k x ATR(14)
4. 평균회귀    : 큰 추세(같은 봉 EMA200) 방향으로만, RSI(2)가 극단(매수 < X, 매도 > 100-X)이면 진입,
                 종가가 SMA5를 넘어서면(매도는 아래로) 청산, 손절 = k x ATR(14), 최대 보유 20봉

사용법: python -m backtests.strategies   (long_period를 먼저 한 번 실행해 틱이 저장돼 있어야 함)
"""
import itertools
from collections import OrderedDict

import numpy as np
import pandas as pd

import backtests.long_period as lb
from indicators import atr, ema, rsi

TRAIN_END = 2024
YEARS = [2022, 2023, 2024, 2025, 2026]
PASS_PF = 1.1


# ---------------------------------------------------------------- 데이터
def resample(m5, rule):
    b = m5.set_index("time").resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return b.reset_index()


def to_ms(ts):
    return np.asarray(ts, dtype="datetime64[ms]").astype(np.int64)


class TickStore:
    """월별 틱 파일을 필요할 때만 불러오고 최근 몇 달만 메모리에 유지"""

    def __init__(self, keep=4):
        self.months = list(lb.months())
        self.cache, self.keep = OrderedDict(), keep
        self.last = (None, None)      # 마지막으로 합친 (월 범위, 배열) - 연속된 거래는 같은 범위를 재사용

    def _month(self, i):
        if i not in self.cache:
            self.cache[i] = lb.load_month(*self.months[i])
            while len(self.cache) > self.keep:
                self.cache.popitem(last=False)
        self.cache.move_to_end(i)
        return self.cache[i]

    def index_of(self, ms):
        ts = pd.Timestamp(ms, unit="ms")
        return self.months.index((ts.year, ts.month))

    def window(self, start_ms, end_ms):
        i0, i1 = self.index_of(start_ms), self.index_of(end_ms)
        i1 = min(max(i1, i0 + 1), len(self.months) - 1)   # 최소 두 달을 묶어서 재사용 기회를 늘림
        if self.last[0] is not None and self.last[0][0] <= i0 and i1 <= self.last[0][1]:
            return self.last[1]
        parts = [self._month(i) for i in range(i0, i1 + 1)]
        arrs = tuple(np.concatenate([p[j] for p in parts]) for j in range(3))
        self.last = ((i0, i1), arrs)
        return arrs


def execute(specs, store):
    """specs: dict(entry, side, exit, sl_abs | sl_dist, tp_r) 목록 (시간은 ms). 반환: 거래 DataFrame"""
    out, busy = [], -1
    for s in specs:
        if s["entry"] < busy or s["exit"] <= s["entry"]:
            continue
        t, bid, ask = store.window(s["entry"], s["exit"])
        k = np.searchsorted(t, s["entry"])
        x = np.searchsorted(t, s["exit"])
        if k >= len(t) - 1 or x >= len(t):
            continue
        side = s["side"]
        e = ask[k] if side == 1 else bid[k]
        sl = s["sl_abs"] if "sl_abs" in s else e - side * s["sl_dist"]
        risk = (e - sl) * side
        if risk <= 0:
            continue
        tp = e + side * s["tp_r"] * risk if s.get("tp_r") else None
        price = bid if side == 1 else ask
        seg = price[k + 1:x]
        if side == 1:
            hit = (seg <= sl) | (seg >= tp) if tp else (seg <= sl)
        else:
            hit = (seg >= sl) | (seg <= tp) if tp else (seg >= sl)
        idx = np.flatnonzero(hit)
        j = k + 1 + idx[0] if idx.size else x
        pnl = (price[j] - e) * side
        out.append((t[k], t[j], side, pnl, pnl / risk, risk))
        busy = s["exit"] if j == x else t[j]      # 시간 청산이면 신호 시각까지만 (같은 시각 반대 진입을 건너뛰지 않게)
    tr = pd.DataFrame(out, columns=["open_ms", "close_ms", "side", "pnl_oz", "r", "risk"])
    tr["close"] = pd.to_datetime(tr.close_ms, unit="ms")
    return tr


# ---------------------------------------------------------------- 전략
def day_sessions(m5, start_hm, end_hm):
    """날짜별로 [start, end) 구간의 5분봉 인덱스"""
    hm = m5.time.dt.hour * 60 + m5.time.dt.minute
    s, e = start_hm[0] * 60 + start_hm[1], end_hm[0] * 60 + end_hm[1]
    sel = m5[(hm >= s) & (hm < e)]
    return {d: g for d, g in sel.groupby(sel.time.dt.date)}


def breakout_specs(m5, rng, trade, close_hm, sl_mode, tp_r):
    t_all = to_ms(m5.time)
    pos = {tm: i for i, tm in enumerate(m5.time)}
    ranges, windows = day_sessions(m5, *rng), day_sessions(m5, *trade)
    specs = []
    for d, w in windows.items():
        r = ranges.get(d)
        if r is None or len(r) < 3:      # 범위 구간 5분봉이 3개(15분) 미만인 날은 제외
            continue
        hi, lo = r.high.max(), r.low.min()
        width = hi - lo
        if width <= 0:
            continue
        exit_ms = to_ms(pd.Timestamp(d) + pd.Timedelta(hours=close_hm[0], minutes=close_hm[1]))
        for row in w.itertuples():
            side = 1 if row.close > hi else -1 if row.close < lo else 0
            if side:
                i = pos[row.time]
                if i + 1 >= len(m5):
                    break
                # 반대편: 범위 반대쪽 끝 / 범위절반: 범위 중간 (손절폭이 절반)
                sl = (lo if side == 1 else hi) if sl_mode == "반대편" else (hi + lo) / 2
                specs.append({"entry": t_all[i + 1], "side": side, "exit": int(exit_ms), "sl_abs": sl, "tp_r": tp_r})
                break
    return specs


def london(m5, sl_mode, tp_r):
    return breakout_specs(m5, ((1, 0), (10, 0)), ((10, 0), (14, 0)), (22, 0), sl_mode, tp_r)


def ny_open(m5, minutes, tp_r):
    end = (16, 30 + minutes) if 30 + minutes < 60 else (17, 30 + minutes - 60)
    return breakout_specs(m5, ((16, 30), end), (end, (19, 30)), (23, 0), "반대편", tp_r)


def bar_exit_specs(bars, entries, exit_fn, sl_dist, max_hold=None):
    """entries: (bar 인덱스, 방향). exit_fn(i, side) -> 청산 신호가 나온 봉 인덱스. 진입/청산은 다음 봉 시가"""
    t = to_ms(bars.time)
    specs = []
    for i, side in entries:
        j = exit_fn(i, side)
        if max_hold is not None:
            j = min(j, i + max_hold) if j is not None else i + max_hold
        if j is None or j + 1 >= len(bars) or i + 1 >= len(bars):
            continue
        specs.append({"entry": t[i + 1], "side": side, "exit": t[j + 1], "sl_dist": sl_dist[i]})
    return specs


def trend(bars, n, k):
    hh = bars.high.rolling(n).max().shift(1)
    ll = bars.low.rolling(n).min().shift(1)
    xh = bars.high.rolling(n // 2).max().shift(1)
    xl = bars.low.rolling(n // 2).min().shift(1)
    c = bars.close.values
    up, dn = (c > hh.values), (c < ll.values)
    ex_long = np.flatnonzero(c < xl.values)
    ex_short = np.flatnonzero(c > xh.values)

    def exit_fn(i, side):
        arr = ex_long if side == 1 else ex_short
        p = np.searchsorted(arr, i + 1)
        return arr[p] if p < len(arr) else None

    entries = [(i, 1) for i in np.flatnonzero(up)] + [(i, -1) for i in np.flatnonzero(dn)]
    entries.sort()
    return bar_exit_specs(bars, entries, exit_fn, (k * atr(bars)).values)


def mean_rev(bars, x, k):
    r2 = rsi(bars.close, 2).values
    e200 = ema(bars.close, 200).values
    sma5 = bars.close.rolling(5).mean().values
    c = bars.close.values
    longs = np.flatnonzero((c > e200) & (r2 < x))
    shorts = np.flatnonzero((c < e200) & (r2 > 100 - x))
    ex_long = np.flatnonzero(c > sma5)
    ex_short = np.flatnonzero(c < sma5)

    def exit_fn(i, side):
        arr = ex_long if side == 1 else ex_short
        p = np.searchsorted(arr, i + 1)
        return arr[p] if p < len(arr) else None

    entries = sorted([(i, 1) for i in longs] + [(i, -1) for i in shorts])
    return bar_exit_specs(bars, entries, exit_fn, (k * atr(bars)).values, max_hold=20)


# ---------------------------------------------------------------- 평가
def pf(x):
    gl = -x[x < 0].sum()
    return round(x[x > 0].sum() / gl, 2) if gl else float("inf")


def evaluate(tr):
    y = tr.close.dt.year
    row = {"거래수": len(tr), "승률%": round((tr.pnl_oz > 0).mean() * 100, 1) if len(tr) else 0,
           "평균R": round(tr.r.mean(), 3) if len(tr) else 0,
           "PF 22-23": pf(tr.pnl_oz[y < TRAIN_END]), "PF 24-26": pf(tr.pnl_oz[y >= TRAIN_END])}
    for yr in YEARS:
        row[f"PF {yr}"] = pf(tr.pnl_oz[y == yr])
    row["연평균 거래"] = round(len(tr) / ((tr.close.iat[-1] - tr.close.iat[0]).days / 365), 0) if len(tr) > 1 else 0
    return row


def main():
    print("봉 만드는 중...", flush=True)
    m5 = lb.bars_from_ticks(5)
    bars = {"M15": resample(m5, "15min"), "H1": resample(m5, "1h"), "H4": resample(m5, "4h")}
    store = TickStore()

    grids = {
        "1. 런던 돌파": [(f"손절 {s}, TP {r or '없음'}", lambda s=s, r=r: london(m5, s, r))
                      for s, r in itertools.product(["반대편", "범위절반"], [1.0, 1.5, 2.0, None])],
        "2. 뉴욕 오픈": [(f"범위 {mi}분, TP {r or '없음'}", lambda mi=mi, r=r: ny_open(m5, mi, r))
                      for mi, r in itertools.product([15, 30], [1.0, 1.5, 2.0, None])],
        "3. 추세추종": [(f"{tf} N{n} 손절 {k}ATR", lambda tf=tf, n=n, k=k: trend(bars[tf], n, k))
                     for tf, n, k in itertools.product(["H1", "H4"], [20, 55], [2.0, 3.0])],
        "4. 평균회귀": [(f"{tf} RSI2<{x} 손절 {k}ATR", lambda tf=tf, x=x, k=k: mean_rev(bars[tf], x, k))
                     for tf, x, k in itertools.product(["M15", "H1"], [5, 10], [2.0, 3.0])],
    }

    all_rows, picks = [], []
    for fam, grid in grids.items():
        best = None
        for name, make in grid:
            tr = execute(make(), store)
            row = {"전략": fam, "설정": name, **evaluate(tr)}
            all_rows.append(row)
            if row["PF 22-23"] != float("inf") and (best is None or row["PF 22-23"] > best["PF 22-23"]):
                best = row
        picks.append(best)
        print(f"완료: {fam}", flush=True)

    res = pd.DataFrame(all_rows)
    res.to_csv("strategies_all.csv", index=False, encoding="utf-8-sig")
    pk = pd.DataFrame(picks)
    pk["통과"] = pk[[f"PF {y}" for y in YEARS]].min(axis=1).ge(PASS_PF).map({True: "통과", False: "-"})
    pk.to_csv("strategies_picked.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n=== 전체 설정 ===")
    print(res.to_string(index=False))
    print("\n=== 2022~2023 PF로 고른 설정 -> 2024~2026 검증 ===")
    print(pk.to_string(index=False))


if __name__ == "__main__":
    main()
