"""+3$에 닿으면 바로 익절하지 않고 이익을 잠근 뒤 더 들고 가는 청산 방식 비교 (2025-01 ~ 현재, 실제 틱, 1,000달러 시작)

1단계 (지금과 같음): 손절 7.5$ / +3$ 중 먼저 닿는 쪽. 손절이면 끝.
2단계 (+3$에 닿은 거래만): 손절선을 진입가 +LOCK$로 올리고, 아래 조건 중 먼저 오는 것에 청산
  trail D : 지금까지 최고 이익에서 D$ 되돌리면 청산 (손절선은 LOCK 아래로 안 내려감)
  sar     : 5분봉 SAR이 반대로 뒤집힌 봉이 끝나면 청산
  rsi     : 5분봉 RSI가 50 반대편으로 넘어간 봉이 끝나면 청산
  dmi     : 5분봉 DMI가 반대 우세(-DI > +DI, 매도는 반대)인 봉이 끝나면 청산
  금요일 23:30 이후 첫 틱에는 무조건 청산
lot = 그날 시작 잔고 1,000달러당 0.05 (마틴 없음, 지금 봇과 같음). 쉬기 규칙/승률 쉬기도 같음.

사용법: python -m backtests.trail_exit [시작 YYYY-MM-DD] [끝 YYYY-MM-DD]   (구간마다 1,000달러로 새로 시작)
"""
import sys

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.add_positions import ATR_REF_BARS, ATR_REF_MIN, CONTRACT, MAX_ATR_MULT, MAX_LOT, SL, TP, floor_lot
from backtests.add_positions_ticks import hit, load_m1_recent
from backtests.gold_rest_others import (FRIDAY_NO_ENTRY, MAX_H1_FLIPS, MIN_ATR, NO_ENTRY_FROM, NO_ENTRY_UNTIL,
                                        WINRATE_MIN, WINRATE_N, h1_flips_by_time, signal_table)
from indicators import atr, psar, rsi

START, END = pd.Timestamp("2025-01-01"), pd.Timestamp("2100-01-01")
BAL0, LOT_PER_1000 = 1000.0, 0.05
INF = np.inf
# 이름: (잠금 $, 트레일 $ (0=안 씀), 봉 조건 (None/sar/rsi/dmi))
CASES = {
    "기준: +3$ 바로 익절": None,
    "잠금 2, 트레일 1": (2.0, 1.0, None),
    "잠금 2, 트레일 2": (2.0, 2.0, None),
    "잠금 2, 트레일 3": (2.0, 3.0, None),
    "잠금 2, SAR 반대": (2.0, 0, "sar"),
    "잠금 2, RSI 반대": (2.0, 0, "rsi"),
    "잠금 2, DMI 반대": (2.0, 0, "dmi"),
    "잠금 2, SAR 반대 + 트레일 3": (2.0, 3.0, "sar"),
    "잠금 2.5, 트레일 1.5": (2.5, 1.5, None),
    "잠금 1, 트레일 2": (1.0, 2.0, None),
}


def dmi(df, n=14):
    up, dn = df.high.diff(), -df.low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = pd.Series(pdm).ewm(alpha=1 / n, adjust=False).mean().values / a.values
    mdi = pd.Series(mdm).ewm(alpha=1 / n, adjust=False).mean().values / a.values
    return pdi, mdi


def signals():
    sig, m5, h1 = signal_table(load_m1_recent())
    end = m5.time + pd.Timedelta(minutes=5)
    ref = pd.Series(atr(m5).values).rolling(ATR_REF_BARS, min_periods=ATR_REF_MIN).median().values
    m = pd.merge_asof(sig[["t"]], pd.DataFrame({"end": end, "ref": ref}), left_on="t", right_on="end",
                      direction="backward")
    sig["ref"] = m.ref.values
    fl = pd.merge_asof(sig[["t"]], h1_flips_by_time(h1)[["end", "flips", "day"]], left_on="t", right_on="end",
                       direction="backward")
    sig["flips"] = np.where(fl.day.values == sig.t.dt.date.values, fl.flips.fillna(0).values, 0)
    # 봉 조건: 방향(1 매수 / -1 매도)마다 "반대 신호가 나온 봉의 끝 시각(ms)" 목록
    end_ms = end.values.astype("datetime64[ms]").astype(np.int64)
    up = psar(m5.high.values, m5.low.values)
    r = rsi(m5.close).values
    pdi, mdi = dmi(m5)
    bear = {"sar": ~up, "rsi": r < 50, "dmi": mdi > pdi}
    bull = {"sar": up, "rsi": r > 50, "dmi": pdi > mdi}
    against = {(c, 1): end_ms[bear[c]] for c in bear} | {(c, -1): end_ms[bull[c]] for c in bull}
    return sig[(sig.t >= START) & (sig.t < END)].reset_index(drop=True), against


def trail_hit(X, pos, side, lock_px, d, fc, chunk=50_000):
    """pos부터 최고 이익에서 d$ 되돌리거나 lock_px에 닿는 첫 idx (fc 넘으면 fc)"""
    peak = -INF                                         # 매도는 부호를 뒤집어 매수처럼 계산
    while pos < fc:
        seg = X[pos:min(pos + chunk, fc)] * side
        cm = np.maximum(np.maximum.accumulate(seg), peak)
        k = np.flatnonzero(seg <= np.maximum(cm - d, lock_px * side))
        if k.size:
            return pos + k[0]
        peak = cm[-1]
        pos += len(seg)
    return fc


def trade(t, bid, ask, k, side, case, fri_ms, against):
    """k번째 틱 진입 -> (청산 idx, $/oz) 또는 None"""
    X = bid if side == 1 else ask
    e = ask[k] if side == 1 else bid[k]
    fc = int(np.searchsorted(t, fri_ms))
    sl_px, tp_px = e - side * SL, e + side * TP
    j = hit(X, k + 1, sl_px, tp_px) if side == 1 else hit(X, k + 1, tp_px, sl_px)
    j = min(j, fc)
    if j >= len(t):
        return None
    oz = (X[j] - e) * side
    if case is None or j == fc or oz < 0:
        return j, oz
    lock, d, cond = case
    lock_px = e + side * lock
    ex = [fc]
    if d:
        ex.append(trail_hit(X, j + 1, side, lock_px, d, fc))
    else:
        ex.append(hit(X, j + 1, lock_px, INF) if side == 1 else hit(X, j + 1, -INF, lock_px))
    if cond:
        a = against[(cond, side)]
        i = np.searchsorted(a, t[j], side="right")
        if i < len(a):
            ex.append(int(np.searchsorted(t, a[i])))
    j2 = min(ex)
    if j2 >= len(t):
        return None
    return j2, (X[j2] - e) * side


class Acc:
    def __init__(self, case):
        self.case = case
        self.bal = self.peak = BAL0
        self.mdd, self.busy, self.day, self.base = 0.0, -1, None, 0.0
        self.results, self.rows = [], []

    def on_signal(self, s, t, bid, ask, against):
        ms = int(np.datetime64(s.t, "ms").astype(np.int64))
        if ms < self.busy:
            return
        if s.t.date() != self.day:
            self.day, self.base = s.t.date(), floor_lot(self.bal / 1000 * LOT_PER_1000)
        mins = s.t.hour * 60 + s.t.minute
        if (s.t.weekday() == 4 and mins >= FRIDAY_NO_ENTRY) or NO_ENTRY_FROM <= mins < NO_ENTRY_UNTIL \
                or s.flips >= MAX_H1_FLIPS or s.atr < MIN_ATR or (s.ref == s.ref and s.atr > MAX_ATR_MULT * s.ref):
            return
        last = self.results[-WINRATE_N:]
        paused = len(last) >= WINRATE_N and np.mean(last) * 100 < WINRATE_MIN
        if not paused and self.base > MAX_LOT:
            return
        k = int(np.searchsorted(t, ms))
        if k >= len(t) - 1 or t[k] - ms > 600_000:
            return
        d = s.t.normalize() + pd.Timedelta(days=(4 - s.t.weekday()) % 7, hours=23, minutes=30)
        res = trade(t, bid, ask, k, s.side, self.case, int(np.datetime64(d, "ms").astype(np.int64)), against)
        if res is None:
            return
        j, oz = res
        self.busy = int(t[j])
        self.results.append(oz > 0)
        if paused:
            return
        profit = oz * self.base * CONTRACT
        self.bal += profit
        self.peak = max(self.peak, self.bal)
        self.mdd = max(self.mdd, (self.peak - self.bal) / self.peak)
        self.rows.append((pd.Timestamp(int(t[k]), unit="ms"), self.day, (t[j] - t[k]) / 60000, oz, profit, self.bal))


def summary(a):
    tr = pd.DataFrame(a.rows, columns=["open", "day", "mins", "oz", "profit", "bal"])
    daily = tr.groupby("day").profit.sum()
    day_start = tr.groupby("day").bal.last() - daily
    gp, gl = tr.profit[tr.profit > 0].sum(), -tr.profit[tr.profit < 0].sum()
    w = tr[tr.oz > 0]
    out = {"거래": len(tr), "승률%": round((tr.oz > 0).mean() * 100, 1), "평균 이익$/oz": round(w.oz.mean(), 2),
           "3$넘게 번 비율%": round((w.oz > TP + 0.01).mean() * 100, 1), "최대 이익$/oz": round(w.oz.max(), 1),
           "이긴 거래 보유(분, 중간값)": round(w.mins.median(), 1), "PF": round(gp / gl, 2) if gl else float("inf"),
           "최종잔고$": round(tr.bal.iat[-1]), "최대낙폭%": round(a.mdd * 100, 1),
           "최악의날%": round((daily / day_start).min() * 100, 1)}
    prev = BAL0
    for y, g in tr.groupby(tr.open.dt.year):
        out[str(y)] = f"{(g.bal.iat[-1] / prev - 1) * 100:+.0f}%"
        prev = g.bal.iat[-1]
    return out


def main():
    global START, END
    if len(sys.argv) > 1:
        START = pd.Timestamp(sys.argv[1])
    if len(sys.argv) > 2:
        END = pd.Timestamp(sys.argv[2])
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    sig, against = signals()
    print(f"신호 {len(sig):,}개", flush=True)
    accs = {n: Acc(c) for n, c in CASES.items()}
    sig_ms = sig.t.values.astype("datetime64[ms]").astype(np.int64)
    ms = [(y, m) for y, m in lb.months() if (START.year, START.month) <= (y, m) <= (END.year, END.month)]
    for i, (y, m) in enumerate(ms):
        s, e = lb.month_range(y, m)
        lo, hi = np.searchsorted(sig_ms, int(s.timestamp() * 1000)), np.searchsorted(sig_ms, int(e.timestamp() * 1000))
        if lo == hi:
            continue
        t, bid, ask = lb.load_month(y, m)
        if i + 1 < len(ms):
            nt, nb, na = lb.load_month(*ms[i + 1])
            t, bid, ask = np.concatenate([t, nt]), np.concatenate([bid, nb]), np.concatenate([ask, na])
            del nt, nb, na
        for r in sig.iloc[lo:hi].itertuples(index=False):
            for a in accs.values():
                a.on_signal(r, t, bid, ask, against)
        del t, bid, ask
        print(f"  {y}-{m:02d} 완료", flush=True)
    res = pd.DataFrame([{"방식": n, **summary(a)} for n, a in accs.items()])
    res.to_csv(f"trail_exit_{START.date()}_{END.date() if END.year < 2100 else 'now'}.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
