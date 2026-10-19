"""회복형 마틴: 다음 한 번만 익절하면 앞의 손실을 전부 메우고 처음 목표(시작 lot의 익절)만큼 남도록 lot을 정함
(2025-01 ~ 현재, 실제 틱, 1,000달러 시작)

lot = 올림((묶음 누적 손실 + 시작 lot x 익절 x 100) / (익절 x 100)), 0.01 단위
익절하면 묶음 끝 -> 다음은 시작 lot. MAX_STEPS번째 진입까지 손절이면 손실 확정하고 시작 lot으로 (묶음은 날짜가 바뀌어도 이어짐)
시작 lot = 묶음 시작 시점 잔고 1,000달러당 PER_1000 lot. 신호/쉬기 규칙은 add_positions_ticks와 같음 (lot 1.0 제한 없음).

사용법: python -m backtests.recovery_martin
"""
import math

import numpy as np
import pandas as pd

import backtests.add_positions_ticks as m
import backtests.long_period as lb
from backtests.add_positions import BAL0, CONTRACT, floor_lot
from backtests.gold_rest_others import (FRIDAY_NO_ENTRY, MAX_H1_FLIPS, MIN_ATR, NO_ENTRY_FROM, NO_ENTRY_UNTIL,
                                        WINRATE_MIN, WINRATE_N)

START = pd.Timestamp("2025-01-01")
SL, TP = 7.5, 3.0
VAR = ("none", 0, 1, SL, TP, 0, 0)
CASES = [(0.01, 3), (0.01, 4), (0.01, 5), (0.02, 3), (0.02, 4), (0.03, 3), (0.03, 4)]


class Recovery:
    def __init__(self, per1000, steps):
        self.per1000, self.steps = per1000, steps
        self.results, self.rows = [], []
        self.busy = -1
        self.bal = self.peak = BAL0
        self.mdd, self.bust = 0.0, None
        self.step, self.chain_loss, self.base = 0, 0.0, 0.0

    def on_signal(self, s, t, bid, ask):
        if self.bust:
            return
        ms = int(np.datetime64(s.t, "ms").astype(np.int64))
        if ms < self.busy:
            return
        mins = s.t.hour * 60 + s.t.minute
        if (s.t.weekday() == 4 and mins >= FRIDAY_NO_ENTRY) or NO_ENTRY_FROM <= mins < NO_ENTRY_UNTIL \
                or s.flips >= MAX_H1_FLIPS or s.atr < MIN_ATR or (s.ref == s.ref and s.atr > 2.0 * s.ref):
            return
        last = self.results[-WINRATE_N:]
        paused = len(last) >= WINRATE_N and np.mean(last) * 100 < WINRATE_MIN
        k = int(np.searchsorted(t, ms))
        if k >= len(t) - 1 or t[k] - ms > 600_000:
            return
        d = s.t.normalize() + pd.Timedelta(days=(4 - s.t.weekday()) % 7, hours=23, minutes=30)
        res = m.tick_basket(t, bid, ask, k, s.side, VAR, 1, int(np.datetime64(d, "ms").astype(np.int64)))
        if res is None:
            return
        j, entries, px = res
        self.busy = int(t[j])
        oz = float((px - entries[0]) * s.side)
        self.results.append(oz > 0)
        if paused:
            return
        if self.step == 0:
            self.base = floor_lot(self.bal / 1000 * self.per1000)
            lot = self.base
        else:
            lot = math.ceil((self.chain_loss + self.base * TP * CONTRACT) / (TP * CONTRACT) * 100 - 1e-9) / 100
        profit = oz * lot * CONTRACT
        self.bal += profit
        self.peak = max(self.peak, self.bal)
        self.mdd = max(self.mdd, (self.peak - self.bal) / self.peak)
        self.rows.append((pd.Timestamp(int(t[k]), unit="ms"), s.t.date(), self.step, lot, oz, profit, self.bal))
        if self.bal <= 0:
            self.bust = s.t
            return
        if oz > 0:
            self.step, self.chain_loss = 0, 0.0
        elif self.step + 1 >= self.steps:
            self.step, self.chain_loss = 0, 0.0
        else:
            self.step += 1
            self.chain_loss -= profit


def summary(a):
    tr = pd.DataFrame(a.rows, columns=["open", "day", "step", "lot", "oz", "profit", "bal"])
    daily = tr.groupby("day").profit.sum()
    day_start = tr.groupby("day").bal.last() - daily
    fails = int(((tr.step == a.steps - 1) & (tr.oz < 0)).sum())
    y = tr.groupby(tr.open.dt.year).bal.last()
    return {"최종잔고$": round(tr.bal.iat[-1]), "2025말": round(y.get(2025, np.nan)), "최대낙폭%": round(a.mdd * 100, 1),
            "최악의날%": round((daily / day_start).min() * 100, 1), "최대 lot": tr.lot.max(),
            "실패(끝까지 손절)": fails, "깡통": str(a.bust.date()) if a.bust is not None else "",
            "단계별 진입수": " ".join(str(int((tr.step == i).sum())) for i in range(a.steps))}


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    m.START = START
    sig = m.signals()
    accs = {f"1,000$당 {p} 시작, 최대 {n}번째": Recovery(p, n) for p, n in CASES}
    sig_ms = sig.t.values.astype("datetime64[ms]").astype(np.int64)
    ms = [(y, mo) for y, mo in lb.months() if (y, mo) >= (START.year, START.month)]
    for i, (y, mo) in enumerate(ms):
        s, e = lb.month_range(y, mo)
        lo, hi = np.searchsorted(sig_ms, int(s.timestamp() * 1000)), np.searchsorted(sig_ms, int(e.timestamp() * 1000))
        t, b, a = lb.load_month(y, mo)
        if i + 1 < len(ms):
            nt, nb, na = lb.load_month(*ms[i + 1])
            t, b, a = np.concatenate([t, nt]), np.concatenate([b, nb]), np.concatenate([a, na])
            del nt, nb, na
        for r in sig.iloc[lo:hi].itertuples(index=False):
            for acc in accs.values():
                acc.on_signal(r, t, b, a)
        del t, b, a
    res = pd.DataFrame([{"방식": n, **summary(a)} for n, a in accs.items()])
    res.to_csv("recovery_martin_2025_2026.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
