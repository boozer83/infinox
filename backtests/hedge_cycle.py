"""HK MARTIN PRO(홍보 영상) 구조를 우리 신호에 붙인 헷지 회복형 사이클 (실제 틱, 1,000달러 시작)

영상에서 확인한 구조:
  1. 신호 방향으로 N개를 GAP달러 간격으로 진입 (불리해질 때마다 추가, 같은 lot)
  2. 그 단계 첫 진입가보다 ZONE달러 불리해지면 반대 방향으로 다음 단계 (lot x2), 최대 MAX_LEVEL단계
     (각 단계도 N개까지 GAP달러 간격 추가)
  3. 열린 포지션 전체 손익(청산 가격 기준)이 목표(시작 lot x TARGET달러 x 100)에 닿으면 전부 청산
  4. 전체 손익이 사이클 시작 잔고의 -LOSS_LIMIT%에 닿으면 전부 청산
  5. 금요일 23:30 이후 첫 틱에 전부 청산 (주말 보유 안 함)
시작 lot = 사이클 시작 잔고 1,000달러당 PER_1000 lot. 신호는 우리 마틴 신호 (filters=True면 쉬기 규칙도 적용).
가격 판정은 중간가(Bid+Ask)/2로 문턱을 계산하고, 체결/청산 손익은 그 틱의 실제 Bid/Ask로 계산.

사용법: python -m backtests.hedge_cycle
"""
import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.add_positions import ATR_REF_BARS, ATR_REF_MIN, floor_lot
from backtests.gold_rest_others import (FRIDAY_NO_ENTRY, MAX_H1_FLIPS, MIN_ATR, NO_ENTRY_FROM, NO_ENTRY_UNTIL,
                                        h1_flips_by_time, load_m1, signal_table)
from backtests.m5_ticks import first_hit
from indicators import atr

BAL0 = 1000.0
START = pd.Timestamp("2022-01-01")
SPREAD_EST = 0.25
INF = np.inf
# 이름: (분할 개수 N, 추가 간격 GAP, 반대 진입 ZONE, 최대 단계, 목표 $/oz(시작 lot 기준), 손실 한도 %, 1,000$당 lot, 쉬기 규칙)
CASES = {
    "N3 존4 목표3 (영상 비슷)": (3, 1.0, 4.0, 5, 3.0, 50, 0.06, True),
    "N3 존4 목표3, 쉬기 없음": (3, 1.0, 4.0, 5, 3.0, 50, 0.06, False),
    "N1 존4 목표3": (1, 1.0, 4.0, 5, 3.0, 50, 0.06, True),
    "N3 존7.5 목표3": (3, 1.0, 7.5, 5, 3.0, 50, 0.06, True),
    "N2 존5 목표3": (2, 1.0, 5.0, 5, 3.0, 50, 0.06, True),
    "N3 존4 목표3, lot 0.03": (3, 1.0, 4.0, 5, 3.0, 50, 0.03, True),
}


def signals():
    sig, m5, h1 = signal_table(load_m1("XAUUSD"))
    end = m5.time + pd.Timedelta(minutes=5)
    ref = pd.Series(atr(m5).values).rolling(ATR_REF_BARS, min_periods=ATR_REF_MIN).median().values
    m = pd.merge_asof(sig[["t"]], pd.DataFrame({"end": end, "ref": ref}), left_on="t", right_on="end",
                      direction="backward")
    sig["ref"] = m.ref.values
    fl = pd.merge_asof(sig[["t"]], h1_flips_by_time(h1)[["end", "flips", "day"]], left_on="t", right_on="end",
                       direction="backward")
    sig["flips"] = np.where(fl.day.values == sig.t.dt.date.values, fl.flips.fillna(0).values, 0)
    return sig[sig.t >= START].reset_index(drop=True)


def cross(mid, pos, x, up):
    """pos부터 mid가 x 이상(up) / 이하로 가는 첫 idx (없으면 len)"""
    if not np.isfinite(x):
        return len(mid)
    j = first_hit(mid, pos, -INF, x) if up else first_hit(mid, pos, x, INF)
    return len(mid) if j < 0 else j


def cycle(t, bid, ask, mid, k, side, base, bal, case, fri_ms):
    """k번째 틱에 사이클 시작 -> (청산 idx, 손익 $, 최대 단계, 최대 포지션 수, 최대 총 lot, 손실 한도 청산 여부) 또는 None"""
    n_split, gap, zone, max_level, target, loss_pct, _, _ = case
    pos_side, pos_lot, pos_e = [], [], []

    def add(j, d, lot):
        pos_side.append(d)
        pos_lot.append(lot)
        pos_e.append(ask[j] if d == 1 else bid[j])

    def pnl_at(j):
        s, l, e = np.array(pos_side), np.array(pos_lot), np.array(pos_e)
        px = np.where(s == 1, bid[j], ask[j])
        return float(np.sum(s * l * (px - e)) * 100)

    T, LL = target * base * 100, loss_pct / 100 * bal
    fc = int(np.searchsorted(t, fri_ms))
    level, d = 0, side
    add(k, d, base)
    first, cnt = pos_e[0], 1
    max_pos, max_lots = 1, base
    p = k + 1
    while True:
        s, l, e = np.array(pos_side), np.array(pos_lot), np.array(pos_e)
        E, C, L = float(np.sum(s * l)), float(np.sum(s * l * e)), float(np.sum(l))
        # 손익(mid) = 100 * (E*mid - C) - 100 * L * SPREAD_EST / 2
        ev = {}
        if abs(E) > 1e-9:
            for name, val in (("target", T), ("loss", -LL)):
                x = (val / 100 + C + L * SPREAD_EST / 2) / E
                up = (E > 0) == (name == "target")
                ev[name] = cross(mid, p, x, up)
        else:
            pl = 100 * (-C) - 100 * L * SPREAD_EST / 2
            if pl >= T:
                ev["target"] = p
        if cnt < n_split:
            ev["add"] = cross(mid, p, first - d * gap * cnt, d == -1)
        if level + 1 < max_level:
            ev["martin"] = cross(mid, p, first - d * zone, d == -1)
        ev["fri"] = fc
        name, j = min(ev.items(), key=lambda x: x[1])
        if j >= len(t):
            return None
        if name in ("target", "loss", "fri"):
            return j, pnl_at(j), level + 1, max_pos, max_lots, name == "loss"
        if name == "add":
            add(j, d, base * 2 ** level)
            cnt += 1
        else:
            level, d = level + 1, -d
            add(j, d, base * 2 ** level)
            first, cnt = pos_e[-1], 1
        max_pos, max_lots = max(max_pos, len(pos_e)), max(max_lots, sum(pos_lot))
        p = j + 1


class Acc:
    def __init__(self, case):
        self.case = case
        self.bal = self.peak = BAL0
        self.mdd, self.busy, self.rows = 0.0, -1, []

    def on_signal(self, s, t, bid, ask, mid):
        if self.bal <= 0:
            return
        ms = int(np.datetime64(s.t, "ms").astype(np.int64))
        if ms < self.busy:
            return
        mins = s.t.hour * 60 + s.t.minute
        if s.t.weekday() == 4 and mins >= FRIDAY_NO_ENTRY:
            return
        if self.case[7] and (NO_ENTRY_FROM <= mins < NO_ENTRY_UNTIL or s.flips >= MAX_H1_FLIPS or s.atr < MIN_ATR
                             or (s.ref == s.ref and s.atr > 2.0 * s.ref)):
            return
        k = int(np.searchsorted(t, ms))
        if k >= len(t) - 1 or t[k] - ms > 600_000:
            return
        base = floor_lot(self.bal / 1000 * self.case[6])
        d = s.t.normalize() + pd.Timedelta(days=(4 - s.t.weekday()) % 7, hours=23, minutes=30)
        res = cycle(t, bid, ask, mid, k, s.side, base, self.bal, self.case, int(np.datetime64(d, "ms").astype(np.int64)))
        if res is None:
            return
        j, pnl, lv, npos, lots, stop = res
        self.busy = int(t[j])
        self.bal += pnl
        self.peak = max(self.peak, self.bal)
        self.mdd = max(self.mdd, (self.peak - self.bal) / self.peak)
        self.rows.append((s.t, pd.Timestamp(int(t[j]), unit="ms"), lv, npos, lots, pnl, self.bal, stop))


def summary(a):
    tr = pd.DataFrame(a.rows, columns=["open", "close", "level", "npos", "lots", "pnl", "bal", "stop"])
    tr["day"] = tr.open.dt.date
    daily = tr.groupby("day").pnl.sum()
    day_start = tr.groupby("day").bal.last() - daily
    out = {"사이클": len(tr), "승률%": round((tr.pnl > 0).mean() * 100, 1), "최종잔고$": round(tr.bal.iat[-1]),
           "최대낙폭%(청산)": round(a.mdd * 100, 1), "최악의날%": round((daily / day_start).min() * 100, 1),
           "50%손실청산": int(tr.stop.sum()), "최대단계": int(tr.level.max()), "최대포지션": int(tr.npos.max()),
           "최대총lot": round(tr.lots.max(), 2),
           "단계별 종료": " ".join(str(int((tr.level == i).sum())) for i in range(1, 6))}
    prev = BAL0
    for y, g in tr.groupby(tr.open.dt.year):
        out[str(y)] = f"{(g.bal.iat[-1] / prev - 1) * 100:+.0f}%"
        prev = g.bal.iat[-1]
    if a.bal <= 0:
        out["깡통"] = str(tr.close.iat[-1].date())
    return out


def main():
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    sig = signals()
    print(f"신호 {len(sig):,}개", flush=True)
    accs = {n: Acc(c) for n, c in CASES.items()}
    sig_ms = sig.t.values.astype("datetime64[ms]").astype(np.int64)
    ms = [(y, m) for y, m in lb.months() if (y, m) >= (START.year, START.month)]
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
        mid = (bid + ask) / 2
        for r in sig.iloc[lo:hi].itertuples(index=False):
            for a in accs.values():
                a.on_signal(r, t, bid, ask, mid)
        del t, bid, ask, mid
        print(f"  {y}-{m:02d} 완료", flush=True)
    res = pd.DataFrame([{"방식": n, **summary(a)} for n, a in accs.items()])
    res.to_csv("hedge_cycle.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
