"""지금 실제로 돌리는 봇 3개를 한 계좌에서 같이 돌리면 (2025-01 ~ 현재, 1,000$ / 160$ 시작, lot 0.01 단위 반영)

마틴 봇(demo_trader): 틱 체결 (trail_exit의 기준 방식 = 지금 규칙), lot = 그날 시작 잔고 1,000$당 0.05 (0.01 내림, 최소 0.01, 최대 1.0)
추세(trend_trader)   : martin_trend_combo.trend_trades (1분봉 체결), lot = 잔고 x 2% / (손절폭 x 100), 0.01 내림, 최소 0.01, 최대 5
                       스왑 지금 MT5 값 적용 (매수 -78.33 / 매도 +26.13 $/lot/밤, 수요일 3배)
아시아돌파           : asia_breakout의 틱 거래 목록, 1단계 lot = 사이클 시작 잔고 1% / (손절폭 x 100), 0.01 내림, 최소 0.01
                       손절이면 x2, 최대 3단계. lot > 1.0이면 건너뜀. 손익은 그날 23:00에 반영
lot은 진입 시점 잔고(청산된 손익만)로 계산. 잔고가 0 이하가 되면 깡통으로 멈춤.
계좌 손실 한도(50%)는 적용하지 않고, 처음 닿은 날짜만 표시.

사용법: python -m backtests.three_bots   (먼저 python -m backtests.asia_breakout 실행 -> asia_breakout_trades.csv)
"""
import heapq
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.long_period as lb
import backtests.trail_exit as te
from backtests.gold_rest_others import load_m1
from backtests.martin_trend_combo import SWAP_LONG, SWAP_SHORT, swap_nights, trend_trades

START = pd.Timestamp("2025-01-01")
CONTRACT = 100
MARTIN_CACHE = Path("three_bots_martin_trades.csv")
COMBOS = {"마틴 봇만": "m", "마틴 + 아시아": "ma", "마틴 + 추세": "mt", "아시아 + 추세": "at", "세 개 다": "mat"}


def floor_lot(x):
    return max(0.01, np.floor(x * 100 + 1e-9) / 100)


def martin_trades():
    """지금 마틴 봇 규칙의 실제 진입 거래 (승률 쉬기 중 가상 진입 제외). 잔고와 상관없이 정해짐"""
    if MARTIN_CACHE.exists():
        return pd.read_csv(MARTIN_CACHE, parse_dates=["open", "close"])
    te.START, te.END = START, pd.Timestamp("2100-01-01")
    sig, against = te.signals()
    acc = te.Acc(None)
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
        for r in sig.iloc[lo:hi].itertuples(index=False):
            acc.on_signal(r, t, bid, ask, against)
        del t, bid, ask
    tr = pd.DataFrame(acc.rows, columns=["open", "day", "mins", "oz", "profit", "bal"])
    tr["close"] = tr.open + pd.to_timedelta(tr.mins, unit="m")
    tr = tr[["open", "close", "oz"]]
    tr.to_csv(MARTIN_CACHE, index=False)
    return tr


def asia_trades():
    a = pd.read_csv("asia_breakout_trades.csv", parse_dates=["day", "t"])
    a = a[a.day >= START]
    return pd.DataFrame({"open": a.t, "close": a.day + pd.Timedelta(hours=23), "oz": a.oz, "sl": a.sl_usd})


def run(m, a, t, bal0, use):
    ev = []
    for kind, df in (("m", m), ("a", a), ("t", t)):
        if kind in use:
            ev += [(np.datetime64(r.open), 0, kind, r) for r in df.itertuples(index=False)]
    heap = [(tm, pri, i, k, r) for i, (tm, pri, k, r) in enumerate(ev)]
    heapq.heapify(heap)
    seq = len(heap)
    bal, out, bust, half = bal0, [], None, None
    peak = bal0
    day, day_bal = None, bal0
    a_level, a_base = 0, 0.0
    while heap and bust is None:
        tm, pri, _, kind, r = heapq.heappop(heap)
        if pri == 0:
            if kind == "m":
                d = pd.Timestamp(tm).date()
                if d != day:
                    day, day_bal = d, bal
                lot = min(1.0, floor_lot(day_bal / 1000 * 0.05))
                pnl = r.oz * lot * CONTRACT
            elif kind == "a":
                if a_level == 0:
                    a_base = floor_lot(bal * 0.01 / (r.sl * CONTRACT))
                lot = round(a_base * 2 ** a_level, 2)
                if lot > 1.0:
                    continue
                pnl = r.oz * lot * CONTRACT
            else:
                lot = min(5.0, floor_lot(bal * 0.02 / (r.risk * CONTRACT)))
                pnl = (r.pnl_oz * CONTRACT + r.nights * (SWAP_LONG if r.side == 1 else SWAP_SHORT)) * lot
            heapq.heappush(heap, (np.datetime64(r.close), -1, seq, kind, (lot, pnl, bal)))
            seq += 1
        else:
            lot, pnl, bal_at_entry = r
            bal += pnl
            peak = max(peak, bal)
            out.append((pd.Timestamp(tm), kind, lot, pnl, bal, pnl / bal_at_entry * 100))
            if kind == "a":
                a_level = 0 if pnl > 0 else (a_level + 1 if a_level + 1 < 3 else 0)
            if half is None and bal <= peak * 0.5:
                half = pd.Timestamp(tm).date()
            if bal <= 0:
                bust = pd.Timestamp(tm).date()
    return pd.DataFrame(out, columns=["close", "kind", "lot", "pnl", "bal", "pct"]), bust, half


def report(name, bal0, d, bust, half):
    eq = np.r_[bal0, d.bal.values]
    peak = np.maximum.accumulate(eq)
    mon = d.groupby(d.close.dt.to_period("M")).agg(p=("pnl", "sum"), b=("bal", "last"))
    mret = mon.p / (mon.b - mon.p) * 100
    out = {"구성": name, "시작$": int(bal0), "최종$": round(d.bal.iat[-1]), "최대낙폭%": round(((peak - eq) / peak).max() * 100, 1),
           "최악의달%": round(mret.min(), 1), "최저잔고$": round(eq.min()), "한 거래 최대손실%": round(d.pct.min(), 1),
           "50%낙폭 처음": str(half or ""), "깡통": str(bust or "")}
    prev = bal0
    for y, g in d.groupby(d.close.dt.year):
        out[str(y)] = f"{(g.bal.iat[-1] / prev - 1) * 100:+.0f}%"
        prev = g.bal.iat[-1]
    for k, n in (("m", "마틴"), ("a", "아시아"), ("t", "추세")):
        g = d[d.kind == k]
        if len(g):
            out[f"{n} 손익$"] = round(g.pnl.sum())
    return out


def main():
    pd.set_option("display.width", 320)
    pd.set_option("display.unicode.east_asian_width", True)
    m = martin_trades()
    print(f"마틴 거래 {len(m)}건", flush=True)
    a = asia_trades()
    t = trend_trades(load_m1("XAUUSD"))
    t = t[t.open >= START].reset_index(drop=True)
    print(f"아시아 {len(a)}건, 추세 {len(t)}건", flush=True)
    rows = []
    for bal0 in (1000.0, 160.0):
        for name, use in COMBOS.items():
            d, bust, half = run(m, a, t, bal0, use)
            rows.append(report(name, bal0, d, bust, half))
    res = pd.DataFrame(rows)
    res.to_csv("three_bots.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
