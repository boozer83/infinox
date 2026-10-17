"""RSI50 + H1 SAR 필터 + 일일 마틴 백테스트 (실제 틱 체결)

마틴 규칙:
- 손절로 끝나면 다음 거래 수량 x2, 익절이면 기본 수량으로 복귀
  (단계 0 = 기본 수량, 단계 1 = x2, 단계 2 = x4 ...)
- 최대 단계(MAX)에서 또 손절이 나면 그날은 거래 정지
- 서버시간 기준 날짜가 바뀌면 단계 0으로 초기화하고 거래 재개
- 잔고가 0 이하가 되면 깡통으로 중단
진입/청산은 m5_ticks.py와 같음 (5분봉 SAR 전환, 실제 Bid/Ask 틱, 고정 SL/TP).

사용법: python -m backtests.daily_martin
"""
import itertools

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from backtests.filters import FILTERS, build, make_signals
from backtests.m5_ticks import END, START, first_hit, load_bars, load_ticks
from config import CONTRACT, SYMBOL, mt5_init

START_BALANCE = 1000.0
FILTER = "RSI50+H1 SAR"
SL_GRID = [5.0, 7.5, 10.0]
TP = 3.0
BASE_LOTS = [0.06, 0.02, 0.01]
MAX_LEVELS = [None, 4, 5]          # None = 마틴 없음
DAY_MS = 86_400_000


def run(sigs, t, bid, ask, sl, tp, base, max_level):
    balance, level, stopped_day = START_BALANCE, 0, None
    cur_day = None
    busy_until = -1
    trades, stop_days = [], []
    min_bal, max_lot = balance, 0.0
    for st, side in sigs:
        if st < busy_until:
            continue
        day = st // DAY_MS
        if day != cur_day:
            cur_day, level = day, 0
        if stopped_day == day:
            continue
        k = np.searchsorted(t, st)
        if k >= len(t) - 1:
            break
        lot = round(base * 2 ** level, 2) if max_level is not None else base
        if side == 1:
            e = ask[k]
            j = first_hit(bid, k + 1, e - sl, e + tp)
            pnl_oz = bid[j] - e if j >= 0 else None
        else:
            e = bid[k]
            j = first_hit(ask, k + 1, e - tp, e + sl)
            pnl_oz = e - ask[j] if j >= 0 else None
        if pnl_oz is None:
            break
        pnl = float(pnl_oz) * lot * CONTRACT
        balance += pnl
        max_lot = max(max_lot, lot)
        min_bal = min(min_bal, balance)
        trades.append((t[k], t[j], level, lot, pnl))
        busy_until = t[j]
        if balance <= 0:
            break
        if max_level is not None:
            if pnl < 0:
                if level >= max_level:
                    stopped_day = day
                    stop_days.append(day)
                else:
                    level += 1
            else:
                level = 0
    tr = pd.DataFrame(trades, columns=["open_ms", "close_ms", "level", "lot", "pnl"])
    tr["close"] = pd.to_datetime(tr.close_ms, unit="ms")
    return tr, balance, min_bal, max_lot, stop_days


def main():
    if not mt5_init():
        raise SystemExit(f"MT5 연결 실패: {mt5.last_error()}")
    try:
        t, bid, ask = load_ticks()
        m5 = load_bars()
        r = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1, START - pd.Timedelta(days=20), END)
    finally:
        mt5.shutdown()
    h1 = pd.DataFrame(r)
    h1["time"] = pd.to_datetime(h1["time"], unit="s")
    sigs = make_signals(build(m5, h1), FILTERS[FILTER])
    print(f"필터 {FILTER}, 신호 {len(sigs):,}개, TP {TP}\n")

    rows = []
    for base, sl, ml in itertools.product(BASE_LOTS, SL_GRID, MAX_LEVELS):
        tr, bal, min_bal, max_lot, stops = run(sigs, t, bid, ask, sl, TP, base, ml)
        eq = START_BALANCE + tr.pnl.cumsum()
        peak = np.maximum.accumulate(np.r_[START_BALANCE, eq.values])[1:]
        m = tr.set_index("close").pnl.resample("ME").sum()
        worst = tr.assign(d=tr.close.dt.date).groupby("d").pnl.sum()
        rows.append({
            "기본lot": base, "SL": sl, "마틴": "없음" if ml is None else f"{ml}단계",
            "거래수": len(tr), "정지일": len(stops),
            "최대lot": round(max_lot, 2),
            "최종잔고": round(bal, 2),
            "MDD%": round(((peak - eq) / peak).max() * 100, 1),
            "최저잔고": round(min_bal, 2),
            "최악의날": round(worst.min(), 2),
            "손실월": int((m < 0).sum()),
            "깡통": pd.Timestamp(tr.close.iat[-1]).strftime("%m-%d") if bal <= 0 else "",
        })
    res = pd.DataFrame(rows)
    res.to_csv("daily_martin_results.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
