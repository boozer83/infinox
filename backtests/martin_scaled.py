"""잔고 비례 lot 시뮬레이션: 2025-01 ~ 현재, 10,000달러 시작 (실제 틱 체결)

1. 마틴 (demo_trader.py 규칙): RSI50 + H1 SAR, SL 7.5 / TP 3, 일일 마틴 5단계 (손절마다 2배, 익절하면 0단계,
   5단계 손절이면 그날 정지, 날짜가 바뀌면 0단계)
   - 고정     : 시작 lot 0.30 고정 (10,000달러 기준 0.03 lot / 1,000달러)
   - 잔고비례 : 그날 시작 잔고 1,000달러당 0.03 lot (0.01 단위 내림, 최소 0.01). 하루 안에서는 같은 시작 lot
2. 추세추종 (trend_trader.py 규칙): H4 N25 진입 / N12 청산, 손절 3 ATR, 거래당 잔고의 2% 위험 (최소 0.01, 최대 5 lot)

사용법: python -m backtests.martin_scaled   (long_period를 먼저 한 번 실행해 틱이 저장돼 있어야 함)
"""
import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.scalp_ideas import m5_bars
from backtests.strategies import TickStore, execute, resample, trend
from config import CONTRACT

START = pd.Timestamp("2025-01-01")
BAL0 = 10_000.0
LOT_PER_1000 = 0.03
MARTIN_MAX_LEVEL = 5
RISK_PCT = 0.02
MAX_LOT = 5.0


def floor_lot(x):
    return max(0.01, np.floor(x * 100 + 1e-9) / 100)


def martin_money(tr, scaled):
    """거래 목록(pnl_oz)으로 일일 마틴 잔고 흐름. 반환: (거래별 기록, 일별 기록)"""
    bal, level, cur, stopped, base = BAL0, 0, None, None, None
    rows = []
    for r in tr.itertuples():
        day = r.open.date()
        if day != cur:
            cur, level = day, 0
            base = floor_lot(bal / 1000 * LOT_PER_1000) if scaled else floor_lot(BAL0 / 1000 * LOT_PER_1000)
        if stopped == day:
            continue
        lot = round(base * 2 ** level, 2)
        profit = r.pnl_oz * lot * CONTRACT
        bal += profit
        rows.append((r.open, day, level, lot, profit, bal))
        if bal <= 0:
            break
        if r.pnl_oz < 0:
            if level >= MARTIN_MAX_LEVEL:
                stopped = day
            else:
                level += 1
        else:
            level = 0
    return pd.DataFrame(rows, columns=["open", "day", "level", "lot", "profit", "balance"])


def trend_money(tr):
    bal, rows = BAL0, []
    for r in tr.itertuples():
        lot = min(MAX_LOT, floor_lot(bal * RISK_PCT / (r.risk * CONTRACT)))
        profit = r.pnl_oz * lot * CONTRACT
        bal += profit
        rows.append((r.close, r.close.date(), 0, lot, profit, bal))
    return pd.DataFrame(rows, columns=["open", "day", "level", "lot", "profit", "balance"])


def summarize(name, d):
    eq = np.r_[BAL0, d.balance.values]
    peak = np.maximum.accumulate(eq)
    mdd = ((peak - eq) / peak).max()
    daily = d.groupby("day").profit.sum()
    d = d.assign(month=pd.to_datetime(d.open).dt.to_period("M"))
    mon = d.groupby("month").agg(손익=("profit", "sum"), 월말잔고=("balance", "last"))
    mon["수익률%"] = (mon.손익 / (mon.월말잔고 - mon.손익) * 100).round(1)
    stops = int(((d.level == MARTIN_MAX_LEVEL) & (d.profit < 0)).sum())
    print(f"===== {name} =====")
    print(f"거래 {len(d):,}건 | 최종 잔고 {d.balance.iat[-1]:,.0f}$ ({(d.balance.iat[-1] / BAL0 - 1) * 100:+.1f}%) | "
          f"최대 낙폭 {mdd:.1%} | 최저 잔고 {eq.min():,.0f}$")
    print(f"최악의 날 {daily.min():,.0f}$ ({daily.idxmin()}) | 손실 월 {(mon.손익 < 0).sum()}/{len(mon)} | "
          f"5단계 손절 {stops}번 | lot 범위 {d.lot.min():g} ~ {d.lot.max():g}")
    print(mon.assign(손익=mon.손익.round(0), 월말잔고=mon.월말잔고.round(0)).to_string())
    print()
    return {"전략": name, "거래수": len(d), "최종잔고": round(d.balance.iat[-1]), "수익률%": round((d.balance.iat[-1] / BAL0 - 1) * 100, 1),
            "최대낙폭%": round(mdd * 100, 1), "최악의날$": round(daily.min()), "손실월": f"{(mon.손익 < 0).sum()}/{len(mon)}",
            "5단계손절": stops}


def main():
    print("봉 만드는 중...", flush=True)
    m5 = m5_bars()
    h1, h4 = resample(m5, "1h"), resample(m5, "4h")

    sigs = [s for s in lb.signals(m5, h1) if s[0] >= int(START.timestamp() * 1000)]
    print(f"마틴 신호 {len(sigs):,}개, 틱 시뮬레이션 중...", flush=True)
    mtr = lb.simulate(sigs, scaled=False)
    mtr["open"] = pd.to_datetime(mtr.open_ms, unit="ms")

    ttr = execute(trend(h4, 25, 3.0), TickStore(keep=2))
    ttr = ttr[ttr.close >= START]

    res = [summarize("마틴 고정 0.30 lot", martin_money(mtr, scaled=False)),
           summarize("마틴 잔고비례 (1,000$당 0.03 lot)", martin_money(mtr, scaled=True)),
           summarize("추세추종 2% 위험", trend_money(ttr))]
    out = pd.DataFrame(res)
    out.to_csv("martin_scaled_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
