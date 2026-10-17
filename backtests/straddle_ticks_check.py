"""양방향 진입 전략을 실제 틱(Bid/Ask)으로 검증

straddle.py의 "항상" 진입, 마틴 없음 설정을 틱 단위로 다시 계산해서
1분봉/5분봉 근사 결과와 비교한다. 스프레드도 가정값이 아니라 실제 호가를 사용.

사용법: python -m backtests.straddle_ticks_check
"""
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import backtests.straddle as bs
from config import CONTRACT, SYMBOL, mt5_init

START = datetime(2026, 10, 12, tzinfo=timezone.utc)
END = datetime(2026, 10, 17, tzinfo=timezone.utc)
SETTINGS = [(1.0, 4.0, "본전"), (1.0, 2.0, "본전"), (2.0, 4.0, "본전"), (1.0, 4.0, "CUT")]
LOT = 0.01


def tick_run(bid, ask, cut, tp, stop):
    sl = 0.0 if stop == "본전" else -cut
    usd = LOT * CONTRACT
    total, cycles, wins = 0.0, 0, 0
    i, n = 0, len(bid)
    while i < n - 1:
        eb, es = ask[i], bid[i]          # 매수는 Ask, 매도는 Bid로 진입
        cyc = 0.0
        legs = {1, -1}
        i += 1
        while i < n and legs:
            pb, ps = bid[i] - eb, es - ask[i]   # 매수는 Bid로, 매도는 Ask로 청산
            if len(legs) == 2:
                if pb <= -cut:
                    legs.discard(1); cyc += pb
                elif ps <= -cut:
                    legs.discard(-1); cyc += ps
            if len(legs) == 1:
                p = pb if 1 in legs else ps
                if p <= sl or p >= tp:
                    legs.clear(); cyc += p
            i += 1
        if legs:   # 데이터 끝
            break
        total += cyc * usd
        cycles += 1
        wins += cyc > 0
    return cycles, wins, total


def main():
    if not mt5_init():
        raise SystemExit(f"MT5 연결 실패: {mt5.last_error()}")
    try:
        ticks = mt5.copy_ticks_range(SYMBOL, START, END, mt5.COPY_TICKS_INFO)
        bars = {}
        for name, tf in bs.TIMEFRAMES.items():
            r = mt5.copy_rates_range(SYMBOL, tf, START, END)
            df = pd.DataFrame(r)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            bars[name] = df
    finally:
        mt5.shutdown()
    if ticks is None or len(ticks) == 0:
        raise SystemExit(f"틱 데이터 없음: {mt5.last_error()}")
    bid, ask = ticks["bid"].astype(float), ticks["ask"].astype(float)
    spread = ask - bid
    print(f"틱 {len(ticks):,}개, 실제 스프레드 평균 {spread.mean():.3f} / 중앙값 {np.median(spread):.3f}")

    rows = []
    for cut, tp, stop in SETTINGS:
        c, w, pnl = tick_run(bid, ask, cut, tp, stop)
        row = {"CUT": cut, "TP": tp, "손절": stop, "틱 사이클": c, "틱 승률%": round(w / c * 100, 1),
               "틱 손익": round(pnl, 2)}
        for name, df in bars.items():
            cy, bal, *_ = bs.run(df, "항상", cut, tp, stop, LOT, 0)
            row[f"{name} 손익"] = round(bal - bs.START_BALANCE, 2)
        rows.append(row)
    print(f"\n기간 {START:%Y-%m-%d} ~ {END:%Y-%m-%d}, 0.01 lot, 마틴 없음")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
