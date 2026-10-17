"""XAUUSD 양방향 동시 진입 + 마틴 백테스트

규칙:
- 진입 시점: TRIGGER가 "SAR"이면 SAR 전환 다음 봉 시가, "항상"이면 포지션이 없을 때 다음 봉 시가
- 매수와 매도를 같은 수량으로 동시에 진입 (매수는 Ask = Bid + 스프레드, 매도는 Bid)
- 먼저 손실이 CUT(달러/온스, 스프레드 포함)에 닿는 쪽을 즉시 청산
- 남은 쪽은 익절 TP 또는 손절에 닿으면 청산
    손절 "본전": 손익이 0 아래로 내려가면 청산 / "CUT": 손실이 CUT에 닿으면 청산
- 두 포지션이 다 청산되면 한 사이클 끝
- 마틴: 사이클 합계가 손실이면 다음 사이클 수량 x2, 수익이면 기본 수량으로 복귀
        MAX_LEVEL번 연속 손실 뒤에는 기본 수량으로 복귀 (손실 확정)
- 잔고가 시작 잔고의 10% 아래로 떨어지면 깡통으로 중단
- 봉 안의 가격 경로는 양봉이면 시가->저가->고가->종가, 음봉이면 시가->고가->저가->종가로 가정하고
  그 순서대로 이벤트를 처리 (청산은 정확히 기준 가격에서 체결된다고 가정)

사용법: python -m backtests.straddle   (MT5 터미널이 켜져 있어야 함)
"""
import itertools

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from backtests.sar_basic import SPREAD, START_BALANCE, TIMEFRAMES, load
from config import CONTRACT, mt5_init
from indicators import psar

BUST_LEVEL = 0.10
TRIGGERS = ["SAR", "항상"]
CUTS = [0.5, 1.0, 2.0]
TPS = [2.0, 4.0]
STOPS = ["본전", "CUT"]
BASE_LOTS = [0.01, 0.03]
MAX_LEVELS = [0, 4, 6]            # 0 = 마틴 없음


def run(df, trigger, cut, tp, stop, base, max_level):
    o, h, l = df.open.values, df.high.values, df.low.values
    if trigger == "SAR":
        up = psar(h, l)
        flip = np.r_[False, up[1:] != up[:-1]]
    balance, level = START_BALANCE, 0
    min_equity, max_lot = balance, 0.0
    legs = {}                 # side -> entry (Bid 기준 진입가: 매수는 Ask)
    lot = base
    cyc_pnl = 0.0
    cycles = []               # (pnl, level)

    def leg_pnl(side, price):   # price = Bid
        e = legs[side]
        return (price - e) if side == 1 else (e - price - SPREAD)

    c_ = df.close.values
    sl = 0.0 if stop == "본전" else -cut
    for i in range(1, len(df) - 1):
        if legs:
            usd = lot * CONTRACT
            path = (o[i], l[i], h[i], c_[i]) if c_[i] >= o[i] else (o[i], h[i], l[i], c_[i])
            for price in path:
                if len(legs) == 2:
                    for s in (1, -1):
                        if leg_pnl(s, price) <= -cut:
                            del legs[s]
                            cyc_pnl += -cut * usd
                            # 잘린 순간 남은 쪽 손익 = CUT - 스프레드x2, 이미 손절선 아래면 그 값으로 바로 청산
                            rest = cut - 2 * SPREAD
                            if rest <= sl:
                                legs.clear()
                                cyc_pnl += rest * usd
                            break
                if len(legs) == 1:
                    (side, _), = legs.items()
                    p = leg_pnl(side, price)
                    if p <= sl or p >= tp:
                        legs.clear()
                        cyc_pnl += (sl if p <= sl else tp) * usd
                if not legs:
                    break
                min_equity = min(min_equity, balance + cyc_pnl + sum(leg_pnl(s, price) for s in legs) * usd)
            if not legs:
                balance += cyc_pnl
                cycles.append((cyc_pnl, level))
                if cyc_pnl < 0 and max_level and level < max_level:
                    level += 1
                else:
                    level = 0
                cyc_pnl = 0.0
                min_equity = min(min_equity, balance)
                if balance < BUST_LEVEL * START_BALANCE:
                    break
        if not legs and (trigger == "항상" or flip[i]):
            lot = round(base * 2 ** level, 2)
            max_lot = max(max_lot, lot * 2)
            legs = {1: o[i+1] + SPREAD, -1: o[i+1]}
    return pd.DataFrame(cycles, columns=["pnl", "level"]), balance, min_equity, max_lot


def main():
    if not mt5_init():
        raise SystemExit(f"MT5 연결 실패: {mt5.last_error()}")
    try:
        data = {name: load(tf) for name, tf in TIMEFRAMES.items()}
    finally:
        mt5.shutdown()

    rows = []
    for name, df in data.items():
        print(f"{name}: {len(df):,}봉  {df.time.iat[0]} ~ {df.time.iat[-1]}")
        for trig, cut, tp, stop, base, ml in itertools.product(TRIGGERS, CUTS, TPS, STOPS, BASE_LOTS, MAX_LEVELS):
            c, bal, mineq, mlot = run(df, trig, cut, tp, stop, base, ml)
            eq = START_BALANCE + c.pnl.cumsum()
            peak = np.maximum.accumulate(np.r_[START_BALANCE, eq.values])[1:]
            rows.append({
                "봉": name, "진입": trig, "CUT": cut, "TP": tp, "손절": stop, "기본lot": base, "마틴": ml,
                "사이클": len(c), "승률": round((c.pnl > 0).mean() * 100, 1) if len(c) else 0,
                "최종잔고": round(bal, 2),
                "MDD%": round(((peak - eq) / peak).max() * 100, 1) if len(c) else 0,
                "최저평가": round(mineq, 2), "최대lot": round(mlot, 2),
                "깡통": "깡통" if bal < BUST_LEVEL * START_BALANCE else "",
            })
    r = pd.DataFrame(rows)
    r.to_csv("straddle_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n전체 {len(r)}개 조합 / 수익 {(r.최종잔고 > START_BALANCE).sum()}개 / 깡통 {(r.깡통 == '깡통').sum()}개")
    for ml in MAX_LEVELS:
        s = r[r.마틴 == ml]
        print(f"  마틴 {ml}: 수익 {(s.최종잔고 > START_BALANCE).sum()}/{len(s)}, 깡통 {(s.깡통 == '깡통').sum()}, "
              f"최종잔고 중앙값 {s.최종잔고.median():.0f}")
    print("\n최종잔고 상위 15개")
    print(r.sort_values("최종잔고", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
