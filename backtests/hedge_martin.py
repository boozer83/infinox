"""XAUUSD 파라볼릭 SAR 마틴 전략 백테스트

규칙 (2.jpg, 3.jpg 마틴 거래내역에서 재구성):
- 포지션이 없을 때 SAR 전환이 나오면 다음 봉 시가에 기본 수량으로 진입 (사이클 시작)
- 사이클 진행 중 SAR이 전환될 때마다 새 방향으로 1개씩 추가 (기존 포지션은 유지 -> 양방향 보유)
- 수량은 PER_LEVEL개 진입마다 2배 (예: 0.06, 0.06, 0.12, 0.12, 0.24, 0.24 ...)
- 한 사이클의 최대 진입 수는 MAX_ENTRIES개
- POS_TP가 있으면 개별 포지션이 그만큼(달러/온스) 이익 나면 그 포지션만 먼저 익절 (실현손익은 사이클에 누적)
- 사이클 손익(실현 + 미실현)이 목표 이상이면 전부 청산
- 사이클 손실이 사이클 시작 잔고의 LOSS_LIMIT(30%)에 닿으면 전부 청산
- 잔고가 시작 잔고의 BUST_LEVEL(10%) 아래로 떨어지면 거기서 중단 (깡통)
- 테스트 끝까지 청산 안 된 사이클은 마지막 종가로 평가해서 따로 표시
- 봉 안에서 손실 한도와 목표가 둘 다 닿으면 손실 한도가 먼저라고 가정 (보수적)

사용법: python -m backtests.hedge_martin   (MT5 터미널이 켜져 있어야 함)
"""
import itertools

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from backtests.sar_basic import SPREAD, START_BALANCE, TIMEFRAMES, load
from config import CONTRACT, mt5_init
from indicators import psar

LOSS_LIMIT = 0.30
BUST_LEVEL = 0.10
PER_LEVEL = 2
BASE_LOTS = [0.01, 0.03, 0.06]
TARGETS = [3.0, 5.0, 10.0]        # 사이클 목표: 기본 수량 기준 금 1온스당 달러 (0.06lot, 3달러 -> 18달러)
MAX_ENTRIES_GRID = [6, 10]        # 6개 = 최대 4배, 10개 = 최대 16배
POS_TP_GRID = [None, 2.0]         # None = 개별 익절 없음


def run(df, base, target_usd, max_entries, pos_tp=None):
    o, h, l = df.open.values, df.high.values, df.low.values
    up = psar(h, l)
    t = df.time.values
    target = target_usd * base * CONTRACT

    balance = START_BALANCE
    pos = []                   # [side, lot, entry]
    cyc_start_bal = cyc_start_t = None
    cycles = []                # (start, end, pnl, kind, entries, max_lot)
    min_equity = balance
    max_lot = 0.0

    def floating(price):
        f = 0.0
        for side, lot, entry in pos:
            f += ((price - entry) if side == 1 else (entry - price - SPREAD)) * lot * CONTRACT
        return f

    realized = 0.0             # 사이클 안에서 개별 익절로 실현된 손익
    n_entries = 0              # 사이클 누적 진입 수 (개별 익절돼도 줄지 않음)
    for i in range(1, len(df) - 1):
        if pos:
            f_lo, f_hi = floating(l[i]), floating(h[i])
            worst, best = realized + min(f_lo, f_hi), realized + max(f_lo, f_hi)
            min_equity = min(min_equity, balance + worst)
            limit = -LOSS_LIMIT * cyc_start_bal
            kind = None
            if worst <= limit:
                pnl, kind = limit, "손절"
            elif best >= target:
                pnl, kind = target, "익절"
            elif pos_tp is not None:
                keep = []
                for side, lot, entry in pos:
                    if (h[i] >= entry + pos_tp) if side == 1 else (l[i] + SPREAD <= entry - pos_tp):
                        realized += pos_tp * lot * CONTRACT
                    else:
                        keep.append([side, lot, entry])
                pos = keep
                if not pos:   # 전부 개별 익절되면 사이클 종료
                    pnl, kind = realized, "익절"
            if kind:
                balance += pnl
                cycles.append((cyc_start_t, t[i], pnl, kind, n_entries, sum(p[1] for p in pos)))
                pos, realized, n_entries = [], 0.0, 0
                if balance < BUST_LEVEL * START_BALANCE:
                    break
        if up[i] != up[i-1] and n_entries < max_entries:
            if n_entries == 0:
                cyc_start_bal, cyc_start_t = balance, t[i+1]
            side = 1 if up[i] else -1
            lot = round(base * 2 ** (n_entries // PER_LEVEL), 2)
            entry = o[i+1] + SPREAD if side == 1 else o[i+1]
            pos.append([side, lot, entry])
            n_entries += 1
            max_lot = max(max_lot, sum(p[1] for p in pos))

    open_pnl = None
    if pos and balance >= BUST_LEVEL * START_BALANCE:
        open_pnl = realized + floating(df.close.values[-1])
        cycles.append((cyc_start_t, t[-1], open_pnl, "미청산", n_entries, sum(p[1] for p in pos)))
        balance += open_pnl
    c = pd.DataFrame(cycles, columns=["start", "end", "pnl", "kind", "entries", "lots"])
    return c, balance, min_equity, max_lot


def summarize(c, balance, min_equity, max_lot, df):
    stuck = c[c.kind == "미청산"]
    eq = START_BALANCE + c.pnl.cumsum()
    peak = np.maximum.accumulate(np.r_[START_BALANCE, eq.values])[1:]
    losses = c[c.kind == "손절"]
    first_loss = pd.Timestamp(losses.end.iat[0]).strftime("%m-%d") if len(losses) else "-"
    busted = balance < BUST_LEVEL * START_BALANCE
    return {
        "사이클": len(c) - len(stuck),
        "익절": (c.kind == "익절").sum(),
        "손절": len(losses),
        "첫손절": first_loss,
        "최종잔고": round(balance, 2),
        "MDD": f"{((peak - eq) / peak).max():.0%}" if len(c) else "0%",
        "최저평가": round(min_equity, 2),
        "최대보유lot": round(max_lot, 2),
        "깡통": pd.Timestamp(c.end.iat[-1]).strftime("%m-%d") if busted else "-",
        "미청산": (f"{pd.Timestamp(stuck.start.iat[0]):%m-%d}부터 {stuck.pnl.iat[0]:+.0f}"
                   if len(stuck) else "-"),
    }


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
        for ptp, base, tgt, mx in itertools.product(POS_TP_GRID, BASE_LOTS, TARGETS, MAX_ENTRIES_GRID):
            c, bal, mineq, mlot = run(df, base, tgt, mx, ptp)
            rows.append({"봉": name, "개별TP": ptp or "-", "기본lot": base, "목표$/oz": tgt, "최대진입": mx,
                         **summarize(c, bal, mineq, mlot, df)})
    print()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
