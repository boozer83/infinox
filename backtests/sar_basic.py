"""XAUUSD 파라볼릭 SAR 전환 전략 백테스트 (일반 모드, 마틴 없음)

규칙 (6.jpg 거래내역에서 재구성):
- SAR(0.02, 0.2)가 아래->위로 전환된 봉이 끝나면 다음 봉 시가에 매수, 위->아래면 매도
- 포지션은 한 번에 1개, 포지션이 있는 동안 새 신호는 무시
- 고정 손절(SL) / 익절(TP), 달러 단위 (금 1달러 = 0.06lot 기준 6달러 손익)
- 한 봉 안에서 SL과 TP가 둘 다 닿으면 SL이 먼저 닿았다고 가정 (보수적)
- 시세는 Bid 기준, 매수는 Ask(= Bid + 스프레드)로 진입하고 Bid로 청산

사용법: python -m backtests.sar_basic   (MT5 터미널이 켜져 있어야 함)
"""
from datetime import datetime, timedelta, timezone
import itertools

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from config import CONTRACT, SYMBOL, mt5_init
from indicators import psar

MONTHS = 6
LOT = 0.06
SPREAD = 0.30           # 달러, 주말엔 실제 스프레드가 0으로 나와서 가정값 사용
START_BALANCE = 1000.0

SL_GRID = [5.0, 7.5, 10.0]
TP_GRID = [2.0, 3.0, 4.0]
TIMEFRAMES = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5}



def load(tf):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=round(MONTHS * 30.5))
    # 1분봉 6개월은 한 번에 요청하면 거부돼서 15일씩 나눠 받음
    parts = []
    s = start
    while s < end:
        e = min(s + timedelta(days=15), end)
        r = mt5.copy_rates_range(SYMBOL, tf, s, e)
        if r is None:
            raise RuntimeError(f"데이터 없음 ({s:%Y-%m-%d}): {mt5.last_error()}")
        parts.append(pd.DataFrame(r))
        s = e
    df = pd.concat(parts).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def run(df, sl, tp):
    o, h, l = df.open.values, df.high.values, df.low.values
    up = psar(h, l)
    trades = []
    pos = None  # (side, entry, idx)
    for i in range(1, len(df) - 1):
        if pos is not None:
            side, entry, _ = pos
            if side == 1:   # 매수: Bid 기준 청산
                hit_sl, hit_tp = l[i] <= entry - sl, h[i] >= entry + tp
            else:           # 매도: Ask(= Bid + 스프레드) 기준 청산
                hit_sl, hit_tp = h[i] + SPREAD >= entry + sl, l[i] + SPREAD <= entry - tp
            if hit_sl or hit_tp:
                pnl = -sl if hit_sl else tp
                trades.append((df.time.iat[pos[2]], df.time.iat[i], side, pnl))
                pos = None
        if pos is None and up[i] != up[i-1]:
            side = 1 if up[i] else -1
            entry = o[i+1] + SPREAD if side == 1 else o[i+1]
            pos = (side, entry, i + 1)
    t = pd.DataFrame(trades, columns=["open", "close", "side", "pnl_usd"])
    t["pnl"] = t.pnl_usd * LOT * CONTRACT
    return t


def stats(t):
    eq = START_BALANCE + t.pnl.cumsum()
    peak = np.maximum.accumulate(np.r_[START_BALANCE, eq.values])[1:]
    mdd = ((peak - eq) / peak).max() if len(t) else 0
    wins = (t.pnl > 0).mean() if len(t) else 0
    gp, gl = t.pnl[t.pnl > 0].sum(), -t.pnl[t.pnl < 0].sum()
    return {
        "거래수": len(t),
        "승률": f"{wins:.1%}",
        "순손익": round(t.pnl.sum(), 2),
        "최종잔고": round(START_BALANCE + t.pnl.sum(), 2),
        "PF": round(gp / gl, 2) if gl else float("inf"),
        "MDD": f"{mdd:.1%}",
        "최저잔고": round(eq.min(), 2) if len(t) else START_BALANCE,
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
        for sl, tp in itertools.product(SL_GRID, TP_GRID):
            t = run(df, sl, tp)
            rows.append({"봉": name, "SL": sl, "TP": tp, **stats(t)})
            if name == "M1" and sl == 7.5 and tp == 3.0:
                t.to_csv("trades_M1_sl7.5_tp3.csv", index=False, encoding="utf-8-sig")
                m = t.set_index("close").pnl.resample("ME").sum().round(2)
                print("\n[M1 SL7.5 TP3] 월별 손익\n" + m.to_string())
    print()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
