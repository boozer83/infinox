"""XAUUSD 5분봉 SAR 일반 모드 백테스트 - 실제 틱으로 체결

규칙:
- 5분봉 SAR(0.02, 0.2)가 전환된 봉이 끝나면, 다음 봉 시작 시각의 첫 틱에 진입
  (매수는 Ask, 매도는 Bid -> 스프레드는 실제 호가 그대로)
- 포지션은 한 번에 1개, 포지션이 있는 동안 새 신호는 무시
- 고정 손절/익절(달러/온스). 매수는 Bid, 매도는 Ask가 처음 닿는 틱에서 그 틱 가격으로 청산
- 시작 잔고 1,000달러, 수량 0.06 lot 고정

틱은 처음 한 번 MT5에서 받아 data/ 폴더에 저장해 두고 다음부터는 저장본을 사용.
사용법: python -m backtests.m5_ticks
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import itertools

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from config import CONTRACT, SYMBOL, mt5_init
from indicators import psar

START = datetime(2026, 4, 17, tzinfo=timezone.utc)
END = datetime(2026, 10, 17, tzinfo=timezone.utc)
LOT = 0.06
START_BALANCE = 1000.0
SL_GRID = [5.0, 7.5, 10.0]
TP_GRID = [2.0, 3.0, 4.0]
DATA = Path(__file__).resolve().parent.parent / "data"


def load_ticks():
    cache = DATA / f"ticks_{SYMBOL}_{START:%Y%m%d}_{END:%Y%m%d}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["t"], z["bid"], z["ask"]
    parts = []
    d = START
    while d < END:
        e = min(d + timedelta(days=7), END)
        t = mt5.copy_ticks_range(SYMBOL, d, e, mt5.COPY_TICKS_INFO)
        if t is None:
            raise RuntimeError(f"틱 데이터 없음 ({d:%Y-%m-%d}): {mt5.last_error()}")
        # 구간 경계의 틱이 두 번 들어오지 않도록 끝 시각 미만만 사용
        parts.append(t[t["time_msc"] < int(e.timestamp() * 1000)][["time_msc", "bid", "ask"]])
        d = e
    a = np.concatenate(parts)
    DATA.mkdir(exist_ok=True)
    np.savez(cache, t=a["time_msc"], bid=a["bid"], ask=a["ask"])
    return a["time_msc"], a["bid"], a["ask"]


def load_bars():
    r = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M5, START, END)
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def signals(df):
    """(진입 시각 ms, 방향) 목록: SAR 전환 봉 다음 봉의 시작 시각"""
    up = psar(df.high.values, df.low.values)
    t = df.time.values.astype("datetime64[ms]").astype(np.int64)
    idx = np.nonzero(up[1:-1] != up[:-2])[0] + 1
    return [(t[i + 1], 1 if up[i] else -1) for i in idx]


def first_hit(arr, start, lo, hi, chunk=50_000):
    """start부터 arr <= lo 또는 arr >= hi인 첫 인덱스 (없으면 -1)"""
    n = len(arr)
    while start < n:
        seg = arr[start:start + chunk]
        k = np.flatnonzero((seg <= lo) | (seg >= hi))
        if k.size:
            return start + k[0]
        start += chunk
    return -1


def run(sigs, t, bid, ask, sl, tp):
    trades = []
    busy_until = -1
    for st, side in sigs:
        if st < busy_until:
            continue
        k = np.searchsorted(t, st)
        if k >= len(t) - 1:
            break
        if side == 1:
            e = ask[k]
            j = first_hit(bid, k + 1, e - sl, e + tp)
            if j < 0:
                break
            pnl = bid[j] - e
        else:
            e = bid[k]
            j = first_hit(ask, k + 1, e - tp, e + sl)
            if j < 0:
                break
            pnl = e - ask[j]
        trades.append((t[k], t[j], side, e, float(pnl)))
        busy_until = t[j]
    df = pd.DataFrame(trades, columns=["open_ms", "close_ms", "side", "entry", "pnl_oz"])
    df["open"] = pd.to_datetime(df.open_ms, unit="ms")
    df["close"] = pd.to_datetime(df.close_ms, unit="ms")
    df["pnl"] = df.pnl_oz * LOT * CONTRACT
    return df.drop(columns=["open_ms", "close_ms"])


def stats(tr):
    eq = START_BALANCE + tr.pnl.cumsum()
    peak = np.maximum.accumulate(np.r_[START_BALANCE, eq.values])[1:]
    gp, gl = tr.pnl[tr.pnl > 0].sum(), -tr.pnl[tr.pnl < 0].sum()
    days = max((tr.close.iat[-1] - tr.open.iat[0]).days, 1)
    return {
        "거래수": len(tr),
        "하루평균": round(len(tr) / days * 7 / 5, 1),
        "승률%": round((tr.pnl > 0).mean() * 100, 1),
        "평균익절$": round(tr.pnl[tr.pnl > 0].mean(), 2),
        "평균손절$": round(tr.pnl[tr.pnl < 0].mean(), 2),
        "순손익": round(tr.pnl.sum(), 2),
        "PF": round(gp / gl, 2) if gl else float("inf"),
        "MDD%": round(((peak - eq) / peak).max() * 100, 1),
        "최저잔고": round(eq.min(), 2),
    }


def main():
    if not mt5_init():
        raise SystemExit(f"MT5 연결 실패: {mt5.last_error()}")
    try:
        t, bid, ask = load_ticks()
        bars = load_bars()
    finally:
        mt5.shutdown()
    print(f"틱 {len(t):,}개, 평균 스프레드 {np.mean(ask - bid):.3f} / 5분봉 {len(bars):,}개")
    sigs = signals(bars)
    print(f"SAR 신호 {len(sigs):,}개\n")

    rows, best = [], None
    for sl, tp in itertools.product(SL_GRID, TP_GRID):
        tr = run(sigs, t, bid, ask, sl, tp)
        s = stats(tr)
        rows.append({"SL": sl, "TP": tp, **s})
        if best is None or s["순손익"] > best[0]:
            best = (s["순손익"], sl, tp, tr)
    print(pd.DataFrame(rows).to_string(index=False))

    _, sl, tp, tr = best
    tr.to_csv(f"trades_m5_ticks_sl{sl}_tp{tp}.csv", index=False, encoding="utf-8-sig")
    m = tr.set_index("close").pnl.resample("ME").agg(["count", "sum"]).round(2)
    m.columns = ["거래수", "손익"]
    print(f"\n[가장 좋은 설정 SL {sl} / TP {tp}] 월별")
    print(m.to_string())


if __name__ == "__main__":
    main()
