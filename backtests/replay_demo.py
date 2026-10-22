"""데모 계좌 실제 SAR 단타 거래 vs 같은 기간 틱 백테스트 비교 (서버 시간)

- 1분봉: MT5에서 직접 받음 (봇이 보는 봉과 같게), 틱: data/ticks/YYYY-MM.npz (이번 달은 새로 받음)
- 규칙 두 가지로 돌림:
    지금 규칙 : 반대 콩 청산 + 과열 거르기 (ext 0.8 / risk 2.8)
    옛 규칙   : 반대 콩 청산 없음, 과열 거르기 없음 (10/21 아침 변경 전)
- 데모 체결(magic 20261017)과 진입 시각/방향/결과를 나란히 출력

사용법: python -m backtests.replay_demo 2026-10-19
"""
import sys
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.long_period as lb
from backtests.ext_filter import features
from backtests.gold_rest_others import (FRIDAY_NO_ENTRY, MAX_H1_FLIPS, MIN_ATR, NO_ENTRY_FROM, NO_ENTRY_UNTIL,
                                        resample)
from backtests.sar_exit import against_times
from backtests.winrate_ideas import trade
from config import SYMBOL, mt5_init

MAGIC = 20261017


def demo_trades(start):
    ds = mt5.history_deals_get(start.to_pydatetime(), datetime.now() + timedelta(days=1)) or []
    opens = {d.position_id: d for d in ds if d.entry == 0 and d.magic == MAGIC}
    rows = []
    for d in ds:
        o = opens.get(d.position_id)
        if d.entry == 1 and o:
            side = 1 if o.type == 0 else -1
            rows.append({"진입": pd.Timestamp(o.time, unit="s"), "방향": side, "진입가": o.price,
                         "청산": pd.Timestamp(d.time, unit="s"), "청산가": d.price,
                         "$/oz": round((d.price - o.price) * side, 2), "사유": d.comment})
    return pd.DataFrame(rows)


def replay(sig, t, bid, ask, against, ext_on):
    busy, rows = 0, []
    for s in sig.itertuples(index=False):
        ms = int(np.datetime64(s.t, "ms").astype(np.int64))
        why = None
        mins = s.t.hour * 60 + s.t.minute
        if ms < busy:
            why = "포지션 보유 중"
        elif (s.t.weekday() == 4 and mins >= FRIDAY_NO_ENTRY) or NO_ENTRY_FROM <= mins < NO_ENTRY_UNTIL:
            why = "진입 금지 시간"
        elif s.flips >= MAX_H1_FLIPS:
            why = "1시간 SAR 전환 많음"
        elif s.atr < MIN_ATR:
            why = "ATR 작음"
        elif s.ref == s.ref and s.atr > 2.0 * s.ref:
            why = "ATR 큼"
        elif ext_on and s.ext > 0.8:
            why = f"과열 ext {s.ext:.2f}"
        elif ext_on and s.risk > 2.8:
            why = f"과열 risk {s.risk:.2f}"
        if why:
            rows.append({"신호": s.t, "방향": s.side, "결과": why})
            continue
        k = int(np.searchsorted(t, ms))
        d = s.t.normalize() + pd.Timedelta(days=(4 - s.t.weekday()) % 7, hours=23, minutes=30)
        fc = int(np.searchsorted(t, int(np.datetime64(d, "ms").astype(np.int64))))
        res = trade(t, bid, ask, k, s.side, fc, against, {}, None)
        if res is None:
            rows.append({"신호": s.t, "방향": s.side, "결과": "아직 안 끝남"})
            continue
        k, j, oz = res
        busy = int(t[j])
        rows.append({"신호": s.t, "방향": s.side, "진입": pd.Timestamp(int(t[k]), unit="ms"),
                     "진입가": ask[k] if s.side == 1 else bid[k], "청산": pd.Timestamp(int(t[j]), unit="ms"),
                     "$/oz": round(oz, 2), "결과": "익절" if oz >= 2.9 else "손절" if oz <= -7.4 else "반대콩"})
    return pd.DataFrame(rows)


def main():
    start = pd.Timestamp(sys.argv[1])
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.max_rows", 500)
    mt5_init()
    lb.download_all()
    r = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, (start - pd.Timedelta(days=40)).to_pydatetime(),
                             datetime.now(timezone.utc) + timedelta(hours=12))
    m1 = pd.DataFrame(r)[["time", "open", "high", "low", "close"]]
    m1["time"] = pd.to_datetime(m1.time, unit="s")
    print(f"1분봉 {m1.time.iat[0]} ~ {m1.time.iat[-1]}")
    m5, h1 = resample(m1, 5), resample(m1, 60)
    fe.START, fe.END = start, pd.Timestamp("2100-01-01")
    sig = fe.bar_close_signals(m5, 5, 0.02, fe.context(m5, h1), True).merge(features(m5), on="t", how="left")
    against = against_times(m5, 5)
    no_against = {1: np.array([], np.int64), -1: np.array([], np.int64)}
    t, bid, ask = lb.load_month(start.year, start.month)
    print(f"틱 ~ {pd.Timestamp(int(t[-1]), unit='ms')}")
    demo = demo_trades(start)
    print("\n=== 데모 실제 거래 ===")
    print(demo.to_string(index=False))
    for name, ag, ext in (("지금 규칙", against, True), ("옛 규칙", no_against, False)):
        print(f"\n=== 백테스트: {name} ===")
        print(replay(sig, t, bid, ask, ag, ext).to_string(index=False))
    mt5.shutdown()


if __name__ == "__main__":
    main()
