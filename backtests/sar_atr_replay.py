"""실제 SAR 단타 거래(데모·실계좌)를 그대로 두고 손절·익절만 ATR로 바꿨으면 (2026-10-27 사용자 요청, 실제 틱, 읽기만)

- MT5 거래내역에서 magic 20261017 진입·청산을 짝지음 (진입 시각·방향·체결가·lot은 실제 그대로)
- 진입 직전 마감된 5분봉의 ATR(14)로 손절 max(7.5, 1.6 ATR) / 익절 max(3, 1.2 ATR)
- 청산: 손절·익절(매수는 Bid, 매도는 Ask) / 5분봉 SAR 반대 전환 봉 마감 / 금요일 23:30 중 먼저
- 같은 엔진으로 고정 7.5 / 3도 돌려서 실제 결과와 맞는지 확인

사용법: python -m backtests.sar_atr_replay demo 2026-10-17     (live면 실계좌 터미널 C:/MT5_LIVE)
"""
import sys
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from indicators import atr, psar

MAGIC = 20261017
PATHS = {"demo": ("C:/Program Files/Infinox MetaTrader 5 Terminal/terminal64.exe", False),
         "live": ("C:/MT5_LIVE/terminal64.exe", True)}
RULES = {"고정 7.5 / 3": (None, None), "ATR 1.6 / 1.2": (1.6, 1.2)}


def deals(start):
    ds = mt5.history_deals_get(start, datetime.now() + timedelta(days=1)) or []
    opens = {d.position_id: d for d in ds if d.entry == 0 and d.magic == MAGIC}
    rows = []
    for d in ds:
        o = opens.get(d.position_id)
        if d.entry == 1 and o:
            rows.append({"open_ms": o.time_msc, "side": 1 if o.type == 0 else -1, "fill": o.price, "lot": o.volume,
                         "close_ms": d.time_msc, "close": d.price, "profit": d.profit, "reason": d.reason})
    return pd.DataFrame(rows).sort_values("open_ms").reset_index(drop=True)


def main():
    acct, start = sys.argv[1], datetime.fromisoformat(sys.argv[2])
    path, portable = PATHS[acct]
    if not mt5.initialize(path=path, portable=portable):
        raise SystemExit(f"MT5 연결 실패 {mt5.last_error()}")
    sym = next(s.name for s in mt5.symbols_get("XAUUSD*") if s.name in ("XAUUSD", "XAUUSD#"))
    tr = deals(start)
    r = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M5, start - timedelta(days=3), datetime.now() + timedelta(days=1))
    m5 = pd.DataFrame(r)
    m5["time"] = pd.to_datetime(m5.time, unit="s")
    a = atr(m5).values
    up = psar(m5.high.values, m5.low.values)
    end_ms = ((m5.time + pd.Timedelta(minutes=5)).values.astype("datetime64[ms]").astype(np.int64))
    # 서버시간 ms: 봉 time(서버 기준 초)과 거래 time_msc는 같은 서버 시계
    out = []
    for x in tr.itertuples(index=False):
        i = int(np.searchsorted(end_ms, x.open_ms, side="right")) - 1     # 진입 직전 마감된 봉
        atr_now = float(a[i])
        later = np.flatnonzero((end_ms > x.open_ms) & ((up == (x.side == -1))))
        sar_ms = int(end_ms[later[0]]) if later.size else None
        tk = mt5.copy_ticks_range(sym, datetime.utcfromtimestamp(x.open_ms / 1000), datetime.utcfromtimestamp(x.open_ms / 1000 + 3 * 86400),
                                  mt5.COPY_TICKS_INFO)
        t, bid, ask = tk["time_msc"], tk["bid"], tk["ask"]
        d = pd.Timestamp(x.open_ms, unit="ms")
        fri = d.normalize() + pd.Timedelta(days=(4 - d.weekday()) % 7, hours=23, minutes=30)
        fri_ms = int(fri.value // 10**6)
        row = {"진입": d, "방향": "매수" if x.side == 1 else "매도", "lot": x.lot, "ATR": round(atr_now, 2),
               "실제 $": round(x.profit, 2)}
        for name, (ka, kb) in RULES.items():
            sl = 7.5 if ka is None else max(7.5, ka * atr_now)
            tp = 3.0 if kb is None else max(3.0, kb * atr_now)
            X = bid if x.side == 1 else ask
            k = int(np.searchsorted(t, x.open_ms, side="right"))
            hit = np.flatnonzero(((X[k:] - x.fill) * x.side <= -sl) | ((X[k:] - x.fill) * x.side >= tp))
            cands = []
            if hit.size:
                cands.append((int(t[k + hit[0]]), float(X[k + hit[0]]), "SL" if (X[k + hit[0]] - x.fill) * x.side < 0 else "TP"))
            for ms, why in ((sar_ms, "SAR"), (fri_ms, "금요일")):
                if ms is not None:
                    j = int(np.searchsorted(t, ms))
                    if j < len(t):
                        cands.append((int(t[j]), float(X[j]), why))
            if not cands:
                row[f"{name} $"], row[f"{name} 사유"] = np.nan, "끝 못 찾음"
                continue
            ms, px, why = min(cands)
            row[f"{name} $"] = round((px - x.fill) * x.side * x.lot * 100, 2)
            row[f"{name} 사유"] = why
        out.append(row)
    mt5.shutdown()
    res = pd.DataFrame(out)
    res.to_csv(f"results/sar_atr_replay_{acct}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.max_rows", 200)
    print(res.to_string(index=False))
    for c in ("실제 $", "고정 7.5 / 3 $", "ATR 1.6 / 1.2 $"):
        print(f"{c}: 합 {res[c].sum():+.2f}, 승률 {(res[c] > 0).mean() * 100:.1f}% ({len(res)}건)")


if __name__ == "__main__":
    main()
