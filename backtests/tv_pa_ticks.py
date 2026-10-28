"""PA Master Grade-A (금 1시간봉) 를 실제 틱으로 체결 (2022-01 ~ 2026-10)

신호는 tv_batch2.pa_master (1분봉 zip으로 만든 1시간봉 + 틱 수 거래량) 그대로.
체결: 신호 다음 봉 시작 시각 이후 첫 틱 (매수 Ask / 매도 Bid). 진입가가 이미 손절선 너머면 건너뜀
      손절: 매수는 Bid가 손절가 이하가 된 첫 틱의 Bid (갭이면 더 나쁜 값), 매도는 Ask가 손절가 이상이 된 첫 틱의 Ask
      익절: 지정가 (매수는 Bid가 익절가에 닿으면 익절가, 매도는 Ask가 닿으면 익절가)
      한 번에 한 거래, 스왑 포함 (tv_single 값). 미끄러짐은 틱 값 그대로 (추가 없음)
비교: 같은 신호의 봉 체결 (tv_batch2.rr_trades)

사용법: python -m backtests.tv_pa_ticks
"""
import numpy as np
import pandas as pd

import backtests.download_symbol as ds
import backtests.long_period as lb
import backtests.tickvol_m1 as tv
import backtests.tv_single as ts
from backtests.tv_batch2 import bars_vol, pa_master, rr_trades

_cache = {}


def month_ticks(y, m, symbol="XAUUSD"):
    key = (symbol, y, m)
    if key not in _cache:
        if len(_cache) >= 2:
            _cache.pop(next(iter(_cache)))
        _cache[key] = lb.load_month(y, m) if symbol == "XAUUSD" else ds.load_month(symbol, y, m)
    return _cache[key]


def next_month(y, m):
    return (y + 1, 1) if m == 12 else (y, m + 1)


def first_hit(t, bid, ask, k, side, sl, tp):
    """k번째 틱부터 처음 손절/익절. (위치, 가격, 이유) 또는 None"""
    if side == 1:
        s = np.flatnonzero(bid[k:] <= sl)
        p = np.flatnonzero(bid[k:] >= tp)
    else:
        s = np.flatnonzero(ask[k:] >= sl)
        p = np.flatnonzero(ask[k:] <= tp)
    i_s = s[0] if s.size else None
    i_p = p[0] if p.size else None
    if i_s is None and i_p is None:
        return None
    if i_p is None or (i_s is not None and i_s <= i_p):
        j = k + i_s
        return j, (bid[j] if side == 1 else ask[j]), "손절"
    return k + i_p, tp, "익절"


def tick_trades(d, side, stop, rr, symbol="XAUUSD"):
    times = d.time.values
    rows, busy = [], np.datetime64("1970-01-01")
    last = lb.months()
    last = list(last)[-1]
    for i in np.flatnonzero(side[:-1]):
        s, sl = int(side[i]), stop[i]
        T = times[i + 1]
        if T < busy:
            continue
        ts_ms = T.astype("datetime64[ms]").astype(np.int64)
        y, m = pd.Timestamp(T).year, pd.Timestamp(T).month
        t, bid, ask = month_ticks(y, m, symbol)
        k = int(np.searchsorted(t, ts_ms))
        if k >= len(t):
            y, m = next_month(y, m)
            if (y, m) > last:
                break
            t, bid, ask = month_ticks(y, m, symbol)
            k = 0
        ent = ask[k] if s == 1 else bid[k]
        et = t[k]
        if (ent - sl) * s <= 0:
            continue
        tp = ent + s * rr * abs(ent - sl)
        while True:
            h = first_hit(t, bid, ask, k, s, sl, tp)
            if h is not None:
                j, px, why = h
                xt = t[j]
                break
            y, m = next_month(y, m)
            if (y, m) > last:
                h = None
                break
            t, bid, ask = month_ticks(y, m, symbol)
            k = 0
        if h is None:
            break
        e64, x64 = np.datetime64(int(et), "ms"), np.datetime64(int(xt), "ms")
        pnl = (px - ent) * s + ts.swap_nights(e64, x64) * (ts.SWAP_LONG if s == 1 else ts.SWAP_SHORT)
        rows.append((e64, x64, s, ent, pnl / ent * 100, why, pnl / abs(ent - sl)))
        busy = x64
    return pd.DataFrame(rows, columns=["open", "close", "side", "entry", "pct", "why", "r"])


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = ts.load_m1("XAUUSD")
    vol = tv.load()
    h1 = bars_vol(m1, vol, 60)
    side, stop, _, _ = pa_master(h1)
    out = []
    for rr in (1.0, 2.0, 3.0):
        for how, tr in (("봉 체결", rr_trades(h1, side, stop, rr)), ("틱 체결", tick_trades(h1, side, stop, rr))):
            out.append({"익절": f"{rr:.0f}R", "체결": how, **ts.summary(tr)})
            if how == "틱 체결":
                tr.to_csv(f"results/tv_pa_ticks_{rr:.0f}r.csv", index=False, encoding="utf-8-sig")
        print(f"{rr:.0f}R 끝", flush=True)
    print(pd.DataFrame(out).drop(columns=["평균%"]).to_string(index=False))


if __name__ == "__main__":
    main()
