"""5분봉 SAR + 필터 조합 백테스트 (실제 틱 체결)

m5_ticks.py와 같은 체결 방식(실제 Bid/Ask 틱, 0.06 lot, 1,000달러 시작)에
진입 필터만 추가한다. 필터는 모두 SAR 전환이 나온 '완성된 봉' 기준 값만 사용 (미래 데이터 안 씀).

필터:
- H1 EMA50   : 마지막으로 완성된 1시간봉 종가가 EMA50 위면 매수만, 아래면 매도만
- H1 SAR     : 마지막으로 완성된 1시간봉 SAR이 위(상승)면 매수만, 아래면 매도만
- ADX>=25    : 5분봉 ADX(14)가 25 이상일 때만 진입 (추세장)
- RSI50      : 매수는 RSI(14) > 50, 매도는 RSI < 50일 때만
- 세션       : 서버시간 09~21시만 (유럽+미국 장)
- 횡보역추세 : ADX < 20이면 SAR 반대 방향으로 진입, ADX >= 25면 SAR 방향 (추세/횡보 모드 흉내)

과최적화 확인용으로 앞 3개월(4/17~7/16)과 뒤 3개월(7/17~10/16) PF를 따로 표시.
사용법: python -m backtests.filters
"""
import itertools

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from backtests.m5_ticks import END, START, load_bars, load_ticks, run, stats
from config import SYMBOL, mt5_init
from indicators import adx, ema, psar, rsi

SPLIT = pd.Timestamp("2026-07-17")
SLTP = [(5.0, 3.0), (7.5, 3.0), (10.0, 3.0)]



def build(m5, h1):
    m5 = m5.copy()
    m5["up"] = psar(m5.high.values, m5.low.values)
    m5["rsi"] = rsi(m5.close)
    m5["adx"] = adx(m5)
    m5["close_time"] = m5.time + pd.Timedelta(minutes=5)

    h1 = h1.copy()
    h1["ema50_dir"] = np.where(h1.close > ema(h1.close, 50), 1, -1)
    h1["sar_dir"] = np.where(psar(h1.high.values, h1.low.values), 1, -1)
    h1["close_time"] = h1.time + pd.Timedelta(hours=1)
    m5 = pd.merge_asof(m5.sort_values("close_time"), h1[["close_time", "ema50_dir", "sar_dir"]],
                       on="close_time", direction="backward")
    return m5.reset_index(drop=True)


FILTERS = {
    "없음":            lambda r, s: s,
    "H1 EMA50":        lambda r, s: s if r.ema50_dir == s else 0,
    "H1 SAR":          lambda r, s: s if r.sar_dir == s else 0,
    "ADX>=25":         lambda r, s: s if r.adx >= 25 else 0,
    "RSI50":           lambda r, s: s if (r.rsi - 50) * s > 0 else 0,
    "세션":            lambda r, s: s if 9 <= r.close_time.hour < 21 else 0,
    "H1 EMA50+ADX":    lambda r, s: s if r.ema50_dir == s and r.adx >= 25 else 0,
    "H1 EMA50+H1 SAR": lambda r, s: s if r.ema50_dir == s and r.sar_dir == s else 0,
    "H1 EMA50+세션":   lambda r, s: s if r.ema50_dir == s and 9 <= r.close_time.hour < 21 else 0,
    "횡보역추세":      lambda r, s: -s if r.adx < 20 else (s if r.adx >= 25 else 0),
    "RSI55/45":        lambda r, s: s if (r.rsi - 50) * s > 5 else 0,
    "RSI50+H1 SAR":    lambda r, s: s if (r.rsi - 50) * s > 0 and r.sar_dir == s else 0,
}


def make_signals(m5, rule):
    t = m5.time.values.astype("datetime64[ms]").astype(np.int64)
    up = m5.up.values
    out = []
    for i in np.nonzero(up[1:-1] != up[:-2])[0] + 1:
        side = rule(m5.iloc[i], 1 if up[i] else -1)
        if side:
            out.append((t[i + 1], side))
    return out


def pf(tr):
    gl = -tr.pnl[tr.pnl < 0].sum()
    return round(tr.pnl[tr.pnl > 0].sum() / gl, 2) if gl else float("inf")


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
    m5 = build(m5, h1)

    rows = []
    for name, rule in FILTERS.items():
        sigs = make_signals(m5, rule)
        for sl, tp in SLTP:
            tr = run(sigs, t, bid, ask, sl, tp)
            s = stats(tr)
            rows.append({"필터": name, "SL": sl, "TP": tp, "거래수": s["거래수"], "하루평균": s["하루평균"],
                         "승률%": s["승률%"], "순손익": s["순손익"], "PF": s["PF"],
                         "앞3개월PF": pf(tr[tr.close < SPLIT]), "뒤3개월PF": pf(tr[tr.close >= SPLIT]),
                         "MDD%": s["MDD%"], "최저잔고": s["최저잔고"]})
        print(f"완료: {name}")
    res = pd.DataFrame(rows)
    res.to_csv("filter_results.csv", index=False, encoding="utf-8-sig")
    print()
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
