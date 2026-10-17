"""RSI50 + 상위봉 SAR 전략 견고성 확인 (실제 틱 체결, 마틴 없음)

기준 설정: 5분봉 SAR(0.02, 0.2), RSI(14) 50 기준, 1시간봉 SAR(0.02, 0.2), TP 3
기준에서 한 가지씩만 바꿔 PF가 유지되는지 본다. 각 줄은 SL 5 / 7.5 / 10의 PF.
진짜 우위라면 설정을 조금 바꿔도 PF가 1보다 확실히 커야 한다.

사용법: python -m backtests.robustness
"""
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from backtests.m5_ticks import END, START, load_bars, load_ticks, run
from config import SYMBOL, mt5_init
from indicators import psar, rsi

SLS = [5.0, 7.5, 10.0]
BASE = dict(sar=(0.02, 0.2), rsi_n=14, rsi_k=0, htf="H1", htf_sar=(0.02, 0.2), tp=3.0, cost=0.0)
SPLIT = pd.Timestamp("2026-07-17")
HTF = {"M15": (mt5.TIMEFRAME_M15, 15), "M30": (mt5.TIMEFRAME_M30, 30),
       "H1": (mt5.TIMEFRAME_H1, 60), "H4": (mt5.TIMEFRAME_H4, 240)}


def signals(m5, htf_bars, p):
    up = psar(m5.high.values, m5.low.values, *p["sar"])
    r = rsi(m5.close, p["rsi_n"]).values
    df, minutes = htf_bars[p["htf"]]
    h = pd.DataFrame({"close_time": df.time + pd.Timedelta(minutes=minutes),
                      "hdir": np.where(psar(df.high.values, df.low.values, *p["htf_sar"]), 1, -1)})
    m = pd.merge_asof(pd.DataFrame({"close_time": m5.time + pd.Timedelta(minutes=5)}), h,
                      on="close_time", direction="backward")
    hdir = m.hdir.values
    t = m5.time.values.astype("datetime64[ms]").astype(np.int64)
    out = []
    for i in np.nonzero(up[1:-1] != up[:-2])[0] + 1:
        s = 1 if up[i] else -1
        if (r[i] - 50) * s > p["rsi_k"] and hdir[i] == s:
            out.append((t[i + 1], s))
    return out


def pf(x):
    gl = -x[x < 0].sum()
    return round(x[x > 0].sum() / gl, 2) if gl else float("inf")


def evaluate(m5, htf_bars, tk, p):
    sigs = signals(m5, htf_bars, p)
    row = {}
    for sl in SLS:
        tr = run(sigs, *tk, sl, p["tp"])
        pnl = tr.pnl - p["cost"] * 0.06 * 100   # 추가 비용(슬리피지): 달러/온스 x 0.06lot
        row[f"PF SL{sl:g}"] = pf(pnl)
        if sl == 7.5:
            row["거래수"] = len(tr)
            row["앞/뒤 PF(SL7.5)"] = f"{pf(pnl[tr.close < SPLIT])} / {pf(pnl[tr.close >= SPLIT])}"
            row["_tr"] = tr.assign(pnl=pnl)
    return row


def main():
    if not mt5_init():
        raise SystemExit(f"MT5 연결 실패: {mt5.last_error()}")
    try:
        tk = load_ticks()
        m5 = load_bars()
        htf_bars = {}
        for k, (tf, minutes) in HTF.items():
            df = pd.DataFrame(mt5.copy_rates_range(SYMBOL, tf, START - pd.Timedelta(days=40), END))
            df["time"] = pd.to_datetime(df["time"], unit="s")
            htf_bars[k] = (df, minutes)
    finally:
        mt5.shutdown()

    tests = [("기준", {})]
    tests += [(f"RSI 기준 {50 + k}/{50 - k}", {"rsi_k": k}) for k in (-5, -2, 2, 5)]
    tests += [(f"RSI 기간 {n}", {"rsi_n": n}) for n in (7, 10, 21)]
    tests += [(f"5분 SAR {a}/{b}", {"sar": (a, b)}) for a, b in ((0.01, 0.1), (0.015, 0.15), (0.03, 0.3))]
    tests += [(f"상위봉 {h}", {"htf": h}) for h in ("M15", "M30", "H4")]
    tests += [(f"상위봉 SAR {a}/{b}", {"htf_sar": (a, b)}) for a, b in ((0.01, 0.1), (0.03, 0.3))]
    tests += [(f"TP {tp:g}", {"tp": tp}) for tp in (2.0, 2.5, 3.5, 4.0)]
    tests += [(f"추가비용 {c}", {"cost": c}) for c in (0.05, 0.10, 0.20)]

    rows, base_tr = [], None
    for name, change in tests:
        r = evaluate(m5, htf_bars, tk, {**BASE, **change})
        if name == "기준":
            base_tr = r["_tr"]
        r.pop("_tr")
        rows.append({"테스트": name, **r})
        print(f"완료: {name}")
    res = pd.DataFrame(rows)[["테스트", "거래수", "PF SL5", "PF SL7.5", "PF SL10", "앞/뒤 PF(SL7.5)"]]
    res.to_csv("robustness_results.csv", index=False, encoding="utf-8-sig")
    print()
    print(res.to_string(index=False))

    m = base_tr.set_index("close").pnl.resample("ME").agg(["count", "sum", pf]).round(2)
    m.columns = ["거래수", "손익(0.06lot)", "PF"]
    print("\n[기준, SL 7.5] 월별")
    print(m.to_string())


if __name__ == "__main__":
    main()
