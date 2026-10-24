"""GPT 봇용 데이터 폴더 만들기 (비밀정보 없이 시세만): ../gpt_data/
종목마다 1분봉 / 5분봉 / 4시간봉 csv.gz, Bid 기준 OHLC + 평균 스프레드(ask-bid) + 틱 수, 시각은 서버시간

사용법: python -m backtests.gpt_export            (전 종목, 이미 만든 종목은 건너뜀)
        python -m backtests.gpt_export XAUUSD     (지정 종목만 다시)
"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.others_scan import load, months_of

OUT = Path(__file__).resolve().parents[2] / "gpt_data"
SYMBOLS = ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD", "XAUEUR", "USOUSD", "UKOUSD", "BTCUSD", "XETUSD", "XRPUSD",
           "US30", "NAS100", "SPX500", "GER40", "UK100", "JPN225",
           "EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "EURJPY", "AUDUSD"]


def m1(sym):
    out = []
    for y, m in months_of(sym):
        t, bid, ask = load(sym, y, m)
        k = t // 60_000
        s = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        e = np.r_[s[1:], len(k)] - 1
        n = np.diff(np.r_[s, len(k)])
        out.append(pd.DataFrame({"open": bid[s], "high": np.maximum.reduceat(bid, s),
                                 "low": np.minimum.reduceat(bid, s), "close": bid[e],
                                 "spread": np.add.reduceat(ask - bid, s) / n, "ticks": n}, index=k[s]))
        del t, bid, ask, k, s, e, n
        gc.collect()
    b = pd.concat(out)
    b = b[~b.index.duplicated()]
    b.index = pd.to_datetime(b.index.values * 60, unit="s")
    b.index.name = "time"
    return b


def resample(b, rule):
    r = b.resample(rule, label="left", closed="left")
    x = pd.DataFrame({"open": r.open.first(), "high": r.high.max(), "low": r.low.min(), "close": r.close.last(),
                      "spread": r.spread.mean(), "ticks": r.ticks.sum()})
    return x.dropna(subset=["open"])


def export(sym):
    d = OUT / sym
    d.mkdir(parents=True, exist_ok=True)
    b = m1(sym)
    digits = 2 if b.close.iat[-1] > 50 else 5
    for name, x in (("M1", b), ("M5", resample(b, "5min")), ("H4", resample(b, "4h"))):
        x.to_csv(d / f"{name}.csv.gz", float_format=f"%.{digits + 1}f", compression="gzip")
    print(f"{sym}: {b.index[0]:%Y-%m-%d} ~ {b.index[-1]:%Y-%m-%d}, 1분봉 {len(b):,}개", flush=True)
    del b
    gc.collect()


def main():
    syms = sys.argv[1:] or [s for s in SYMBOLS if not (OUT / s / "H4.csv.gz").exists()]
    for s in syms:
        export(s)


if __name__ == "__main__":
    main()
