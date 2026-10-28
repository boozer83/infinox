"""금에서 괜찮았던 지표 전략을 다른 종목 12개에도 (2022-01 ~ 2026-10, 같은 규칙·기본 설정)

종목: 금(XAUUSD, 기준), XAUEUR, 은, 나스닥, 독일, 일본, 비트코인, 이더리움, 솔라나, 리플, 도지, BNB, 에이다
봉: data/bars/<종목>_M1_ticks_clean.pkl (틱으로 만든 1분봉 Bid) -> 1시간 / 4시간봉
스프레드: 그 종목 틱의 (월, 시간대)별 중앙값 (data/spread_hourly.csv에 저장). 미끄러짐 = 그 종목 평균 스프레드의 10%
스왑: 금만 넣음 (다른 종목 스왑 값이 없어서 0) -> 다른 종목은 조금 좋게 나옴
전략 (tv_single / tv_guppy / tv_overlap과 같은 계산):
  SuperTrend 10/3, EMA 9/21, Awesome Osc, Guppy 매수+매도(조기 신호 끔), Squeeze Mom (항상 보유),
  RSI 14 30/70 역 (역추세 대표), 지금 추세 봇 (25봉 돌파 / 12봉 청산 / 3 ATR, 구름 있음·없음)

사용법: python -m backtests.tv_symbols
"""
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.download_symbol as ds
import backtests.tv_guppy as tg
import backtests.tv_overlap as tov
import backtests.tv_single as ts

BARS = Path(__file__).resolve().parent.parent / "data" / "bars"
SPREAD_CSV = BARS.parent / "spread_hourly.csv"
SYMS = ["XAUUSD", "XAUEUR", "XAGUSD", "NAS100", "GER40", "JPN225",
        "BTCUSD", "XETUSD", "SOLUSD", "XRPUSD", "DOGUSD", "BNBUSD", "ADAUSD"]
GOLD_SWAP = (ts.SWAP_LONG, ts.SWAP_SHORT)
STRATS = {
    "SuperTrend 10/3": lambda d: ts.supertrend_dir(d),
    "EMA 9/21": ts.INDICATORS["EMA 9/21 교차"][1],
    "Awesome Osc": ts.INDICATORS["Awesome Osc"][1],
    "Guppy 매수+매도": lambda d: tg.signals(d, early=False, both=True),
    "Squeeze Mom": ts.INDICATORS["Squeeze Mom (LazyBear)"][1],
    "RSI 30/70 역": ts.INDICATORS["RSI 14 30/70 역"][1],
}


def spread_table():
    """종목·월·시간대별 스프레드 중앙값"""
    old = pd.read_csv(SPREAD_CSV) if SPREAD_CSV.exists() else pd.DataFrame(columns=["sym", "ym", "hour", "sp"])
    rows = []
    for s in SYMS[1:]:
        if s in set(old.sym):
            continue
        for f in sorted((ds.ROOT / s).glob("*.npz")):
            with np.load(f) as z:
                t = z["t0"][0] + np.cumsum(z["dt"], dtype=np.int64)
                sp = z["sp"] / z["scale"][0]
            hr = (t // 3_600_000) % 24
            g = pd.Series(sp).groupby(hr).median()
            rows += [(s, f.stem, int(h), v) for h, v in g.items()]
        print(f"  스프레드 {s}", flush=True)
    if rows:
        old = pd.concat([old, pd.DataFrame(rows, columns=["sym", "ym", "hour", "sp"])])
        old.to_csv(SPREAD_CSV, index=False)
    return old


def load_sym(s, sp_tab):
    if s == "XAUUSD":
        return ts.load_m1("XAUUSD")
    m1 = pd.read_pickle(BARS / f"{s}_M1_ticks_clean.pkl")
    t = sp_tab[sp_tab.sym == s].set_index(["ym", "hour"]).sp
    key = pd.MultiIndex.from_arrays([m1.time.dt.strftime("%Y-%m"), m1.time.dt.hour])
    m1["sp"] = t.reindex(key).values
    m1["sp"] = m1.sp.ffill().bfill()
    return m1


def main():
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.unicode.east_asian_width", True)
    sp_tab = spread_table()
    out = []
    for s in SYMS:
        m1 = load_sym(s, sp_tab)
        ts.SLIP = 0.02 if s == "XAUUSD" else 0.1 * m1.sp.median()
        ts.SWAP_LONG, ts.SWAP_SHORT = GOLD_SWAP if s == "XAUUSD" else (0.0, 0.0)
        for tf, mins in (("H1", 60), ("H4", 240)):
            d = ts.bars(m1, mins)
            bh = (d.close.iat[-1] / d.close.iat[0] - 1) * 100
            cost = (d.sp / d.close).median() * 100
            cases = {n: ts.trades(d, f(d)) for n, f in STRATS.items()}
            cases["추세 봇 (구름 없이)"] = tov.trend_bot(d, False)[1]
            cases["추세 봇 + 구름"] = tov.trend_bot(d, True)[1]
            for n, tr in cases.items():
                if len(tr) < 10:
                    continue
                out.append({"종목": s, "봉": tf, "전략": n, "보유%": round(bh), "스프레드%": round(cost, 3),
                            **ts.summary(tr)})
        print(f"{s} 끝 (보유 {bh:+.0f}%, 스프레드 {cost:.3f}%)", flush=True)
    res = pd.DataFrame(out)
    yc = [str(y) for y in ts.YEARS]
    res["이긴 해"] = (res[yc] > 1).sum(axis=1)
    res.to_csv("results/tv_symbols.csv", index=False, encoding="utf-8-sig")
    for tf in ("H1", "H4"):
        r = res[res.봉 == tf]
        print(f"\n===== {tf}: 종목별 PF =====")
        print(r.pivot(index="전략", columns="종목", values="PF")[SYMS].to_string())
        print(f"\n===== {tf}: 종목별 매도 PF =====")
        print(r.pivot(index="전략", columns="종목", values="매도PF")[SYMS].to_string())
        print(f"\n===== {tf}: 5년 중 PF 1 넘은 해 =====")
        print(r.pivot(index="전략", columns="종목", values="이긴 해")[SYMS].to_string())


if __name__ == "__main__":
    main()
