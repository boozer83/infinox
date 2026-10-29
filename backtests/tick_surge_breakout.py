"""초단타: 틱 급증 + 전고점(전저점) 돌파 (2026-10-29, 사용자가 보낸 주식 스캘핑 코드의 '거래량 폭증 돌파'를 금에 맞춤, 실제 틱)

금 틱에는 거래량이 없어서 거래량 대신 틱 수:
  급증 = 최근 10초 초당 틱 수 / 최근 5분 초당 틱 수 >= M
  돌파 = Bid가 직전 L틱의 최고 Bid를 처음 넘음 (매수) / 최저 Bid 아래로 처음 내려감 (매도). 주식 코드와 달리 매도도
  둘 다 맞는 틱에 신호, 다음 틱 체결 (매수 = Ask, 매도 = Bid)
청산은 backtests.big_bar_scalp.simulate와 같음 (익절 TP$, 손절 SL$ 또는 SL=0이면 진입 때 자리보다 나빠지면, 최대 30분)
한 번에 1개, 결과 $/oz. 거래는 모아두지 않고 (조건, 연도)별 합계만 쌓음
결과 results/tick_surge[_nospread].csv

사용법: python -m backtests.tick_surge_breakout [nospread]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.big_bar_scalp import simulate

RES = Path(__file__).resolve().parent.parent / "results"
NOSPREAD = "nospread" in sys.argv[1:]
LS = (300, 1000)
MS = (2.0, 3.0, 5.0)
TPS = (0.5, 1.0, 2.0)
SLS = (0.0, 0.5, 1.0, 2.0)


def surge_triggers(t, bid, look, mult):
    n10 = np.arange(len(t)) - np.searchsorted(t, t - 10_000)
    n300 = np.arange(len(t)) - np.searchsorted(t, t - 300_000)
    rate = (n10 / 10) / np.maximum(n300 / 300, 1e-9)
    s = pd.Series(bid)
    hi = s.rolling(look).max().shift(1).values
    lo = s.rolling(look).min().shift(1).values
    up, dn = bid > hi, bid < lo
    up &= ~np.r_[False, up[:-1]]                       # 처음 넘은 틱만
    dn &= ~np.r_[False, dn[:-1]]
    surge = rate >= mult
    idx = np.flatnonzero((up | dn) & surge)
    side = np.where(up[idx], 1, -1)
    return idx, side


def main():
    acc = {}
    for y, m in lb.months():
        try:
            t, bid, ask = lb.load_month(y, m)
        except FileNotFoundError:
            continue
        if NOSPREAD:
            ask = bid
        for look in LS:
            for mult in MS:
                idx, side = surge_triggers(t, bid, look, mult)
                sigs = list(zip(idx, side, np.zeros(len(idx), int)))
                for tp in TPS:
                    for sl in SLS:
                        r = simulate(t, bid, ask, sigs, tp, sl)
                        oz = np.array([x[2] for x in r])
                        c = acc.setdefault((look, mult, tp, sl, y), [0, 0, 0.0, 0.0])
                        c[0] += len(oz)
                        c[1] += int((oz > 0).sum())
                        c[2] += float(oz.sum())
                        c[3] += float(sum(x[3] for x in r))
        del t, bid, ask
        print(f"  {y}-{m:02d}", flush=True)
    df = pd.DataFrame([k + tuple(v) for k, v in acc.items()], columns=["L", "M", "TP", "SL", "y", "n", "win", "oz", "secs"])
    keys = ["L", "M", "TP", "SL"]
    g = df.groupby(keys)[["n", "win", "oz", "secs"]].sum()
    res = pd.DataFrame({"건수": g.n, "승률": (g.win / g.n * 100).round(1), "oz합": g.oz.round(1),
                        "oz건": (g.oz / g.n).round(4), "평균초": (g.secs / g.n).round(1)})
    for yr in sorted(df.y.unique()):
        res[f"oz{yr % 100}"] = df[df.y == yr].set_index(keys).oz.round(1)
    res = res.reset_index()
    res.to_csv(RES / f"tick_surge{'_nospread' if NOSPREAD else ''}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 500)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
