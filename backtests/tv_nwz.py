"""설명만 받은 지표 2개 (원본 코드 없음, 공개된 일반 계산으로 근사) 를 금에 (2022-01 ~ 2026-10)

Nadaraya-Watson Envelope (LuxAlgo 계열, 다시 그리지 않는 판):
  가운데 = 지난 500봉 종가의 가우시안 가중 평균 (h = 8), 폭 = |종가 - 가운데| 499봉 평균 x 3
  (원본 기본판은 과거 선을 다시 그려서 백테스트가 불가능 -> 실시간과 같은 끝점 계산만 씀)
Reversal Z-Score Bands: z = (종가 - EMA(n)) / 표준편차(n), n = 100 / 200 (원본 기본값 모름)
신호 (사용자 설명대로): 아래 밴드 밖으로 나갔다가 안으로 들어오면 매수, 위 밴드는 매도, 가운데선에서 청산
비교: 밖으로 나가는 순간 바로 진입 (LuxAlgo 삼각형 위치)
체결·비용: tv_single과 같음

사용법: python -m backtests.tv_nwz
"""
import numpy as np
import pandas as pd

import backtests.tv_single as ts
from indicators import ema


def nw_env(d, h=8.0, n=500, mult=3.0):
    w = np.exp(-np.arange(n) ** 2 / (2 * h * h))
    c = d.close.values
    num = np.convolve(c, w)[:len(c)]
    den = np.convolve(np.ones(len(c)), w)[:len(c)]
    mid = num / den
    mid[:n] = np.nan
    mae = pd.Series(np.abs(c - mid)).rolling(n - 1).mean().values * mult
    return mid + mae, mid - mae, mid


def zbands(d, n, k=2.0):
    m = ema(d.close, n)
    sd = d.close.rolling(n).std(ddof=0)
    return (m + k * sd).values, (m - k * sd).values, m.values


def touch(d, up, lo, mid):
    c = d.close.values
    return ts.hold(ts.cross_dn(c, lo), ts.cross_up(c, up), c >= mid, c <= mid)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = ts.load_m1("XAUUSD")
    out = []
    for tf, mins in {"M15": 15, "H1": 60, "H4": 240}.items():
        d = ts.bars(m1, mins)
        bands = {"NW Envelope": nw_env(d), "Z-Score EMA100 ±2": zbands(d, 100), "Z-Score EMA200 ±2": zbands(d, 200)}
        for name, (up, lo, mid) in bands.items():
            for how, want in (("밴드 안으로 돌아오면", ts.band_revert(d, up, lo, mid)), ("밴드 밖으로 나가면", touch(d, up, lo, mid))):
                tr = ts.trades(d, want)
                if len(tr) >= 5:
                    out.append({"봉": tf, "지표": name, "진입": how, **ts.summary(tr)})
        print(f"{tf} 끝", flush=True)
    res = pd.DataFrame(out)
    res.to_csv("results/tv_nwz.csv", index=False, encoding="utf-8-sig")
    print(res.drop(columns=["평균%"]).to_string(index=False))


if __name__ == "__main__":
    main()
