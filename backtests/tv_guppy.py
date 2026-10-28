"""Super Guppy Strategy (트레이딩뷰, Daryl Guppy EMA 리본) 를 금에 (2022-01 ~ 2026-10)

원본 규칙: 느린 EMA 15개(24, 27, ..., 66)가 큰 순서대로 정렬 = 초록, 반대 = 빨강, 아니면 회색
  조기 신호 켜짐(기본): 매수 = 회색->초록 또는 빨강->회색, 청산 = 초록->회색 또는 회색->빨강
  조기 신호 꺼짐       : 매수 = 회색->초록, 청산 = 초록->회색
  원본 strategy 코드는 매수만 (useShorts는 표시용). 대칭판: 조기 신호 매도 신호에 매도로 갈아탐 / 빨강 구간 매도
비용·체결: tv_single과 같음 (다음 봉 시가, 실제 스프레드, 미끄러짐 0.02, 스왑)
비교: 같은 거래 수·보유 봉 수의 무작위 매수 (tv_sqz.random_baseline)

사용법: python -m backtests.tv_guppy
"""
import numpy as np
import pandas as pd

import backtests.tv_single as ts
from backtests.tv_sqz import random_baseline
from indicators import ema

TFS = {"M15": 15, "H1": 60, "H4": 240, "D1": 1440}
SLOW = range(24, 67, 3)


def ribbon(d):
    e = np.column_stack([ema(d.close, n).values for n in SLOW])
    up = (np.diff(e, axis=1) < 0).all(axis=1)          # 짧은 EMA가 더 큼
    dn = (np.diff(e, axis=1) > 0).all(axis=1)
    return np.where(up, 1, np.where(dn, -1, 0))


def signals(d, early, both):
    c = ribbon(d)
    p = np.r_[0, c[:-1]]
    if early:
        go_long = ((c == 1) & (p == 0)) | ((c == 0) & (p == -1))
        go_short = ((c == 0) & (p == 1)) | ((c == -1) & (p == 0))
        if both:
            return ts.hold(go_long, go_short)
        return ts.hold(go_long, np.zeros(len(d), bool), go_short)
    open_l, close_l = (c == 1) & (p == 0), (c == 0) & (p == 1)
    open_s, close_s = (c == -1) & (p == 0), (c == 0) & (p == -1)
    if both:
        return ts.hold(open_l, open_s, close_l, close_s)
    return ts.hold(open_l, np.zeros(len(d), bool), close_l)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = ts.load_m1("XAUUSD")
    rng = np.random.default_rng(1)
    out = []
    for tf, mins in TFS.items():
        d = ts.bars(m1, mins)
        for early in (True, False):
            for both in (False, True):
                tr = ts.trades(d, signals(d, early, both))
                if len(tr) < 10:
                    continue
                row = {"봉": tf, "조기 신호": "켬(기본)" if early else "끔",
                       "방식": "매수+매도" if both else "매수만 (원본)", **ts.summary(tr)}
                if not both:
                    med, p95 = random_baseline(d, tr, rng)
                    row["무작위매수 PF 중간"], row["무작위 상위5%"] = round(med, 2), round(p95, 2)
                out.append(row)
        print(f"{tf} 끝", flush=True)
    res = pd.DataFrame(out)
    res.to_csv("results/tv_guppy.csv", index=False, encoding="utf-8-sig")
    print(res.drop(columns=["평균%"]).to_string(index=False))


if __name__ == "__main__":
    main()
