"""트레이딩뷰 인기 지표 3개(SuperTrend / Slow Stochastic / High Volume Boxes)를 SAR 단타에 써 보면
(실제 틱, 1,000달러 시작, 1,000달러당 0.05 고정 lot, TP 3 / SL 7.5, 쉬기 규칙·승률 쉬기·금요일 청산은 지금 봇과 같음)

  기준          : 5분봉 SAR 전환 + RSI50 + H1 SAR (지금 SAR 단타)
  ST 진입       : 5분봉 SuperTrend(10, 3) 전환 + RSI50 + H1 SAR
  기준+ST 필터  : 기준 신호 중 5분봉 SuperTrend 방향이 같은 것만
  Stoch 진입    : 5분봉 Slow Stochastic(14,3,3) %K가 %D를 20 아래에서 위로 교차 → 매수, 80 위에서 아래로 → 매도,
                  H1 SAR 방향과 같을 때만 (추세 안의 눌림목)
  기준+Stoch 필터: 기준 신호 중 매수는 %K < 80, 매도는 %K > 20 (과매수에 사지 않기)
  HVB 돌파      : High Volume Boxes 비슷한 버전 (원본 코드가 없어서 직접 만든 근사).
                  5분봉 틱 수가 하루(288봉) 중앙값의 2배를 넘는 봉이 좌우 10봉 고점/저점(피벗)이면 그 가격을 저항/지지로 두고,
                  종가가 가장 최근 저항 위로 마감 → 매수, 지지 아래로 → 매도 (H1 SAR 방향과 같을 때만)
틱 수(거래량 대신)는 틱 파일에서 5분 단위로 세서 data/m5_tickvol.csv에 저장

사용법: python -m backtests.indicator_test [시작 YYYY-MM-DD] [끝 YYYY-MM-DD]   (구간마다 1,000달러로 새로 시작)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.long_period as lb
from backtests.add_positions_ticks import load_m1_recent
from backtests.gold_rest_others import resample
from indicators import atr, psar, rsi

TICKVOL_CSV = Path("data/m5_tickvol.csv")


def supertrend(df, n=10, mult=3.0):
    """KivancOzbilgic SuperTrend과 같은 계산 (ATR Wilder). 각 봉 추세 +1 / -1"""
    a = atr(df, n).values
    src = ((df.high + df.low) / 2).values
    c = df.close.values
    up, dn = src - mult * a, src + mult * a
    tr = np.ones(len(c), int)
    for i in range(1, len(c)):
        if c[i - 1] > up[i - 1]:
            up[i] = max(up[i], up[i - 1])
        if c[i - 1] < dn[i - 1]:
            dn[i] = min(dn[i], dn[i - 1])
        if tr[i - 1] == -1 and c[i] > dn[i - 1]:
            tr[i] = 1
        elif tr[i - 1] == 1 and c[i] < up[i - 1]:
            tr[i] = -1
        else:
            tr[i] = tr[i - 1]
    return tr


def slow_stoch(df, k=14, d=3, smooth=3):
    ll, hh = df.low.rolling(k).min(), df.high.rolling(k).max()
    fast = 100 * (df.close - ll) / (hh - ll)
    sk = fast.rolling(smooth).mean()
    return sk.values, sk.rolling(d).mean().values


def tick_volume():
    """5분봉 틱 수 (봉 시작 시각 -> 개수)"""
    if TICKVOL_CSV.exists():
        v = pd.read_csv(TICKVOL_CSV, parse_dates=["time"])
        return v.set_index("time").vol
    parts = []
    for y, m in lb.months():
        if y < 2024:
            continue
        t, _, _ = lb.load_month(y, m)
        b = t // 300_000
        u, cnt = np.unique(b, return_counts=True)
        parts.append(pd.Series(cnt, index=pd.to_datetime(u * 300_000, unit="ms")))
        del t
        print(f"  틱 수 {y}-{m:02d}", flush=True)
    v = pd.concat(parts)
    v = v.groupby(level=0).sum()
    v.index.name = "time"
    v.rename("vol").reset_index().to_csv(TICKVOL_CSV, index=False)
    return v


def hvb_rows(m5, vol, piv=10, mult=2.0):
    v = vol.reindex(m5.time).fillna(0).values
    med = pd.Series(v).rolling(288, min_periods=100).median().values
    hi, lo, c = m5.high.values, m5.low.values, m5.close.values
    a = atr(m5).values
    end = (m5.time + pd.Timedelta(minutes=5)).values
    res, sup, rows = np.nan, np.nan, []
    for i in range(2 * piv, len(c)):
        p = i - piv                     # i번째 봉이 끝나야 p의 좌우 10봉이 확인됨
        if v[p] > mult * med[p]:
            if hi[p] == hi[p - piv:i + 1].max():
                res = hi[p]
            if lo[p] == lo[p - piv:i + 1].min():
                sup = lo[p]
        if res == res and c[i] > res:
            rows.append((end[i], 1, True, a[i]))
            res = np.nan
        elif sup == sup and c[i] < sup:
            rows.append((end[i], -1, True, a[i]))
            sup = np.nan
    return rows


def run(sig, label):
    """거래가 없거나 너무 적어서 요약이 안 되면 신호 수만"""
    try:
        return fe.run(sig, label)
    except (IndexError, ZeroDivisionError):
        return {"방식": label, "신호": len(sig), "거래": 0}


def main():
    if len(sys.argv) > 1:
        fe.START = pd.Timestamp(sys.argv[1])
    if len(sys.argv) > 2:
        fe.END = pd.Timestamp(sys.argv[2])
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = load_m1_recent()
    m5, h1 = resample(m1, 5), resample(m1, 60)
    del m1
    ctx = fe.context(m5, h1)
    end = m5.time + pd.Timedelta(minutes=5)
    up = psar(m5.high.values, m5.low.values)
    r = rsi(m5.close).values
    a = atr(m5).values
    st = supertrend(m5)
    sk, sd = slow_stoch(m5)

    i = np.nonzero(up[1:] != up[:-1])[0] + 1
    side = np.where(up[i], 1, -1)
    base = [(end.iat[j], s, (r[j] - 50) * s > 0, a[j]) for j, s in zip(i, side)]
    j = np.nonzero(st[1:] != st[:-1])[0] + 1
    st_rows = [(end.iat[k], st[k], (r[k] - 50) * st[k] > 0, a[k]) for k in j]
    st_filt = [x for x, k in zip(base, i) if st[k] == x[1]]
    sto_filt = [x for x, k in zip(base, i) if (sk[k] < 80 if x[1] == 1 else sk[k] > 20)]
    cu = (sk[1:] > sd[1:]) & (sk[:-1] <= sd[:-1]) & (sd[:-1] < 20)
    cd = (sk[1:] < sd[1:]) & (sk[:-1] >= sd[:-1]) & (sd[:-1] > 80)
    sto_rows = [(end.iat[k + 1], 1, True, a[k + 1]) for k in np.nonzero(cu)[0]] + \
               [(end.iat[k + 1], -1, True, a[k + 1]) for k in np.nonzero(cd)[0]]
    sigs = {
        "기준 (SAR 단타)": fe.attach(base, ctx),
        "SuperTrend 진입": fe.attach(st_rows, ctx),
        "기준 + SuperTrend 필터": fe.attach(st_filt, ctx),
        "Slow Stoch 진입 (추세 눌림)": fe.attach(sto_rows, ctx),
        "기준 + Stoch 필터": fe.attach(sto_filt, ctx),
        "HVB 돌파 (근사)": fe.attach(hvb_rows(m5, tick_volume()), ctx),
    }
    res = pd.DataFrame([run(s, n) for n, s in sigs.items()])
    tag = f"{fe.START.date()}_{fe.END.date() if fe.END.year < 2100 else 'now'}"
    res.to_csv(f"indicator_test_{tag}.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
