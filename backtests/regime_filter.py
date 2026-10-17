"""시장 상태(레짐) 필터 탐색 - 2022~2023년으로 기준을 정하고 2024~2026년으로 검증

기본 전략: long_period.py (RSI50 + H1 SAR, 고정 SL 7.5 / TP 3, 실제 틱 체결)
각 신호 시점에 '이미 완성된' 데이터로만 시장 상태 지표를 계산:
  ATR$      : 일봉 ATR(14), 달러
  ATR%      : 일봉 ATR(14) / 종가 (금값 수준과 무관한 변동성)
  ATR비율   : 일봉 ATR(14) / 일봉 ATR(100) (최근 변동성이 평소보다 큰지)
  일봉ER    : 일봉 효율비율(10일) = |10일 순변화| / 10일 변화 절대값 합 (1에 가까울수록 한 방향 추세)
  H1 ER     : 1시간봉 효율비율(24개)
  일봉ADX   : 일봉 ADX(14)
  추세거리  : (종가 - 일봉 SMA50) / ATR14 의 절대값 (큰 추세 위에 있는지)

1단계: 2022~2023년 거래만 보고 지표마다 '지표 >= 기준' 또는 '지표 <= 기준' 중 PF가 가장 좋은 기준을 고름
       (기준 후보는 2022~2023 분포의 20~80% 분위수, 거래가 25% 이상 남는 것만)
2단계: 그 기준을 그대로 2024 / 2025 / 2026에 적용한 PF를 확인
3단계: 검증을 통과한 필터는 실제로 다시 시뮬레이션해서 월별 수익률 출력

사용법: python -m backtests.regime_filter   (long_period를 먼저 한 번 실행해 틱이 저장돼 있어야 함)
"""
import numpy as np
import pandas as pd

import backtests.long_period as lb
from indicators import adx

TRAIN_END = pd.Timestamp("2024-01-01")
QUANTILES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MIN_KEEP = 0.25


def efficiency(close, n):
    return (close - close.shift(n)).abs() / close.diff().abs().rolling(n).sum()


def regime_table(m5, h1):
    d = m5.set_index("time").resample("D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    atr100 = tr.ewm(alpha=1/100, adjust=False).mean()
    daily = pd.DataFrame({
        "avail": d.time + pd.Timedelta(days=1),   # 그날 일봉이 끝난 뒤부터 사용
        "ATR$": atr14,
        "ATR%": atr14 / d.close * 100,
        "ATR비율": atr14 / atr100,
        "일봉ER": efficiency(d.close, 10),
        "일봉ADX": adx(d),
        "추세거리": ((d.close - d.close.rolling(50).mean()) / atr14).abs(),
    })
    hourly = pd.DataFrame({"avail": h1.time + pd.Timedelta(hours=1), "H1 ER": efficiency(h1.close, 24)})
    return daily, hourly


def attach(times_ms, daily, hourly):
    q = pd.DataFrame({"avail": pd.to_datetime(np.asarray(times_ms), unit="ms").astype("datetime64[us]")})
    daily = daily.assign(avail=daily.avail.astype("datetime64[us]"))
    hourly = hourly.assign(avail=hourly.avail.astype("datetime64[us]"))
    q = pd.merge_asof(q, daily, on="avail", direction="backward")
    q = pd.merge_asof(q, hourly, on="avail", direction="backward")
    return q.drop(columns="avail")


def pf(x):
    gl = -x[x < 0].sum()
    return x[x > 0].sum() / gl if gl else np.inf


def main():
    m5 = lb.bars_from_ticks(5)
    h1 = lb.bars_from_ticks(60)
    daily, hourly = regime_table(m5, h1)
    sigs = lb.signals(m5, h1)

    # 1단계용: 필터 없이 시뮬레이션한 거래에 지표를 붙임 (필터가 거래를 빼면 이후 진입이 조금 달라지는 건 3단계에서 정확히 다시 계산)
    base = lb.simulate(sigs, scaled=False)
    feats = attach(base.open_ms.values, daily, hourly)
    base = pd.concat([base.reset_index(drop=True), feats], axis=1)
    train = base[base.close < TRAIN_END]
    years = [2024, 2025, 2026]
    print(f"기본 전략 PF  2022-23: {pf(train.pnl_oz):.3f}  " +
          "  ".join(f"{y}: {pf(base.pnl_oz[base.close.dt.year == y]):.3f}" for y in years) + "\n")

    rows = []
    for f in feats.columns:
        best = None
        for q in QUANTILES:
            th = train[f].quantile(q)
            for op in (">=", "<="):
                keep = train[f] >= th if op == ">=" else train[f] <= th
                if keep.mean() < MIN_KEEP:
                    continue
                p = pf(train.pnl_oz[keep])
                if best is None or p > best[0]:
                    best = (p, op, th, keep.mean())
        p_tr, op, th, kept = best
        row = {"지표": f, "규칙": f"{op} {th:.3g}", "남는비율": f"{kept:.0%}", "PF 22-23": round(p_tr, 3)}
        for y in years:
            g = base[base.close.dt.year == y]
            keep = g[f] >= th if op == ">=" else g[f] <= th
            row[f"PF {y}"] = round(pf(g.pnl_oz[keep]), 3)
            row[f"거래 {y}"] = int(keep.sum())
        test = base[base.close >= TRAIN_END]
        keep = test[f] >= th if op == ">=" else test[f] <= th
        row["PF 24-26"] = round(pf(test.pnl_oz[keep]), 3)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv("regime_results.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))

    # 3단계: 2022-23 PF가 1을 넘은 필터만 정확히 다시 시뮬레이션
    for _, r in res[res["PF 22-23"] > 1.0].iterrows():
        f, (op, th) = r["지표"], r["규칙"].split(" ")
        th = float(th)
        sf = attach([s[0] for s in sigs], daily, hourly)[f].values
        keep = sf >= th if op == ">=" else sf <= th
        tr = lb.simulate([s for s, k in zip(sigs, keep) if k], scaled=False)
        m = tr.groupby("month").pnl_oz.sum() * lb.FIXED_LOT * lb.CONTRACT / 1000 * 100
        p = pd.DataFrame({"y": m.index.year, "mo": m.index.month, "pct": m.values}).pivot(
            index="y", columns="mo", values="pct").round(1)
        p["연간"] = p.sum(axis=1).round(1)
        eq = 1000 + (tr.pnl_oz * lb.FIXED_LOT * lb.CONTRACT).cumsum()
        print(f"\n===== {f} {op} {th:.3g} (정확 재계산) =====")
        print(f"거래 {len(tr):,}건, PF {pf(tr.pnl_oz):.3f}, 최저잔고 {eq.min():.0f}, "
              f"손실월 {(m < 0).sum()}/{len(m)}")
        print("월별 수익률 % (0.02lot, 시작 1,000달러 대비)")
        print(p.to_string())


if __name__ == "__main__":
    main()
