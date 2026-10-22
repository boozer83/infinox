"""장세(추세 / 일반 / 횡보) 판단 지표 비교 (2022-01 ~ 2026-10, 5분봉 마감마다)

HK 자동주행모드는 장세로 분할 수·lot을 바꿈 (추세 1 / 일반 2 / 횡보 3). 판단 기준은 ❓이라 여러 지표를 비교.
지표 (모두 그 시각까지 완성된 봉만 사용, 클수록 '추세' 쪽이 되게 부호를 맞춤):
  ADX 1시간 / 15분 / 5분, 직진도(효율비) 2시간 / 4시간 / 하루, 볼린저 폭(1시간, 100봉 중앙값 대비),
  ATR 비율(5분 ATR / 20일 중앙값), Choppiness(1시간, 작을수록 추세 -> 부호 반대), 1시간 SAR 전환 수(24시간, 부호 반대),
  4시간 구름 밖 거리(ATR 단위), EMA50 기울기(1시간, ATR 단위), 회귀 R²(2시간), 분산비(5분 6봉, 하루),
  지금 규칙 (hedge_martin_v2.regime_at: ADX 30/20 + 직진도 0.35/0.12)
평가 1 (앞으로 맞히나): 다음 2시간·4시간 직진도, 다음 2시간 폭/ATR 과의 순위 상관 (연도별 최소값도),
  지표 상위 1/3 -> 실제 다음 2시간이 상위 1/3(추세)일 확률, 지표 하위 1/3 -> 실제 하위 1/3(횡보)일 확률 (아무렇게나 하면 33%)
평가 2 (마틴에 쓸모 있나): results/martin_cycles_*.csv (martin_split cycles 모드, 장세와 무관하게 분할 3 고정)의
  사이클 시작 시각 지표 1/3 구간별 한도·동결 비율, 5마틴 비율, 평균 손익%

사용법: python -m backtests.regime_check    (results/regime_check.csv, regime_cycles.csv)
"""
import zipfile

import numpy as np
import pandas as pd

import backtests.hedge_martin_v2 as hm2
from backtests.gold_rest_others import BARS, resample
from indicators import adx, atr, psar

OUT = BARS.parent.parent / "results"
YEARS = range(2022, 2027)


def load_m1():
    parts = []
    for y in YEARS:
        with zipfile.ZipFile(BARS / f"XAUUSD_M1_{y}.csv.zip") as z:
            parts.append(pd.read_csv(z.open(z.namelist()[0]), usecols=["time", "open", "high", "low", "close"]))
    m1 = pd.concat(parts)
    m1["time"] = pd.to_datetime(m1.time)
    return m1.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def eff(c, n):
    return (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()


def r2(c, n):
    x = np.arange(n) - (n - 1) / 2
    sxx = (x ** 2).sum()
    cov = c.rolling(n).apply(lambda v: (x * (v - v.mean())).sum(), raw=True)
    var = c.rolling(n).var(ddof=0) * n
    return cov ** 2 / (sxx * var)


def on_end(df, minutes, cols):
    """상위 봉 지표를 봉 마감 시각 기준으로"""
    return pd.DataFrame({"end": df.time + pd.Timedelta(minutes=minutes), **cols})


def features(m1):
    m5, m15, h1, h4 = resample(m1, 5), resample(m1, 15), resample(m1, 60), resample(m1, 240)
    c5 = m5.close
    a5 = atr(m5)
    f = pd.DataFrame({"t": m5.time + pd.Timedelta(minutes=5)})
    f["ADX 5분"] = adx(m5).values
    f["직진도 2시간"] = eff(c5, 24).values
    f["직진도 4시간"] = eff(c5, 48).values
    f["직진도 하루"] = eff(c5, 288).values
    f["ATR 비율"] = (a5 / a5.rolling(5760, min_periods=1000).median()).values
    f["회귀 R² 2시간"] = r2(c5, 24).values
    r1 = c5.diff()
    f["분산비 (6봉)"] = ((c5 - c5.shift(6)).rolling(288).var() / (6 * r1.rolling(288).var())).values
    # 상위 봉
    ah1 = atr(h1)
    bb = h1.close.rolling(20).std() * 4 / h1.close
    tr = pd.concat([h1.high - h1.low, (h1.high - h1.close.shift()).abs(), (h1.low - h1.close.shift()).abs()], axis=1).max(axis=1)
    chop = 100 * np.log10(tr.rolling(14).sum() / (h1.high.rolling(14).max() - h1.low.rolling(14).min())) / np.log10(14)
    up = psar(h1.high.values, h1.low.values)
    flips = pd.Series(np.r_[False, up[1:] != up[:-1]].astype(int)).rolling(24).sum()
    e50 = h1.close.ewm(span=50, adjust=False).mean()
    hh = on_end(h1, 60, {"ADX 1시간": adx(h1).values, "볼린저 폭 1시간": (bb / bb.rolling(100).median()).values,
                         "Choppiness 1시간 (반대)": -chop.values, "1시간 SAR 전환 24h (반대)": -flips.values,
                         "EMA50 기울기 1시간": ((e50 - e50.shift(5)).abs() / ah1).values})
    m15f = on_end(m15, 15, {"ADX 15분": adx(m15).values})
    mid = lambda d, n: (d.high.rolling(n).max() + d.low.rolling(n).min()) / 2
    top = pd.concat([((mid(h4, 9) + mid(h4, 26)) / 2).shift(26), mid(h4, 52).shift(26)], axis=1)
    hi, lo = top.max(axis=1), top.min(axis=1)
    out = np.maximum(h4.close - hi, lo - h4.close).clip(lower=0) / atr(h4)
    h4f = on_end(h4, 240, {"4시간 구름 밖 거리": out.values})
    for d in (hh, m15f, h4f):
        f = pd.merge_asof(f, d, left_on="t", right_on="end", direction="backward").drop(columns="end")
    f["지금 규칙 (ADX+직진도)"] = 2 - hm2.regime_at(m5, h1, f.t)      # 2 추세 / 1 일반 / 0 횡보
    # 앞으로 (평가용, 지표에는 안 씀)
    f["앞 직진도 2시간"] = eff(c5, 24).shift(-24).values
    f["앞 직진도 4시간"] = eff(c5, 48).shift(-48).values
    f["앞 폭/ATR 2시간"] = ((m5.high.rolling(24).max().shift(-24) - m5.low.rolling(24).min().shift(-24)) / a5).values
    return f


def sp(a, b):
    """순위 상관 (scipy 없이)"""
    ok = a.notna() & b.notna()
    return a[ok].rank().corr(b[ok].rank())


def tercile(s):
    r = s.rank(pct=True, method="average")
    return pd.cut(r, [0, 1 / 3, 2 / 3, 1.0001], labels=[0, 1, 2], include_lowest=True).astype(float)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    allf = features(load_m1())
    f = allf[allf.t.dt.weekday < 5].dropna(subset=["앞 직진도 4시간"]).iloc[::3]     # 15분마다 (겹침 줄이기)
    inds = [c for c in f.columns if c not in ("t",) and not c.startswith("앞 ")]
    fut = tercile(f["앞 직진도 2시간"])
    rows = []
    for c in inds:
        x = f[c]
        ok = x.notna()
        yr = f.t.dt.year
        by = [sp(x[ok & (yr == y)], f["앞 직진도 2시간"][ok & (yr == y)]) for y in YEARS]
        q = tercile(x[ok]) if x.nunique() > 3 else x[ok]
        top, bot = (q == q.max()), (q == q.min())
        rows.append({"지표": c, "상관 앞 직진도 2h": round(sp(x, f["앞 직진도 2시간"]), 3),
                     "연도별 최소": round(min(by), 3), "연도별 최대": round(max(by), 3),
                     "상관 앞 직진도 4h": round(sp(x, f["앞 직진도 4시간"]), 3),
                     "상관 앞 폭/ATR": round(sp(x, f["앞 폭/ATR 2시간"]), 3),
                     "추세라 할 때 맞음%": round((fut[ok][top] == 2).mean() * 100, 1),
                     "횡보라 할 때 맞음%": round((fut[ok][bot] == 0).mean() * 100, 1)})
    res = pd.DataFrame(rows).sort_values("상관 앞 직진도 2h", ascending=False)
    print("평가 1: 앞으로의 장세를 맞히나 (아무렇게나 = 상관 0, 맞음 33%)")
    print(res.to_string(index=False))
    res.to_csv(OUT / "regime_check.csv", index=False, encoding="utf-8-sig")

    files = sorted(OUT.glob("martin_cycles_*.csv"))
    if not files:
        print("martin_cycles_*.csv 없음: python -m backtests.martin_split 2025 cycles 먼저")
        return
    cy = pd.concat([pd.read_csv(p) for p in files])
    cy = cy[cy["2배 조건"] == "전환+다참"]
    cy["t"] = pd.to_datetime(cy.t0, unit="ms").astype(f.t.dtype)
    cy = pd.merge_asof(cy.sort_values("t"), allf[["t"] + inds], on="t", direction="backward")
    cy["나쁨"] = cy.이유.isin(["limit", "freeze"])
    rows = []
    for c in inds:
        q = tercile(cy[c]) if cy[c].nunique() > 3 else cy[c]
        g = cy.groupby(q)
        lab = {0: "횡보쪽", 1: "중간", 2: "추세쪽"}
        row = {"지표": c}
        for k, v in g:
            row[f"{lab.get(int(k), k)} 사이클"] = len(v)
            row[f"{lab.get(int(k), k)} 한도·동결%"] = round(v.나쁨.mean() * 100, 1)
            row[f"{lab.get(int(k), k)} 5마틴%"] = round((v.마틴 >= 5).mean() * 100, 1)
            row[f"{lab.get(int(k), k)} 평균손익%"] = round(v["손익%"].mean(), 2)
        rows.append(row)
    res2 = pd.DataFrame(rows)
    print(f"\n평가 2: 마틴 사이클 {len(cy)}개, 시작 때 지표 1/3 구간별 (전체 한도·동결 {cy.나쁨.mean() * 100:.1f}%)")
    print(res2.to_string(index=False))
    res2.to_csv(OUT / "regime_cycles.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
