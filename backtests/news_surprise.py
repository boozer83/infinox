"""뉴스 서프라이즈 방향 매매 (실제 틱, USD·EUR·GBP high + moderate, 2022-01 ~ 2026-10)

서프라이즈 S: 같은 시각 발표 지표마다 z = (실제 - 예측) / 그 지표의 2022~2024 표준편차 (±3에서 자름),
  지표별 금 방향(극성) = 2022~2024에 z와 발표 15분 움직임 상관의 부호 (12번 이상, |상관| >= 0.2인 지표만), S = 극성 x z 합
  -> 2022~2024로 극성을 정하고 2025~2026은 확인 기간 (극성 표: results/news_surprise_polarity.csv)
진입: 발표 L초 뒤 (실제치를 받는 데 걸리는 시간) S 방향으로 한쪽만, 0.3초 늦게 체결, 매수 Ask / 매도 Bid
  거르기: |S| >= 문턱, 지표끼리 방향이 엇갈리면(+0.5 넘는 것과 -0.5 넘는 것이 같이 있음) 안 함,
          '첫 움직임 같음' = 진입 때 중간가가 발표 직전보다 S 방향에 있을 때만
청산: 발표 T분 / 손절 = 진입가 대비 X% 역행 / 익절 Y%
비용: 틱 Bid/Ask + 체결마다 추가 미끄러짐 0.1$
결과 $/oz

사용법: python -m backtests.news_surprise      (먼저 news_moves2 실행, results/news_surprise.csv)
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.long_period as lb

ROOT = Path(__file__).resolve().parents[1]
LAGS = (1, 3, 5, 10)
THRS = (1.0, 1.5, 2.0, 3.0)
HOLDS = (5, 10, 15, 30)
STOPS = (None, 0.05, 0.1)
TAKES = (None, 0.1, 0.2)
SLIP = 0.1


def surprise(d):
    rows = []
    for i, s in d.sur.dropna().items():
        for it in s.split("|"):
            k, z = it.rsplit(":", 1)
            rows.append((i, k, float(z)))
    s = pd.DataFrame(rows, columns=["i", "k", "z"]).join(d[["m15", "y"]], on="i")
    st = s[s.y <= 2024].groupby("k").apply(lambda g: pd.Series({"n": len(g), "corr": g.z.corr(g.m15)}))
    pol = np.sign(st["corr"]).where((st.n >= 12) & (st["corr"].abs() >= 0.2), 0)
    pol.to_csv(ROOT / "results" / "news_surprise_polarity.csv", encoding="utf-8-sig")
    s["pz"] = s.z.clip(-3, 3) * s.k.map(pol).fillna(0)
    agg = s.groupby("i").pz.agg(S="sum", npos=lambda v: (v > 0.5).sum(), nneg=lambda v: (v < -0.5).sum())
    return d.join(agg)


def sim(t, b, a, ms, side, lag, hold, stop, take):
    mid = (a + b) / 2
    r = int(np.searchsorted(t, ms))
    ref = mid[r - 1]
    k = int(np.searchsorted(t, ms + lag * 1000 + 300))
    end = int(np.searchsorted(t, ms + hold * 60_000))
    if k >= end:
        return None
    first_ok = (mid[k] - ref) * side > 0
    e = a[k] if side > 0 else b[k]
    p = (b[k:end] - e) if side > 0 else (e - a[k:end])
    j = len(p) - 1
    hits = []
    if stop:
        hits.append(np.flatnonzero(p <= -e * stop / 100))
    if take:
        hits.append(np.flatnonzero(p >= e * take / 100))
    for h in hits:
        if h.size:
            j = min(j, int(h[0]))
    return p[j] - 2 * SLIP, first_ok


def main():
    d = pd.read_csv(ROOT / "results" / "news_moves2.csv", parse_dates=["time"])
    d["y"] = d.time.dt.year
    d["m15"] = d.d15m / d.ref * 100
    d = surprise(d)
    cand = d[(d.S.abs() >= min(THRS)) & (d.npos.fillna(0) * d.nneg.fillna(0) == 0)].copy()
    ms_all = cand.time.values.astype("datetime64[ms]").astype(np.int64)
    combos = list(itertools.product(LAGS, HOLDS, STOPS, TAKES))
    res = {c: {} for c in combos}
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        idx = np.flatnonzero((ms_all >= int(s.timestamp() * 1000)) & (ms_all < int(e.timestamp() * 1000)))
        if not len(idx):
            continue
        t, bid, ask = lb.load_month(y, m)
        for i in idx:
            ms, side = int(ms_all[i]), int(np.sign(cand.S.iloc[i]))
            k0, k1 = np.searchsorted(t, [ms - 60_000, ms + 1_860_000])
            tt, bb, aa = t[k0:k1], bid[k0:k1], ask[k0:k1]
            if len(tt) < 10 or np.searchsorted(tt, ms) == 0:
                continue
            for c in combos:
                res[c][cand.index[i]] = sim(tt, bb, aa, ms, side, *c)
        print(f"  {y}-{m:02d} 완료", flush=True)
    rows = []
    for c in combos:
        for thr, first_need in itertools.product(THRS, (False, True)):
            v, yy = [], []
            for i, r in res[c].items():
                if r is None or abs(cand.S[i]) < thr or (first_need and not r[1]):
                    continue
                v.append(r[0])
                yy.append(cand.y[i])
            v, yy = np.array(v), np.array(yy)
            if not len(v):
                continue
            row = {"지연초": c[0], "보유분": c[1], "손절%": c[2], "익절%": c[3], "|S|문턱": thr, "첫움직임같음": first_need,
                   "매매 수": len(v), "손실 수": int((v < 0).sum()), "승률%": round((v > 0).mean() * 100, 1)}
            for y in range(2022, 2027):
                row[str(y)] = round(v[yy == y].sum(), 1)
            row["2022~24"] = round(v[yy <= 2024].sum(), 1)
            row["2025~26"] = round(v[yy >= 2025].sum(), 1)
            row["손실 수 2025~26"] = int((v[yy >= 2025] < 0).sum())
            row["매매 수 2025~26"] = int((yy >= 2025).sum())
            row["5년"] = round(v.sum(), 1)
            row["최악 1번"] = round(v.min(), 2)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "news_surprise.csv", index=False, encoding="utf-8-sig")
    cand.to_csv(ROOT / "results" / "news_surprise_events.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    print(out.sort_values("2022~24", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
