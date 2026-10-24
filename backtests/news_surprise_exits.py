"""서프라이즈 방향 진입 x 청산 방법 전부 (실제 틱, USD·EUR·GBP high + moderate, 2022-01 ~ 2026-10)

진입: news_surprise.py와 같음. 발표 L초 뒤(+0.3초 체결) S 방향 한쪽, |S| >= 문턱, 지표끼리 엇갈리면 제외
청산: news_grid.py의 63가지 ($ 고정) + 가격 비례 (금값이 1,800 -> 4,000대로 올라서 $ 고정은 해마다 기준이 달라짐)
  익절손절% TP / SL : 진입가의 TP% 익절, SL% 손절 (발표 30분이 최대)
  추적% Z T분       : 최고 손익에서 진입가의 Z% 밀리면, 늦어도 T분
  시간 T분 되돌림% R : 발표 T분, 중간가가 발표 직전 가격 반대쪽으로 R% 넘어가면 즉시 (서프라이즈와 반대로 간 것)
  본전이동% A T분    : 이익이 A% 넘으면 손절을 본전(+0.02%)으로, 늦어도 T분 (처음 손절 0.1%)
비용: 틱 Bid/Ask + 체결마다 0.1$ 미끄러짐. 결과 $/oz. 고르기 = 2022~2024, 확인 = 2025~2026

사용법: python -m backtests.news_surprise_exits    (먼저 news_moves2, news_surprise 실행, results/news_surprise_exits.csv)
"""
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.news_grid import EXITS, exit_at, first, name_e

ROOT = Path(__file__).resolve().parents[1]
LAGS = (1, 1.5, 3, 5, 10)
THRS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
SLIP, DELAY, MAXH = 0.1, 300, 1_800_000
PCT_EXITS = ([("익절손절%", tp, sl) for tp in (0.1, 0.2, 0.3, 0.5) for sl in (0.05, 0.1, 0.2)] +
             [("추적%", z, t) for z in (0.05, 0.1, 0.2) for t in (15, 30)] +
             [("시간되돌림%", t, r) for t in (10, 15, 30) for r in (0.03, 0.05, 0.1)] +
             [("본전이동%", a, t) for a in (0.05, 0.1, 0.2) for t in (15, 30)])
ALL_EXITS = EXITS + PCT_EXITS


def name_x(e):
    if e[0] == "익절손절%":
        return f"익절 {e[1]}% 손절 {e[2]}%"
    if e[0] == "추적%":
        return f"추적 {e[1]}% {e[2]}분"
    if e[0] == "시간되돌림%":
        return f"시간 {e[2 - 1]}분 되돌림 {e[2]}%"
    if e[0] == "본전이동%":
        return f"본전이동 {e[1]}% {e[2]}분"
    return name_e(e)


def exit_pct(e, p, f, tm, ms, ref, entry):
    u = entry / 100
    if e[0] == "익절손절%":
        return exit_at(("익절손절", e[1] * u, e[2] * u), p, f, tm, ms, 0.0)
    if e[0] == "추적%":
        return exit_at(("추적", e[1] * u, e[2]), p, f, tm, ms, 0.0)
    if e[0] == "시간되돌림%":
        return exit_at(("시간", e[1], e[2] * ref / 100), p, f, tm, ms, 0.0)
    j = min(int(np.searchsorted(tm, ms + e[2] * 60_000)), len(p) - 1)     # 본전이동
    on = first(p[:j + 1] >= e[1] * u)
    k = first(p[:j + 1] <= -0.1 * u)
    if on is not None and (k is None or on < k):
        k = first(p[:j + 1] <= 0.02 * u, on)
    return j if k is None else k


def month_job(args):
    y, m, items = args
    try:
        t, bid, ask = lb.load_month(y, m)
    except FileNotFoundError:
        return []
    out = []
    for i, ms, side in items:
        k0, k1 = np.searchsorted(t, [ms - 60_000, ms + MAXH])
        tt, b, a = t[k0:k1], bid[k0:k1], ask[k0:k1]
        r = int(np.searchsorted(tt, ms))
        if len(tt) < 10 or r == 0:
            continue
        mid = (a + b) / 2
        ref = mid[r - 1]
        res = np.full((len(LAGS), len(ALL_EXITS)), np.nan)
        for li, lag in enumerate(LAGS):
            k = int(np.searchsorted(tt, ms + lag * 1000 + DELAY))
            if k >= len(tt) - 2:
                continue
            e0 = a[k] if side > 0 else b[k]
            p = (b[k:] - e0) if side > 0 else (e0 - a[k:])
            f = (mid[k:] - ref) * side
            tm = tt[k:]
            for ei, e in enumerate(ALL_EXITS):
                j = exit_pct(e, p, f, tm, ms, ref, e0) if e in PCT_EXITS else exit_at(e, p, f, tm, ms, 0.0)
                res[li, ei] = p[j] - 2 * SLIP
        out.append((i, res))
    return out


def main():
    c = pd.read_csv(ROOT / "results" / "news_surprise_events.csv", parse_dates=["time"])
    c = c[(c.S.abs() >= min(THRS)) & (c.npos.fillna(0) * c.nneg.fillna(0) == 0)].reset_index(drop=True)
    ms = c.time.values.astype("datetime64[ms]").astype(np.int64)
    jobs = []
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        idx = np.flatnonzero((ms >= int(s.timestamp() * 1000)) & (ms < int(e.timestamp() * 1000)))
        if len(idx):
            jobs.append((y, m, [(int(i), int(ms[i]), int(np.sign(c.S[i]))) for i in idx]))
    raw = np.full((len(c), len(LAGS), len(ALL_EXITS)), np.nan)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 4))) as ex:
        for (y, m, _), res in zip(jobs, ex.map(month_job, jobs)):
            for i, r in res:
                raw[i] = r
            print(f"  {y}-{m:02d} 완료", flush=True)
    years, absS = c.y.values, c.S.abs().values
    rows = []
    for (li, lag), thr, (ei, e) in itertools.product(enumerate(LAGS), THRS, enumerate(ALL_EXITS)):
        v = raw[:, li, ei]
        k = ~np.isnan(v) & (absS >= thr)
        vv, yy = v[k], years[k]
        row = {"지연초": lag, "|S|문턱": thr, "청산": name_x(e), "매매 수": int(k.sum()),
               "손실률%": round((vv < 0).mean() * 100, 1)}
        for yr in range(2022, 2027):
            row[str(yr)] = round(vv[yy == yr].sum(), 1)
        row["2022~24"] = round(vv[yy <= 2024].sum(), 1)
        row["2025~26"] = round(vv[yy >= 2025].sum(), 1)
        row["손실률 2025~26%"] = round((vv[yy >= 2025] < 0).mean() * 100, 1)
        row["5년"] = round(vv.sum(), 1)
        row["최악 1번"] = round(vv.min(), 1)
        row["플러스 해"] = sum(row[str(yr)] > 0 for yr in range(2022, 2027))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "news_surprise_exits.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    cols = ["지연초", "|S|문턱", "청산", "매매 수", "손실률%", "2022", "2023", "2024", "2025", "2026",
            "2022~24", "2025~26", "손실률 2025~26%", "5년", "최악 1번"]
    for lag in (1, 5):
        d = out[out["지연초"] == lag]
        print(f"\n=== 지연 {lag}초: 2022~24로 고른 상위 12")
        print(d.sort_values("2022~24", ascending=False).head(12)[cols].to_string(index=False))
        print(f"--- 지연 {lag}초: 해마다 플러스 + 2022~24 손실률 낮은 순 (매매 60번 이상)")
        g = d[(d["플러스 해"] == 5) & (d["매매 수"] >= 60)].sort_values("손실률%")
        print(g.head(12)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
