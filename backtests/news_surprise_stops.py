"""서프라이즈 진입 뒤 '아니다' 판단 (실제 틱): 시장 반응 / SAR(콩) 방향

진입: 발표 1.5초 뒤 +0.3초, |S| >= 1.5, 엇갈림 없음 (news_surprise_types와 같음). 체결마다 0.1$
기본 청산: 큰이익추적 5/2.5/15분, 30분 보유 (발표 30분 직전 틱까지)
덧붙이는 '아니다' 판단 (먼저 걸리면 그 틱에 청산)
  반응 T초 X$   : 발표 T초 뒤 중간가가 발표 직전 가격 대비 서프라이즈 방향으로 X$ 못 갔으면 청산
                  (X = -1: 반대로 1$ 넘게 감, 0: 반대쪽에 있음, 1·2: 반응이 약함)
  SAR 1분/5분 봉마감 : 진입 뒤 마감된 봉의 SAR 방향이 포지션과 반대면 다음 틱에 청산 (Bid 봉, PSAR 0.02/0.2)
  SAR 1분/5분 닿음   : 진입 뒤 가격이 SAR 점을 넘어서는 틱에 청산 (진입 때 SAR이 같은 방향인 것만)
  SAR 같은 방향만 진입: 진입 때 마지막 마감 봉의 SAR이 서프라이즈 방향과 같을 때만 (청산은 기본 그대로)
결과: results/news_surprise_stops.csv. 고르기 2022~2024, 확인 2025~2026

사용법: python -m backtests.news_surprise_stops
"""
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.news_grid import exit_at, first
from indicators import psar_levels

ROOT = Path(__file__).resolve().parents[1]
LAG_MS, DELAY, SLIP, HOLD = 1500, 300, 0.1, 1_800_000
BASES = {"큰이익추적": ("큰이익추적", 5, 2.5, 15), "30분": ("시간", 30, None)}
REACT = [(t, x) for t in (15, 30, 60, 120) for x in (-1, 0, 1, 2)]
_cache = {}


def month(y, m):
    if (y, m) not in _cache:
        if len(_cache) > 2:
            _cache.pop(next(iter(_cache)))
        try:
            _cache[(y, m)] = lb.load_month(y, m)
        except FileNotFoundError:
            _cache[(y, m)] = None
    return _cache[(y, m)]


def window(ms0, before, after):
    ts = pd.Timestamp(ms0, unit="ms")
    prev = (ts - pd.Timedelta(milliseconds=before))
    parts = []
    for y, m in sorted({(prev.year, prev.month), (ts.year, ts.month)}):
        d = month(y, m)
        if d is None:
            continue
        t, b, a = d
        k0, k1 = np.searchsorted(t, [ms0 - before, ms0 + after])
        parts.append((t[k0:k1], b[k0:k1], a[k0:k1]))
    if not parts:
        return None
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def sar_bars(t, b, minutes):
    """Bid 봉 SAR: (봉 시작 ms, 봉 끝 ms, 추세(+1/-1), 그 봉에서 뚫어야 했던 SAR 값)"""
    key = t // (minutes * 60_000)
    s = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    h, lo = np.maximum.reduceat(b, s), np.minimum.reduceat(b, s)
    up, _, trig = psar_levels(h, lo)
    start = key[s] * minutes * 60_000
    return start, start + minutes * 60_000, np.where(up, 1, -1), trig


def stop_sar_close(bars, tm, entry_ms, side):
    start, end, d, _ = bars
    done = np.flatnonzero((end > entry_ms) & (d != side))      # 진입 뒤 마감된 봉 중 반대 방향
    if not done.size:
        return None
    k = int(np.searchsorted(tm, end[done[0]]))
    return k if k < len(tm) else None


def stop_sar_touch(bars, tm, bid, entry_ms, side):
    start, end, d, trig = bars
    i0 = int(np.searchsorted(end, entry_ms, side="right"))     # 진입이 들어 있는 봉
    for i in range(i0, len(start)):
        if d[i - 1] != side:                                    # 이미 반대면 그 봉 시작에서
            k = int(np.searchsorted(tm, start[i]))
            return k if k < len(tm) else None
        a, z = np.searchsorted(tm, [max(start[i], entry_ms), end[i]])
        if a >= z:
            continue
        hit = np.flatnonzero(bid[a:z] < trig[i]) if side > 0 else np.flatnonzero(bid[a:z] > trig[i])
        if hit.size:
            return int(a + hit[0])
    return None


def main():
    c = pd.read_csv(ROOT / "results" / "news_surprise_events.csv", parse_dates=["time"])
    c = c[(c.S.abs() >= 1.5) & (c.npos.fillna(0) * c.nneg.fillna(0) == 0)].reset_index(drop=True)
    ms_all = c.time.values.astype("datetime64[ms]").astype(np.int64)
    rows = []
    for i in range(len(c)):
        ms0, side = int(ms_all[i]), int(np.sign(c.S[i]))
        w = window(ms0, 8 * 3600_000, HOLD)
        if w is None:
            continue
        t, b, a = w
        r = int(np.searchsorted(t, ms0))
        if r == 0 or len(t) - r < 10 or t[0] > ms0 - 4 * 3600_000:
            continue
        ref = (a[r - 1] + b[r - 1]) / 2
        k = int(np.searchsorted(t, ms0 + LAG_MS + DELAY))
        e0 = a[k] if side > 0 else b[k]
        tm, bid, ask = t[k:], b[k:], a[k:]
        p = (bid - e0) if side > 0 else (e0 - ask)
        f = ((ask + bid) / 2 - ref) * side
        bars = {mm: sar_bars(t, b, mm) for mm in (1, 5)}
        sar_now = {}
        for mm, br in bars.items():
            j = int(np.searchsorted(br[1], tm[0], side="right")) - 1   # 진입 때 마지막 마감 봉
            sar_now[mm] = br[2][j] if j >= 0 else 0
        stops = {"없음": None}
        for tt, x in REACT:
            j = int(np.searchsorted(tm, ms0 + tt * 1000))
            stops[f"반응 {tt}초 {x:+d}$"] = j if j < len(tm) and f[j] < x else None
        for mm in (1, 5):
            stops[f"SAR {mm}분 봉마감"] = stop_sar_close(bars[mm], tm, tm[0], side)
            stops[f"SAR {mm}분 닿음"] = stop_sar_touch(bars[mm], tm, bid, tm[0], side) if sar_now[mm] == side else "제외"
        row = {"time": c.time[i], "key": c.key[i], "S": c.S[i], "y": c.y[i], "sar1": sar_now[1] == side, "sar5": sar_now[5] == side}
        for bn, ex in BASES.items():
            j0 = exit_at(ex, p, f, tm, ms0, 0.0)
            for sn, js in stops.items():
                if js == "제외":
                    row[f"{bn} | {sn}"] = np.nan
                    continue
                j = j0 if js is None else min(j0, js)
                row[f"{bn} | {sn}"] = p[j] - 2 * SLIP
        rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(c)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results" / "news_surprise_stops_events.csv", index=False, encoding="utf-8-sig")
    out = []
    for thr in (1.5, 2.0):
        for col in [x for x in d.columns if " | " in x]:
            for filt in ("", "SAR 1분 같은 방향만", "SAR 5분 같은 방향만"):
                k = (d.S.abs() >= thr) & d[col].notna()
                if filt:
                    k &= d["sar1" if "1분" in filt else "sar5"]
                v, yy = d[col][k].values, d.y[k].values
                if not len(v):
                    continue
                base, stop = col.split(" | ")
                o = {"|S|": thr, "청산": base, "아니다 판단": stop, "진입 거르기": filt or "-", "매매": len(v),
                     "승률%": round((v > 0).mean() * 100, 1), "질 때 평균": round(v[v < 0].mean(), 2)}
                for yr in range(2022, 2027):
                    o[str(yr)] = round(v[yy == yr].sum(), 1)
                o["2022~24"] = round(v[yy <= 2024].sum(), 1)
                o["2025~26"] = round(v[yy >= 2025].sum(), 1)
                o["5년"] = round(v.sum(), 1)
                o["최악"] = round(v.min(), 1)
                out.append(o)
    out = pd.DataFrame(out)
    out.to_csv(ROOT / "results" / "news_surprise_stops.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    for thr in (1.5, 2.0):
        for bn in BASES:
            x = out[(out["|S|"] == thr) & (out["청산"] == bn)]
            print(f"\n=== |S| >= {thr}, 기본 청산 {bn}: 2022~24 순")
            print(x.sort_values("2022~24", ascending=False).head(14).drop(columns=["|S|", "청산"]).to_string(index=False))
            base = x[(x["아니다 판단"] == "없음") & (x["진입 거르기"] == "-")]
            print("--- 기준"); print(base.drop(columns=["|S|", "청산"]).to_string(index=False))


if __name__ == "__main__":
    main()
