"""index_intraday 후보 2개를 실제 틱으로 다시 확인 (2022-01 ~ 2026-10)

NAS100 첫 30분 방향 : 본장 첫 틱 Bid -> 시가 30분 직전 마지막 틱 Bid 움직임이 일봉 ATR x M 넘으면 그 방향으로
                      시가 30분 + 0.3초 뒤 첫 틱에 진입 (매수 Ask / 매도 Bid). 손절: 매수는 Bid <= 진입 Bid - ATR x S,
                      매도는 Ask >= 진입 Ask + ATR x S (그 틱 가격에 청산). 마감 직전 마지막 틱에 청산 (매수 Bid / 매도 Ask)
GER40 ORB 30분 매도만 : 본장 처음 30분 틱 Bid 고가·저가, 그 뒤 1분봉(틱 Bid로 만듦) 종가가 저가 밑이면 다음 분 + 0.3초 첫 틱 Bid에 매도
                      (진입 시각이 시가 + 3시간 전까지), 손절 = Bid >= 범위 가운데면 그 틱 Ask, 마감 직전 마지막 틱 Ask로 청산
세션·휴장일·조기 마감: index_intraday.sessions와 같음. ATR: index_intraday.prep과 같은 값 (전날까지 본장 고저폭 14일 평균). 미끄러짐: 체결마다 0 / 0.5 / 1.0 포인트 따로
결과: results/index_ticks.csv (날마다), 연도별 요약 출력

사용법: python -m backtests.index_ticks
"""
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.index_intraday as ii
from backtests.download_symbol import load_month

ROOT = Path(__file__).resolve().parents[1]
DELAY = 300
SLIPS = (0.0, 0.5, 1.0)
CASES = {"NAS100": ("첫30분", 0.2, 1.0), "GER40": ("ORB매도", 30, None)}


def day_list(sym):
    m1 = pd.read_csv(ii.DATA / sym / "M1.csv.gz", index_col="time", parse_dates=True)
    m1 = m1[~m1.index.duplicated()].sort_index()
    ss = ii.sessions(sym, m1)
    t = m1.index.values
    h, l = m1.high.values, m1.low.values
    out, rng = [], []
    for d, a, b in ss:
        atr = np.mean(rng[-14:]) if len(rng) >= 14 else np.nan
        rng.append(h[a:b].max() - l[a:b].min())
        if not np.isnan(atr):
            out.append((d, pd.Timestamp(t[a]), pd.Timestamp(t[b - 1]) + pd.Timedelta(minutes=1), atr))
    return out


def ms(ts):
    return int(ts.value // 1_000_000)


def nas_first30(t, b, a, so, sc, atr, m, s):
    k0 = int(np.searchsorted(t, so))
    k30 = int(np.searchsorted(t, so + 30 * 60_000))
    ke = int(np.searchsorted(t, sc)) - 1
    if k0 >= k30 or k30 >= ke:
        return None
    mv = b[k30 - 1] - b[k0]
    if abs(mv) < m * atr:
        return None
    side = 1 if mv > 0 else -1
    k = int(np.searchsorted(t, so + 30 * 60_000 + DELAY))
    if k >= ke:
        return None
    if side > 0:
        e, stop = a[k], b[k] - s * atr
        hit = np.flatnonzero(b[k:ke + 1] <= stop)
        j = k + int(hit[0]) if hit.size else ke
        return side, b[j] - e, j < ke
    e, stop = b[k], a[k] + s * atr
    hit = np.flatnonzero(a[k:ke + 1] >= stop)
    j = k + int(hit[0]) if hit.size else ke
    return side, e - a[j], j < ke


def ger_orb_short(t, b, a, so, sc, n):
    k0 = int(np.searchsorted(t, so))
    kn = int(np.searchsorted(t, so + n * 60_000))
    ke = int(np.searchsorted(t, sc)) - 1
    if kn - k0 < 10 or kn >= ke:
        return None
    hi, lo = b[k0:kn].max(), b[k0:kn].min()
    mid = (hi + lo) / 2
    minute = (t[kn:ke + 1] - so) // 60_000
    last = np.flatnonzero(np.r_[minute[1:] != minute[:-1], True])        # 분마다 마지막 틱
    for q in last:
        mnt = minute[q]
        if mnt + 1 >= 180:                       # 진입(다음 분)이 시가 + 3시간 전이어야
            return None
        if b[kn + q] < lo:
            k = int(np.searchsorted(t, so + (mnt + 1) * 60_000 + DELAY))
            if k >= ke:
                return None
            e = b[k]
            hit = np.flatnonzero(b[k:ke + 1] >= mid)
            j = k + int(hit[0]) if hit.size else ke
            return -1, e - a[j], j < ke
    return None


def main():
    rows = []
    for sym, case in CASES.items():
        days = day_list(sym)
        cache = {}
        for d, so_ts, sc_ts, atr in days:
            key = (so_ts.year, so_ts.month)
            if key not in cache:
                cache.clear()
                try:
                    cache[key] = load_month(sym, *key)
                except FileNotFoundError:
                    cache[key] = None
            tk = cache[key]
            if tk is None:
                continue
            t, b, a = tk
            so, sc = ms(so_ts), ms(sc_ts)
            r = nas_first30(t, b, a, so, sc, atr, case[1], case[2]) if case[0] == "첫30분" else ger_orb_short(t, b, a, so, sc, case[1])
            if r is None:
                continue
            side, pts, stopped = r
            rows.append({"종목": sym, "전략": case[0], "날짜": d, "y": d.year, "방향": side, "pts": pts, "손절": stopped, "atr": atr})
        print(sym, "끝", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results" / "index_ticks.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    for (sym, strat), g in d.groupby(["종목", "전략"]):
        print(f"\n=== {sym} {strat} (틱)")
        out = []
        for slip in SLIPS:
            v = g.pts - 2 * slip
            row = {"미끄러짐pt": slip}
            for yr in range(2022, 2027):
                row[f"{yr}"] = round(v[g.y == yr].sum())
            gp, lp = v[v > 0].sum(), -v[v < 0].sum()
            row.update({"매매": len(v), "승률%": round((v > 0).mean() * 100, 1), "PF": round(gp / lp, 2),
                        "2022~24 PF": round(v[(g.y <= 2024) & (v > 0)].sum() / -v[(g.y <= 2024) & (v < 0)].sum(), 2),
                        "2025~26 PF": round(v[(g.y >= 2025) & (v > 0)].sum() / -v[(g.y >= 2025) & (v < 0)].sum(), 2),
                        "이길때": round(v[v > 0].mean(), 1), "질때": round(v[v < 0].mean(), 1), "최악": round(v.min(), 1),
                        "손절 비율%": round(g["손절"].mean() * 100, 1)})
            out.append(row)
        print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
