"""일봉 변동성 압축 돌파를 여러 종목에 같은 규칙으로 (2026-10-31, idea_screen2 4번 금 일봉이 5년 19건이라 표본 늘리기)

규칙 고정 (금 idea_screen2 '일봉 볼린저폭 청산 3R' 그대로, 종목별 조정 없음):
  압축 = 전날 볼린저(20,2) 폭 / 중간선이 최근 100일 중 하위 20%
  진입 = 일봉 종가가 직전 20일 고가(저가)를 넘으면 다음 날 첫 1분봉 시가 (매수 Ask = Bid + 스프레드)
  손절 = 2 x 일봉 ATR(14, 신호일), 익절 3R, 60일 지나면 그날 종가 청산. 종목마다 한 번에 1개
체결: 1분봉 (Bid 봉 + 그 분 평균 스프레드). 같은 1분봉에 손절·익절 둘 다 닿으면 손절로 침 (보수적). 스왑 없음
결과는 R(초기 위험 대비)로: 종목별 / 전체 풀링, 2022~23 / 2024~26 / 연도별

사용법: python -m backtests.compress_symbols
"""
import gc

import numpy as np
import pandas as pd

import backtests.tv_symbols as tsy
from indicators import atr

SYMS = ["XAUUSD", "XAGUSD", "XAUEUR", "NAS100", "GER40", "JPN225", "BTCUSD", "XETUSD"]


def signals(d):
    mid, sd = d.close.rolling(20).mean(), d.close.rolling(20).std()
    sq = ((4 * sd / mid).rolling(100).rank(pct=True) <= 0.2).shift(1, fill_value=False).values
    hh, ll = d.high.rolling(20).max().shift(1).values, d.low.rolling(20).min().shift(1).values
    c = d.close.values
    return sorted([(i, 1) for i in np.flatnonzero(sq & (c > hh))] + [(i, -1) for i in np.flatnonzero(sq & (c < ll))])


def run(m1):
    d = m1.set_index("time").resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    a = atr(d).values
    t = m1.time.values
    o, h, lo, cl, sp = (m1[c].values for c in ("open", "high", "low", "close", "sp"))
    out, busy = [], np.datetime64("1970-01-01")
    for i, side in signals(d):
        if i + 1 >= len(d) or not a[i] > 0:
            continue
        day1 = np.datetime64(d.time.iat[i + 1])
        if day1 < busy:
            continue
        k = np.searchsorted(t, day1)
        end_i = min(i + 60, len(d) - 1)
        kend = np.searchsorted(t, np.datetime64(d.time.iat[end_i] + pd.Timedelta(days=1)))
        if k >= len(t) or kend <= k:
            continue
        e = o[k] + sp[k] if side == 1 else o[k]
        risk = 2 * a[i]
        sl, tp = e - side * risk, e + side * 3 * risk
        if side == 1:
            hit_sl, hit_tp = lo[k:kend] <= sl, h[k:kend] >= tp
        else:
            hit_sl, hit_tp = h[k:kend] + sp[k:kend] >= sl, lo[k:kend] + sp[k:kend] <= tp
        js, jt = np.flatnonzero(hit_sl), np.flatnonzero(hit_tp)
        j_sl = js[0] if js.size else None
        j_tp = jt[0] if jt.size else None
        if j_sl is not None and (j_tp is None or j_sl <= j_tp):
            px, j, why = sl, k + j_sl, "SL"
        elif j_tp is not None:
            px, j, why = tp, k + j_tp, "TP"
        else:
            j = kend - 1
            px, why = (cl[j] if side == 1 else cl[j] + sp[j]), "TIME"
        r = (px - e) * side / risk
        out.append({"open": pd.Timestamp(day1), "close": pd.Timestamp(t[j]), "side": side, "r": r, "why": why})
        busy = t[j]
    return pd.DataFrame(out)


def pf(r):
    g = -r[r < 0].sum()
    return round(r[r > 0].sum() / g, 2) if g else float("inf")


def row(name, tr):
    y = tr.open.dt.year
    res = {"종목": name, "거래": len(tr), "승률%": round((tr.r > 0).mean() * 100, 1), "PF(R)": pf(tr.r), "합R": round(tr.r.sum(), 1),
           "PF 22-23": pf(tr.r[y < 2024]), "PF 24-26": pf(tr.r[y >= 2024])}
    for yr in range(2022, 2027):
        res[str(yr)] = f"{pf(tr.r[y == yr])} ({(y == yr).sum()})"
    return res


def main():
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    sp_tab = tsy.spread_table()
    rows, allt = [], []
    for s in SYMS:
        if s == "XAGUSD":       # 은은 틱 pkl 대신 1분봉 zip (spread = 포인트, 0.001)
            m1 = pd.concat([pd.read_csv(f, parse_dates=["time"]) for f in sorted(tsy.BARS.glob("XAGUSD_M1_20*.csv.zip"))])
            m1 = m1.drop_duplicates("time").sort_values("time").reset_index(drop=True)
            m1["sp"] = m1.spread * 0.001
        else:
            m1 = tsy.load_sym(s, sp_tab)
        tr = run(m1)
        del m1
        gc.collect()
        if tr.empty:
            print(s, "거래 없음", flush=True)
            continue
        rows.append(row(s, tr))
        allt.append(tr.assign(sym=s))
        print(rows[-1], flush=True)
    pool = pd.concat(allt, ignore_index=True)
    rows.append(row("전체 풀링", pool))
    rows.append(row("금 빼고 풀링", pool[pool.sym != "XAUUSD"]))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print("\n청산 이유:", pool.why.value_counts().to_dict())


if __name__ == "__main__":
    main()
