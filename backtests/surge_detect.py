"""급등락 포착 -> 박스 전략(SAR 단타·헤지) 쉬기 + 단기 추세 진입 (2026-10-27 사용자 아이디어, 2022-01 ~ 2026-10-16)

10/27 저녁: 4일 박스(3,975~4,030) 위로 2시간 반 만에 +38$ -> SAR 단타 손절, 아시아 손절, 헤지 매도 물림

포착 규칙 (5분봉 Bid, 봉이 끝난 순간 판단, 일봉 ATR(14)은 전날까지 값):
  속도 n분 k  : 최근 n분 동안 종가가 일봉 ATR의 k배 이상 움직임
  박스 N시간 w : 종가가 직전 N시간 최고가(최저가)를 뚫음, 박스 폭이 일봉 ATR의 w배 이하 (inf = 폭 상관없음)
  같은 규칙은 한 번 뜨면 4시간 동안 다시 안 셈
1. 단기 추세: 포착 다음 봉 시가에 그 방향 진입, 비용 0.4$/oz (스프레드+슬리피지), 봉 안에서는 손절 먼저
     고정  : 손절 일봉 ATR x0.25 / 익절 x0.5 / 4시간 지나면 청산
     SAR   : 손절 일봉 ATR x0.25, 5분봉 SAR이 반대로 바뀐 봉 종가에 청산 (최대 8시간)
     n시간 : 손절 없이 1 / 4시간 뒤 청산 (그 방향으로 계속 가는지 참고)
2. 쉬기: 포착 뒤 H시간 안에 시작한 SAR 단타 거래(results/sar_us_now_YYYY.csv) / 헤지 사이클(scratchpad hedge_cycles)의 성적
     -> 이 거래들이 마이너스면 쉬는 게 이득

사용법: python -m backtests.surge_detect   (결과 results/surge_trend.csv, results/surge_pause.csv)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.scalp_ideas import m5_bars
from indicators import psar

RES = Path(__file__).resolve().parent.parent / "results"
COST = 0.4
COOL = 48                      # 4시간 (5분봉 48개)
END = pd.Timestamp("2026-10-17")
HEDGE_DIR = None               # hedge_cycles_YYYY.csv 폴더 (인자 --hedge=폴더)


def daily_atr(m5):
    d = m5.set_index("time").resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    pc = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    a = tr.rolling(14).mean().shift(1)                       # 전날까지
    return m5.time.dt.normalize().map(a).values


def detectors(m5, datr):
    c, h, lo = m5.close.values, m5.high.values, m5.low.values
    out = {}
    for n in (30, 60, 120):
        b = n // 5
        mv = np.r_[np.full(b, np.nan), c[b:] - c[:-b]] / datr
        for k in (0.3, 0.4, 0.5, 0.7):
            out[f"속도 {n}분 x{k}"] = np.where(mv >= k, 1, np.where(mv <= -k, -1, 0))
    for nh in (12, 24, 48):
        b = nh * 12
        hh = pd.Series(h).rolling(b).max().shift(1).values
        ll = pd.Series(lo).rolling(b).min().shift(1).values
        width = (hh - ll) / datr
        for w in (0.5, 0.7, 1.0, np.inf):
            ok = width <= w
            out[f"박스 {nh}시간 폭x{w:g}"] = np.where(ok & (c > hh), 1, np.where(ok & (c < ll), -1, 0))
    b3 = 36                                                   # 3시간
    c3 = np.r_[np.full(b3, np.nan), c[:-b3]]
    hh3 = pd.Series(h).rolling(b3).max().shift(1).values      # 3시간 박스
    ll3 = pd.Series(lo).rolling(b3).min().shift(1).values
    mid3 = (hh3 + ll3) / 2
    out["3시간 박스 돌파"] = np.where(c > hh3, 1, np.where(c < ll3, -1, 0))
    for m in (15, 30):                                        # m분 전엔 박스 반대쪽 절반에 있었음 = 짧은 시간에 박스를 가로지름
        cm = np.r_[np.full(m // 5, np.nan), c[:-(m // 5)]]
        out[f"3시간 박스 {m}분 안에 가로질러 돌파"] = np.where((c > hh3) & (cm < mid3), 1, np.where((c < ll3) & (cm > mid3), -1, 0))
        for w in (0.3, 0.5):
            ok = (hh3 - ll3) <= w * datr
            out[f"3시간 박스(폭 ATRx{w} 이하) {m}분 안에 가로질러 돌파"] = np.where(
                ok & (c > hh3) & (cm < mid3), 1, np.where(ok & (c < ll3) & (cm > mid3), -1, 0))
    for p in (3, 6, 12, 24):                                  # 고점 양옆 p봉 (15분 / 30분 / 1시간 / 2시간)
        s1h, s2h = swings(h, p, True)
        s1l, s2l = swings(lo, p, False)
        top, bot = np.fmax(s1h, s2h), np.fmin(s1l, s2l)
        out[f"3시간 안에 전고점 뚫음 (양옆{p * 5}분)"] = np.where((c > s1h) & (c3 < s1h), 1, np.where((c < s1l) & (c3 > s1l), -1, 0))
        out[f"3시간 안에 전고점+전전고점 뚫음 (양옆{p * 5}분)"] = np.where(
            (c > top) & (c3 < np.fmin(s1h, s2h)), 1, np.where((c < bot) & (c3 > np.fmax(s1l, s2l)), -1, 0))
    return out


def swings(x, p, high):
    """봉마다 그때까지 확인된(고점 뒤 p봉 지난) 마지막 고점 s1, 그 전 고점 s2"""
    r = pd.Series(x).rolling(2 * p + 1, center=True)
    ext = (r.max() if high else r.min()).values
    piv = np.flatnonzero(x == ext)
    s1, s2 = np.full(len(x), np.nan), np.full(len(x), np.nan)
    conf = piv + p
    keep = conf < len(x)
    piv, conf = piv[keep], conf[keep]
    s1[conf] = x[piv]
    s2[conf[1:]] = x[piv[:-1]]
    return pd.Series(s1).ffill().values, pd.Series(s2).ffill().values


def events(sig):
    """4시간 쿨다운 적용한 (봉 번호, 방향)"""
    ev, last = [], -10**9
    for i in np.flatnonzero(np.nan_to_num(sig) != 0):
        if i - last >= COOL:
            ev.append((i, int(sig[i])))
            last = i
    return ev


def trade(m5v, up, datr, i, d, mode):
    o, h, lo, c = m5v
    n = len(c)
    if i + 1 >= n or not np.isfinite(datr[i]):
        return None
    e = o[i + 1]
    if mode in ("1시간", "4시간"):
        j = min(n - 1, i + (12 if mode == "1시간" else 48))
        return (c[j] - e) * d - COST
    sl = 0.25 * datr[i]
    tp = 0.5 * datr[i] if mode == "고정" else np.inf
    hold = 48 if mode == "고정" else 96
    for j in range(i + 1, min(n, i + 1 + hold)):
        adverse = (e - lo[j]) if d == 1 else (h[j] - e)
        fav = (h[j] - e) if d == 1 else (e - lo[j])
        if adverse >= sl:
            return -sl - COST
        if fav >= tp:
            return tp - COST
        if mode == "SAR" and up[j] != (d == 1):
            return (c[j] - e) * d - COST
    j = min(n - 1, i + hold)
    return (c[j] - e) * d - COST


def pf(x):
    g, l_ = x[x > 0].sum(), -x[x < 0].sum()
    return round(g / l_, 2) if l_ > 0 else np.inf


def main():
    global HEDGE_DIR
    HEDGE_DIR = next((Path(a[8:]) for a in sys.argv if a.startswith("--hedge=")), None)
    m5 = m5_bars()
    m5 = m5[m5.time < END].reset_index(drop=True)
    datr = daily_atr(m5)
    up = psar(m5.high.values, m5.low.values)
    m5v = (m5.open.values, m5.high.values, m5.low.values, m5.close.values)
    tclose = (m5.time + pd.Timedelta(minutes=5)).values
    yr = m5.time.dt.year.values
    sar = pd.concat([pd.read_csv(RES / f"sar_us_now_{y}.csv", parse_dates=["open"]) for y in range(2022, 2027)])
    hed = None
    if HEDGE_DIR:
        hed = pd.concat([pd.read_csv(HEDGE_DIR / f"hedge_cycles_{y}.csv", parse_dates=["start"]) for y in range(2022, 2027)])
    years = range(2022, 2027)
    trows, prows = [], []
    only = next((a[7:] for a in sys.argv if a.startswith("--only=")), "")
    for name, sig in detectors(m5, datr).items():
        if only not in name:
            continue
        ev = events(sig)
        if not ev:
            continue
        idx = np.array([i for i, _ in ev])
        for mode in ("고정", "SAR", "1시간", "4시간"):
            p = np.array([trade(m5v, up, datr, i, d, mode) for i, d in ev], dtype=float)
            ok = np.isfinite(p)
            py = yr[idx][ok]
            p = p[ok]
            row = {"규칙": name, "청산": mode, "건수": len(p), "연평균": round(len(p) / 4.8), "승률%": round((p > 0).mean() * 100, 1),
                   "oz합": round(p.sum(), 1), "PF": pf(p)}
            row.update({f"oz {y}": round(p[py == y].sum(), 1) for y in years})
            row["플러스 해"] = sum(p[py == y].sum() > 0 for y in years)
            trows.append(row)
        et = tclose[idx]
        for hours in (1, 2, 4):
            win = np.timedelta64(hours, "h")
            k = np.searchsorted(et, sar.open.values, side="right") - 1
            hit = (k >= 0) & (sar.open.values - et[np.clip(k, 0, None)] < win)
            s = sar[hit]
            row = {"규칙": name, "쉬는 시간": f"{hours}시간", "단타 쉬는 건수": len(s), "단타 승률%": round((s.oz > 0).mean() * 100, 1),
                   "단타 oz": round(s.oz.sum(), 1)}
            row.update({f"단타 {y}": round(s.oz[s.open.dt.year == y].sum(), 1) for y in years})
            if hed is not None:
                k = np.searchsorted(et, hed.start.values, side="right") - 1
                hh = hed[(k >= 0) & (hed.start.values - et[np.clip(k, 0, None)] < win)]
                row.update({"헤지 쉬는 사이클": len(hh), "그중 -30%": int((hh.why == "limit").sum()), "헤지 %합": round(hh.pct.sum(), 1),
                            "헤지 전체 -30%": int((hed.why == "limit").sum())})
            prows.append(row)
        print("완료", name, len(ev), flush=True)
    tr, pa = pd.DataFrame(trows), pd.DataFrame(prows)
    tr.to_csv(RES / "surge_trend.csv", index=False, encoding="utf-8-sig")
    pa.to_csv(RES / "surge_pause.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.unicode.east_asian_width", True)
    print(tr.sort_values("oz합", ascending=False).head(30).to_string(index=False))
    print(pa.sort_values("단타 oz").head(30).to_string(index=False))


if __name__ == "__main__":
    main()
