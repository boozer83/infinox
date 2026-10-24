"""뉴스 발표 뒤 10분 방향의 공통점 찾기 (실제 틱, high USD·EUR·GBP, 2022-01 ~ 2026-10-23)

발표마다 발표 시각 중간가(m0) 기준으로
  1분 / 10분 뒤 중간가 변화, 10분 안 최대 위·아래, 중간가가 2$ 먼저 닿은 쪽 (HK 판단 기준, 5분 안),
  발표 전 30분 추세, 예상치 대비 실제치(서프라이즈) 부호
10분 방향: 10분 뒤 변화가 +1$ 이상이면 위, -1$ 이하면 아래, 그 사이는 횡보
확인하는 공통점
  1) 첫 1분 방향이 10분까지 가나   2) 2$ 먼저 닿은 쪽이 10분까지 가나
  3) 발표 전 30분 추세를 따라가나 / 거스르나   4) 서프라이즈 부호와 맞나 (발표 종류별)
  5) 10분 안 최대폭 중 10분 끝에 얼마나 남나
  6) 받은 거래 사진(HK 계좌)의 뉴스 시각만 따로
10/22 이후 틱은 저장 파일에 없어서 MT5에서 바로 받음. 캘린더 실제치는 10/21 내보낸 것까지만 있음
결과: results/news_dir10.csv (발표마다)

사용법: python -m backtests.news_dir10
"""
import gc
import os
import re
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import backtests.long_period as lb

OUT = lb.TICK_DIR.parent.parent / "results"
END = "2026-10-24"
FLAT = 1.0
# 받은 거래 사진의 뉴스 시각 (서버 시간)
HK_TIMES = ["2026-10-20 09:00", "2026-10-21 09:00", "2026-10-21 09:30", "2026-10-21 16:00",
            "2026-10-23 09:00", "2026-10-23 10:30", "2026-10-23 17:00"]


def num(s):
    if not isinstance(s, str) or not s.strip():
        return np.nan
    m = re.match(r"^\s*(-?[\d.]+)\s*([KMBT%]?)", s.replace(",", ""))
    return float(m.group(1)) if m else np.nan


def events():
    f = os.path.join(os.environ["APPDATA"], "MetaQuotes/Terminal/Common/Files/calendar_all.csv")
    c = pd.read_csv(f, encoding="cp949", dtype={"actual": str, "forecast": str})
    c = c[c.currency.isin(["USD", "EUR", "GBP"]) & (c.importance == "high")].copy()
    c["t"] = pd.to_datetime(c.time, format="%Y.%m.%d %H:%M")
    c = c[(c.t.dt.hour > 0) | (c.t.dt.minute > 0)]
    dst = c.t.dt.tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward").map(lambda x: bool(x.dst()))
    c["t"] = c.t - pd.to_timedelta((~dst.values).astype(int), unit="h")
    c = c[(c.t >= "2022-01-03") & (c.t < END) & (c.t.dt.weekday < 5)]
    c["key"] = c.currency + " " + c.event.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    c["sur"] = np.sign(c.actual.map(num) - c.forecast.map(num))
    # 같은 시각 여러 발표: 이름은 합치고, 서프라이즈는 첫 번째 것
    g = c.sort_values("t").groupby("t")
    ev = pd.DataFrame({"key": g.key.first(), "keys": g.key.agg(" / ".join), "n": g.size(), "sur": g.sur.first()})
    return ev.reset_index()


def recent_ticks():
    mt5.initialize()
    mt5.symbol_select("XAUUSD#", True)
    x = mt5.copy_ticks_range("XAUUSD#", datetime(2026, 10, 22, tzinfo=timezone.utc),
                             datetime(2026, 10, 24, 12, tzinfo=timezone.utc), mt5.COPY_TICKS_INFO)
    if x is None:
        raise RuntimeError(f"MT5 틱 없음: {mt5.last_error()}")
    return x["time_msc"], x["bid"], x["ask"]


def at(t, mid, ms):
    k = int(np.searchsorted(t, ms, side="right")) - 1
    return mid[k] if k >= 0 else np.nan


def measure(t, bid, ask, ms):
    k0 = int(np.searchsorted(t, ms))
    if k0 >= len(t) or t[k0] - ms > 20_000 or k0 == 0:
        return None
    k10 = int(np.searchsorted(t, ms + 600_000))
    if k10 - k0 < 20:
        return None
    mid = (bid + ask) / 2
    m0 = mid[k0 - 1]
    seg = mid[k0:k10] - m0
    st = t[k0:k10]
    r = {"가격": round(m0, 2),
         "1분": round(at(t, mid, ms + 60_000) - m0, 2),
         "10분": round(at(t, mid, ms + 600_000) - m0, 2),
         "최대위": round(seg.max(), 2), "최대아래": round(seg.min(), 2),
         "전30분": round(m0 - at(t, mid, ms - 1_800_000), 2)}
    in5 = st < ms + 300_000
    up, dn = np.flatnonzero(in5 & (seg >= 2)), np.flatnonzero(in5 & (seg <= -2))
    fu, fd = (up[0] if up.size else 10 ** 9), (dn[0] if dn.size else 10 ** 9)
    r["2$먼저"] = 0 if fu == fd else (1 if fu < fd else -1)
    r["2$초"] = round((st[min(fu, fd)] - ms) / 1000, 1) if r["2$먼저"] else np.nan
    return r


def side(x):
    return np.where(x >= FLAT, 1, np.where(x <= -FLAT, -1, 0))


def pct(a):
    return f"{a.mean() * 100:.0f}% ({a.sum()}/{len(a)})"


def main():
    ev = events()
    ms_all = ev.t.values.astype("datetime64[ms]").astype(np.int64)
    rows = []
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        idx = np.flatnonzero((ms_all >= int(s.timestamp() * 1000)) & (ms_all < int(e.timestamp() * 1000)))
        if not len(idx):
            continue
        t, bid, ask = lb.load_month(y, m)
        if (y, m) == (2026, 10):
            rt, rb, ra = recent_ticks()
            keep = t < rt[0]
            t, bid, ask = np.concatenate([t[keep], rt]), np.concatenate([bid[keep], rb]), np.concatenate([ask[keep], ra])
        for i in idx:
            r = measure(t, bid, ask, int(ms_all[i]))
            if r:
                r["i"] = i
                rows.append(r)
        del t, bid, ask
        gc.collect()
    d = pd.DataFrame(rows).set_index("i").join(ev).sort_values("t")
    d["방향10"] = side(d["10분"])
    d["방향1"] = side(d["1분"])
    d["전추세"] = side(d["전30분"])
    d["남은비율"] = np.where(d["10분"] >= 0, d["10분"] / d.최대위.where(d.최대위 > 0), d["10분"] / d.최대아래.where(d.최대아래 < 0))
    d.to_csv(OUT / "news_dir10.csv", index=False, encoding="utf-8-sig")

    print(f"발표 {len(d)}개 ({d.t.min():%Y-%m-%d} ~ {d.t.max():%Y-%m-%d})")
    print("10분 방향:", d.방향10.map({1: "위", -1: "아래", 0: "횡보"}).value_counts().to_dict(),
          f"| 10분 |변화| 중간값 {d['10분'].abs().median():.2f}$, 10분 안 최대폭 중간값 "
          f"{np.maximum(d.최대위, -d.최대아래).median():.2f}$")

    a = d[(d.방향1 != 0) & (d.방향10 != 0)]
    print("\n1) 1분 방향(±1$) = 10분 방향:", pct(a.방향1 == a.방향10))
    b = d[(d["2$먼저"] != 0)]
    same = b["2$먼저"] == b.방향10
    print("2) 2$ 먼저 닿은 쪽 = 10분 방향:", pct(same), "| 반대로 끝남:", pct(b["2$먼저"] == -b.방향10),
          f"| 그쪽으로 10분 평균 {(b['2$먼저'] * b['10분']).mean():+.2f}$, 2$ 닿기까지 중간값 {b['2$초'].median():.0f}초")
    for yr, g in b.groupby(b.t.dt.year):
        print(f"     {yr}: {pct(g['2$먼저'] == g.방향10)}, 평균 {(g['2$먼저'] * g['10분']).mean():+.2f}$")
    c = d[(d.전추세 != 0) & (d.방향10 != 0)]
    print("3) 발표 전 30분 추세 방향으로 10분:", pct(c.전추세 == c.방향10))
    rr = d.남은비율.dropna()
    print(f"5) 10분 끝에 남은 비율 (같은 쪽 최대폭 대비) 중간값 {rr.median() * 100:.0f}%,"
          f" 최대폭의 절반 이상 남은 경우 {pct(rr >= 0.5)}")

    print("\n4) 서프라이즈(실제-예상) 부호와 10분 방향 (발표 종류별, 10번 이상)")
    s = d[(d.sur != 0) & d.sur.notna() & (d.방향10 != 0) & (d.n == 1)]
    k = s.assign(m=(s.sur == s.방향10)).groupby("key").agg(횟수=("m", "size"), 같은방향=("m", "mean"),
                                                         평균10분=("10분", lambda x: x.abs().mean()))
    k = k[k.횟수 >= 10]
    k["일관성"] = np.maximum(k.같은방향, 1 - k.같은방향)
    k["금 반응"] = np.where(k.같은방향 >= 0.5, "좋게 나오면 금↑", "좋게 나오면 금↓")
    k = k.sort_values("일관성", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print(k.round(2).head(25).to_string())
    usd = s[s.key.str.startswith("USD")]
    print("   USD 전체: 예상보다 높게 나오면 금 10분", pct(usd.sur == -usd.방향10).replace("%", "% 아래"))

    print("\n6) 받은 거래 사진의 뉴스 시각")
    hk = d[d.t.isin(pd.to_datetime(HK_TIMES))]
    cols = ["t", "keys", "전30분", "1분", "2$먼저", "2$초", "최대위", "최대아래", "10분", "sur"]
    print(hk[cols].to_string(index=False))
    miss = sorted(set(pd.to_datetime(HK_TIMES)) - set(hk.t))
    if miss:
        print("   캘린더 high에 없는 시각:", [f"{x:%m-%d %H:%M}" for x in miss])


if __name__ == "__main__":
    main()
