"""나스닥(NAS100)·독일(GER40) 전용 장중 전략 찾기 (1분봉 = 실제 틱으로 만든 Bid 봉 + 그 봉 평균 스프레드, 2022-01 ~ 2026-10)

본장 시각 (서버시간 = 뉴욕 + 7시간): NAS100 16:30 ~ 23:00 (뉴욕 09:30 ~ 16:00, 조기 마감일 13:00)
                                    GER40 Xetra 09:00 ~ 17:30 을 날마다 서버시간으로 바꿈 (보통 10:00 ~ 18:30)
  휴장일(common/exchange_calendar.py)은 뺌, 'N분'은 봉 개수가 아니라 본장 첫 봉 시각부터 잰 시간 (2026-10-24 GPT 검증으로 고침)
전략 (하루 한 번, 본장 안에서만, 마감 봉에 청산)
  ORB N분      : 본장 처음 N분 고가·저가, 그 뒤 1분봉 종가가 넘으면 다음 봉 시가에 진입 (진입은 시가 + 3시간 전까지)
                 손절 = 범위 반대쪽(opp) 또는 범위 가운데(mid), 익절 = 범위 x k (없으면 마감까지)
  갭 메우기/따라가기 : 갭 = 본장 첫 봉 시가 - 전날 본장 마지막 봉 종가, |갭| >= g x 일봉 ATR(전날까지)이면
                 시가 + w분 봉 시가에 진입. 메우기는 전날 종가에 익절, 손절 = 진입가에서 갭 x s 반대. 따라가기는 손절만
                 청산: 본장 시가 + h분 또는 마감
  첫 30분 방향  : 시가 -> 시가+30분 움직임이 일봉 ATR x m 넘으면 그 방향으로 시가+30분에 진입, 마감 / 손절 ATR x s
  RSI2 되돌림   : 본장 중 5분봉 RSI(2) < 10 매수 / > 90 매도, RSI(2)가 50 넘으면 청산, 손절 ATR x s, 최대 h분, 하루 여러 번
체결: 신호 봉 다음 1분봉 시가. 매수 = Bid + 스프레드 + 미끄러짐, 매도 = Bid - 미끄러짐 (청산도 같은 방식, 매수 청산 Bid)
      한 봉 안에서 손절·익절이 둘 다 닿으면 손절 먼저 (보수적). 미끄러짐 SLIP = 그 봉 스프레드의 절반을 체결마다 더함
결과: 포인트와 '일봉 ATR 단위(R)'. 고르기 = 2022~2024, 확인 = 2025~2026. results/index_intraday.csv

사용법: python -m backtests.index_intraday
"""
import itertools
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common.exchange_calendar import de_closed, us_close_time, us_closed
from indicators import rsi

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "gpt_data"
NY, BER = ZoneInfo("America/New_York"), ZoneInfo("Europe/Berlin")
ENTRY_UNTIL = 180     # 진입은 본장 시작 + 180분 전까지 (ORB)
SESS = {"NAS100": (NY, dtime(9, 30), dtime(16, 0)), "GER40": (BER, dtime(9, 0), dtime(17, 30))}


def server(day, tz, t):
    """그 날짜 현지 시각 -> 서버시간 (뉴욕 + 7시간)"""
    loc = datetime.combine(day, t, tzinfo=tz)
    return (loc.astimezone(NY) + timedelta(hours=7)).replace(tzinfo=None)


def sessions(sym, m1):
    tz, o, c = SESS[sym]
    days = sorted(set(m1.index.date))
    t = m1.index.values
    out = []
    for d in days:
        if (us_closed(d) if sym == "NAS100" else de_closed(d)):       # 휴장일 (CFD 시세만 있음)
            continue
        cl = us_close_time(d, c) if sym == "NAS100" else c             # 미국 조기 마감 13:00
        so, sc = np.datetime64(server(d, tz, o)), np.datetime64(server(d, tz, cl))
        a, b = np.searchsorted(t, [so, sc])
        if b - a < 60 or t[a] - so > np.timedelta64(5, "m"):
            continue
        out.append((d, a, b))
    return out


class Day:
    __slots__ = ("o", "h", "l", "c", "sp", "m", "atr", "prev_close", "y")


def prep(sym):
    m1 = pd.read_csv(DATA / sym / "M1.csv.gz", index_col="time", parse_dates=True)
    m1 = m1[~m1.index.duplicated()].sort_index()
    ss = sessions(sym, m1)
    o, h, l, c, sp = (m1[x].values for x in ("open", "high", "low", "close", "spread"))
    rows, prev = [], None
    rng = []
    t = m1.index.values
    for d, a, b in ss:
        x = Day()
        x.o, x.h, x.l, x.c, x.sp = o[a:b], h[a:b], l[a:b], c[a:b], sp[a:b]
        x.m = ((t[a:b] - t[a]) // np.timedelta64(1, "m")).astype(int)    # 본장 첫 봉부터 몇 분째 (봉이 빠져도 시각 기준)
        x.atr = np.mean(rng[-14:]) if len(rng) >= 14 else np.nan
        x.prev_close, x.y = prev, d.year
        prev = c[b - 1]
        rng.append(h[a:b].max() - l[a:b].min())
        if not np.isnan(x.atr) and x.prev_close is not None:
            rows.append(x)
    return rows


def fill(x, k, side, entering):
    """k번째 봉 시가 체결가. 진입 매수 / 청산 매도(=매도 진입 청산) 구분"""
    sl = x.sp[k] / 2
    buy = (side > 0) == entering
    return x.o[k] + x.sp[k] + sl if buy else x.o[k] - sl


def run_trade(x, k, side, stop=None, take=None, until=None):
    """k번째 봉 시가 진입 -> 손익 포인트. stop/take는 Bid 기준 가격, until = 이 봉 시가에 청산 (없으면 마지막 봉 종가)"""
    e = fill(x, k, side, True)
    n = len(x.o)
    end = min(until if until is not None else n, n)
    for j in range(k, end):
        lo, hi = x.l[j], x.h[j]
        # 손절·익절 가격은 Bid 기준. 매도 청산은 Ask(= Bid + 스프레드), 손절은 미끄러짐도 더함
        if stop is not None and ((side > 0 and lo <= stop) or (side < 0 and hi >= stop)):
            px = min(stop, x.o[j]) - x.sp[j] / 2 if side > 0 else max(stop, x.o[j]) + x.sp[j] * 1.5
            return (px - e) if side > 0 else (e - px), j
        if take is not None and ((side > 0 and hi >= take) or (side < 0 and lo <= take)):
            px = take if side > 0 else take + x.sp[j]
            return (px - e) if side > 0 else (e - px), j
    if end < n:
        px = fill(x, end, side, False)
    else:
        px = x.c[n - 1] - x.sp[n - 1] / 2 if side > 0 else x.c[n - 1] + x.sp[n - 1] * 1.5
    return (px - e) if side > 0 else (e - px), end


def at(x, minute):
    """본장 시작 minute분 이후 첫 봉 번호"""
    return int(np.searchsorted(x.m, minute))


def orb(x, n, stop_mode, k_tp, only=0):
    kn = at(x, n)
    if len(x.o) <= kn + 2:
        return []
    hi, lo = x.h[:kn].max(), x.l[:kn].min()
    r = hi - lo
    if r <= 0:
        return []
    for j in range(kn, len(x.o) - 1):
        if x.m[j + 1] >= ENTRY_UNTIL:              # 진입 봉이 시가 + 3시간 전이어야
            break
        side = 1 if x.c[j] > hi else (-1 if x.c[j] < lo else 0)
        if side and (not only or side == only):
            stop = (lo if side > 0 else hi) if stop_mode == "opp" else (hi + lo) / 2
            take = (hi + k_tp * r if side > 0 else lo - k_tp * r) if k_tp else None
            return [run_trade(x, j + 1, side, stop, take)[0]]
    return []


def gap(x, mode, g, w, s, hmin):
    gp = x.o[0] - x.prev_close
    w, hmin = at(x, w), (at(x, hmin) if hmin < 10_000 else None)
    if abs(gp) < g * x.atr or w + 1 >= len(x.o):
        return []
    side = -np.sign(gp) if mode == "fill" else np.sign(gp)
    e_ref = x.o[w]
    if mode == "fill" and ((gp > 0 and x.l[:w + 1].min() <= x.prev_close) or (gp < 0 and x.h[:w + 1].max() >= x.prev_close)):
        return []
    stop = e_ref - side * abs(gp) * s
    take = x.prev_close if mode == "fill" else None
    return [run_trade(x, w, int(side), stop, take, hmin)[0]]


def first30(x, m, s):
    k = at(x, 30)
    if k < 20 or k >= len(x.o) - 5:
        return []
    mv = x.c[k - 1] - x.o[0]
    if abs(mv) < m * x.atr:
        return []
    side = int(np.sign(mv))
    stop = x.o[k] - side * s * x.atr
    return [run_trade(x, k, side, stop)[0]]


def rsi2(x, lo, s, hmin):
    bucket = x.m // 5
    last = np.flatnonzero(np.r_[bucket[1:] != bucket[:-1], True])      # 5분 칸마다 마지막 봉
    n = len(last)
    if n < 10:
        return []
    r = rsi(pd.Series(x.c[last]), 2).values
    out, j = [], 2
    while j < n - 1:
        side = 1 if r[j] < lo else (-1 if r[j] > 100 - lo else 0)
        if not side:
            j += 1
            continue
        k = last[j] + 1
        exit_bar = next((last[q] + 1 for q in range(j + 1, n) if (side > 0 and r[q] > 50) or (side < 0 and r[q] < 50)), None)
        until = min(exit_bar or len(x.o), at(x, x.m[k] + hmin))
        pts, end = run_trade(x, k, side, x.o[k] - side * s * x.atr, None, until)
        out.append(pts)
        j = int(np.searchsorted(last, end)) + 1
    return out


def configs():
    yield from (("ORB", (n, sm, k, only)) for n, sm, k, only in
                itertools.product((5, 15, 30, 60), ("opp", "mid"), (0, 1, 2), (0, 1, -1)))
    yield from (("갭", (mode, g, w, s, h)) for mode, g, w, s, h in
                itertools.product(("fill", "follow"), (0.1, 0.25, 0.5), (0, 5, 15), (1.0, 2.0), (60, 180, 10_000)))
    yield from (("첫30분", (m, s)) for m, s in itertools.product((0.0, 0.1, 0.2, 0.3), (0.5, 1.0, 9.0)))
    yield from (("RSI2", (lo, s, h)) for lo, s, h in itertools.product((5, 10, 20), (0.5, 1.0), (30, 90, 240)))


FUN = {"ORB": orb, "갭": gap, "첫30분": first30, "RSI2": rsi2}


def main():
    rows = []
    for sym in SESS:
        days = prep(sym)
        print(sym, len(days), "일", flush=True)
        for name, p in configs():
            res = [(x.y, v, v / x.atr) for x in days for v in FUN[name](x, *p)]
            if len(res) < 30:
                continue
            d = pd.DataFrame(res, columns=["y", "pts", "R"])
            row = {"종목": sym, "전략": name, "설정": str(p), "매매": len(d), "승률%": round((d.pts > 0).mean() * 100, 1)}
            for per, k in (("IS", d.y <= 2024), ("OOS", d.y >= 2025)):
                v = d[k]
                g, l_ = v.pts[v.pts > 0].sum(), -v.pts[v.pts < 0].sum()
                row[f"{per} 매매"] = len(v)
                row[f"{per} PF"] = round(g / l_, 2) if l_ else np.nan
                row[f"{per} R합"] = round(v.R.sum(), 1)
            for yr in range(2022, 2027):
                row[f"{yr} R"] = round(d.R[d.y == yr].sum(), 1)
            row["플러스 해"] = sum(row[f"{yr} R"] > 0 for yr in range(2022, 2027))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "index_intraday.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    for sym in SESS:
        x = out[out["종목"] == sym]
        print(f"\n=== {sym}: 2022~24 PF 상위 (IS 60번 이상)")
        print(x[x["IS 매매"] >= 60].sort_values("IS PF", ascending=False).head(15).drop(columns=["종목"]).to_string(index=False))
        print(f"--- {sym}: 해마다 플러스")
        print(x[x["플러스 해"] == 5].sort_values("IS PF", ascending=False).head(10).drop(columns=["종목"]).to_string(index=False))


if __name__ == "__main__":
    main()
