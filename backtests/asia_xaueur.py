"""아시아 돌파 v2를 XAUEUR에 두 번째 봇으로 돌려도 되는지 검증 (실제 틱, 2022-01 ~ 2026-10)

규칙은 asia_breakout_trader.py v2 그대로 (all_symbols.asia_row와 같은 계산):
  레인지 01:05~08:55, 09:00~17:55 첫 돌파, 5일 모멘텀, U = 0.5 x 일봉 ATR(14), 익절 U / 손절 0.75U,
  가중치 1.5(전일 고저 돌파)/0.5, 마틴 3단계(0 이하면 x2), 23:00 청산, 1만$ 시작, 1% x 가중치 / U (최소 lot 무시)
XAUEUR는 거래 결과를 U 단위로 계산하므로 유로 표시 가격이어도 % 손익은 그대로 (환율 영향 없음).

1. 구간 나누기: IS 2022~2024 / OOS 2025-01~2026-10
2. 민감도: 레인지 끝 08:00/09:00/10:00 x 손절 0.6/0.75/1.0U x 익절 0.8/1.0/1.2U x 모멘텀 필터 켬/끔
3. 스프레드: 실제 스프레드 비교, 스프레드 1.5배 (Ask를 0.5 x 스프레드만큼 더 벌림. 봉은 Bid라 신호는 같음)
4. 금 아시아와 상관: 일별 손익 상관, 같은 날 같은 방향 비율, 한 계좌 합산 (각자 1% 기준, 마틴 따로)
5. 스왑(당일 청산 확인), XAUEUR와 금 거래 시간이 다른 날

사용법: python -m backtests.asia_xaueur          (틱 결과를 results/asia_xaueur/outcomes.pkl에 저장해 두고 씀)
        python -m backtests.asia_xaueur --fresh  (틱부터 다시)
"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.all_symbols as al
from backtests.asia_breakout import ATR_MULT, ENTRY_UNTIL, EXIT_AT, MIN_RANGE_BARS, MOM_DAYS, RANGE_FROM
from backtests.gold_rest_others import resample

OUT = Path(__file__).resolve().parent.parent / "results" / "asia_xaueur"
SYMS = ["XAUEUR", "XAUUSD"]
RANGE_ENDS = [480, 540, 600]          # 08:00 / 09:00(기본) / 10:00 - 레인지 끝이자 진입 시작
SLS = [0.6, 0.75, 1.0]
TPS = [0.8, 1.0, 1.2]
SPREAD_MULTS = [1.0, 1.5]
BASE = dict(re=540, sl=0.75, tp=1.0, mom=True, sm=1.0)
OOS_FROM = pd.Timestamp("2025-01-01")
BAL0 = 10_000.0


# ---------------------------------------------------------------- 신호
def signals(m1):
    """레인지 끝마다 그날 첫 돌파 (모멘텀 필터 전, mom 열로 나중에 거름)"""
    m1 = m1.copy()
    m1["day"] = m1.time.dt.date
    d1 = m1.groupby("day").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    pc = d1.close.shift()
    tr = np.maximum(d1.high - d1.low, np.maximum((d1.high - pc).abs(), (d1.low - pc).abs()))
    tr.iat[0] = d1.high.iat[0] - d1.low.iat[0]
    a = tr.ewm(alpha=1 / 14, adjust=False).mean()
    feat = pd.DataFrame({"atr": a.shift(), "mom": np.sign(d1.close - d1.close.shift(MOM_DAYS)).shift(),
                         "ph": d1.high.shift(), "pl": d1.low.shift()})
    m5 = resample(m1, 5)
    m5["day"] = m5.time.dt.date
    m5["mins"] = m5.time.dt.hour * 60 + m5.time.dt.minute
    rows = []
    for day, g in m5.groupby("day"):
        if day not in feat.index or not np.isfinite(feat.atr[day]) or d1.index.get_loc(day) < 30:
            continue
        f = feat.loc[day]
        for re in RANGE_ENDS:
            r = g[(g.mins >= RANGE_FROM) & (g.mins < re)]
            if len(r) < MIN_RANGE_BARS:
                continue
            hi, lo = r.high.max(), r.low.min()
            w = g[(g.mins >= re) & (g.mins < ENTRY_UNTIL)]
            b = w[(w.close > hi) | (w.close < lo)]
            if b.empty:
                continue
            bar = b.iloc[0]
            side = 1 if bar.close > hi else -1
            strong = bar.close > f.ph if side == 1 else bar.close < f.pl
            rows.append((re, pd.Timestamp(day), bar.time + pd.Timedelta(minutes=5), side, f.mom, f.atr, bool(strong)))
    return pd.DataFrame(rows, columns=["re", "day", "t", "side", "mom", "atr", "strong"])


# ---------------------------------------------------------------- 틱 결과
def outcomes(sym, sig):
    """신호 x 손절 x 익절 x 스프레드 배수 -> ru (U 단위 결과). 경로의 누적 최고/최저로 한 번에 계산"""
    res = []
    for y, m in al.month_list(sym):
        g = sig[(sig.day.dt.year == y) & (sig.day.dt.month == m)]
        if g.empty:
            continue
        t, bid, ask = al.load(sym, y, m)
        sp = ask - bid
        for r in g.itertuples():
            u = r.atr * ATR_MULT
            ems = al.to_ms(np.datetime64(r.t))
            xms = al.to_ms(np.datetime64(r.day + pd.Timedelta(minutes=EXIT_AT)))
            k = int(np.searchsorted(t, ems))
            if k >= len(t) or t[k] - ems > 600_000 or t[k] >= xms:
                continue
            fc = int(np.searchsorted(t, xms))
            if fc >= len(t):
                continue
            for smul in SPREAD_MULTS:
                a = ask if smul == 1.0 else ask + (smul - 1) * sp
                if r.side == 1:
                    e, path = a[k], (bid[k + 1:fc] - a[k])
                    xpx = bid[fc] - e
                else:
                    e, path = bid[k], (bid[k] - a[k + 1:fc])
                    xpx = e - a[fc]
                if len(path):
                    cmax, cmin = np.maximum.accumulate(path), np.minimum.accumulate(path)
                for sl in SLS:
                    for tp in TPS:
                        jt = int(np.searchsorted(cmax, tp * u)) if len(path) else 0
                        js = int(np.searchsorted(-cmin, sl * u)) if len(path) else 0
                        n = len(path)
                        if jt >= n and js >= n:
                            ru, why, jx = xpx / u, "TIME", fc
                        elif js <= jt:
                            ru, why, jx = path[js] / u, "SL", k + 1 + js
                        else:
                            ru, why, jx = path[jt] / u, "TP", k + 1 + jt
                        res.append((sym, r.re, r.day, int(t[k]), r.side, r.mom, r.strong, sl, tp, smul, ru, why,
                                    int(t[jx]), (ask[k] - bid[k]) / u, ask[k] - bid[k], u, bid[k]))
        del t, bid, ask, sp
        gc.collect()
        print(f"  {sym} {y}-{m:02d}", end="\r", flush=True)
    print()
    return pd.DataFrame(res, columns=["sym", "re", "day", "en_ms", "side", "mom", "strong", "sl", "tp", "sm", "ru",
                                      "why", "ex_ms", "sp_u", "sp", "u", "px"])


def get_outcomes(fresh):
    f = OUT / "outcomes.pkl"
    if f.exists() and not fresh:
        return pd.read_pickle(f)
    parts = []
    for sym in SYMS:
        m1 = al.m1_bars(sym)
        sig = signals(m1)
        print(f"[{sym}] 신호 {len(sig)}개 (레인지 끝 3가지, 모멘텀 전)", flush=True)
        parts.append(outcomes(sym, sig))
        del m1
        gc.collect()
    d = pd.concat(parts, ignore_index=True)
    d.to_pickle(f)
    return d


def pick(d, sym, re=540, sl=0.75, tp=1.0, mom=True, sm=1.0):
    x = d[(d.sym == sym) & (d.re == re) & (d.sl == sl) & (d.tp == tp) & (d.sm == sm)]
    if mom:
        x = x[x.side == x.mom]
    return x.sort_values("en_ms").reset_index(drop=True)


# ---------------------------------------------------------------- 자금관리
def martin(trs, bal0=BAL0, risk=0.01):
    """trs: {이름: 거래표}. 한 계좌, 이름마다 마틴 단계 따로. 진입 순서대로, 1단계 진입 때 잔고가 사이클 잔고.
    -> (시각별 잔고 표)"""
    ev = pd.concat([x.assign(key=k) for k, x in trs.items()]).sort_values("en_ms")
    bal = bal0
    lv = {k: 0 for k in trs}
    cyc = {k: bal0 for k in trs}
    out = []
    for r in ev.itertuples():
        if lv[r.key] == 0:
            cyc[r.key] = bal
        bal += cyc[r.key] * risk * (1.5 if r.strong else 0.5) * 2 ** lv[r.key] * r.ru
        out.append((pd.Timestamp(r.ex_ms, unit="ms"), r.day, bal))
        lv[r.key] = (lv[r.key] + 1 if lv[r.key] + 1 < 3 else 0) if r.ru <= 0 else 0
    return pd.DataFrame(out, columns=["t", "day", "bal"])


def worst_month(b, bal0=BAL0):
    s = b.set_index("t").bal.resample("ME").last().dropna()
    r = s / np.r_[bal0, s.values[:-1]] - 1
    return round(r.min() * 100, 1), str(r.idxmin().strftime("%Y-%m"))


def stats(x, bal0=BAL0):
    b = martin({"a": x}, bal0)
    wm = worst_month(b, bal0) if len(b) else (np.nan, "")
    return {"거래": len(x), "승률%": round((x.ru > 0).mean() * 100, 1), "PF": al.pf(x.ru.values),
            "평균U": round(x.ru.mean(), 3), "최종/시작": round(b.bal.iat[-1] / bal0, 2) if len(b) else 1.0,
            "최대낙폭%": al.mdd(b.bal, bal0) if len(b) else 0.0, "최악의달%": wm[0]}


# ---------------------------------------------------------------- 1. IS / OOS
def part1(d):
    rows = []
    for sym in SYMS:
        x = pick(d, sym)
        full = martin({"a": x})
        for name, g in (("전체", x), ("IS 2022~24", x[x.day < OOS_FROM]), ("OOS 2025~26.10", x[x.day >= OOS_FROM])):
            rows.append({"종목": sym, "구간": name, **stats(g)})
        rows[-3].update(al.yearly(full.t, full.bal, BAL0))
    return pd.DataFrame(rows).fillna("")


# ---------------------------------------------------------------- 2. 민감도
def part2(d, sym="XAUEUR", sm=1.0):
    rows = []
    for re in RANGE_ENDS:
        for mom in (True, False):
            for sl in SLS:
                for tp in TPS:
                    x = pick(d, sym, re, sl, tp, mom, sm)
                    s = stats(x)
                    rows.append({"레인지끝": f"{re // 60:02d}:00", "모멘텀": "켬" if mom else "끔", "손절U": sl, "익절U": tp,
                                 **s, "IS평균U": round(x[x.day < OOS_FROM].ru.mean(), 3),
                                 "OOS평균U": round(x[x.day >= OOS_FROM].ru.mean(), 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. 스프레드
def part3(d):
    rows = []
    for sym in SYMS:
        x = pick(d, sym)
        x15 = pick(d, sym, sm=1.5)
        s1, s15 = stats(x), stats(x15)
        rows.append({"종목": sym, "진입 스프레드 평균": round(x.sp.mean(), 3), "중간값": round(x.sp.median(), 3),
                     "가격 대비 bp": round((x.sp / x.px).mean() * 1e4, 2), "U 대비 %": round(x.sp_u.mean() * 100, 2),
                     "평균U(1x)": s1["평균U"], "최종/시작(1x)": s1["최종/시작"], "낙폭%(1x)": s1["최대낙폭%"],
                     "평균U(1.5x)": s15["평균U"], "PF(1.5x)": s15["PF"], "최종/시작(1.5x)": s15["최종/시작"],
                     "낙폭%(1.5x)": s15["최대낙폭%"], "OOS평균U(1.5x)": round(x15[x15.day >= OOS_FROM].ru.mean(), 3)})
    return pd.DataFrame(rows)


def session_spread(sym):
    """09:00~18:00 전체 틱 스프레드 (진입 순간만이 아닌 평소 수준)"""
    out = []
    for y, m in al.month_list(sym):
        t, bid, ask = al.load(sym, y, m)
        mins = (t // 60_000) % 1440
        w = (mins >= 540) & (mins < 1080)
        sp = (ask - bid)[w]
        out.append((y, np.median(sp), np.median(sp / bid[w]) * 1e4))
        del t, bid, ask
    o = pd.DataFrame(out, columns=["y", "sp", "bp"]).groupby("y").mean().round(3)
    o.insert(0, "종목", sym)
    return o.reset_index()


# ---------------------------------------------------------------- 4. 금과 합치기
def part4(d):
    e, g = pick(d, "XAUEUR"), pick(d, "XAUUSD")
    de = e.assign(w=np.where(e.strong, 1.5, 0.5) * e.ru).set_index("day")
    dg = g.assign(w=np.where(g.strong, 1.5, 0.5) * g.ru).set_index("day")
    both = de.join(dg, lsuffix="_e", rsuffix="_g", how="inner")
    union = pd.concat([de.w.rename("e").groupby(level=0).sum(), dg.w.rename("g").groupby(level=0).sum()], axis=1, sort=True).fillna(0)
    info = {"XAUEUR 거래일": len(de), "금 거래일": len(dg), "둘 다 거래한 날": len(both),
            "같은 방향 %": round((both.side_e == both.side_g).mean() * 100, 1),
            "같은 날 결과 같음(둘 다 +/둘 다 -) %": round(((both.ru_e > 0) == (both.ru_g > 0)).mean() * 100, 1),
            "일별 손익 상관 (어느 쪽이든 거래한 날, 없으면 0)": round(union.e.corr(union.g), 2),
            "일별 손익 상관 (둘 다 거래한 날)": round(both.w_e.corr(both.w_g), 2)}
    rows = []
    for name, trs, risk in (("금만 1%", {"g": g}, 0.01), ("XAUEUR만 1%", {"e": e}, 0.01),
                            ("금 + XAUEUR 각 1% (한 계좌)", {"g": g, "e": e}, 0.01),
                            ("금 + XAUEUR 각 0.7%", {"g": g, "e": e}, 0.007),
                            ("금 + XAUEUR 각 0.6%", {"g": g, "e": e}, 0.006),
                            ("금 + XAUEUR 각 0.5%", {"g": g, "e": e}, 0.005)):
        b = martin(trs, risk=risk)
        day = b.groupby("day").bal.last()
        dr = day / np.r_[BAL0, day.values[:-1]] - 1
        wm = worst_month(b)
        r = {"구성": name, "거래": len(b), "최종$": round(b.bal.iat[-1]), "최대낙폭%": al.mdd(b.bal, BAL0),
             "최악의달%": wm[0], "최악의달": wm[1],
             "최악의날%": round(dr.min() * 100, 1), "수익/낙폭": round((b.bal.iat[-1] / BAL0 - 1) * 100 / al.mdd(b.bal, BAL0), 1),
             **al.yearly(b.t, b.bal, BAL0)}
        bi, bo = b[b.day < OOS_FROM], b[b.day >= OOS_FROM]
        r["OOS 낙폭%"] = al.mdd(bo.bal, bi.bal.iat[-1])
        rows.append(r)
    return info, pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. 스왑 / 거래시간
def part5(d):
    out = {}
    for sym in SYMS:
        x = pick(d, sym)
        ex = pd.to_datetime(x.ex_ms, unit="ms")
        late = ex > x.day + pd.Timedelta(minutes=EXIT_AT + 10)
        out[sym] = {"거래": len(x), "TP/SL/TIME": "/".join(str((x.why == w).sum()) for w in ("TP", "SL", "TIME")),
                    "청산 최대 시각": str(ex.dt.strftime("%H:%M:%S").max()),
                    "23:10 넘어 청산": int(late.sum()), "다음날로 넘어감": int((ex.dt.normalize() > x.day).sum())}
    hours = {}
    for sym in SYMS:
        m1 = al.m1_bars(sym)
        m1 = m1[m1.time >= "2022-01-01"]
        g = m1.groupby(m1.time.dt.date).time
        hours[sym] = pd.DataFrame({"first": g.min().dt.strftime("%H:%M"), "last": g.max().dt.strftime("%H:%M"),
                                   "bars": g.size()})
        del m1
    h = hours["XAUEUR"].join(hours["XAUUSD"], lsuffix="_eur", rsuffix="_usd", how="outer")
    h = h[pd.to_datetime(h.index).weekday < 5]
    diff = h[(h.first_eur != h.first_usd) | (h.last_eur != h.last_usd) | ((h.bars_eur - h.bars_usd).abs() > 15)]
    return out, h, diff


# ---------------------------------------------------------------- 실행
def show(title, df):
    print(f"\n=== {title} ===")
    print(df.to_string(index=False) if isinstance(df, pd.DataFrame) else df)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    d = get_outcomes("--fresh" in sys.argv)

    p1 = part1(d)
    p1.to_csv(OUT / "1_is_oos.csv", index=False, encoding="utf-8-sig")
    show("1. IS / OOS (기본 규칙, 각 구간 1만$ 새로 시작, 연도별은 전체 연속)", p1)

    p2 = part2(d)
    p2.to_csv(OUT / "2_sensitivity_xaueur.csv", index=False, encoding="utf-8-sig")
    show("2. XAUEUR 민감도 (54가지)", p2)
    for col in ("평균U", "최종/시작"):
        pv = p2.pivot_table(index=["레인지끝", "모멘텀", "손절U"], columns="익절U", values=col)
        show(f"2. XAUEUR {col} 격자", pv.reset_index())
    p2g = part2(d, "XAUUSD")
    p2g.to_csv(OUT / "2_sensitivity_xauusd.csv", index=False, encoding="utf-8-sig")
    show("2. (비교) 금 평균U 격자",
         p2g.pivot_table(index=["레인지끝", "모멘텀", "손절U"], columns="익절U", values="평균U").reset_index())
    p2s = part2(d, "XAUEUR", 1.5)
    p2s.to_csv(OUT / "2_sensitivity_xaueur_spread15.csv", index=False, encoding="utf-8-sig")

    p3 = part3(d)
    ss = pd.concat([session_spread(s) for s in SYMS])
    p3.to_csv(OUT / "3_spread.csv", index=False, encoding="utf-8-sig")
    ss.to_csv(OUT / "3_spread_by_year.csv", index=False, encoding="utf-8-sig")
    show("3. 스프레드 (진입 순간) 와 1.5배", p3)
    show("3. 09:00~18:00 틱 스프레드 중간값 (달별 중간값의 연평균)", ss)

    info, p4 = part4(d)
    p4.to_csv(OUT / "4_combined.csv", index=False, encoding="utf-8-sig")
    pd.Series(info).to_csv(OUT / "4_correlation.csv", encoding="utf-8-sig", header=["값"])
    show("4. 금 아시아와의 관계", "\n".join(f"  {k}: {v}" for k, v in info.items()))
    show("4. 한 계좌 합산 (각자 1% 기준, 마틴 따로)", p4)

    p5, hrs, diff = part5(d)
    hrs.to_csv(OUT / "5_session_hours.csv", encoding="utf-8-sig")
    diff.to_csv(OUT / "5_session_diff_days.csv", encoding="utf-8-sig")
    show("5. 청산 시각 (스왑)", pd.DataFrame(p5).T.reset_index().rename(columns={"index": "종목"}))
    show(f"5. 거래 시간이 금과 다른 평일 {len(diff)}일 (전체 {len(hrs)}일)", diff.reset_index())


if __name__ == "__main__":
    main()
