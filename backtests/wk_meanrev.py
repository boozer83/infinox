"""BTCUSD 주말 전용 단타 - 평균회귀(되돌림) 아이디어 4가지 (실제 틱, 2022-01 ~ 2026-10)

주말 창 (서버시간): 토요일 01:00 ~ 일요일 23:50. 새 진입은 일요일 23:30 전까지, 23:50 첫 틱에 강제 청산.
신호는 완성된 1분봉/5분봉(Bid)으로만 판단하고, 봉이 끝난 뒤 첫 틱에 진입 (매수 Ask, 매도 Bid).
청산: 매수는 Bid, 매도는 Ask. 손절은 처음 닿은 틱 가격(건너뛰면 그 틱 가격), 익절은 지정가(그 레벨),
      시간 손절·조건 청산·23:50은 그 시각 첫 틱 가격. 손절/익절 거리는 진입 체결가 기준 (스프레드는 틱 그대로).
포지션은 아이디어마다 한 번에 1개. 토->일 자정을 넘기면 스왑 (매수 -25%/년, 매도 -15%/년, 하루치) 차감.

A. 볼린저 z 되돌림 : 봉 종가가 20봉 평균에서 표준편차 k배 넘게 벗어나면 반대로, 익절 = 신호 때 평균까지,
                     손절 s x ATR(14), 60분 시간 손절.  (1분봉/5분봉 x k 2/2.5/3 x s 1.5/2.5 = 12개)
B. RSI(2) 극단     : 5분봉 RSI(2) < L 매수 / > 100-L 매도, RSI(2)가 50을 넘으면(매도는 밑돌면) 그 봉 끝에 청산,
                     손절 2.5 x ATR(14), 시간 손절 T분, EMA200 방향만(선택).  (L 5/10 x T 30/90 x 필터 유무 = 8개)
C. 1분봉 급등락 되돌림 : 몸통 |종가-시가| > k x 직전 ATR(1분봉 1440 = 하루), 반대로. 손절 = 몸통 1배, 익절 = 몸통 f배,
                     시간 손절 H분.  (k 3/4/6 x f 0.4/0.7 x H 15/60 = 12개)
D. 주말 기준값 되돌림 : 기준값 = 토 01:00 시가 또는 주말 시간가중평균(5분봉 종가 평균), 5분봉 종가가 기준값에서
                     k x ATR(5분봉 288) 넘게 처음 벗어나면 반대로. 익절·손절 = 벗어난 거리의 절반, 시간 손절 H분,
                     토 03:00 이후만.  (기준값 2 x k 3/5 x H 60/180 = 8개)
합계 40개. 표본 안(IS) = 2022~2024 주말에서만 고르고, 표본 밖(OOS) = 2025-01 ~ 2026-10 로 확인.
결과는 1 BTC당 USD, R(R = 처음 손절 거리), 거래당 1% 위험 기준 % (단순 합), bp.
IS 1등 설정을 XETUSD(ETH)에도 같은 값으로 돌려 확인.

사용법: python -m backtests.wk_meanrev        (-> results/weekend/meanrev_*.csv)
"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.all_symbols import load, m1_bars, month_list
from backtests.gold_rest_others import resample
from indicators import atr, ema, rsi

OUT = Path(__file__).resolve().parent.parent / "results" / "weekend"
DAY, HOUR, MIN = 86_400_000, 3_600_000, 60_000
SWAP = {1: 0.25 / 365, -1: 0.15 / 365}
IS_END = pd.Timestamp("2025-01-01")
MIN_IS_TRADES = 150


# ---------------------------------------------------------------- 시간
def sat_of(ms):
    """ms -> (그 주말 토요일 00:00 ms, 토/일이면 True)"""
    day = ms // DAY
    dow = (day + 3) % 7          # 1970-01-01 = 목요일 -> 월=0 ... 토=5, 일=6
    return (day - (dow - 5)) * DAY, dow >= 5


def entry_ok(start_ms, end_ms):
    """봉 시작이 토 01:00 이후, 봉 끝(=진입 시각)이 일 23:30 전"""
    sat, wk = sat_of(start_ms)
    return wk & (start_ms >= sat + HOUR) & (end_ms < sat + DAY + 23 * HOUR + 30 * MIN), sat


def to_ms(times):
    return np.asarray(times, "datetime64[ms]").astype(np.int64)


# ---------------------------------------------------------------- 신호
def bars_ms(b, minutes):
    s = to_ms(b.time.values)
    return s, s + minutes * MIN


def sig_frame(start, end, side, sl, tp, ok, hold, cond=None):
    m = ok & (side != 0) & np.isfinite(sl) & (sl > 0)
    sat, _ = sat_of(start)
    d = pd.DataFrame({"ts": end[m], "side": side[m].astype(int), "sl": sl[m], "tp": tp[m], "wk": sat[m]})
    d["hold"] = hold
    return d.sort_values("ts").reset_index(drop=True), cond


def idea_bb(m1, m5, tf, k, s):
    b = m1 if tf == 1 else m5
    st, en = bars_ms(b, tf)
    c = b.close
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    z = ((c - ma) / sd).values
    a = atr(b).values
    side = np.where(z < -k, 1, np.where(z > k, -1, 0))
    ok, _ = entry_ok(st, en)
    return sig_frame(st, en, side, s * a, np.abs(c.values - ma.values), ok, 60)


def idea_rsi(m1, m5, lvl, hold, filt):
    st, en = bars_ms(m5, 5)
    r = rsi(m5.close, 2).values
    a = atr(m5).values
    side = np.where(r < lvl, 1, np.where(r > 100 - lvl, -1, 0))
    if filt:
        e = ema(m5.close, 200).values
        side = np.where(((side == 1) & (m5.close.values > e)) | ((side == -1) & (m5.close.values < e)), side, 0)
    ok, _ = entry_ok(st, en)
    cond = {1: np.sort(en[r > 50]), -1: np.sort(en[r < 50])}   # 매수 청산 / 매도 청산 시각 (봉 끝)
    return sig_frame(st, en, side, 2.5 * a, np.full(len(a), np.nan), ok, hold, cond)


def idea_spike(m1, m5, k, f, hold):
    st, en = bars_ms(m1, 1)
    body = (m1.close - m1.open).values
    a = atr(m1, 1440).shift(1).values
    side = np.where(np.abs(body) > k * a, -np.sign(body), 0)
    ok, _ = entry_ok(st, en)
    return sig_frame(st, en, side, np.abs(body), f * np.abs(body), ok, hold)


def idea_anchor(m1, m5, anchor, k, hold):
    st, en = bars_ms(m5, 5)
    sat, wk = sat_of(st)
    inwk = wk & (st >= sat + HOUR)
    c = m5.close.values
    grp = pd.Series(np.where(inwk, sat, -1))
    if anchor == "open":
        ref = pd.Series(np.where(inwk, m5.open.values, np.nan)).groupby(grp).transform("first").values
    else:
        ref = pd.Series(np.where(inwk, c, np.nan)).groupby(grp).transform(lambda x: x.expanding().mean()).values
    ref = np.where(inwk, ref, np.nan)
    a = atr(m5, 288).values
    dev = (c - ref) / a
    prev = np.r_[np.nan, dev[:-1]]
    prev = np.where(np.r_[False, (grp.values[1:] == grp.values[:-1])], prev, 0.0)  # 주말 첫 봉은 직전 0
    cross = (np.abs(dev) > k) & ~(np.abs(prev) > k)
    side = np.where(cross, -np.sign(dev), 0)
    ok, _ = entry_ok(st, en)
    ok &= st >= sat + 3 * HOUR
    d = np.abs(c - ref) * 0.5
    return sig_frame(st, en, side, d, d, ok, hold)


def configs():
    cf = []
    for tf in (1, 5):
        for k in (2.0, 2.5, 3.0):
            for s in (1.5, 2.5):
                cf.append(("A_bb", f"M{tf} k{k} sl{s}ATR", idea_bb, (tf, k, s)))
    for lvl in (5, 10):
        for hold in (30, 90):
            for filt in (False, True):
                cf.append(("B_rsi2", f"L{lvl} T{hold} {'ema200' if filt else 'nofilt'}", idea_rsi, (lvl, hold, filt)))
    for k in (3, 4, 6):
        for f in (0.4, 0.7):
            for hold in (15, 60):
                cf.append(("C_spike", f"k{k} tp{f} H{hold}", idea_spike, (k, f, hold)))
    for anc in ("open", "twap"):
        for k in (3, 5):
            for hold in (60, 180):
                cf.append(("D_anchor", f"{anc} k{k} H{hold}", idea_anchor, (anc, k, hold)))
    return cf


# ---------------------------------------------------------------- 틱 실행
def run_weekend(sig, cond, t, b, a, sat):
    close_ms = sat + DAY + 23 * HOUR + 50 * MIN
    sun0 = sat + DAY
    n = len(t)
    out, busy = [], -1
    for ts, side, sl, tp, hold in zip(sig.ts.values, sig.side.values, sig.sl.values, sig.tp.values, sig.hold.values):
        if ts < busy:
            continue
        k = np.searchsorted(t, ts)
        if k >= n - 1 or t[k] >= close_ms:
            continue
        if side > 0:
            e, xs, slp, tpp = a[k], b, a[k] - sl, a[k] + tp
        else:
            e, xs, slp, tpp = b[k], a, b[k] + sl, b[k] - tp
        end_ms = min(t[k] + hold * MIN, close_ms)
        if cond is not None:
            arr = cond[side]
            j = np.searchsorted(arr, t[k], "right")
            if j < len(arr):
                end_ms = min(end_ms, arr[j])
        kend = min(np.searchsorted(t, end_ms), n - 1)
        seg = xs[k + 1:kend]
        if side > 0:
            hs, ht = seg <= slp, seg >= tpp
        else:
            hs, ht = seg >= slp, seg <= tpp
        h = np.flatnonzero(hs | ht)
        if h.size:
            i = k + 1 + h[0]
            x, why = (xs[i], "SL") if hs[h[0]] else (tpp, "TP")
        else:
            i, x, why = kend, xs[kend], "TIME"
        pnl = (x - e) * side
        if t[k] < sun0 <= t[i]:
            pnl -= SWAP[side] * e
        out.append((sat, t[k], t[i], side, e, x, sl, pnl, why))
        busy = t[i]
    return out


def weekend_ticks(sym):
    """주말(토 01:00 ~ 일 24:00) 틱을 주말 단위로 내보냄 (달을 넘는 주말도 이어 붙임)"""
    pend = {}
    for y, m in month_list(sym):
        t, bid, ask = load(sym, y, m)
        sat, wk = sat_of(t)
        keep = wk & (t >= sat + HOUR)
        t, bid, ask, sat = t[keep], bid[keep], ask[keep], sat[keep]
        for s in np.unique(sat):
            mm = sat == s
            pend.setdefault(int(s), []).append((t[mm], bid[mm], ask[mm]))
        del t, bid, ask, sat
        gc.collect()
        month_end = to_ms(np.datetime64(pd.Timestamp(y, m, 1) + pd.offsets.MonthBegin(1)))
        for s in sorted(pend):
            if s + 2 * DAY <= month_end:
                parts = pend.pop(s)
                yield s, np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), \
                    np.concatenate([p[2] for p in parts])
    for s in sorted(pend):
        parts = pend.pop(s)
        yield s, *(np.concatenate([p[j] for p in parts]) for j in range(3))


def backtest(sym, cfs):
    m1 = m1_bars(sym)
    m5 = resample(m1, 5)
    sigs = []
    for fam, name, fn, args in cfs:
        d, cond = fn(m1, m5, *args)
        sigs.append((fam, name, {s: g for s, g in d.groupby("wk")}, cond))
    rows = []
    nwk = 0
    for s, t, b, a in weekend_ticks(sym):
        nwk += 1
        for fam, name, by, cond in sigs:
            if s in by:
                for r in run_weekend(by[s], cond, t, b, a, s):
                    rows.append((fam, name) + r)
        print(f"\r{sym} 주말 {nwk} {pd.Timestamp(s, unit='ms').date()}", end="", file=sys.stderr)
    print(file=sys.stderr)
    tr = pd.DataFrame(rows, columns=["family", "config", "wk", "t_in", "t_out", "side", "entry", "exit", "risk",
                                     "pnl", "why"])
    tr["R"] = tr.pnl / tr.risk
    tr["bp"] = tr.pnl / tr.entry * 1e4
    tr["wkdate"] = pd.to_datetime(tr.wk, unit="ms")
    tr["year"] = tr.wkdate.dt.year
    return tr, nwk


# ---------------------------------------------------------------- 요약
def stats(g, nwk):
    if len(g) == 0:
        return dict(trades=0)
    R = g.R.values
    gp, gl = R[R > 0].sum(), -R[R < 0].sum()
    cum = np.cumsum(R)
    dd = (np.maximum.accumulate(np.r_[0, cum])[1:] - cum).max()
    wkR = g.groupby("wk").R.sum()
    return dict(trades=len(g), per_wk=round(len(g) / nwk, 2), pf_R=round(gp / gl, 3) if gl else np.inf,
                pf_usd=round(g.pnl[g.pnl > 0].sum() / -g.pnl[g.pnl < 0].sum(), 3),
                win=round((R > 0).mean() * 100, 1), sumR=round(R.sum(), 1), pct_1risk=round(R.sum(), 1),
                usd_1btc=round(g.pnl.sum()), avg_bp=round(g.bp.mean(), 2), avgR=round(R.mean(), 3),
                maxddR=round(dd, 1), worst_wkR=round(wkR.min(), 1))


def summarize(tr, cfs, nwk_is, nwk_oos):
    rows = []
    for fam, name, _, _ in cfs:
        g = tr[(tr.family == fam) & (tr.config == name)]
        for per, gg, nw in (("IS", g[g.wkdate < IS_END], nwk_is), ("OOS", g[g.wkdate >= IS_END], nwk_oos)):
            rows.append(dict(family=fam, config=name, period=per, **stats(gg, nw)))
    return pd.DataFrame(rows)


def yearly(tr, picks, nwk_year):
    rows = []
    for fam, name in picks:
        g = tr[(tr.family == fam) & (tr.config == name)]
        for y, gy in g.groupby("year"):
            rows.append(dict(family=fam, config=name, year=y, **stats(gy, nwk_year.get(y, 1))))
    return pd.DataFrame(rows)


def week_counts(sym):
    """주말 수 (전체/IS/OOS/연도별) - 틱이 있는 달 기준"""
    sats = set()
    for y, m in month_list(sym):
        d = pd.date_range(pd.Timestamp(y, m, 1), pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0))
        sats.update(x for x in d if x.dayofweek == 5)
    s = pd.Series(sorted(sats))
    s = s[s <= pd.Timestamp("2026-10-24")]
    return (s < IS_END).sum(), (s >= IS_END).sum(), s.dt.year.value_counts().to_dict()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    cfs = configs()
    tr, _ = backtest("BTCUSD", cfs)
    nis, noos, ny = week_counts("BTCUSD")
    tr.to_csv(OUT / "meanrev_trades_BTCUSD.csv", index=False)
    sm = summarize(tr, cfs, nis, noos)
    sm.to_csv(OUT / "meanrev_summary_BTCUSD.csv", index=False)
    print(sm.to_string())

    # IS에서만 고르기: 거래 150회 이상 중 PF(R) 순
    isd = sm[(sm.period == "IS") & (sm.trades >= MIN_IS_TRADES)].sort_values("pf_R", ascending=False)
    best = isd.groupby("family").head(1)
    top3 = isd.head(3)
    picks = list(dict.fromkeys(list(zip(best.family, best.config)) + list(zip(top3.family, top3.config))))
    sel = sm.set_index(["family", "config"]).loc[picks].reset_index()
    sel["pick"] = [("IS-best " if (f, c) in set(zip(best.family, best.config)) else "") +
                   ("top3" if (f, c) in set(zip(top3.family, top3.config)) else "") for f, c in sel[["family", "config"]].values]
    sel.to_csv(OUT / "meanrev_picks_BTCUSD.csv", index=False)
    print("\n== IS에서 고른 설정 (IS / OOS)\n", sel.to_string())
    yr = yearly(tr, picks, ny)
    yr.to_csv(OUT / "meanrev_yearly_BTCUSD.csv", index=False)
    print("\n== 연도별\n", yr[["family", "config", "year", "trades", "pf_R", "win", "sumR", "avg_bp", "maxddR",
                              "worst_wkR"]].to_string())

    # ETH 확인 (IS-best 설정 그대로)
    ecf = [c for c in cfs if (c[0], c[1]) in set(zip(best.family, best.config))]
    del tr
    gc.collect()
    etr, _ = backtest("XETUSD", ecf)
    eis, eoos, ey = week_counts("XETUSD")
    etr.to_csv(OUT / "meanrev_trades_XETUSD.csv", index=False)
    esm = summarize(etr, ecf, eis, eoos)
    esm.to_csv(OUT / "meanrev_eth_XETUSD.csv", index=False)
    print("\n== XETUSD (같은 설정)\n", esm.to_string())
    eyr = yearly(etr, [(c[0], c[1]) for c in ecf], ey)
    eyr.to_csv(OUT / "meanrev_eth_yearly_XETUSD.csv", index=False)
    print(eyr[["family", "config", "year", "trades", "pf_R", "sumR", "avg_bp"]].to_string())


if __name__ == "__main__":
    main()
