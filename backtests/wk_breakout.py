"""주말 전용 BTCUSD 단타 - 돌파·모멘텀 계열 (실제 틱, 2022-01 ~ 2026-10)

주말(토 01:00 ~ 일 23:50, 서버시간)에만 거래. 하루 단위로 끊음:
  토요일 01:00 ~ 23:30 진입, 23:50 이후 첫 틱에 강제 청산 / 일요일 00:00 ~ 23:30 진입, 23:50 강제 청산
  -> 자정을 넘겨 들고 가지 않으므로 스왑 없음. 포지션은 한 번에 1개.
신호는 완성된 봉(1분봉 Bid로 만든 5/15/60/240분봉, 평일 봉은 지표 준비용)으로만, 진입은 봉 끝 이후 첫 틱
(매수 Ask, 매도 Bid, 신호 후 60초 안에 틱이 없으면 버림). 손절·익절·트레일링은 틱마다 확인, 건너뛰면 그 틱 가격.
트레일링은 진입 후 '이전 틱까지'의 최고(최저) Bid(Ask)에서 거리만큼 (미래 틱 안 봄).

아이디어 (각 8개, 모두 40개):
  A. ORB      : 그날 첫 N시간(토 01시~ / 일 00시~) Bid 고가·저가, 이후 Bid가 처음 넘는 틱에 그 방향 진입(하루 1번),
                손절 = 반대편 또는 중간, 익절 = 1R/2R, 시간청산 = 그날 23:50
  B. 돈치안    : M5/M15 종가가 직전 N봉 최고가(최저가) 돌파, 손절 1.5 ATR, 익절 2R(12봉) 또는 트레일 1.5 ATR(24봉)
  C. 스퀴즈    : 볼린저(20,2) 폭이 직전 288봉 중 하위 p% 인 상태에서 종가가 밴드 밖, 손절 = 중심선,
                익절 2R 또는 트레일 1R, 24봉 시간청산
  D. 급등락 추종: 1분봉 범위가 직전 60봉 평균의 k배 + 종가가 위 30%(아래 30%) -> 그 방향,
                손절 = 그 봉 반대 끝 또는 중간, 익절 1R(30분) 또는 트레일 1R(60분)
  E. 눌림 추세  : H1/H4 종가 > EMA50 (아래면 매도), M5 RSI가 최근 6봉 안에 lo 아래(위면 100-lo 위)였다가 50 재돌파,
                손절 1.5 ATR(M5), 익절 1R/2R, 2시간 시간청산
검증: IS = 2022-01 ~ 2024-12 주말에서만 설정 고름(거래 100회 이상 중 PF 최고), OOS = 2025-01 ~ 2026-10.
R = 처음 손절 거리, 1% 위험 고정이면 % 수익 = R 합. 수수료 없음, 스프레드는 틱 그대로.
ETH(XETUSD)는 같은 설정으로 확인만.

사용법: python -m backtests.wk_breakout     (-> results/weekend/breakout_*.csv)
"""
import gc
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.all_symbols import load, m1_bars, month_list
from backtests.gold_rest_others import resample
from indicators import atr, ema, rsi

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "weekend"
OUT = ROOT / "results" / "weekend"
DAY = 86_400_000
H = 3_600_000
MIN = 60_000
LAST_ENTRY = 23 * H + 30 * MIN
CLOSE_AT = 23 * H + 50 * MIN
IS_END = pd.Timestamp("2025-01-01")
MIN_IS_TRADES = 100


# ---------------------------------------------------------------- 데이터
def dow(t):
    return ((t // DAY) + 3) % 7  # 월=0 ... 토=5, 일=6


def weekend_ticks(sym, y, m):
    """그 달의 주말 틱(토 01:00 ~ 일 24:00)만, data/weekend/<종목>/에 저장해 두고 씀"""
    f = CACHE / sym / f"{y}-{m:02d}.npz"
    if f.exists():
        z = np.load(f)
        return z["t"], z["b"], z["a"]
    t, b, a = load(sym, y, m)
    d, tod = dow(t), t % DAY
    k = ((d == 5) & (tod >= H)) | (d == 6)
    t, b, a = t[k], b[k], a[k]
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, t=t, b=b, a=a)
    return t, b, a


def entry_ok(t):
    d, tod = dow(t), t % DAY
    return ((d == 5) & (tod >= H) & (tod <= LAST_ENTRY)) | ((d == 6) & (tod <= LAST_ENTRY))


# ---------------------------------------------------------------- 신호 (DataFrame: t, dir, stop_abs, stop_dist)
def sig_frame(t_end, d, stop_abs=None, stop_dist=None):
    n = len(t_end)
    df = pd.DataFrame({"t": t_end, "dir": d,
                       "stop_abs": np.nan if stop_abs is None else stop_abs,
                       "stop_dist": np.nan if stop_dist is None else stop_dist})
    df = df[(df.dir != 0) & entry_ok(df.t.values)]
    return df.reset_index(drop=True) if n else df


def bars_ms(b, tf):
    return (b.time.values.astype("datetime64[ms]").astype(np.int64) + tf * MIN)


def donchian(m1, tf, n):
    b = resample(m1, tf)
    hh = b.high.shift(1).rolling(n).max()
    ll = b.low.shift(1).rolling(n).min()
    d = np.where(b.close > hh, 1, np.where(b.close < ll, -1, 0))
    return sig_frame(bars_ms(b, tf), d, stop_dist=1.5 * atr(b).values)


def squeeze(m1, tf, p):
    b = resample(m1, tf)
    mid = b.close.rolling(20).mean()
    sd = b.close.rolling(20).std()
    w = 4 * sd / mid
    rank = w.rolling(288).rank(pct=True).shift(1)  # 직전 봉 폭의 순위
    low = rank <= p / 100
    d = np.where(low & (b.close > mid + 2 * sd), 1, np.where(low & (b.close < mid - 2 * sd), -1, 0))
    return sig_frame(bars_ms(b, tf), d, stop_abs=mid.values)


def surge(m1, k, half):
    b = m1
    rg = b.high - b.low
    avg = rg.shift(1).rolling(60).mean()
    pos = (b.close - b.low) / rg.replace(0, np.nan)
    big = rg >= k * avg
    d = np.where(big & (pos >= 0.7), 1, np.where(big & (pos <= 0.3), -1, 0))
    if half:
        st = ((b.high + b.low) / 2).values
    else:
        st = np.where(d == 1, b.low.values, b.high.values)
    return sig_frame(bars_ms(b, 1), d, stop_abs=st)


def pullback(m1, lo, htf):
    b = resample(m1, 5)
    hb = resample(m1, htf)
    he = pd.DataFrame({"end": hb.time + pd.Timedelta(minutes=htf),
                       "tr": np.where(hb.close > ema(hb.close, 50), 1, -1)})
    end = b.time + pd.Timedelta(minutes=5)
    tr = pd.merge_asof(pd.DataFrame({"end": end}), he, on="end", direction="backward").tr.values
    r = rsi(b.close)
    mn, mx = r.rolling(6).min().shift(1), r.rolling(6).max().shift(1)
    up = (tr == 1) & (mn < lo) & (r.shift(1) < 50) & (r >= 50)
    dn = (tr == -1) & (mx > 100 - lo) & (r.shift(1) > 50) & (r <= 50)
    d = np.where(up, 1, np.where(dn, -1, 0))
    return sig_frame(bars_ms(b, 5), d, stop_dist=1.5 * atr(b).values)


def orb_ranges(m1, n_hours):
    """날마다 (범위 끝 ms, 고가, 저가). 토 01시 / 일 00시 시작"""
    tm = m1.time.values.astype("datetime64[ms]").astype(np.int64)
    d, tod = dow(tm), tm % DAY
    start = np.where(d == 5, H, 0)
    k = ((d == 5) | (d == 6)) & (tod >= start) & (tod < start + n_hours * H)
    g = pd.DataFrame({"day": tm[k] // DAY, "s": start[k], "h": m1.high.values[k], "l": m1.low.values[k]})
    r = g.groupby("day").agg(s=("s", "first"), h=("h", "max"), l=("l", "min")).reset_index()
    r["end"] = r.day * DAY + r.s + n_hours * H
    return r


# ---------------------------------------------------------------- 체결
def close_idx(t):
    """틱마다 그날 23:50 이후 첫 틱 위치 (없으면 그날 마지막 틱)"""
    day = t // DAY
    u, first = np.unique(day, return_index=True)
    c = np.searchsorted(t, u * DAY + CLOSE_AT)
    last = np.r_[first[1:], len(t)] - 1
    c = np.minimum(c, last)
    return c[np.searchsorted(u, day)]


def trade(t, b, a, ci, i, d, stop_abs, stop_dist, tp_r, trail_r, tmax):
    """i번째 틱 진입. (청산 틱, 진입가, 청산가, R 거리, 이유) 또는 None"""
    en = a[i] if d == 1 else b[i]
    if not np.isnan(stop_abs):
        sl = stop_abs
        R = (en - sl) * d
    else:
        R = stop_dist
        sl = en - d * R
    if not (R > 0) or np.isnan(R):
        return None
    end = ci[i]
    if tmax:
        end = min(end, np.searchsorted(t, t[i] + tmax * MIN))
    end = min(end, len(t) - 1)
    if end <= i:
        return None
    px = b[i + 1:end + 1] if d == 1 else a[i + 1:end + 1]
    if d == 1:
        line = np.full(len(px), sl)
        if trail_r:
            run = np.maximum.accumulate(b[i:end])
            line = np.maximum(line, run - trail_r * R)
        hit = px <= line
        if tp_r:
            hit |= px >= en + tp_r * R
    else:
        line = np.full(len(px), sl)
        if trail_r:
            run = np.minimum.accumulate(a[i:end])
            line = np.minimum(line, run + trail_r * R)
        hit = px >= line
        if tp_r:
            hit |= px <= en - tp_r * R
    k = np.flatnonzero(hit)
    j = k[0] if k.size else len(px) - 1
    return i + 1 + j, en, px[j], R, ("hit" if k.size else "time")


def run_month(t, b, a, sigs, ex):
    """sigs: 이 달 신호(t 오름차순, idx 포함). ex = (tp_r, trail_r, tmax분)"""
    if len(t) == 0 or len(sigs) == 0:
        return []
    ci = close_idx(t)
    rows, free = [], 0
    for s in sigs.itertuples(index=False):
        i = s.idx
        if i < free or i >= len(t) or t[i] - s.t > MIN or t[i] // DAY != s.t // DAY:
            continue
        r = trade(t, b, a, ci, i, s.dir, s.stop_abs, s.stop_dist, *ex)
        if r is None:
            continue
        j, en, xp, R, why = r
        pnl = (xp - en) * s.dir
        rows.append((t[i], t[j], s.dir, en, xp, R, pnl, why))
        free = j + 1
    return rows


def orb_month(t, b, rng):
    """이 달 틱으로 ORB 첫 돌파 틱 찾기 -> 신호표"""
    out = []
    for r in rng.itertuples(index=False):
        i0 = np.searchsorted(t, r.end)
        i1 = np.searchsorted(t, r.day * DAY + LAST_ENTRY, side="right")
        if i0 >= i1:
            continue
        seg = b[i0:i1]
        k = np.flatnonzero((seg > r.h) | (seg < r.l))
        if not k.size:
            continue
        i = i0 + k[0]
        d = 1 if b[i] > r.h else -1
        out.append((t[i], i, d, r.h, r.l))
    return out


# ---------------------------------------------------------------- 설정 40개
def configs():
    c = []
    for n, st, tp in itertools.product([2, 4], ["opp", "mid"], [1, 2]):
        c.append(("A_ORB", f"N{n}h_sl{st}_tp{tp}R", ("orb", n, st), (tp, 0, 0)))
    for tf, n, ex in itertools.product([5, 15], [20, 48], ["tp", "trail"]):
        e = (2, 0, 12 * tf) if ex == "tp" else (0, 1, 24 * tf)  # 트레일 1R = 1.5 ATR
        c.append(("B_Donchian", f"M{tf}_N{n}_{ex}", ("don", tf, n), e))
    for tf, p, ex in itertools.product([5, 15], [10, 25], ["tp", "trail"]):
        e = (2, 0, 24 * tf) if ex == "tp" else (0, 1, 24 * tf)
        c.append(("C_Squeeze", f"M{tf}_p{p}_{ex}", ("sq", tf, p), e))
    for k, half, ex in itertools.product([3, 5], [False, True], ["tp", "trail"]):
        e = (1, 0, 30) if ex == "tp" else (0, 1, 60)
        c.append(("D_Surge", f"k{k}_{'half' if half else 'full'}_{ex}", ("su", k, half), e))
    for lo, htf, tp in itertools.product([35, 40], [60, 240], [1, 2]):
        c.append(("E_Pullback", f"rsi{lo}_H{htf // 60}_tp{tp}R", ("pb", lo, htf), (tp, 0, 120)))
    return c


def make_signals(m1, key):
    kind = key[0]
    if kind == "don":
        return donchian(m1, key[1], key[2])
    if kind == "sq":
        return squeeze(m1, key[1], key[2])
    if kind == "su":
        return surge(m1, key[1], key[2])
    if kind == "pb":
        return pullback(m1, key[1], key[2])
    raise ValueError(kind)


def backtest(sym, cfgs):
    m1 = m1_bars(sym)
    sigs = {c[2]: make_signals(m1, c[2]) for c in cfgs if c[2][0] != "orb"}
    rngs = {n: orb_ranges(m1, n) for n in {c[2][1] for c in cfgs if c[2][0] == "orb"}}
    trades = {c[:2]: [] for c in cfgs}
    for y, m in month_list(sym):
        t, b, a = weekend_ticks(sym, y, m)
        if len(t) == 0:
            continue
        lo, hi = t[0], t[-1]
        orbs = {}
        for n, r in rngs.items():
            rr = r[(r.end >= lo) & (r.end <= hi)]
            orbs[n] = orb_month(t, b, rr)
        for c in cfgs:
            key = c[2]
            if key[0] == "orb":
                o = orbs[key[1]]
                if not o:
                    continue
                tt, ii, dd, hh, ll = map(np.array, zip(*o))
                st = np.where(dd == 1, ll, hh) if key[2] == "opp" else (hh + ll) / 2
                s = pd.DataFrame({"t": tt, "idx": ii, "dir": dd, "stop_abs": st, "stop_dist": np.nan})
            else:
                s = sigs[key]
                s = s[(s.t >= lo) & (s.t <= hi)].copy()
                s["idx"] = np.searchsorted(t, s.t.values)
            trades[c[:2]] += run_month(t, b, a, s, c[3])
        del t, b, a
        gc.collect()
        print(sym, y, m, flush=True)
    out = []
    for (idea, name), rows in trades.items():
        df = pd.DataFrame(rows, columns=["entry_ms", "exit_ms", "dir", "entry", "exit", "R_usd", "pnl", "why"])
        df["idea"], df["config"] = idea, name
        out.append(df)
    tr = pd.concat(out, ignore_index=True)
    tr["entry_time"] = pd.to_datetime(tr.entry_ms, unit="ms")
    tr["exit_time"] = pd.to_datetime(tr.exit_ms, unit="ms")
    tr["R"] = tr.pnl / tr.R_usd
    tr["bp"] = tr.pnl / tr.entry * 1e4
    sat = tr.entry_time.dt.normalize() - pd.to_timedelta((tr.entry_time.dt.dayofweek == 6).astype(int), unit="D")
    tr["weekend"] = sat
    return tr


# ---------------------------------------------------------------- 통계
def pf(x):
    g, l = x[x > 0].sum(), -x[x < 0].sum()
    return g / l if l > 0 else np.inf


def stats(df, n_weekends):
    if len(df) == 0:
        return dict(trades=0)
    cr = df.R.cumsum().values
    dd = float(np.max(np.maximum.accumulate(np.r_[0, cr]) - np.r_[0, cr]))
    wk = df.groupby("weekend").R.sum()
    return dict(trades=len(df), per_wk=round(len(df) / n_weekends, 2), pf_usd=round(pf(df.pnl), 3),
                pf_R=round(pf(df.R), 3), win=round((df.pnl > 0).mean() * 100, 1), avg_bp=round(df.bp.mean(), 2),
                usd=round(df.pnl.sum()), sum_R=round(df.R.sum(), 1), pct_1risk=round(df.R.sum(), 1),
                avg_R=round(df.R.mean(), 3), mdd_R=round(dd, 1), worst_wk_R=round(wk.min(), 1))


def summarize(tr, n_wk):
    rows = []
    for (idea, name), g in tr.groupby(["idea", "config"], sort=False):
        g = g.sort_values("entry_ms")
        is_, oos = g[g.entry_time < IS_END], g[g.entry_time >= IS_END]
        r = {"idea": idea, "config": name}
        r.update({f"IS_{k}": v for k, v in stats(is_, n_wk["IS"]).items()})
        r.update({f"OOS_{k}": v for k, v in stats(oos, n_wk["OOS"]).items()})
        for y in range(2022, 2027):
            gy = g[g.entry_time.dt.year == y]
            r[f"R_{y}"] = round(gy.R.sum(), 1)
            r[f"PF_{y}"] = round(pf(gy.pnl), 2) if len(gy) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def n_weekends(sym):
    ms = month_list(sym)
    sats = pd.date_range(f"{ms[0][0]}-{ms[0][1]:02d}-01", "2026-10-24", freq="W-SAT")
    return {"IS": int((sats < IS_END).sum()), "OOS": int((sats >= IS_END).sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfgs = configs()
    tr = backtest("BTCUSD", cfgs)
    tr.to_csv(OUT / "breakout_trades_BTCUSD.csv", index=False)
    s = summarize(tr, n_weekends("BTCUSD"))
    s.to_csv(OUT / "breakout_summary_BTCUSD.csv", index=False, encoding="utf-8-sig")
    ok = s[s.IS_trades >= MIN_IS_TRADES]
    best = ok.sort_values("IS_pf_usd", ascending=False).groupby("idea").head(1)
    top3 = ok.sort_values("IS_pf_usd", ascending=False).head(3)
    pick = pd.concat([best, top3]).drop_duplicates(["idea", "config"])
    pick.to_csv(OUT / "breakout_picks_BTCUSD.csv", index=False, encoding="utf-8-sig")
    cols = ["idea", "config", "IS_trades", "IS_pf_usd", "IS_avg_bp", "IS_sum_R", "OOS_trades", "OOS_pf_usd",
            "OOS_avg_bp", "OOS_sum_R", "OOS_mdd_R", "OOS_worst_wk_R"] + [f"R_{y}" for y in range(2022, 2027)]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(s[cols].to_string())
    print("\nIS 최고(아이디어별) + IS 상위 3\n", pick[cols].to_string())
    # ETH 같은 설정으로 확인
    ecfg = [c for c in cfgs if (c[0], c[1]) in set(zip(best.idea, best.config))]
    te = backtest("XETUSD", ecfg)
    te.to_csv(OUT / "breakout_trades_XETUSD.csv", index=False)
    se = summarize(te, n_weekends("XETUSD"))
    se.to_csv(OUT / "breakout_eth_check.csv", index=False, encoding="utf-8-sig")
    print("\nETH 확인\n", se[cols].to_string())


if __name__ == "__main__":
    main()
