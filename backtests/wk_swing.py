"""주말 전용 BTCUSD 스윙 - 몇 시간 ~ 주말 내내 들고 가기 (실제 틱, 2022-01 ~ 2026-10)

목적: 주말 단타(분~3시간)는 스프레드에 다 먹혀서 실패 -> 거래 수를 줄이고 오래 들고 가면 되는지 확인.
규칙: 진입은 주말 창 안에서만(토 세션 시작 ~ 일 20:00), 모든 포지션은 일 23:50 이후 첫 틱에 청산.
      신호는 완성된 봉/이미 아는 정보만, 진입 = 신호 뒤 첫 틱 (매수 Ask / 매도 Bid, 5분 안에 틱 없으면 버림).
      손절·트레일은 틱마다 (매수 Bid / 매도 Ask), 갭이면 그 틱 가격. 트레일 = 이전 틱까지 최고(최저)가에서 거리.
      토->일 자정을 넘기면 스왑 1일치 (매수 -25%/년, 매도 -15%/년 x 진입가 / 365). 수수료 없음.
      한 전략당 한 번에 1포지션. 모든 설정에 손절이 있음 -> R = 처음 손절 거리, 거래당 1% 위험이면 % = R 합.
      일봉 ATR(14)은 금요일까지 끝난 일봉(서버시간)으로.

아이디어 (모두 38개):
  A. 금요일 추세 들고 가기 (12): 세션 첫 틱 + 5분에 진입, 일 23:50 청산.
     방향 = fri(금 종가 - 목 종가) / wk(금 종가 - 이번주 월 시가) / ema(금 종가 - 일봉 EMA20) 부호,
     follow(그 방향) / fade(반대), 손절 1.5 또는 3 x 일봉 ATR
  B. 주말 봉 돈치안 돌파 (8): H1/H4 종가가 직전 N봉(20/55) 최고(최저) 돌파 -> 그 방향,
     손절 = k x ATR(그 봉) (k = 2/3), 같은 거리로 트레일, 일 23:50 청산. 끝나면 다음 신호 다시 받음.
  C. 토요일 범위 -> 일요일 돌파 (6): 토요일 주말 틱 Bid 고가·저가, 일 00:00~20:00 처음 넘는 틱에
     follow: 손절 반대편/중간, 시간청산 또는 1R 트레일 / fade: 반대로, 손절 0.5 x 범위, 익절 중간 또는 시간청산
  D. 토요일 움직임 -> 일요일 (8): 일 00:00 첫 틱에 (토 마지막 Bid - 기준) 부호, 기준 = 금 종가 / 세션 첫 Bid,
     follow/fade, 손절 1 또는 2 x 일봉 ATR, 일 23:50 청산 (자정 안 넘김 -> 스왑 없음)
  E. 변동성 국면 + 주간 추세 (4): A의 wk 신호, 손절 1.5 ATR, 일봉 ATR(5)/ATR(50) > 1 (거친 주) 또는 < 1 (조용한 주)일 때만
검증: IS = 2022~2024 주말 (거래 40회 이상 중 PF 최고), OOS = 2025-01 ~ 2026-10. ETH(XETUSD)는 같은 설정으로 확인만.

사용법: python -m backtests.wk_swing     (-> results/weekend/swing_*.csv)
"""
import gc
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.all_symbols import m1_bars, month_list
from backtests.gold_rest_others import resample
from backtests.wk_breakout import DAY, H, MIN, dow, weekend_ticks
from indicators import atr, ema

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "weekend"
CLOSE_AT = 23 * H + 50 * MIN        # 일 23:50 청산
LAST_ENTRY = 20 * H                 # 일 20:00 이후 진입 없음
IS_END = pd.Timestamp("2025-01-01")
MIN_IS_TRADES = 40
SWAP = {1: 0.25 / 365, -1: 0.15 / 365}


# ---------------------------------------------------------------- 주말 단위 틱
def weekends(sym):
    """(토요일 day 번호, t, b, a)를 주말마다. 달을 넘기는 주말은 합쳐서"""
    buf = {}
    for y, m in month_list(sym):
        t, b, a = weekend_ticks(sym, y, m)
        if len(t):
            day = t // DAY
            key = day - (dow(t) == 6)
            u, st = np.unique(key, return_index=True)
            en = np.r_[st[1:], len(t)]
            for k, s, e in zip(u, st, en):
                p = (t[s:e].copy(), b[s:e].copy(), a[s:e].copy())
                if k in buf:
                    p = tuple(np.concatenate([x, y_]) for x, y_ in zip(buf[k], p))
                buf[k] = p
            for k in sorted(buf):
                if k < u[-1]:
                    yield (k, *buf.pop(k))
        del t, b, a
        gc.collect()
        print(sym, y, m, flush=True)
    for k in sorted(buf):
        yield (k, *buf.pop(k))


# ---------------------------------------------------------------- 주별 특징 (금요일까지 아는 것만)
def daily_features(m1):
    d = resample(m1, 1440)
    d["day"] = d.time.values.astype("datetime64[D]").astype(np.int64)
    d["atr14"] = atr(d).values
    d["atr5"], d["atr50"] = atr(d, 5).values, atr(d, 50).values
    d["ema20"] = ema(d.close, 20).values
    d = d.set_index("day")
    return d


def week_info(d, sat):
    """sat = 토요일 day 번호. 금(sat-1), 목(sat-2), 월(sat-5) 일봉"""
    fri, thu, mon = sat - 1, sat - 2, sat - 5
    if fri not in d.index or thu not in d.index:
        return None
    f = d.loc[fri]
    mo = d.loc[mon].open if mon in d.index else np.nan
    return dict(fri_close=f.close, sig_fri=np.sign(f.close - d.loc[thu].close), sig_wk=np.sign(f.close - mo),
                sig_ema=np.sign(f.close - f.ema20), atr=f.atr14, vol=f.atr5 / f.atr50)


def bar_signals(m1, tf, n, k):
    b = resample(m1, tf)
    hh = b.high.shift(1).rolling(n).max()
    ll = b.low.shift(1).rolling(n).min()
    d = np.where(b.close > hh, 1, np.where(b.close < ll, -1, 0))
    t_end = b.time.values.astype("datetime64[ms]").astype(np.int64) + tf * MIN
    s = pd.DataFrame({"t": t_end, "dir": d, "R": k * atr(b).values})
    return s[s.dir != 0].reset_index(drop=True)


# ---------------------------------------------------------------- 체결
def sim(t, b, a, i, d, R, ci, trail=False, tp=None):
    """i번째 틱 진입, 처음 손절 거리 R(가격), trail=True면 R 거리 트레일, tp = 익절 가격.
    (청산 틱, 진입가, 청산가, 이유) 또는 None"""
    if not (R > 0) or i >= ci:
        return None
    en = a[i] if d == 1 else b[i]
    sl = en - d * R
    if d == 1:
        px = b[i + 1:ci + 1]
        line = np.full(len(px), sl)
        if trail:
            line = np.maximum(line, np.maximum.accumulate(b[i:ci]) - R)
        hit = px <= line
        if tp is not None:
            hit |= px >= tp
    else:
        px = a[i + 1:ci + 1]
        line = np.full(len(px), sl)
        if trail:
            line = np.minimum(line, np.minimum.accumulate(a[i:ci]) + R)
        hit = px >= line
        if tp is not None:
            hit |= px <= tp
    k = np.flatnonzero(hit)
    j = k[0] if k.size else len(px) - 1
    return i + 1 + j, en, px[j], ("hit" if k.size else "time")


def row(t, sat, i, d, R, res):
    j, en, xp, why = res
    nights = int(t[j] // DAY - t[i] // DAY)
    swap = en * SWAP[d] * nights
    pnl = (xp - en) * d - swap
    return dict(weekend=sat, entry_ms=t[i], exit_ms=t[j], dir=d, entry=en, exit=xp, R_usd=R,
                swap=swap, pnl=pnl, why=why)


# ---------------------------------------------------------------- 설정 38개
def configs():
    c = []
    for s, mode, k in itertools.product(["fri", "wk", "ema"], ["follow", "fade"], [1.5, 3.0]):
        c.append(("A_carry", f"{s}_{mode}_sl{k}", dict(sig=s, mode=mode, k=k)))
    for tf, n, k in itertools.product([60, 240], [20, 55], [2.0, 3.0]):
        c.append(("B_donchian", f"H{tf // 60}_N{n}_k{k:g}", dict(tf=tf, n=n, k=k)))
    for st, ex in itertools.product(["opp", "mid"], ["time", "trail"]):
        c.append(("C_satrange", f"follow_sl{st}_{ex}", dict(mode="follow", st=st, ex=ex)))
    for ex in ["mid", "time"]:
        c.append(("C_satrange", f"fade_tp{ex}", dict(mode="fade", ex=ex)))
    for ref, mode, k in itertools.product(["fri", "open"], ["follow", "fade"], [1.0, 2.0]):
        c.append(("D_satmove", f"ref{ref}_{mode}_sl{k:g}", dict(ref=ref, mode=mode, k=k)))
    for v, mode in itertools.product(["hi", "lo"], ["follow", "fade"]):
        c.append(("E_volwk", f"vol{v}_{mode}", dict(vol=v, mode=mode)))
    return c


def backtest(sym, cfgs):
    m1 = m1_bars(sym)
    d = daily_features(m1)
    bsig = {}
    for idea, name, p in cfgs:
        if idea == "B_donchian":
            bsig[name] = bar_signals(m1, p["tf"], p["n"], p["k"])
    del m1
    rows = []
    for sat, t, b, a in weekends(sym):
        if len(t) < 100:
            continue
        wi = week_info(d, sat)
        sun = (sat + 1) * DAY
        ci = min(np.searchsorted(t, sun + CLOSE_AT), len(t) - 1)
        last_ent = sun + LAST_ENTRY

        def add(idea, name, i, dd, R, res):
            if res is not None:
                r = row(t, sat, i, dd, R, res)
                r.update(idea=idea, config=name)
                rows.append(r)

        i_open = np.searchsorted(t, t[0] + 5 * MIN)
        sat_k = t < sun
        i_sun = np.searchsorted(t, sun)
        has_sat, has_sun = sat_k.any() and t[0] < sun - 2 * H, i_sun < len(t) and t[i_sun] < sun + 5 * MIN
        for idea, name, p in cfgs:
            if idea in ("A_carry", "E_volwk") and wi is not None:
                sig = p.get("sig", "wk")
                dd = wi[f"sig_{sig}"]
                if idea == "E_volwk":
                    if np.isnan(wi["vol"]) or (wi["vol"] > 1) != (p["vol"] == "hi"):
                        continue
                if not dd or np.isnan(dd) or t[i_open] > last_ent:
                    continue
                dd = int(dd) * (1 if p["mode"] == "follow" else -1)
                R = p.get("k", 1.5) * wi["atr"]
                add(idea, name, i_open, dd, R, sim(t, b, a, i_open, dd, R, ci))
            elif idea == "B_donchian":
                s = bsig[name]
                s = s[(s.t >= t[0]) & (s.t <= last_ent)]
                free = 0
                for x in s.itertuples(index=False):
                    i = np.searchsorted(t, x.t)
                    if i < free or i >= len(t) or t[i] - x.t > 5 * MIN:
                        continue
                    res = sim(t, b, a, i, x.dir, x.R, ci, trail=True)
                    add(idea, name, i, x.dir, x.R, res)
                    if res is not None:
                        free = res[0] + 1
            elif idea == "C_satrange" and has_sat and has_sun:
                hi, lo = b[:i_sun].max(), b[:i_sun].min()
                rng = hi - lo
                i1 = np.searchsorted(t, last_ent, side="right")
                k = np.flatnonzero((b[i_sun:i1] > hi) | (b[i_sun:i1] < lo))
                if not k.size or rng <= 0:
                    continue
                i = i_sun + k[0]
                up = 1 if b[i] > hi else -1
                mid = (hi + lo) / 2
                if p["mode"] == "follow":
                    en = a[i] if up == 1 else b[i]
                    stop = (lo if up == 1 else hi) if p["st"] == "opp" else mid
                    R = (en - stop) * up
                    add(idea, name, i, up, R, sim(t, b, a, i, up, R, ci, trail=p["ex"] == "trail"))
                else:
                    dd = -up
                    R = 0.5 * rng
                    tp = mid if p["ex"] == "mid" else None
                    add(idea, name, i, dd, R, sim(t, b, a, i, dd, R, ci, tp=tp))
            elif idea == "D_satmove" and has_sat and has_sun and wi is not None:
                ref = wi["fri_close"] if p["ref"] == "fri" else b[0]
                dd = int(np.sign(b[i_sun - 1] - ref))
                if not dd:
                    continue
                dd *= 1 if p["mode"] == "follow" else -1
                R = p["k"] * wi["atr"]
                add(idea, name, i_sun, dd, R, sim(t, b, a, i_sun, dd, R, ci))
        del t, b, a
    tr = pd.DataFrame(rows)
    tr["weekend"] = pd.to_datetime(tr.weekend * DAY, unit="ms")
    tr["entry_time"] = pd.to_datetime(tr.entry_ms, unit="ms")
    tr["exit_time"] = pd.to_datetime(tr.exit_ms, unit="ms")
    tr["R"] = tr.pnl / tr.R_usd
    tr["bp"] = tr.pnl / tr.entry * 1e4
    tr["hold_h"] = (tr.exit_ms - tr.entry_ms) / H
    return tr.sort_values(["idea", "config", "entry_ms"]).reset_index(drop=True)


# ---------------------------------------------------------------- 통계
def pf(x):
    g, l = x[x > 0].sum(), -x[x < 0].sum()
    return g / l if l > 0 else np.inf


def stats(df, years):
    if len(df) == 0:
        return dict(trades=0)
    cr = df.R.cumsum().values
    dd = float(np.max(np.maximum.accumulate(np.r_[0, cr]) - np.r_[0, cr]))
    wk = df.groupby("weekend").R.sum()
    tst = df.R.mean() / df.R.std(ddof=1) * np.sqrt(len(df)) if len(df) > 1 and df.R.std() > 0 else np.nan
    return dict(trades=len(df), pf=round(pf(df.pnl), 2), win=round((df.pnl > 0).mean() * 100, 1),
                avg_bp=round(df.bp.mean(), 1), avg_R=round(df.R.mean(), 3), t=round(tst, 2),
                pct_yr=round(df.R.sum() / years, 1), mdd_pct=round(dd, 1), worst_wk=round(wk.min(), 2),
                hold_h=round(df.hold_h.mean(), 1))


def summarize(tr):
    out = []
    for (idea, name), g in tr.groupby(["idea", "config"], sort=False):
        is_, oos = g[g.entry_time < IS_END], g[g.entry_time >= IS_END]
        r = {"idea": idea, "config": name}
        r.update({f"IS_{k}": v for k, v in stats(is_, 2.83).items()})   # 2022-03 ~ 2024-12
        r.update({f"OOS_{k}": v for k, v in stats(oos, 1.81).items()})  # 2025-01 ~ 2026-10
        for y in range(2022, 2027):
            gy = g[g.entry_time.dt.year == y]
            r[f"R_{y}"] = round(gy.R.sum(), 1)
        out.append(r)
    return pd.DataFrame(out)


COLS = ["idea", "config", "IS_trades", "IS_pf", "IS_avg_bp", "IS_t", "IS_pct_yr", "OOS_trades", "OOS_pf", "OOS_win",
        "OOS_avg_bp", "OOS_t", "OOS_pct_yr", "OOS_mdd_pct", "OOS_worst_wk", "OOS_hold_h"] + [f"R_{y}" for y in range(2022, 2027)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    cfgs = configs()
    tr = backtest("BTCUSD", cfgs)
    tr.to_csv(OUT / "swing_trades_BTCUSD.csv", index=False)
    s = summarize(tr)
    s.to_csv(OUT / "swing_summary_BTCUSD.csv", index=False, encoding="utf-8-sig")
    ok = s[s.IS_trades >= MIN_IS_TRADES]
    best = ok.sort_values("IS_pf", ascending=False).groupby("idea").head(1)
    top3 = ok.sort_values("IS_pf", ascending=False).head(3)
    pick = pd.concat([best.assign(pick="idea_best"), top3.assign(pick="IS_top3")]).drop_duplicates(["idea", "config"])
    pick.to_csv(OUT / "swing_picks_BTCUSD.csv", index=False, encoding="utf-8-sig")
    print(s[COLS].to_string())
    print("\nIS 최고(아이디어별) + IS 상위 3\n", pick[COLS + ["pick"]].to_string())
    ecfg = [c for c in cfgs if (c[0], c[1]) in set(zip(pick.idea, pick.config))]
    te = backtest("XETUSD", ecfg)
    te.to_csv(OUT / "swing_trades_XETUSD.csv", index=False)
    se = summarize(te)
    se.to_csv(OUT / "swing_eth_check.csv", index=False, encoding="utf-8-sig")
    print("\nETH 확인\n", se[COLS].to_string())


if __name__ == "__main__":
    main()
