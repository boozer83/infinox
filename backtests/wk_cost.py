"""주말 BTCUSD 단타 4계열 - 스프레드(거래 비용) 민감도 (실제 틱, 2022-01 ~ 2026-10)

질문: 스프레드가 훨씬 좁은 브로커(수수료 없음)라면 wk_breakout / wk_meanrev / wk_timing / wk_grid 전략이 될까?
다른 브로커 자료가 없으므로 같은 틱에서 비용만 바꿔 다시 돌림.

방법:
  - 체결 가격을 중간값 mid = (Bid+Ask)/2 와 고정 스프레드 S(bp, 그 틱 가격 대비)로 다시 만듦:
    Bid' = mid x (1 - S/2/1e4), Ask' = mid x (1 + S/2/1e4)  (매수 Ask', 매도 Bid')
    S = 실제(브로커 틱 그대로, 기준선 - 원래 결과와 같아야 함), 2.0, 1.0, 0.5, 0.2, 0 bp
  - 신호·지연·손절·익절·시간 청산·스왑은 원래 파일 함수 그대로. 원래 파일은 고치지 않고 여기서 감쌈:
      breakout : 월별 틱 캐시(data/weekend/) + run_month 에 Bid'/Ask' 전달. ORB 돌파 판단은 원래 Bid로 (신호 그대로)
      meanrev  : 같은 캐시를 주말 단위로 묶어 run_weekend 에 전달 (원래는 전체 틱을 load 했지만 거르는 조건 같음)
      timing   : load_wk 를 바꿔 끼움 (신호는 mid 라 그대로). S1 방향 = 원래처럼 실제 스프레드 IS로, 최고 1시간 = timing_hours.csv
      grid     : weekends() 틱에 Bid'/Ask' 전달. tick_signals 는 원래 Bid/Ask로 (스프레드 거르기 신호 그대로)
  - 설정 고르기는 원래 결과(IS = 2022~2024, 실제 스프레드)의 IS 최고 + IS 상위 3 그대로 (pick 열).
    따로 'S=0.5bp에서 IS로 다시 고르기'(resel_0.5) 한 줄씩 (계열마다 원래 기준: breakout PF$, meanrev PF(R),
    timing t값, grid PF$, 최소 거래 수도 원래대로).
  - 손익분기 스프레드: OOS 평균 bp를 S(0~2bp)에 직선으로 맞춰 평균 = 0 되는 S. 총 우위(gross) = S=0의 평균 bp.

사용법: python -m backtests.wk_cost      (-> results/weekend/cost_sensitivity.csv, cost_breakeven.csv)
"""
import gc
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "results" / "weekend"
SYM = "BTCUSD"
SPREADS = ["real", 2.0, 1.0, 0.5, 0.2, 0.0]
IS_END = pd.Timestamp("2025-01-01")
DAY, HOUR = 86_400_000, 3_600_000


def fix(b, a, S):
    """S = 'real'이면 그대로, 아니면 mid 둘레 고정 스프레드 S bp"""
    if S == "real":
        return b, a
    mid = (b + a) / 2
    h = mid * (S / 2e4)
    return mid - h, mid + h


# ---------------------------------------------------------------- 통계 (계열 공통)
def _pf(x):
    gl = -x[x < 0].sum()
    return round(float(x[x > 0].sum() / gl), 3) if gl > 0 else np.inf


def _stats(g, p):
    if len(g) == 0:
        return {f"{p}n": 0}
    bp = g.bp.values
    r = {f"{p}n": len(g), f"{p}PF": _pf(g.usd.values), f"{p}avg_bp": round(bp.mean(), 2),
         f"{p}win": round((g.usd.values > 0).mean() * 100, 1),
         f"{p}t": round(bp.mean() / bp.std(ddof=1) * np.sqrt(len(bp)), 2) if len(bp) > 2 else np.nan}
    if "R" in g and g.R.notna().any():
        r[f"{p}PF_R"] = _pf(g.R.values)
        r[f"{p}sum_R"] = round(g.R.sum(), 1)
    return r


def summarize(tr, fam):
    """tr: idea, config, S, wk(토요일 Timestamp), usd, bp, R"""
    rows = []
    tr["year"] = tr.wk.dt.year
    for (idea, cfg, S), g in tr.groupby(["idea", "config", "S"], sort=False):
        r = dict(family=fam, idea=idea, config=cfg, S=S)
        r.update(_stats(g[g.wk < IS_END], "IS_"))
        r.update(_stats(g[g.wk >= IS_END], "OOS_"))
        for y in range(2022, 2027):
            gy = g[g.year == y]
            r[f"n_{y}"] = len(gy)
            r[f"PF_{y}"] = _pf(gy.usd.values) if len(gy) else np.nan
            r[f"bp_{y}"] = round(gy.bp.mean(), 2) if len(gy) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def sat_ts(ms):
    day = np.asarray(ms) // DAY
    dow = (day + 3) % 7
    return pd.to_datetime((day - (dow - 5)) * DAY, unit="ms")


# ---------------------------------------------------------------- 1. 돌파 (wk_breakout)
def run_breakout():
    import backtests.wk_breakout as B
    from backtests.all_symbols import m1_bars, month_list
    cfgs = B.configs()
    m1 = m1_bars(SYM)
    sigs = {c[2]: B.make_signals(m1, c[2]) for c in cfgs if c[2][0] != "orb"}
    rngs = {n: B.orb_ranges(m1, n) for n in {c[2][1] for c in cfgs if c[2][0] == "orb"}}
    del m1
    rows = []
    for y, m in month_list(SYM):
        t, b, a = B.weekend_ticks(SYM, y, m)
        if len(t) == 0:
            continue
        lo, hi = t[0], t[-1]
        orbs = {n: B.orb_month(t, b, r[(r.end >= lo) & (r.end <= hi)]) for n, r in rngs.items()}  # 원래 Bid로 신호
        sframes = {}
        for c in cfgs:
            key = c[2]
            if key[0] == "orb":
                o = orbs[key[1]]
                if not o:
                    continue
                tt, ii, dd, hh, ll = map(np.array, zip(*o))
                st = np.where(dd == 1, ll, hh) if key[2] == "opp" else (hh + ll) / 2
                sframes[c[:2]] = pd.DataFrame({"t": tt, "idx": ii, "dir": dd, "stop_abs": st, "stop_dist": np.nan})
            else:
                s = sigs[key]
                s = s[(s.t >= lo) & (s.t <= hi)].copy()
                s["idx"] = np.searchsorted(t, s.t.values)
                sframes[c[:2]] = s
        for S in SPREADS:
            bb, aa = fix(b, a, S)
            for c in cfgs:
                if c[:2] in sframes:
                    for r in B.run_month(t, bb, aa, sframes[c[:2]], c[3]):
                        rows.append((c[0], c[1], S, r[0], r[3], r[5], r[6]))
        del t, b, a
        gc.collect()
        print("breakout", y, m, file=sys.stderr, flush=True)
    tr = pd.DataFrame(rows, columns=["idea", "config", "S", "t_in", "entry", "R_usd", "usd"])
    tr["bp"] = tr.usd / tr.entry * 1e4
    tr["R"] = tr.usd / tr.R_usd
    tr["wk"] = sat_ts(tr.t_in.values)
    return summarize(tr, "breakout")


# ---------------------------------------------------------------- 2. 평균회귀 (wk_meanrev)
def mr_weekends():
    """data/weekend/ 월 캐시(토 01:00~일 24:00)를 주말 단위로 이어 붙여 내보냄 (wk_meanrev.weekend_ticks 와 같은 틱)"""
    import backtests.wk_breakout as B
    import backtests.wk_meanrev as M
    from backtests.all_symbols import month_list
    pend = {}
    for y, m in month_list(SYM):
        t, bid, ask = B.weekend_ticks(SYM, y, m)
        sat, _ = M.sat_of(t)
        for s in np.unique(sat):
            mm = sat == s
            pend.setdefault(int(s), []).append((t[mm], bid[mm], ask[mm]))
        month_end = M.to_ms(np.datetime64(pd.Timestamp(y, m, 1) + pd.offsets.MonthBegin(1)))
        for s in sorted(pend):
            if s + 2 * DAY <= month_end:
                parts = pend.pop(s)
                yield s, *(np.concatenate([p[j] for p in parts]) for j in range(3))
    for s in sorted(pend):
        parts = pend.pop(s)
        yield s, *(np.concatenate([p[j] for p in parts]) for j in range(3))


def run_meanrev():
    import backtests.wk_meanrev as M
    from backtests.all_symbols import m1_bars
    from backtests.gold_rest_others import resample
    cfs = M.configs()
    m1 = m1_bars(SYM)
    m5 = resample(m1, 5)
    sigs = []
    for fam, name, fn, args in cfs:
        d, cond = fn(m1, m5, *args)
        sigs.append((fam, name, {s: g for s, g in d.groupby("wk")}, cond))
    del m1, m5
    rows = []
    for s, t, b, a in mr_weekends():
        for S in SPREADS:
            bb, aa = fix(b, a, S)
            for fam, name, by, cond in sigs:
                if s in by:
                    for r in M.run_weekend(by[s], cond, t, bb, aa, s):
                        rows.append((fam, name, S, r[0], r[4], r[6], r[7]))   # sat, entry, risk, pnl
    print("meanrev done", file=sys.stderr, flush=True)
    tr = pd.DataFrame(rows, columns=["idea", "config", "S", "wk", "entry", "risk", "usd"])
    tr["bp"] = tr.usd / tr.entry * 1e4
    tr["R"] = tr.usd / tr.risk
    tr["wk"] = pd.to_datetime(tr.wk, unit="ms")
    return summarize(tr, "meanrev")


# ---------------------------------------------------------------- 3. 시간대 (wk_timing)
def run_timing():
    import backtests.wk_timing as T
    orig = T.load_wk
    wks = [s for s in T.weekends(SYM) if s + 48 * HOUR <= pd.Timestamp("2026-10-24").value // 1_000_000]
    _, both = T.run_all(SYM, wks, {})                       # S1 방향: 원래처럼 실제 스프레드 IS 평균으로
    s1_dirs = {}
    for a_, b_ in T.S1_BLOCKS:
        lg, sh = T.split(both[(a_, b_, 1)])[0], T.split(both[(a_, b_, -1)])[0]
        s1_dirs[(a_, b_)] = 1 if (lg.bp.mean() if len(lg) else 0) >= (sh.bp.mean() if len(sh) else 0) else -1
    hh = pd.read_csv(OUT / "timing_hours.csv", encoding="utf-8-sig")
    hh = hh[hh.IS_n >= 50]
    k = hh.IS_t.abs().idxmax()
    best_hour = (int(hh.h[k]), 1 if hh.IS_t[k] > 0 else -1)
    out = []
    for S in SPREADS:
        T.load_wk = lambda sym, s, S=S: (lambda t, b, a: (t, *fix(b, a, S)))(*orig(sym, s))
        trades, _ = T.run_all(SYM, wks, s1_dirs, best_hour)
        for (f, n), tr in trades.items():
            if len(tr):
                out.append(pd.DataFrame({"idea": f, "config": n, "S": S, "wk": pd.to_datetime(tr.sat),
                                         "usd": tr.usd.values, "bp": tr.bp.values, "R": tr.R.values}))
        print("timing", S, file=sys.stderr, flush=True)
    T.load_wk = orig
    return summarize(pd.concat(out, ignore_index=True), "timing")


# ---------------------------------------------------------------- 4. 그리드·틱 (wk_grid)
def run_grid():
    import backtests.wk_grid as G
    W = G.weekends(SYM)
    orig_sig = G.tick_signals
    origs = {id(t): (b, a) for _, t, b, a in W}
    G.tick_signals = lambda t, b, a, cfg: orig_sig(t, *origs[id(t)], cfg)   # 신호는 원래 Bid/Ask로
    out = []
    for S in SPREADS:
        W2 = [(s0, t, *fix(b, a, S)) for s0, t, b, a in W]
        for name, cfgs, risk in G.IDEAS:
            for cfg in cfgs:
                df = G.run_cfg(W2, cfg, risk)
                if len(df):
                    out.append(pd.DataFrame({"idea": name, "config": str(cfg), "S": S,
                                             "wk": pd.to_datetime(df.wk.values, unit="ms"),
                                             "usd": df.usd.values, "bp": df.bp.values, "R": df.R.values}))
        del W2
        gc.collect()
        print("grid", S, file=sys.stderr, flush=True)
    G.tick_signals = orig_sig
    return summarize(pd.concat(out, ignore_index=True), "grid")


RUNNERS = {"breakout": run_breakout, "meanrev": run_meanrev, "timing": run_timing, "grid": run_grid}


def _run(name):
    return RUNNERS[name]()


# ---------------------------------------------------------------- 원래 고른 설정
def original_picks():
    """(family, idea, config) -> 'IS-best' / 'top3' / 'IS-best top3'"""
    p = {}

    def add(k, tag):
        p[k] = (p.get(k, "") + " " + tag).strip()
    s = pd.read_csv(OUT / "breakout_summary_BTCUSD.csv", encoding="utf-8-sig")
    ok = s[s.IS_trades >= 100].sort_values("IS_pf_usd", ascending=False)
    for r in ok.groupby("idea").head(1).itertuples():
        add(("breakout", r.idea, r.config), "IS-best")
    for r in ok.head(3).itertuples():
        add(("breakout", r.idea, r.config), "top3")
    m = pd.read_csv(OUT / "meanrev_picks_BTCUSD.csv")
    for r in m[m.period == "IS"].itertuples():
        add(("meanrev", r.family, r.config), r.pick.strip())
    t = pd.read_csv(OUT / "timing_configs.csv", encoding="utf-8-sig")
    for r in t[t.IS_rank <= 3].itertuples():
        add(("timing", r.family, r.config), "IS-best" if r.IS_rank == 1 else "top3")
    g = pd.read_csv(OUT / "grid_top3.csv", encoding="utf-8-sig")
    for r in g.itertuples():
        add(("grid", r.아이디어, r.설정), "IS-best" if r.IS순위 == 1 else "top3")
    return p


# 계열마다 원래 고르는 기준 (열, 최소 IS 거래)
SEL = {"breakout": ("IS_PF", 100), "meanrev": ("IS_PF_R", 150), "timing": ("IS_t", 0), "grid": ("IS_PF", 100)}


def breakeven(res):
    rows = []
    for (fam, idea, cfg), g in res[res.S != "real"].groupby(["family", "idea", "config"], sort=False):
        g = g.dropna(subset=["OOS_avg_bp"]).sort_values("S")
        x, y = g.S.astype(float).values, g.OOS_avg_bp.values
        r = dict(family=fam, idea=idea, config=cfg, pick=g.pick.iloc[0])
        if len(x) >= 2:
            sl, ic = np.polyfit(x, y, 1)
            r.update(OOS_gross_bp_S0=round(float(y[x == 0][0]) if (x == 0).any() else ic, 2), slope=round(sl, 2),
                     OOS_breakeven_S_bp=round(-ic / sl, 2) if ic > 0 and sl < 0 else 0.0)
        gi = g.dropna(subset=["IS_avg_bp"])
        if len(gi) >= 2:
            sl2, ic2 = np.polyfit(gi.S.astype(float).values, gi.IS_avg_bp.values, 1)
            r.update(IS_gross_bp_S0=round(float(gi.IS_avg_bp[gi.S.astype(float) == 0].iloc[0]), 2),
                     IS_breakeven_S_bp=round(-ic2 / sl2, 2) if ic2 > 0 and sl2 < 0 else 0.0)
        real = res[(res.family == fam) & (res.idea == idea) & (res.config == cfg) & (res.S == "real")]
        if len(real):
            r.update(OOS_real_avg_bp=real.OOS_avg_bp.iloc[0], OOS_real_PF=real.OOS_PF.iloc[0])
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(RUNNERS)
    with ProcessPoolExecutor(len(names)) as ex:
        parts = list(ex.map(_run, names))
    res = pd.concat(parts, ignore_index=True)
    picks = original_picks()
    res["pick"] = [picks.get((f, i, c), "") for f, i, c in res[["family", "idea", "config"]].values]
    # S = 0.5bp 에서 IS로 다시 고르기 (계열·아이디어마다 1개)
    for fam, (col, mn) in SEL.items():
        h = res[(res.family == fam) & (res.S == 0.5) & (res.IS_n >= max(mn, 1))]
        if col not in h:
            continue
        for _, g in h.groupby("idea"):
            r = g.sort_values(col, ascending=False).iloc[0]
            k = (res.family == fam) & (res.idea == r.idea) & (res.config == r.config)
            res.loc[k, "pick"] = (res.loc[k, "pick"] + " resel_0.5").str.strip()
    res.to_csv(OUT / "cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    be = breakeven(res)
    be.to_csv(OUT / "cost_breakeven.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
    sel = res[res.pick != ""]
    cols = ["family", "idea", "config", "pick", "S", "IS_n", "IS_PF", "IS_avg_bp", "OOS_n", "OOS_PF", "OOS_avg_bp"] + \
        [f"bp_{y}" for y in range(2022, 2027)]
    print(sel[cols].to_string())
    print(be[be.pick != ""].to_string())


if __name__ == "__main__":
    main()
