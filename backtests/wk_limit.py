"""주말 BTCUSD RSI(2) 되돌림 단타(B_rsi2) - 지정가(리밋) 진입/청산이면 지금 브로커(인피녹스, 수수료 없음,
스프레드 ~1.5bp)에서 수익이 나는가? (실제 틱, 2022-01 ~ 2026-10, 보수적으로)

신호·손절·시간 손절·RSI 청산은 wk_meanrev.idea_rsi / run_weekend 그대로 (L5 T90, 필터 없음 / EMA200):
  5분봉(Bid) RSI(2) < 5 매수 / > 95 매도, RSI(2)가 50을 넘으면(매도는 밑돌면) 그 봉 끝에 청산,
  손절 2.5 x ATR(14) (체결가 기준), 시간 손절 90분, 23:50 강제 청산, 토->일 자정 넘기면 스왑.
  포지션(또는 걸어둔 주문) 한 번에 1개.

진입 방식:
  mkt      : 원래대로 신호 후 첫 틱 시장가 (매수 Ask, 매도 Bid)
  lim      : 신호(봉 끝) + 지연 lat 초 뒤 첫 틱(배치 틱)의 Bid 기준 매수 지정가 L = Bid - off
             (매도는 Ask + off). off = x bp (0/0.25/0.5/1.0) 또는 스프레드 비율 (s-0.5 = 중간값에 걸기,
             s0.5 / s1.0 = Bid 아래 스프레드 절반/1배).
             체결: 배치 틱 '다음' 틱부터 Ask <= L (매수, 매도는 Bid >= L) 이면 L에 체결 (touch),
                   더 엄격하게 Ask <= L - 0.1bp 까지 내려와야 체결 (thru, 대기열 뒤라고 가정).
             취소: N분(1/3/5/10) 지나거나, 그 전에 RSI 청산 조건(50 넘음) 봉이 끝나거나, 일 23:30.
             걸어둔 동안 새 신호는 무시.
청산 방식:
  mkt      : 원래대로 (손절은 서버 손절 = 닿은 틱 가격, RSI/시간 청산은 그 시각 + 지연 뒤 첫 틱 시장가)
  pex      : RSI/시간 청산 시점에 반대쪽 호가로 지정가 청산 (매수 포지션 = 그때 Ask에 매도 지정가) 1분 대기,
             안 되면 시장가. 대기 중에도 손절 살아 있음. 23:50 강제 청산은 시장가.
  tpK      : 원래 전략엔 가격 목표가 없으므로 진입가 ± K x ATR(14) 에 지정가 익절을 추가로 걸어 둠
             (K = 0.5 / 1.0, 체결 규칙은 진입과 같음 touch/thru). 나머지는 원래 청산.
  tp0.5+pex: 둘 다.
지연: 0 / 1 / 2초 (지정가 배치와 시장가 청산 주문이 늦게 도착, 서버 손절·지정가 체결은 지연 없음).
스프레드: real = 브로커 틱 그대로 (IS 2022~2024는 3~13bp로 넓음), fix1.5 = 중간값 둘레 1.5bp 고정
          (지금 브로커 수준을 IS 기간에도 가정).
역선택 확인: 지정가 신호마다 '같은 시점에 시장가로 들어갔다면'(cf) 손익을 같이 계산 -> 체결된 신호 vs 안 된 신호.

IS = 2022-01 ~ 2024-12 주말, OOS = 2025-01 ~ 2026-10. bp = 손익 / 진입가 x 1e4 (거래당), t = 평균/표준오차.
통과 조건: OOS t > 2 이고 연도별(2022~2026) 평균 bp 모두 > 0 (real 스프레드).

사용법: python -m backtests.wk_limit   (-> results/weekend/limit_summary.csv, limit_yearly.csv, limit_adverse.csv,
                                           limit_weekly_best.csv, limit_latency.csv)
"""
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "weekend"
SYM = "BTCUSD"
DAY, HOUR, MIN = 86_400_000, 3_600_000, 60_000
SWAP = {1: 0.25 / 365, -1: 0.15 / 365}
IS_END = pd.Timestamp("2025-01-01")
THRU_BP = 0.1

CFGS = {"nofilt": (5, 90, False), "ema200": (5, 90, True)}
SPREADS = ["real", 1.5]
LATS = [0, 1, 2]
OFFS = [("bp", 0.0), ("bp", 0.25), ("bp", 0.5), ("bp", 1.0), ("spr", -0.5), ("spr", 0.5), ("spr", 1.0)]
WAITS = [1, 3, 5, 10]
FILLS = ["touch", "thru"]
EXITS = ["mkt", "pex", "tp0.5", "tp1.0", "tp0.5+pex"]


def entries():
    out = [("mkt", None, None, None)]
    for off, w, f in itertools.product(OFFS, WAITS, FILLS):
        out.append(("lim", off, w, f))
    return out


def variants():
    v = []
    for cfg, S, lat in itertools.product(CFGS, SPREADS, LATS):
        for en in entries():
            for ex in EXITS:
                v.append((cfg, S, lat) + en + (ex,))
    return v


def vname(v):
    cfg, S, lat, kind, off, w, f, ex = v
    if kind == "mkt":
        e = "mkt"
    else:
        o = f"{off[1]}bp" if off[0] == "bp" else ("mid" if off[1] == -0.5 else f"{off[1]}spr")
        e = f"lim {o} {w}m {f}"
    return cfg, str(S), lat, e, ex


# ---------------------------------------------------------------- 데이터
def fix(b, a, S):
    if S == "real":
        return b, a
    mid = (b + a) / 2
    h = mid * (S / 2e4)
    return mid - h, mid + h


_MCACHE = {}


def month_ticks(y, m):
    import backtests.wk_breakout as B
    k = (y, m)
    if k not in _MCACHE:
        if len(_MCACHE) > 2:
            _MCACHE.pop(next(iter(_MCACHE)))
        _MCACHE[k] = B.weekend_ticks(SYM, y, m)
    return _MCACHE[k]


def weekend(sat):
    """토요일 00:00 ms -> 그 주말 틱 (달을 넘으면 이어 붙임)"""
    parts = []
    for d in (pd.Timestamp(sat, unit="ms"), pd.Timestamp(sat + DAY, unit="ms")):
        if (d.year, d.month) not in [p[0] for p in parts]:
            parts.append(((d.year, d.month), month_ticks(d.year, d.month)))
    t = np.concatenate([p[1][0] for p in parts])
    b = np.concatenate([p[1][1] for p in parts])
    a = np.concatenate([p[1][2] for p in parts])
    k = (t >= sat + HOUR) & (t < sat + 2 * DAY)
    return t[k], b[k], a[k]


def all_sats():
    from backtests.all_symbols import month_list
    import backtests.wk_meanrev as M
    s = set()
    for y, m in month_list(SYM):
        f = ROOT / "data" / "weekend" / SYM / f"{y}-{m:02d}.npz"
        if not f.exists():
            continue
        t = np.load(f)["t"]
        if len(t):
            s.update(np.unique(M.sat_of(t)[0]).tolist())
    return sorted(int(x) for x in s)


# ---------------------------------------------------------------- 실행 (한 거래)
def hit_fill(px, L, side, thr):
    """지정가 L 체결 틱 (매수: Ask <= L - thr, 매도: Bid >= L + thr). px = 매수면 Ask, 매도면 Bid 구간"""
    h = np.flatnonzero(px <= L - thr) if side > 0 else np.flatnonzero(px >= L + thr)
    return h[0] if h.size else -1


def run_exit(t, b, a, kf, e, side, sl, hold, cond, close_ms, lat, ex, thr_rule):
    """체결 틱 kf, 체결가 e 에서 청산까지. -> (청산 틱, 청산가, 이유)"""
    n = len(t)
    tf = t[kf]
    xs = b if side > 0 else a           # 청산 체결 호가 (매수 포지션은 Bid에 팔고)
    opp = a if side > 0 else b          # 반대 호가 (지정가 청산 걸 곳)
    slp = e - side * sl
    atr = sl / 2.5
    tpk = 0.5 if ex.startswith("tp0.5") else (1.0 if ex.startswith("tp1.0") else None)
    tpp = e + side * tpk * atr if tpk else None
    end_ms = min(tf + hold * MIN, close_ms)
    arr = cond[side]
    j = np.searchsorted(arr, tf, "right")
    if j < len(arr):
        end_ms = min(end_ms, arr[j])
    kend = min(np.searchsorted(t, end_ms + lat * 1000), n - 1)

    def scan(k0, k1):
        seg = xs[k0:k1]
        hs = seg <= slp if side > 0 else seg >= slp
        i_s = np.flatnonzero(hs)
        i_s = i_s[0] if i_s.size else -1
        i_t = -1
        if tpp is not None:
            thr = tpp * THRU_BP / 1e4 if thr_rule == "thru" else 0.0
            ht = seg >= tpp + thr if side > 0 else seg <= tpp - thr
            q = np.flatnonzero(ht)
            i_t = q[0] if q.size else -1
        if i_s < 0 and i_t < 0:
            return None
        if i_t < 0 or (0 <= i_s <= i_t):
            return k0 + i_s, xs[k0 + i_s], "SL"
        return k0 + i_t, tpp, "TP"

    r = scan(kf + 1, kend)
    if r is not None:
        return r
    if "pex" in ex and end_ms < close_ms and kend < n - 1:
        Lx = opp[kend]
        kw = min(np.searchsorted(t, min(t[kend] + MIN, close_ms)), n - 1)
        r = scan(kend + 1, kw)
        thr = Lx * THRU_BP / 1e4 if thr_rule == "thru" else 0.0
        seg = xs[kend + 1:kw]
        q = np.flatnonzero(seg >= Lx + thr) if side > 0 else np.flatnonzero(seg <= Lx - thr)
        ip = kend + 1 + q[0] if q.size else n
        if r is not None and r[0] <= ip:
            return r
        if q.size:
            return ip, Lx, "PEX"
        kx = min(np.searchsorted(t, t[kw] + lat * 1000), n - 1)
        return kx, xs[kx], "PEX_MKT"
    return kend, xs[kend], "TIME"


def pnl_of(e, x, side, tf, tx, sun0):
    p = (x - e) * side
    if tf < sun0 <= tx:
        p -= SWAP[side] * e
    return p


def run_weekend_var(sig, cond, t, b, a, sat, v):
    """한 주말 x 한 변형. -> 행 (ts, filled, bp, bp_cf, why)"""
    cfg, S, lat, kind, off, wait, fill, ex = v
    close_ms = sat + DAY + 23 * HOUR + 50 * MIN
    last_entry = sat + DAY + 23 * HOUR + 30 * MIN
    sun0 = sat + DAY
    n = len(t)
    rows, busy = [], -1
    for ts, side, sl, hold in sig:
        if ts < busy:
            continue
        k0 = np.searchsorted(t, ts + lat * 1000)
        if k0 >= n - 1 or t[k0] >= close_ms:
            continue
        # 같은 시점 시장가 (cf)
        e_m = a[k0] if side > 0 else b[k0]
        ki, xi, _ = run_exit(t, b, a, k0, e_m, side, sl, hold, cond, close_ms, lat, ex, fill or "touch")
        bp_cf = pnl_of(e_m, xi, side, t[k0], t[ki], sun0) / e_m * 1e4
        if kind == "mkt":
            rows.append((ts, 1, bp_cf, bp_cf, 0))
            busy = t[ki]
            continue
        ref = b[k0] if side > 0 else a[k0]
        spr = a[k0] - b[k0]
        o = ref * off[1] / 1e4 if off[0] == "bp" else spr * off[1]
        L = ref - side * o
        exp_ms = min(t[k0] + wait * MIN, last_entry)
        arr = cond[side]
        j = np.searchsorted(arr, ts, "right")
        if j < len(arr):
            exp_ms = min(exp_ms, arr[j])
        kexp = min(np.searchsorted(t, exp_ms), n)
        thr = L * THRU_BP / 1e4 if fill == "thru" else 0.0
        px = a if side > 0 else b
        h = hit_fill(px[k0 + 1:kexp], L, side, thr)
        if h < 0:
            rows.append((ts, 0, np.nan, bp_cf, 0))
            busy = exp_ms
            continue
        kf = k0 + 1 + h
        kx, x, why = run_exit(t, b, a, kf, L, side, sl, hold, cond, close_ms, lat, ex, fill)
        rows.append((ts, 1, pnl_of(L, x, side, t[kf], t[kx], sun0) / L * 1e4, bp_cf, 0))
        busy = t[kx]
    return rows


# ---------------------------------------------------------------- 워커
_G = {}


def _init(sigs, conds, vlist):
    _G.update(sigs=sigs, conds=conds, vlist=vlist)


def _work(sats):
    sigs, conds, vlist = _G["sigs"], _G["conds"], _G["vlist"]
    out = []
    for sat in sats:
        t, b0, a0 = weekend(sat)
        if len(t) < 10:
            continue
        for S in SPREADS:
            b, a = fix(b0, a0, S)
            for vi, v in enumerate(vlist):
                if v[1] != S:
                    continue
                sg = sigs[v[0]].get(sat)
                if sg is None:
                    continue
                for ts, fl, bp, cf, _ in run_weekend_var(sg, conds[v[0]], t, b, a, sat, v):
                    out.append((vi, sat, ts, fl, bp, cf))
    arr = np.array(out, dtype=np.float64) if out else np.zeros((0, 6))
    print(f"  {pd.Timestamp(sats[0], unit='ms').date()} ~ {len(sats)}주말 끝", file=sys.stderr, flush=True)
    return arr


# ---------------------------------------------------------------- 통계
def tstat(x):
    return round(x.mean() / x.std(ddof=1) * np.sqrt(len(x)), 2) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def pf(x):
    gl = -x[x < 0].sum()
    return round(float(x[x > 0].sum() / gl), 3) if gl > 0 else np.inf


def stats(g, nwk, p):
    f = g[g.filled == 1]
    bp = f.bp.values
    r = {f"{p}sig": len(g), f"{p}fill": round(len(f) / len(g) * 100, 1) if len(g) else np.nan,
         f"{p}n": len(f), f"{p}per_wk": round(len(f) / max(nwk, 1), 1)}
    if len(f) < 3:
        return r
    wk = f.groupby("sat").bp.sum().reindex(g.sat.unique(), fill_value=0.0)
    r.update({f"{p}PF": pf(bp), f"{p}avg_bp": round(bp.mean(), 3), f"{p}t": tstat(bp),
              f"{p}win": round((bp > 0).mean() * 100, 1), f"{p}sum_bp": round(bp.sum()),
              f"{p}wk_pos": round((wk > 0).mean() * 100), f"{p}wk_p10": round(wk.quantile(0.1), 1),
              f"{p}wk_med": round(wk.median(), 1), f"{p}worst_wk": round(wk.min(), 1)})
    return r


def main():
    import backtests.wk_meanrev as M
    from backtests.all_symbols import m1_bars
    from backtests.gold_rest_others import resample
    OUT.mkdir(parents=True, exist_ok=True)
    m1 = m1_bars(SYM)
    m5 = resample(m1, 5)
    sigs, conds = {}, {}
    for c, (lvl, hold, filt) in CFGS.items():
        d, cond = M.idea_rsi(m1, m5, lvl, hold, filt)
        sigs[c] = {int(s): list(zip(g.ts.values.astype(np.int64), g.side.values, g.sl.values, g.hold.values))
                   for s, g in d.groupby("wk")}
        conds[c] = cond
    del m1, m5
    sats = all_sats()
    print(f"주말 {len(sats)}개, 변형 {len(variants())}개", file=sys.stderr)
    vlist = variants()
    only = sys.argv[1:]  # 빠른 확인: python -m backtests.wk_limit quick
    if only and only[0] == "quick":
        vlist = [v for v in vlist if v[2] == 1 and v[1] == "real" and v[0] == "nofilt"]
        sats = sats[-30:]
    chunks = [sats[i:i + 6] for i in range(0, len(sats), 6)]
    with ProcessPoolExecutor(14, initializer=_init, initargs=(sigs, conds, vlist)) as ex:
        parts = list(ex.map(_work, chunks))
    arr = np.concatenate(parts)
    df = pd.DataFrame(arr, columns=["vi", "sat", "ts", "filled", "bp", "bp_cf"])
    df["vi"] = df.vi.astype(int)
    df["sat"] = df.sat.astype(np.int64)
    df["year"] = pd.to_datetime(df.sat, unit="ms").dt.year
    sat_dt = pd.to_datetime(pd.Series(sats), unit="ms")
    nwk_is, nwk_oos = int((sat_dt < IS_END).sum()), int((sat_dt >= IS_END).sum())
    nwk_y = sat_dt.dt.year.value_counts().to_dict()
    is_cut = int(IS_END.value // 1_000_000)

    rows, yrows, adv = [], [], []
    for vi, g in df.groupby("vi"):
        cfg, S, lat, en, exn = vname(vlist[vi])
        r = dict(cfg=cfg, S=S, lat=lat, entry=en, exit=exn)
        r.update(stats(g[g.sat < is_cut], nwk_is, "IS_"))
        r.update(stats(g[g.sat >= is_cut], nwk_oos, "OOS_"))
        ys = {}
        for y, gy in g[g.filled == 1].groupby("year"):
            ys[y] = gy.bp.values
            r[f"bp_{y}"] = round(gy.bp.mean(), 2)
            r[f"t_{y}"] = tstat(gy.bp.values)
        r["min_year_bp"] = round(min(v.mean() for v in ys.values()), 2) if ys else np.nan
        r["min_oos_year_bp"] = round(min(ys[y].mean() for y in ys if y >= 2025), 2) if any(y >= 2025 for y in ys) \
            else np.nan
        rows.append(r)
        for per, gg in (("IS", g[g.sat < is_cut]), ("OOS", g[g.sat >= is_cut])):
            fl, uf = gg[gg.filled == 1], gg[gg.filled == 0]
            adv.append(dict(cfg=cfg, S=S, lat=lat, entry=en, exit=exn, period=per, sig=len(gg),
                            fill_pct=round(len(fl) / max(len(gg), 1) * 100, 1),
                            cf_bp_filled=round(fl.bp_cf.mean(), 2), cf_bp_unfilled=round(uf.bp_cf.mean(), 2),
                            cf_bp_all=round(gg.bp_cf.mean(), 2), lim_bp_filled=round(fl.bp.mean(), 2),
                            gain_vs_cf_filled=round((fl.bp - fl.bp_cf).mean(), 2),
                            lim_sum_per_sig=round(fl.bp.sum() / max(len(gg), 1), 3),
                            cf_sum_per_sig=round(gg.bp_cf.sum() / max(len(gg), 1), 3)))
    sm = pd.DataFrame(rows)
    sm["pass"] = (sm.OOS_t > 2) & (sm.min_year_bp > 0)
    sm["pass_oos_only"] = (sm.OOS_t > 2) & (sm.min_oos_year_bp > 0)
    sm.to_csv(OUT / "limit_summary.csv", index=False, encoding="utf-8-sig")
    ad = pd.DataFrame(adv)
    ad.to_csv(OUT / "limit_adverse.csv", index=False, encoding="utf-8-sig")

    # 지연 민감도: 같은 진입/청산을 lat 0/1/2로 나란히
    lt = sm.pivot_table(index=["cfg", "S", "entry", "exit"], columns="lat", values=["OOS_avg_bp", "OOS_t", "IS_avg_bp"])
    lt.columns = [f"{a}_lat{b}" for a, b in lt.columns]
    lt.reset_index().to_csv(OUT / "limit_latency.csv", index=False, encoding="utf-8-sig")

    # 연도별 (real, lat1, OOS t 상위 20 + 시장가 기준선)
    base = sm[(sm.S == "real") & (sm.lat == 1)]
    top = base.sort_values("OOS_t", ascending=False).head(20)
    sel = pd.concat([base[base.entry == "mkt"], top]).drop_duplicates(["cfg", "entry", "exit"])
    sel.to_csv(OUT / "limit_yearly.csv", index=False, encoding="utf-8-sig")

    # 주말별 분포 (real, lat1, OOS t 최고 5개 + 시장가 기준선 2개)
    wrows = []
    for _, r in pd.concat([base[(base.entry == "mkt") & (base.exit == "mkt")], top.head(5)]).iterrows():
        vi = [i for i, v in enumerate(vlist) if vname(v) == (r.cfg, r.S, r.lat, r.entry, r.exit)][0]
        g = df[(df.vi == vi) & (df.filled == 1)].groupby("sat").agg(n=("bp", "size"), sum_bp=("bp", "sum"))
        g = g.reindex(sats, fill_value=0)
        for s, x in g.iterrows():
            wrows.append(dict(cfg=r.cfg, entry=r.entry, exit=r.exit, sat=pd.Timestamp(s, unit="ms").date(),
                              n=int(x.n), sum_bp=round(x.sum_bp, 2)))
    pd.DataFrame(wrows).to_csv(OUT / "limit_weekly_best.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 280, "display.max_columns", 50, "display.max_rows", 300)
    cols = ["cfg", "S", "lat", "entry", "exit", "IS_fill", "IS_n", "IS_avg_bp", "IS_t", "OOS_fill", "OOS_n",
            "OOS_per_wk", "OOS_PF", "OOS_avg_bp", "OOS_t", "OOS_worst_wk", "OOS_wk_pos"] + \
        [f"bp_{y}" for y in range(2022, 2027)] + ["pass", "pass_oos_only"]
    print("== 기준선 (시장가 진입)\n", sm[sm.entry == "mkt"][cols].to_string())
    print("\n== real, lat1, OOS t 상위 25\n", base.sort_values("OOS_t", ascending=False).head(25)[cols].to_string())
    print("\n== fix1.5, lat1, IS+OOS 평균 상위 15\n",
          sm[(sm.S == "1.5") & (sm.lat == 1)].sort_values("OOS_t", ascending=False).head(15)[cols].to_string())
    print("\n통과(real, 모든 해 +, OOS t>2):", int(sm[sm.S == "real"]["pass"].sum()),
          " / OOS 해만:", int(sm[sm.S == "real"]["pass_oos_only"].sum()))


if __name__ == "__main__":
    main()
