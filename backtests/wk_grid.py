"""주말 BTCUSD 단타 - 범위 그리드(마틴 없음) + 틱 미시구조 (실제 틱, 2022-01 ~ 2026-10)

주말 창(서버시간): 토 01:00 ~ 일 23:50. 일 23:30 이후 새 진입 없음, 일 23:50 첫 틱에 전부 청산.
체결: 신호는 그 틱까지의 정보로만, 주문은 300ms 뒤 첫 틱에 (매수 Ask / 매도 Bid). 청산도 실제 틱 (매수 Bid / 매도 Ask).
      지정가 그리드: 매수 지정가 L은 Ask <= L일 때 L에, 매도 지정가는 Bid >= L일 때.
스왑: 토->일 자정을 넘기면 매수 -25%/년, 매도 -15%/년 x 명목 / 365.  수수료 없음.

0. 짧은 구간 모멘텀/되돌림 확인: s초 동안 순이동(mid) 다음 h초 수익률의 방향 (중간값 기준)
1. 범위 그리드 (grid)   : 세션(토 01:00~23:59, 일 00:00~23:50) 처음 R시간 mid 범위, 가운데 기준 위아래 N=3단계 지정가
                           (아래 매수, 위 매도, 칸 = 범위/6, 최소 4bp), 수명 life시간 뒤 전부 정리하고 새 범위,
                           단계마다 익절 = 한 칸, 범위 밖으로 X칸 넘게 나가면 전부 손절하고 그 자리부터 새 범위 (마틴 없음)
2. 틱 방향 (tickdir)    : 최근 s초 mid 순이동 >= k bp면 따라가기 또는 반대로(fade), 익절/손절 bp, 시간 손절
3. 스프레드 거르기       : 2번에 '지금 스프레드 <= 최근 스프레드 중간값' 조건 (spread)
4. 틱 수 급증 (surge)   : 최근 60초 틱 수가 직전 30분 평균의 m배 넘고 순이동 k bp 이상 -> 따라가기/반대
IS = 2022-01 ~ 2024-12 주말, OOS = 2025-01 ~ 2026-10. 설정은 IS로만 고르고 OOS·연도별·XETUSD(같은 설정)로 확인.
크기(복리 없이 더함): 그리드는 사이클 최악 손실(3단계 다 찬 뒤 손절) = 잔고 2%, 틱 단타는 손절 거리+진입 스프레드 = 잔고 0.5%.
자료: BTC 틱은 실제로 2022-03부터. 하루가 통째로 빈 주말(자료 빠짐 3번, 진행 중인 2026-10-24)은 뺌.

사용법: python -m backtests.wk_grid          (전부 -> results/weekend/grid_*.csv)
        python -m backtests.wk_grid cache    (주말 틱만 뽑아 data/weekend_grid/ 에 저장)
"""
import gc
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.all_symbols import load, month_list

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "weekend_grid"
OUT = ROOT / "results" / "weekend"
DAY = 86_400_000
HOUR = 3_600_000
LAT = 300
OPEN_MS = 1 * HOUR                      # 토 01:00
LAST_ENTRY = DAY + 23 * HOUR + 30 * 60_000   # 일 23:30 (토 00:00 기준)
CLOSE_MS = DAY + 23 * HOUR + 50 * 60_000     # 일 23:50
MIDNIGHT = DAY
SWAP_LONG, SWAP_SHORT = 0.25 / 365, 0.15 / 365
RISK_GRID = 0.02                 # 그리드: 단계 전부 찬 뒤 손절 = 잔고 2%
RISK_TICK = 0.005                # 틱 단타: 손절 거리 = 잔고 0.5%
OOS_START = int(pd.Timestamp("2025-01-01").value // 10**6)


# ---------------------------------------------------------------- 주말 틱 뽑기
def _extract(args):
    sym, y, m = args
    t, bid, ask = load(sym, y, m)
    dow = (t // DAY + 3) % 7               # 월=0 ... 토=5, 일=6
    k = (dow == 5) | (dow == 6)
    r = t[k], bid[k], ask[k]
    del t, bid, ask
    gc.collect()
    return r


def build_cache(sym):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{sym}.npz"
    if f.exists():
        return
    ms = month_list(sym)
    parts = []
    with ProcessPoolExecutor(6) as ex:
        for r in ex.map(_extract, [(sym, y, m) for y, m in ms]):
            parts.append(r)
    t = np.concatenate([p[0] for p in parts])
    o = np.argsort(t, kind="stable")
    np.savez(f, t=t[o], bid=np.concatenate([p[1] for p in parts])[o], ask=np.concatenate([p[2] for p in parts])[o])


def weekends(sym):
    """[(토 00:00 ms, t, bid, ask)]  - 토 01:00~일 23:50 틱이 있는 주말"""
    with np.load(CACHE / f"{sym}.npz") as z:
        t, bid, ask = z["t"], z["bid"], z["ask"]
    day = t // DAY
    sat = day - ((day + 3) % 7 == 6)        # 일요일이면 전날(토)
    out = []
    st = np.r_[0, np.flatnonzero(np.diff(sat)) + 1, len(t)]
    for a, b in zip(st[:-1], st[1:]):
        s0 = int(sat[a]) * DAY
        tt = t[a:b] - s0
        k = (tt >= OPEN_MS - 30 * 60_000) & (tt <= CLOSE_MS + 60_000)   # 미시구조 준비용 30분 앞
        sat_n = ((tt >= OPEN_MS) & (tt < DAY)).sum()
        sun_n = ((tt >= DAY) & (tt < CLOSE_MS)).sum()
        if sat_n < 1000 or sun_n < 1000:          # 하루가 통째로 빈 주말(자료 빠짐, 진행 중인 주말) 빼기
            continue
        out.append((s0, tt[k], bid[a:b][k], ask[a:b][k]))
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "cache":
    for s in ("BTCUSD", "XETUSD"):
        build_cache(s)
        print(s, "ok")


# ---------------------------------------------------------------- 공통
def first_true(mask):
    k = np.flatnonzero(mask)
    return int(k[0]) if k.size else -1


def swap_cost(side, ent_t, ex_t, price):
    """토->일 자정을 넘기면 하룻밤 스왑 ($/BTC)"""
    if ent_t < MIDNIGHT <= ex_t:
        return price * (SWAP_LONG if side > 0 else SWAP_SHORT)
    return 0.0


# ---------------------------------------------------------------- 0. 짧은 구간 모멘텀/되돌림
def momentum_table(W):
    rows = []
    sec = np.arange(OPEN_MS, CLOSE_MS, 1000)
    lm_all, oos_all = [], []
    for s0, t, b, a in W:
        i = np.searchsorted(t, sec, "right") - 1
        lm_all.append(np.log((a[i] + b[i]) / 2) * 1e4)
        oos_all.append(s0 >= OOS_START)
    for s in (5, 15, 60):
        for h in (5, 30, 120):
            X, Y, O = [], [], []
            for lm, oo in zip(lm_all, oos_all):
                x = lm[s:-h] - lm[:-s - h]
                y = lm[s + h:] - lm[s:-h]
                X.append(x[::5]), Y.append(y[::5]), O.append(np.full(len(x[::5]), oo))
            X, Y, O = map(np.concatenate, (X, Y, O))
            for k in (6, 10, 20):
                sel = np.abs(X) >= k
                c = np.sign(X[sel]) * Y[sel]
                rows.append(dict(s초=s, h초=h, k_bp=k, n=int(sel.sum()), 따라간_다음수익_bp=round(c.mean(), 2),
                                 IS=round(c[~O[sel]].mean(), 2), OOS=round(c[O[sel]].mean(), 2)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2~4. 틱 신호 (1초마다 판단)
def tick_signals(t, b, a, cfg):
    """1초 격자에서 신호 -> (판단 틱 인덱스, 방향) 배열"""
    sec = np.arange(OPEN_MS, LAST_ENTRY, 1000)
    i = np.searchsorted(t, sec, "right") - 1
    ok = i >= 0
    sec, i = sec[ok], i[ok]
    mid = (a + b) / 2
    s = cfg["s"]
    j = np.searchsorted(t, sec - s * 1000, "right") - 1
    j = np.maximum(j, 0)
    mv = (mid[i] / mid[j] - 1) * 1e4
    sig = np.abs(mv) >= cfg["k"]
    if cfg.get("spread"):          # 지금 스프레드 <= 최근 10분(600초) 스프레드 중간값
        sp = a[i] - b[i]
        med = pd.Series(sp).rolling(600, min_periods=60).median().values
        sig &= sp <= med
    if cfg.get("surge"):           # 최근 60초 틱 수 >= 직전 30분 1분 평균 x m
        n60 = i - (np.searchsorted(t, sec - 60_000, "right") - 1)
        n30 = (np.searchsorted(t, sec - 60_000, "right") - 1) - (np.searchsorted(t, sec - 1_860_000, "right") - 1)
        sig &= n60 >= cfg["surge"] * np.maximum(n30, 1) / 30
    d = np.sign(mv) * (1 if cfg["mode"] == "follow" else -1)
    return i[sig], d[sig].astype(int)


def run_tick(W, cfg):
    tp, sl, T = cfg["tp"] / 1e4, cfg["sl"] / 1e4, cfg["T"] * 1000
    out = []
    for s0, t, b, a in W:
        di, dd = tick_signals(t, b, a, cfg)
        free = -1
        cend = np.searchsorted(t, CLOSE_MS, "left")
        if cend >= len(t):
            cend = len(t) - 1
        p = 0
        while True:
            p = np.searchsorted(t[di], free, "right") if free >= 0 else 0
            if p >= len(di):
                break
            dt = t[di[p]]
            e = np.searchsorted(t, dt + LAT, "left")
            if e >= cend:
                break
            side = dd[p]
            px = a[e] if side > 0 else b[e]
            end = min(np.searchsorted(t, t[e] + T, "left"), cend)
            if side > 0:
                seg = b[e + 1:end]
                k = first_true((seg >= px * (1 + tp)) | (seg <= px * (1 - sl)))
            else:
                seg = a[e + 1:end]
                k = first_true((seg <= px * (1 - tp)) | (seg >= px * (1 + sl)))
            x = e + 1 + k if k >= 0 else end
            xp = b[x] if side > 0 else a[x]
            pnl = side * (xp - px) - swap_cost(side, t[e], t[x], px)
            # 크기: 손절 거리(+스프레드)를 위험 1R로
            out.append(dict(wk=s0, t=t[e], side=side, entry=px, exit=xp, usd=pnl, bp=pnl / px * 1e4,
                            risk_bp=cfg["sl"] + (a[e] - b[e]) / px * 1e4, hold=(t[x] - t[e]) / 1000))
            free = t[x]
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 1. 범위 그리드
SESSIONS = [(OPEN_MS, DAY - 60_000 - 10 * 60_000, DAY - 60_000),      # 토 01:00~23:49 진입, 23:59 청산 (스왑 없음)
            (DAY, LAST_ENTRY, CLOSE_MS)]                                 # 일 00:00~23:30 진입, 23:50 청산


def grid_session(t, b, a, start, last, close, cfg, s0, out):
    R, N, X, gmin = cfg["R"] * HOUR, cfg["N"], cfg["X"], cfg["gmin"] / 1e4
    cend = min(np.searchsorted(t, close, "left"), len(t) - 1)
    f0 = start
    while True:
        i0, i1 = np.searchsorted(t, f0, "left"), np.searchsorted(t, f0 + R, "left")
        if f0 + R >= last or i1 >= cend or i1 - i0 < 50:
            return
        mid = (a[i0:i1] + b[i0:i1]) / 2
        hi, lo = mid.max(), mid.min()
        c = (hi + lo) / 2
        g = max((hi - lo) / (2 * N), gmin * c)
        M = N + X
        ilast = min(np.searchsorted(t, last, "left"), cend)
        iend = cend                                      # 그리드 수명 (life 시간) 끝 또는 세션 끝
        if cfg.get("life"):
            iend = min(np.searchsorted(t, f0 + R + cfg["life"] * HOUR, "left"), cend)
        tt, aa, bb = t[i1:iend + 1], a[i1:iend + 1], b[i1:iend + 1]
        ca = np.clip(np.ceil((aa - c) / g), -M - 1, M + 1).astype(int)
        fb = np.clip(np.floor((bb - c) / g), -M - 1, M + 1).astype(int)
        ch = np.r_[0, np.flatnonzero((np.diff(ca) != 0) | (np.diff(fb) != 0)) + 1]
        # 상태: open[j] = 진입가 (j<0 매수, j>0 매도), armed 시각
        pos = {}
        arm = {j: 0 for j in list(range(-N, 0)) + list(range(1, N + 1))}
        worst = g * (N * M - N * (N + 1) / 2)
        cyc = dict(wk=s0, start=int(f0), c=c, g_bp=g / c * 1e4, fills=0, tp=0, usd=0.0, stop=0, worst=worst)
        stopped = False
        for k in ch:                                     # 호가가 칸을 넘는 틱만
            ti, A, B = tt[k], ca[k], fb[k]
            gi = i1 + k
            if B <= -M - 1 or A >= M + 1:               # 범위 밖 X칸 넘으면 전부 손절
                for j, px in pos.items():
                    cyc["usd"] += (bb[k] - px) if j < 0 else (px - aa[k])
                pos = {}
                cyc["stop"] = 1
                stopped = True
                break
            for j in list(pos):                          # 익절 (한 칸)
                px = pos[j]
                if (j < 0 and B >= j + 1) or (j > 0 and A <= j - 1):
                    cyc["usd"] += (c + (j + 1) * g - px) if j < 0 else (px - c - (j - 1) * g)
                    cyc["tp"] += 1
                    del pos[j]
                    arm[j] = ti + LAT
            if gi < ilast:                               # 지정가 체결
                for j in arm:
                    if j in pos or ti < arm[j]:
                        continue
                    if (j < 0 and A <= j) or (j > 0 and B >= j):
                        # 처음 걸 때 이미 가격이 넘어가 있으면 그 자리 시장가 (매수 Ask / 매도 Bid)
                        pos[j] = c + j * g if k else (aa[k] if j < 0 else bb[k])
                        cyc["fills"] += 1
        if not stopped:                                  # 수명/세션 끝 청산 (시장가)
            for j, px in pos.items():
                cyc["usd"] += (b[iend] - px) if j < 0 else (px - a[iend])
        if cyc["fills"]:
            cyc["bp"] = cyc["usd"] / c * 1e4
            cyc["R"] = cyc["usd"] / worst
            out.append(cyc)
        if not stopped and iend >= cend:
            return
        f0 = int(tt[k]) if stopped else int(t[iend])     # 손절/수명 끝 뒤 새 범위


def run_grid(W, cfg):
    out = []
    for s0, t, b, a in W:
        for st, la, cl in SESSIONS:
            grid_session(t, b, a, st, la, cl, cfg, s0, out)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 성적
def year_of(ms):
    return pd.to_datetime(ms, unit="ms").year


def stats(df, risk, part=""):
    """df: usd, bp, R, wk.  잔고% = R x 위험% (복리 없이 더함)"""
    if df is None or len(df) == 0:
        return {f"{part}n": 0}
    pct = df.R.values * risk * 100
    gp, gl = df.usd[df.usd > 0].sum(), -df.usd[df.usd < 0].sum()
    eq = np.cumsum(pct)
    dd = (np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()
    wk = pd.Series(pct).groupby(df.wk.values).sum()
    return {f"{part}n": len(df), f"{part}PF": round(gp / gl, 2) if gl > 0 else np.inf,
            f"{part}승률": round((df.usd > 0).mean() * 100, 1), f"{part}평균bp": round(df.bp.mean(), 2),
            f"{part}평균usd": round(df.usd.mean(), 1), f"{part}잔고%": round(pct.sum(), 1),
            f"{part}주말당%": round(wk.mean(), 2), f"{part}MDD%": round(dd, 1), f"{part}최악주말%": round(wk.min(), 1)}


def summarize(name, cfg, df, risk):
    r = dict(아이디어=name, 설정=str(cfg))
    if len(df):
        is_ = df[df.wk < OOS_START]
        oos = df[df.wk >= OOS_START]
        r.update(stats(is_, risk, "IS_"))
        r.update(stats(oos, risk, "OOS_"))
        y = year_of(df.wk.values)
        for yr in range(2022, 2027):
            d = df[y == yr]
            r[f"{yr}%"] = round(d.R.sum() * risk * 100, 1)
            gl = -d.usd[d.usd < 0].sum()
            r[f"{yr}PF"] = round(d.usd[d.usd > 0].sum() / gl, 2) if gl > 0 else np.nan
    return r


def tick_df(W, cfg):
    df = run_tick(W, cfg)
    if len(df):
        df["R"] = df.bp / df.risk_bp
    return df


TICK_CFGS = [dict(s=s, k=k, mode=m, tp=tp, sl=sl, T=T)
             for s, k in ((5, 10), (5, 20), (30, 10), (30, 20))
             for m in ("fade", "follow")
             for tp, sl, T in ((6, 6, 60), (10, 10, 300))]
SURGE_CFGS = [dict(s=60, k=10, surge=m, mode=md, tp=tp, sl=sl, T=T)
              for m in (3, 5) for md in ("fade", "follow") for tp, sl, T in ((6, 6, 60), (10, 10, 300))]
GRID_CFGS = [dict(R=R, N=3, X=X, gmin=4, life=life) for R in (1, 2, 4) for X in (1, 3) for life in (2, None)]
SPREAD_CFGS = [dict(s=s, k=20, mode=m, tp=10, sl=10, T=300, spread=1) for s in (5, 30) for m in ("fade", "follow")]


# ---------------------------------------------------------------- 실행
IDEAS = [("1_그리드", GRID_CFGS, RISK_GRID), ("2_틱방향", TICK_CFGS, RISK_TICK),
         ("3_스프레드", SPREAD_CFGS, RISK_TICK), ("4_틱급증", SURGE_CFGS, RISK_TICK)]


def run_cfg(W, cfg, risk):
    return run_grid(W, cfg) if risk == RISK_GRID else tick_df(W, cfg)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    W = weekends("BTCUSD")
    print("주말", len(W))
    momentum_table(W).to_csv(OUT / "grid_momentum.csv", index=False, encoding="utf-8-sig")
    rows, best = [], []
    for name, cfgs, risk in IDEAS:
        part = []
        for cfg in cfgs:
            r = summarize(name, cfg, run_cfg(W, cfg, risk), risk)
            part.append(r)
            print(name, cfg, r.get("IS_PF"), r.get("IS_n"), r.get("OOS_PF"), r.get("OOS_n"), flush=True)
        d = pd.DataFrame(part)
        d["IS순위"] = d.IS_PF.where(d.IS_n >= 100, -1).rank(ascending=False, method="first").astype(int)
        rows.append(d)
        top = d.sort_values("IS순위").iloc[0]
        best.append((name, eval(top["설정"]), risk))
    allc = pd.concat(rows)
    allc.to_csv(OUT / "grid_configs.csv", index=False, encoding="utf-8-sig")
    allc[allc.IS순위 <= 3].sort_values(["아이디어", "IS순위"]).to_csv(OUT / "grid_top3.csv", index=False,
                                                                 encoding="utf-8-sig")
    del W
    gc.collect()
    E = weekends("XETUSD")
    eth = []
    for name, cfg, risk in best:
        r = summarize(name + "_ETH", cfg, run_cfg(E, cfg, risk), risk)
        eth.append(r)
    pd.concat([allc[allc.IS순위 == 1], pd.DataFrame(eth)]).to_csv(OUT / "grid_best.csv", index=False,
                                                                  encoding="utf-8-sig")
    print(pd.concat([allc[allc.IS순위 <= 3], pd.DataFrame(eth)]).to_string())


if __name__ == "__main__" and len(sys.argv) == 1:
    main()
