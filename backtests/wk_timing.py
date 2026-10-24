"""주말 비트코인(BTCUSD) 단타 - 시간대·주말 구조 연구 (실제 틱, 2022-01 ~ 2026-10)

주말(서버시간 토 01:00 ~ 일 23:50)에만 거래, 월요일 전에 전부 청산. 수수료 없음, 스프레드는 틱 그대로.
체결: 신호는 진입 틱 이전 정보만. 진입 = 신호 시각 이후 첫 틱 (매수 Ask / 매도 Bid),
      청산도 실제 틱 (매수 Bid / 매도 Ask), 손절·익절은 처음 넘는 틱의 가격 (갭이면 그 틱 가격).
      토->일 자정을 넘겨 들고 있으면 스왑 (매수 -25%/년, 매도 -15%/년 / 365, 하루치).
      한 전략당 한 번에 1포지션.
주의: 브로커 주말 거래 시간이 바뀜 - 2022-01~02 주말 거래 없음, 2022-03~07 토 16:00~일 16:00(일부 24:00),
      2022-08~2024-11 토 08:00~일 24:00, 2024-11 말부터 토 01:00~일 24:00.
      그래서 '토 01~08시' 구간은 표본 안(IS)에 없음. 세션 첫 틱 갭 = 금요일 종가 이후 닫혀 있던 동안의 움직임.

0. 기술 통계: 주말 시간대(토/일 x 시)별 평균 수익률·t값·평균 |수익률|·스프레드, 5/15/60분 수익률의
   1차 자기상관 (주말 vs 평일), 5분봉 고저폭 (주말 vs 평일)
1. 시간대 계절성 (S1): 4시간 블록 10개 + (S1h) IS t값이 가장 큰 1시간 하나 (토 08/12/16/20시, 일 00/04/08/12/16/20시 시작),
   IS 평균 방향으로 블록 시작에 진입, 블록 끝에 청산 (토 블록 끝은 23:55로 - 스왑 피함, 일 20시 블록은 23:50)
2. 금요일 종가 자석 (S2): 금요일 마지막 틱 중간값 C. 15분마다 (직전 틱 중간값 - C)가 th x 일봉 ATR 이상 벌어지면
   fade = C 쪽으로 진입, 익절 C, 손절 sl x ATR / follow = 벌어진 쪽으로, 익절·손절 sl x ATR. 일 23:50 청산, 주말 1번.
3. 일요일 저녁 (S3): 일 H시 진입 -> 23:50 청산. 방향 = 매수 / 매도 / 주말 움직임(현재 - C) 따라 / 반대.
4. 세션 시작 갭 (S4): 세션 첫 틱 중간값 - C (토 00:00 정비 / 예전엔 08:00까지 닫힘)가 g x ATR 이상이면
   첫 틱 1분 뒤(토 01:00 이후) 진입, 갭 방향 따라/반대, H시간 뒤 청산 (토 23:55 넘지 않음).
결과: USD/1BTC, bp, R (S2는 손절 거리, 시간 청산은 일봉 ATR x sqrt(보유시간/24)를 1R로),
      %자산 = 거래당 1% 위험 x R 단순 합. IS = 2022~2024 주말, OOS = 2025-01~2026-10. 설정 고르기는 IS만.

사용법: python -m backtests.wk_timing            (BTCUSD 전부 + XETUSD 확인 -> results/weekend/timing_*.csv)
        python -m backtests.wk_timing cache      (주말 틱 캐시만 만들기)
"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.all_symbols import BARS, load, m1_bars, month_list
from backtests.gold_rest_others import resample
from backtests.m5_ticks import first_hit
from indicators import atr

OUT = Path(__file__).resolve().parent.parent / "results" / "weekend"
H = 3_600_000
MIN = 60_000
OOS_FROM = pd.Timestamp("2025-01-01")
W_START, LAST_ENTRY, FORCE = 1 * H, 47 * H + 30 * MIN, 47 * H + 50 * MIN   # 토 00:00 기준
SWAP = {1: 0.25 / 365, -1: 0.15 / 365}


# ---------------------------------------------------------------- 데이터
def cache_dir(sym):
    return BARS / "weekend" / sym


def build_cache(sym):
    """금 22:00 ~ 월 00:00 틱을 주말(토요일 날짜)별로 저장"""
    d = cache_dir(sym)
    d.mkdir(parents=True, exist_ok=True)
    buf = {}
    ms = month_list(sym)
    for y, m in ms:
        t, bid, ask = load(sym, y, m)
        dt = pd.to_datetime(t, unit="ms")
        dow, hr = dt.dayofweek.values, dt.hour.values
        keep = (dow >= 5) | ((dow == 4) & (hr >= 22))
        t, bid, ask, dt = t[keep], bid[keep], ask[keep], dt[keep]
        sat = (dt.normalize() + pd.to_timedelta(5 - dt.dayofweek.values, unit="D")).values.astype("datetime64[ms]").astype(np.int64)
        for s in np.unique(sat):
            k = sat == s
            buf.setdefault(int(s), []).append((t[k], bid[k], ask[k]))
        month_end = pd.Timestamp(y + (m == 12), m % 12 + 1, 1).value // 1_000_000
        for s in [s for s in buf if s + 48 * H <= month_end]:
            parts = buf.pop(s)
            np.savez(d / f"{s}.npz", t=np.concatenate([p[0] for p in parts]),
                     bid=np.concatenate([p[1] for p in parts]), ask=np.concatenate([p[2] for p in parts]))
        del t, bid, ask, dt
        gc.collect()
        print(sym, y, m, flush=True)
    for s, parts in buf.items():
        np.savez(d / f"{s}.npz", t=np.concatenate([p[0] for p in parts]),
                 bid=np.concatenate([p[1] for p in parts]), ask=np.concatenate([p[2] for p in parts]))


def weekends(sym):
    d = cache_dir(sym)
    if not d.exists():
        build_cache(sym)
    return sorted(int(f.stem) for f in d.glob("*.npz"))


def load_wk(sym, s):
    z = np.load(cache_dir(sym) / f"{s}.npz")
    return z["t"], z["bid"], z["ask"]


def daily_atr(sym):
    """토요일 시각(ms) -> 금요일까지 일봉 ATR14 (가격 대비 비율)"""
    d1 = resample(m1_bars(sym), 1440)
    a = (atr(d1) / d1.close).values
    end = d1.time.values.astype("datetime64[ms]").astype(np.int64) + 1440 * MIN   # 일봉 끝 시각
    return end, a


def atr_at(atr_tab, s):
    end, a = atr_tab
    k = np.searchsorted(end, s, "right") - 1
    return float(a[k]) if k >= 0 else np.nan


# ---------------------------------------------------------------- 체결
class Wk:
    """주말 하나의 틱 + 자주 쓰는 값"""
    def __init__(self, sym, s, atr_tab):
        self.s = s
        self.t, self.b, self.a = load_wk(sym, s)
        self.mid = (self.b + self.a) / 2
        f = np.searchsorted(self.t, s)          # 첫 주말 틱
        self.i0 = f
        self.C = self.mid[f - 1] if f > 0 and self.t[f - 1] >= s - 2 * H else np.nan
        self.ok = f < len(self.t)
        self.t0 = self.t[f] if self.ok else None
        self.atr = atr_at(atr_tab, s)
        self.n = len(self.t)

    def idx(self, ms, late=5 * MIN):
        """ms 이후 첫 주말 틱 (late 넘게 늦으면 -1)"""
        i = max(np.searchsorted(self.t, ms), self.i0)
        return i if i < self.n and self.t[i] - ms <= late else -1

    def last_mid(self, ms):
        """ms 직전 주말 틱 중간값 (없으면 nan)"""
        k = np.searchsorted(self.t, ms) - 1
        return self.mid[k] if k >= self.i0 else np.nan


def swap_cost(w, d, i, j, entry):
    return entry * SWAP[d] if w.t[i] < w.s + 24 * H <= w.t[j] else 0.0


def trade(w, d, i, j, risk, kind, extra=None):
    entry = w.a[i] if d > 0 else w.b[i]
    exitp = w.b[j] if d > 0 else w.a[j]
    usd = d * (exitp - entry) - swap_cost(w, d, i, j, entry)
    r = usd / risk
    row = dict(sat=pd.Timestamp(w.s, unit="ms"), entry_time=pd.Timestamp(w.t[i], unit="ms"),
               exit_time=pd.Timestamp(w.t[j], unit="ms"), dir=d, entry=entry, exit=exitp, usd=usd,
               bp=usd / entry * 1e4, R=r, pct=r, kind=kind)
    if extra:
        row.update(extra)
    return row


def time_trade(w, d, t_in, t_out):
    if t_in < w.s + W_START or t_in > w.s + LAST_ENTRY:
        return None
    t_out = min(t_out, w.s + FORCE)
    i, j = w.idx(t_in), w.idx(t_out, 10 * MIN)
    if i < 0 or j < 0 or j <= i or np.isnan(w.atr):
        return None
    hours = (t_out - t_in) / H
    risk = w.atr * w.mid[i] * np.sqrt(hours / 24)
    return trade(w, d, i, j, risk, "time")


def bracket(w, d, i, sl, tp, t_end):
    """i에서 진입, 손절/익절 가격, t_end(이후 첫 틱)에 시간 청산"""
    e = w.idx(t_end, 10 * MIN)
    if e < 0:
        e = w.n - 1                                   # 세션이 일찍 끝난 주말: 마지막 틱
    if e <= i:
        return None
    if d > 0:
        k = first_hit(w.b[i + 1:e], 0, sl, tp)
    else:
        k = first_hit(w.a[i + 1:e], 0, tp, sl)
    j = e if k < 0 else i + 1 + k
    entry = w.a[i] if d > 0 else w.b[i]
    return trade(w, d, i, j, abs(entry - sl), "stop" if k >= 0 else "time")


# ---------------------------------------------------------------- 전략
S1_BLOCKS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40), (40, 44), (44, 48)]
S2_CFGS = [("fade", th, sl) for th in (0.5, 1.0, 1.5) for sl in (0.5, 1.0)] + [("follow", 1.0, 0.5), ("follow", 1.0, 1.0)]
S3_CFGS = [(h, mode) for h in (18, 20, 22) for mode in ("long", "short", "follow", "fade")]
S4_CFGS = [(mode, g, hh) for mode in ("follow", "fade") for g in (0.0, 0.25) for hh in (1, 4)]


def block_name(a, b):
    day = lambda x: ("토" if x < 24 else "일") + f"{x % 24:02d}"
    return f"{day(a)}-{day(b) if b % 24 else ('토24' if b == 24 else '일24')}"


def block_times(w, a, b):
    t_in = w.s + a * H
    t_out = w.s + b * H - (5 * MIN if b == 24 else 0)   # 일 24시는 time_trade에서 23:50으로
    return t_in, t_out


def s1(w, a, b, d):
    t_in, t_out = block_times(w, a, b)
    return time_trade(w, d, t_in, t_out)


def s2(w, mode, th, sl):
    if np.isnan(w.C) or np.isnan(w.atr) or not w.ok:
        return None
    A = w.atr * w.C
    start = max(w.s + W_START, (w.t0 // (15 * MIN) + 1) * 15 * MIN)
    for T in range(int(start), int(w.s + LAST_ENTRY) + 1, 15 * MIN):
        p = w.last_mid(T)
        if np.isnan(p) or abs(p - w.C) < th * A:
            continue
        up = p > w.C
        d = (-1 if up else 1) if mode == "fade" else (1 if up else -1)
        i = w.idx(T)
        if i < 0:
            continue
        entry = w.a[i] if d > 0 else w.b[i]
        if mode == "fade":
            tp = w.C
            if d * (tp - entry) <= 0:
                continue
        else:
            tp = entry + d * sl * A
        stop = entry - d * sl * A
        return bracket(w, d, i, stop, tp, w.s + FORCE)
    return None


def s3(w, h, mode):
    t_in = w.s + (24 + h) * H
    if mode in ("follow", "fade"):
        p = w.last_mid(t_in)
        if np.isnan(p) or np.isnan(w.C) or p == w.C:
            return None
        d = int(np.sign(p - w.C)) * (1 if mode == "follow" else -1)
    else:
        d = 1 if mode == "long" else -1
    return time_trade(w, d, t_in, w.s + FORCE)


def s4(w, mode, g, hh):
    if np.isnan(w.C) or not w.ok or np.isnan(w.atr):
        return None
    gap = w.mid[w.i0] - w.C
    if gap == 0 or abs(gap) < g * w.atr * w.C:
        return None
    d = int(np.sign(gap)) * (1 if mode == "follow" else -1)
    t_in = max(w.t0 + MIN, w.s + W_START)
    t_out = min(t_in + hh * H, w.s + 24 * H - 5 * MIN)
    return time_trade(w, d, t_in, t_out)


def all_configs(s1_dirs, best_hour=None):
    cf = []
    if best_hour:
        h, d = best_hour
        cf.append(("S1h 최고1시간", f"{block_name(h, h + 1)} {'매수' if d > 0 else '매도'}",
                   lambda w, h=h, d=d: s1(w, h, h + 1, d)))
    for (a, b) in S1_BLOCKS:
        d = s1_dirs.get((a, b), 1)
        cf.append(("S1 계절성", f"{block_name(a, b)} {'매수' if d > 0 else '매도'}", lambda w, a=a, b=b, d=d: s1(w, a, b, d)))
    for c in S2_CFGS:
        cf.append(("S2 금요일종가", f"{c[0]} th{c[1]} sl{c[2]}", lambda w, c=c: s2(w, *c)))
    for c in S3_CFGS:
        cf.append(("S3 일요일저녁", f"일{c[0]}시 {c[1]}", lambda w, c=c: s3(w, *c)))
    for c in S4_CFGS:
        cf.append(("S4 시작갭", f"{c[0]} g{c[1]} {c[2]}h", lambda w, c=c: s4(w, *c)))
    return cf


# ---------------------------------------------------------------- 통계
def pf(x):
    gl = -x[x < 0].sum()
    return round(float(x[x > 0].sum() / gl), 2) if gl else float("inf")


def stats(tr, n_wk):
    if len(tr) == 0:
        return dict(trades=0)
    bp = tr.bp.values
    cum = np.cumsum(tr.pct.values)
    dd = float((np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum]).max())
    wk = tr.groupby("sat").pct.sum()
    return dict(trades=len(tr), per_wk=round(len(tr) / n_wk, 2), PF=pf(tr.usd.values), win=round((bp > 0).mean() * 100, 1),
                avg_bp=round(bp.mean(), 1), t=round(bp.mean() / bp.std(ddof=1) * np.sqrt(len(bp)), 2) if len(bp) > 2 else np.nan,
                usd=round(tr.usd.sum()), avg_R=round(tr.R.mean(), 3), pct=round(cum[-1], 1), mdd_pct=round(dd, 1),
                worst_wk=round(wk.min(), 1))


def yearly_str(tr):
    out = {}
    for y, g in tr.groupby(tr.sat.dt.year):
        out[str(y)] = f"{g.pct.sum():+.1f}% PF{pf(g.usd.values)} n{len(g)}"
    return out


# ---------------------------------------------------------------- 0. 기술 통계
def descriptive(sym, wks):
    rows = []
    for s in wks:
        t, b, a = load_wk(sym, s)
        mid = (a + b) / 2
        for h in range(48):
            i = np.searchsorted(t, s + h * H)
            j = np.searchsorted(t, s + (h + 1) * H)
            if i >= len(t) or j >= len(t) or t[i] - (s + h * H) > 3 * MIN or t[j] - (s + (h + 1) * H) > 3 * MIN:
                continue
            sp = (a[i:j] - b[i:j]) / mid[i:j] * 1e4
            rows.append((s, h, (mid[j] / mid[i] - 1) * 1e4, np.median(sp), np.median(a[i:j] - b[i:j])))
    d = pd.DataFrame(rows, columns=["s", "h", "ret", "sp_bp", "sp_usd"])
    d["oos"] = pd.to_datetime(d.s, unit="ms") >= OOS_FROM
    out = []
    for h, g in d.groupby("h"):
        r = dict(h=h, hour=("토" if h < 24 else "일") + f"{h % 24:02d}")
        for nm, gg in (("IS", g[~g.oos]), ("OOS", g[g.oos])):
            x = gg.ret.values
            r.update({f"{nm}_n": len(x), f"{nm}_mean_bp": round(x.mean(), 1) if len(x) else np.nan,
                      f"{nm}_t": round(x.mean() / x.std(ddof=1) * np.sqrt(len(x)), 2) if len(x) > 2 else np.nan,
                      f"{nm}_abs_bp": round(np.abs(x).mean(), 1) if len(x) else np.nan,
                      f"{nm}_spread_bp": round(gg.sp_bp.median(), 2) if len(x) else np.nan,
                      f"{nm}_spread_usd": round(gg.sp_usd.median(), 1) if len(x) else np.nan})
        out.append(r)
    hours = pd.DataFrame(out)

    m1 = m1_bars(sym)
    ac = []
    for k in (1, 5, 15, 60):
        bk = resample(m1, k) if k > 1 else m1[["time", "open", "high", "low", "close"]]
        tm = bk.time.values.astype("datetime64[ms]").astype(np.int64)
        c = bk.close.values
        r = np.log(c[1:] / c[:-1])
        good = np.diff(tm) == k * MIN                     # r[i] = bar i -> bar i+1 (끝 시각 기준 연속)
        tt = pd.to_datetime(tm[1:], unit="ms")
        dow, hr = tt.dayofweek.values, tt.hour.values
        wkend = dow >= 5
        wkday = (dow <= 4) & (hr >= 1) & (hr <= 22)
        pair = good[1:] & good[:-1]
        oos = (tt >= OOS_FROM)[1:]
        for nm, msk in (("주말", wkend), ("평일", wkday)):
            for pn, pm in (("IS", ~oos), ("OOS", oos)):
                m = pair & msk[1:] & msk[:-1] & pm
                x, y = r[:-1][m], r[1:][m]
                rho = np.corrcoef(x, y)[0, 1] if len(x) > 10 else np.nan
                rng = ((bk.high - bk.low) / bk.close).values[1:][:-1][m] * 1e4
                ac.append(dict(min=k, part=nm, period=pn, pairs=len(x), rho1=round(rho, 3),
                               t=round(rho * np.sqrt(len(x)), 1), median_range_bp=round(np.median(rng), 1)))
    return hours, pd.DataFrame(ac)


# ---------------------------------------------------------------- 실행
def run_all(sym, wks, s1_dirs, best_hour=None):
    atr_tab = daily_atr(sym)
    cfgs = all_configs(s1_dirs, best_hour)
    trades = {(f, n): [] for f, n, _ in cfgs}
    s1_both = {(a, b, d): [] for a, b in S1_BLOCKS for d in (1, -1)}
    for s in wks:
        w = Wk(sym, s, atr_tab)
        if not w.ok:
            continue
        for f, n, fn in cfgs:
            r = fn(w)
            if r:
                trades[(f, n)].append(r)
        if not s1_dirs:
            for a, b in S1_BLOCKS:
                for d in (1, -1):
                    r = s1(w, a, b, d)
                    if r:
                        s1_both[(a, b, d)].append(r)
        del w
    to_df = lambda v: pd.DataFrame(v) if v else pd.DataFrame(columns=["sat", "usd", "bp", "R", "pct"])
    return {k: to_df(v) for k, v in trades.items()}, {k: to_df(v) for k, v in s1_both.items()}


def split(tr):
    if len(tr) == 0:
        return tr, tr
    oos = tr.sat >= OOS_FROM
    return tr[~oos], tr[oos]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sym = "BTCUSD"
    wks = [s for s in weekends(sym) if s + 48 * H <= pd.Timestamp("2026-10-24").value // 1_000_000]
    is_wk = sum(pd.Timestamp(s, unit="ms") < OOS_FROM for s in wks)
    oos_wk = len(wks) - is_wk
    print(f"주말 {len(wks)}개 (IS {is_wk}, OOS {oos_wk})")

    hours, ac = descriptive(sym, wks)
    hours.to_csv(OUT / "timing_hours.csv", index=False, encoding="utf-8-sig")
    ac.to_csv(OUT / "timing_autocorr.csv", index=False, encoding="utf-8-sig")
    print(hours.to_string())
    print(ac.to_string())

    # S1 방향은 IS 평균(스프레드 포함 전, 중간값 수익률)으로 정함
    _, both = run_all(sym, wks, {})
    s1_dirs = {}
    for a, b in S1_BLOCKS:
        lg = split(both[(a, b, 1)])[0]
        sh = split(both[(a, b, -1)])[0]
        s1_dirs[(a, b)] = 1 if (lg.bp.mean() if len(lg) else 0) >= (sh.bp.mean() if len(sh) else 0) else -1
    # S1h: IS t값 절댓값이 가장 큰 1시간 (IS 표본 50주 이상, 토 08시~일 22시)
    hh = hours[hours.IS_n >= 50].copy()
    k = hh.IS_t.abs().idxmax()
    best_hour = (int(hh.h[k]), 1 if hh.IS_t[k] > 0 else -1)
    trades, _ = run_all(sym, wks, s1_dirs, best_hour)

    rows = []
    for (f, n), tr in trades.items():
        i, o = split(tr)
        r = dict(family=f, config=n)
        r.update({f"IS_{k}": v for k, v in stats(i, is_wk).items()})
        r.update({f"OOS_{k}": v for k, v in stats(o, oos_wk).items()})
        if len(tr):
            r.update(yearly_str(tr))
        rows.append(r)
    res = pd.DataFrame(rows)
    res["IS_rank"] = res.groupby("family").IS_t.rank(ascending=False, method="first")
    res.to_csv(OUT / "timing_configs.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    cols = ["family", "config", "IS_rank", "IS_trades", "IS_PF", "IS_avg_bp", "IS_t", "IS_pct", "OOS_trades", "OOS_PF",
            "OOS_avg_bp", "OOS_t", "OOS_pct", "OOS_mdd_pct", "OOS_worst_wk"]
    print(res[cols].sort_values(["family", "IS_rank"]).to_string())
    print(res[["family", "config", "IS_rank"] + [c for c in res.columns if c.isdigit()]].sort_values(["family", "IS_rank"]).to_string())

    # IS 최고 설정 거래 기록 + ETH 확인
    best = res[res.IS_rank == 1]
    pd.concat([trades[(f, n)].assign(family=f, config=n) for f, n in zip(best.family, best.config)]).to_csv(
        OUT / "timing_trades_best.csv", index=False, encoding="utf-8-sig")
    eth = "XETUSD"
    ewks = [s for s in weekends(eth) if s + 48 * H <= pd.Timestamp("2026-10-24").value // 1_000_000]
    e_is = sum(pd.Timestamp(s, unit="ms") < OOS_FROM for s in ewks)
    etr, _ = run_all(eth, ewks, s1_dirs, best_hour)
    erows = []
    for f, n in zip(best.family, best.config):
        tr = etr[(f, n)]
        i, o = split(tr)
        r = dict(symbol=eth, family=f, config=n)
        r.update({f"IS_{k}": v for k, v in stats(i, e_is).items()})
        r.update({f"OOS_{k}": v for k, v in stats(o, len(ewks) - e_is).items()})
        if len(tr):
            r.update(yearly_str(tr))
        erows.append(r)
    er = pd.DataFrame(erows)
    er.to_csv(OUT / "timing_eth.csv", index=False, encoding="utf-8-sig")
    print(er.to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cache":
        for s in ("BTCUSD", "XETUSD"):
            build_cache(s)
    else:
        main()
