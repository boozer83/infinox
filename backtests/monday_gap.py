"""월요일 시가 갭 연구: 주말 코인 움직임 -> 평일 종목 월요일 방향? 갭 메우기/따라가기? (실제 틱, 2022-01 ~ 2026-10)

종목: XAUUSD, XAUEUR, NAS100, GER40, JPN225 (주말 거래 없음). 예측 변수: BTCUSD, XETUSD 주말 수익률.
매주:
  금 종가 = 주말 쉬는 시간(40시간 넘는 틱 공백) 직전 마지막 틱 Bid,  월 시가 = 그 뒤 첫 틱 (월요일만, 공휴일로 화요일 열면 뺌)
  갭 = 월 첫 Bid - 금 마지막 Bid,  갭(ATR) = 갭 / 일봉 ATR14 (월요일 전 날들만, 서버시간 날짜 기준)
  BTC 주말 수익률 = 그 종목 금 종가 시각 -> 월 시가 시각의 BTC 1분봉 종가 변화 (ETH도 같은 방식)
거래 (월요일 하루, 한 주 1번):
  진입 = 시가 + 대기(0/5/15/30분) 후 첫 틱 (매수 Ask, 매도 Bid) -> 열 때 스프레드가 넓어서 기다리는 경우도 시험
  청산 = 시간(1시간 / 3시간 / 09:00 아시아 끝 / 22:00), 손절(진입가에서 일봉 ATR x 0.5 / 1.0 / 없음),
         갭 메우기는 Bid가 금 종가 Bid에 닿으면 익절 (매수는 Bid로, 매도는 Ask로 청산)
  A. 갭 메우기: 갭 반대 방향 (진입 전에 이미 메워졌으면 거래 안 함)
  B. 갭 따라가기: 갭 방향
  C. BTC 방향: 주말 BTC 수익률 부호 방향
  조건: |갭| >= 0 / 0.1 / 0.25 / 0.5 ATR (C는 |BTC| >= 0 / 1 / 3 / 5 %)
검증: IS = 2022-01 ~ 2024-12, OOS = 2025-01 ~ 2026-10. 손익은 일봉 ATR 단위와 가격 단위(1단위 계약, 스프레드 포함).
t = 평균 / (표준편차 / √n) (ATR 단위). 수수료·스왑 없음.

사용법: python -m backtests.monday_gap     (-> results/monday_gap/*.csv)
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.all_symbols import load, m1_bars, month_list

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "monday_gap"
OUT = ROOT / "results" / "monday_gap"
SYMS = ["XAUUSD", "XAUEUR", "NAS100", "GER40", "JPN225"]
H = 3_600_000
MIN = 60_000
DAY = 86_400_000
IS_END = pd.Timestamp("2025-01-01")
WAITS = [0, 5, 15, 30]                       # 분
EXITS = ["1h", "3h", "09:00", "22:00"]
SLS = [0.5, 1.0, None]                       # 일봉 ATR 배수
GAP_MIN = [0.0, 0.1, 0.25, 0.5]              # 갭 / ATR
BTC_MIN = [0.0, 0.01, 0.03, 0.05]            # BTC 주말 수익률


def dow(t):
    return ((t // DAY) + 3) % 7  # 월=0


# ---------------------------------------------------------------- 1. 주마다 금 종가·월 시가·월요일 틱 뽑기
def extract(sym):
    """data/monday_gap/<종목>.npz: 주 표(open_t, fri_t, fri_b, fri_a) + 월요일 틱(시가 ~ 그날 23:00)"""
    f = CACHE / f"{sym}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    rows, tt, bb, aa, off = [], [], [], [], [0]
    tail = None
    for y, m in month_list(sym):
        if (y, m) < (2022, 1):
            continue
        t, b, a = load(sym, y, m)
        if tail is not None:
            t, b, a = np.r_[tail[0], t], np.r_[tail[1], b], np.r_[tail[2], a]
        gap = np.r_[0, np.diff(t)]
        for o in np.flatnonzero(gap > 40 * H):
            if dow(t[o]) != 0:
                continue
            end = (t[o] // DAY) * DAY + 23 * H
            e = np.searchsorted(t, end)
            rows.append((t[o], t[o - 1], b[o - 1], a[o - 1]))
            tt.append(t[o:e]), bb.append(b[o:e]), aa.append(a[o:e])
            off.append(off[-1] + e - o)
        k = t >= t[-1] - 3 * DAY
        tail = (t[k], b[k], a[k])
        print(f"  {sym} {y}-{m:02d}", flush=True)
    r = np.array(rows)
    d = dict(open_t=r[:, 0].astype(np.int64), fri_t=r[:, 1].astype(np.int64), fri_b=r[:, 2], fri_a=r[:, 3],
             t=np.concatenate(tt), b=np.concatenate(bb), a=np.concatenate(aa), off=np.array(off))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, **d)
    return d


def daily_atr(sym):
    """서버 날짜 일봉 ATR14 (그날 포함) -> 다음 날부터 씀"""
    m1 = m1_bars(sym)
    g = m1.set_index("time").resample("1D")
    d = pd.DataFrame({"high": g.high.max(), "low": g.low.min(), "close": g.close.last()}).dropna()
    pc = d.close.shift(1)
    tr = np.maximum(d.high - d.low, np.maximum((d.high - pc).abs(), (d.low - pc).abs()))
    return tr.rolling(14).mean()


def price_at(m1, ms):
    """그 시각 이전에 끝난 마지막 1분봉 종가"""
    end = m1.time.values.astype("datetime64[ms]").astype(np.int64) + MIN
    i = np.searchsorted(end, ms, side="right") - 1
    return m1.close.values[np.clip(i, 0, None)]


def week_table(sym, d, btc, eth):
    w = pd.DataFrame({"open_t": d["open_t"], "fri_t": d["fri_t"], "fri_b": d["fri_b"], "fri_a": d["fri_a"]})
    w["date"] = pd.to_datetime(w.open_t, unit="ms").dt.normalize()
    w["open_b"] = d["b"][d["off"][:-1]]
    w["open_a"] = d["a"][d["off"][:-1]]
    w["open_sp"] = w.open_a - w.open_b
    at = daily_atr(sym)
    w["atr"] = at.reindex(w.date - pd.Timedelta(days=1), method="ffill").values  # 월요일 전날까지
    w["gap"] = w.open_b - w.fri_b
    w["gap_atr"] = w.gap / w.atr
    for name, m1 in (("btc", btc), ("eth", eth)):
        w[name] = price_at(m1, w.open_t.values) / price_at(m1, w.fri_t.values) - 1
    # 월 시가 -> 22:00 Bid 수익(ATR), 갭이 그날 메워졌는지
    t, b = d["t"], d["b"]
    oc, filled = [], []
    for i in range(len(w)):
        s, e = d["off"][i], d["off"][i + 1]
        tt, bb = t[s:e], b[s:e]
        j = np.searchsorted(tt, (tt[0] // DAY) * DAY + 22 * H) - 1
        oc.append(bb[max(j, 0)] - bb[0])
        g = w.gap.iat[i]
        filled.append(bool((bb <= w.fri_b.iat[i]).any() if g > 0 else (bb >= w.fri_b.iat[i]).any()))
    w["oc_atr"] = np.array(oc) / w.atr
    w["filled"] = filled
    return w


# ---------------------------------------------------------------- 2. 거래 시뮬레이션 (틱)
def trade(tt, bb, aa, t0, d, exit_at, sl, target):
    """t0 이후 첫 틱 진입, d=+1 매수 / -1 매도. 반환 (가격 손익, 진입 시 스프레드, 청산 이유) 또는 None"""
    e = np.searchsorted(tt, t0)
    if e >= len(tt) or tt[e] >= exit_at:
        return None
    entry = aa[e] if d > 0 else bb[e]
    x = np.searchsorted(tt, exit_at) - 1           # 시간 청산: 그 시각 직전 마지막 틱
    if x <= e:
        return None
    sb, sa = bb[e + 1:x + 1], aa[e + 1:x + 1]
    n = len(sb)
    cand = [(n - 1, "시간")]
    if sl is not None:
        hit = np.flatnonzero(sb <= entry - sl) if d > 0 else np.flatnonzero(sa >= entry + sl)
        if len(hit):
            cand.append((hit[0], "손절"))
    if target is not None:
        hit = np.flatnonzero(sb >= target) if d > 0 else np.flatnonzero(sb <= target)
        if len(hit):
            cand.append((hit[0], "메움"))
    k, why = min(cand, key=lambda c: c[0])
    out = sb[k] if d > 0 else sa[k]
    return d * (out - entry), aa[e] - bb[e], why


def exit_time(t_open, ex):
    if ex == "1h":
        return t_open + H
    if ex == "3h":
        return t_open + 3 * H
    day = (t_open // DAY) * DAY
    return day + (9 if ex == "09:00" else 22) * H


def run_configs(sym, d, w):
    rows = []
    t, b, a = d["t"], d["b"], d["a"]
    recs = list(w.itertuples())
    for strat, wait, ex, sl in itertools.product(["메우기", "따라가기", "BTC"], WAITS, EXITS, SLS):
        for i, r in enumerate(recs):
            if not np.isfinite(r.atr) or r.gap == 0:
                continue
            s, e = d["off"][i], d["off"][i + 1]
            tt, bb, aa = t[s:e], b[s:e], a[s:e]
            t0 = tt[0] + wait * MIN
            xa = exit_time(tt[0], ex)
            if strat == "BTC":
                dd = 1 if r.btc > 0 else -1
            else:
                dd = -1 if r.gap > 0 else 1
                if strat == "따라가기":
                    dd = -dd
            target = r.fri_b if strat == "메우기" else None
            if strat == "메우기" and wait:
                k = np.searchsorted(tt, t0)
                pre = bb[:k]
                if ((pre <= r.fri_b).any() if r.gap > 0 else (pre >= r.fri_b).any()):
                    continue                        # 기다리는 동안 이미 메워짐
            res = trade(tt, bb, aa, t0, dd, xa, None if sl is None else sl * r.atr, target)
            if res is None:
                continue
            pnl, sp, why = res
            rows.append((sym, strat, wait, ex, sl, r.date, r.gap_atr, r.btc, pnl, pnl / r.atr, sp / r.atr, why))
    return pd.DataFrame(rows, columns=["sym", "strat", "wait", "exit", "sl", "date", "gap_atr", "btc", "pnl",
                                       "r_atr", "sp_atr", "why"])


# ---------------------------------------------------------------- 3. 통계
def stats(x, px=None):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return dict(n=n)
    gl = -x[x < 0].sum()
    out = dict(n=n, win=round((x > 0).mean() * 100, 1), pf=round(x[x > 0].sum() / gl, 2) if gl else np.inf,
               avg_atr=round(x.mean(), 4), t=round(x.mean() / (x.std(ddof=1) / np.sqrt(n)), 2) if x.std() else 0)
    if px is not None:
        out["avg_px"] = round(float(np.mean(px)), 3)
    return out


def filt(tr):
    """조건 묶음: (이름, 마스크)"""
    out = []
    for g in GAP_MIN:
        out.append((f"|갭|>={g}ATR", tr.gap_atr.abs() >= g))
    return out


def summarize(tr):
    rows = []
    for key, g in tr.groupby(["sym", "strat", "wait", "exit", "sl"], dropna=False):
        conds = ([(f"|BTC|>={c:.0%}", g.btc.abs() >= c) for c in BTC_MIN] if key[1] == "BTC" else filt(g))
        for cname, msk in conds:
            gg = g[msk]
            iis, oos = gg[gg.date < IS_END], gg[gg.date >= IS_END]
            s_is, s_oos = stats(iis.r_atr, iis.pnl), stats(oos.r_atr, oos.pnl)
            yr = gg.groupby(gg.date.dt.year).r_atr.sum()
            rows.append(dict(sym=key[0], strat=key[1], wait=key[2], exit=key[3], sl=key[4], cond=cname,
                             **{f"is_{k}": v for k, v in s_is.items()}, **{f"oos_{k}": v for k, v in s_oos.items()},
                             years_pos=f"{(yr > 0).sum()}/{len(yr)}",
                             **{f"y{y}": round(v, 2) for y, v in yr.items()},
                             open_sp_atr=round(gg.sp_atr.median(), 3) if len(gg) else np.nan))
    return pd.DataFrame(rows)


def corr_t(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    r = np.corrcoef(x, y)[0, 1]
    return round(r, 3), round(r * np.sqrt((len(x) - 2) / (1 - r * r)), 2), len(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    btc, eth = m1_bars("BTCUSD"), m1_bars("XETUSD")
    all_tr, weeks, pred = [], [], []
    for sym in SYMS:
        d = extract(sym)
        w = week_table(sym, d, btc, eth)
        w.insert(0, "sym", sym)
        weeks.append(w)
        for per, ww in (("IS", w[w.date < IS_END]), ("OOS", w[w.date >= IS_END]), ("전체", w)):
            for xn in ("btc", "eth"):
                for yn in ("gap_atr", "oc_atr"):
                    r, tv, n = corr_t(ww[xn].values, ww[yn].values)
                    pred.append(dict(sym=sym, period=per, x=xn, y=yn, corr=r, t=tv, n=n))
            pred.append(dict(sym=sym, period=per, x="gap_atr", y="oc_atr",
                             **dict(zip(["corr", "t", "n"], corr_t(ww.gap_atr.values, ww.oc_atr.values)))))
            big = ww[ww.gap_atr.abs() >= 0.25]
            pred.append(dict(sym=sym, period=per, x="메움 비율(|갭|>=0.25ATR)", y="filled", corr=round(big.filled.mean(), 3),
                             t=np.nan, n=len(big)))
        tr = run_configs(sym, d, w)
        all_tr.append(tr)
        print(f"{sym}: 주 {len(w)}, 거래표 {len(tr)}", flush=True)
    weeks = pd.concat(weeks)
    weeks.to_csv(OUT / "weeks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pred).to_csv(OUT / "predict_corr.csv", index=False, encoding="utf-8-sig")
    tr = pd.concat(all_tr)
    tr.to_pickle(CACHE / "trades.pkl")
    s = summarize(tr)
    # 5종목 합친 같은 규칙 (ATR 단위)
    tp = tr.copy()
    tp["sym"] = "5종목합"
    s = pd.concat([s, summarize(tp)])
    s.to_csv(OUT / "configs.csv", index=False, encoding="utf-8-sig")
    print(f"설정 수: {len(s)} (종목별 {len(s[s.sym != '5종목합'])}, 합침 {len(s[s.sym == '5종목합'])})")

    # IS에서 고르기: (종목, 전략)마다 IS 거래 40회 이상 중 t 최고 -> OOS 확인
    pick = s[s.is_n >= 40].sort_values("is_t", ascending=False).groupby(["sym", "strat"]).head(1)
    pick.to_csv(OUT / "is_best.csv", index=False, encoding="utf-8-sig")
    cols = ["sym", "strat", "wait", "exit", "sl", "cond", "is_n", "is_win", "is_pf", "is_avg_atr", "is_t",
            "oos_n", "oos_win", "oos_pf", "oos_avg_atr", "oos_avg_px", "oos_t", "years_pos"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n[IS 최고 t 설정 -> OOS]")
    print(pick[cols].to_string(index=False))
    ok = s[(s.is_t > 2) & (s.oos_t > 2)]
    ok = ok[ok.years_pos.str[0].astype(int) >= 4]
    ok.to_csv(OUT / "pass.csv", index=False, encoding="utf-8-sig")
    print(f"\n[IS t>2 그리고 OOS t>2 그리고 5년 중 4년+ 플러스] {len(ok)} / {len(s)}")
    print(ok[cols].head(40).to_string(index=False))
    print(f"\n[OOS만 t>2] {(s.oos_t > 2).sum()} / {len(s)},  [OOS t<-2] {(s.oos_t < -2).sum()}")
    print("\n[예측 상관]")
    print(pd.DataFrame(pred).to_string(index=False))


if __name__ == "__main__":
    main()
