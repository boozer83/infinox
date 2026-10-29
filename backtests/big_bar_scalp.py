"""초단타: 큰 봉이 그려지는 중이면 그 방향으로, 익절 아주 작게 / 손절 0 (2026-10-29 사용자 아이디어, 실제 틱)

진입: 5분봉이 그려지는 중에 (Bid - 봉 시가)가 직전 5분봉 ATR(14) x K 이상이면 그 방향으로 (봉마다 최대 1번)
      DOM이 켜져 있으면 직전 60초 틱 방향(오른 틱 비율 >= 0.55 매수 / <= 0.45 매도)도 같은 방향일 때만
      신호 틱 다음 틱에 체결 (주문 지연 흉내). 매수 = Ask, 매도 = Bid
청산: 익절 = 진입가 ± TP$
      손절 SL=0: 가격이 진입할 때 자리보다 나빠지면 (매수면 Bid가 진입 순간 Bid 미만) -> 손실은 스프레드만큼
      손절 SL>0: 진입가 ∓ SL$ (비교용)
      최대 MAX_MIN분 지나면 그 자리 청산
한 번에 1개. 결과 $/oz (0.01 lot = 1oz). results/big_bar_scalp.csv

연속 틱 (사용자 아이디어 2): Bid가 N틱 연속 같은 방향으로 바뀌면 (안 바뀐 틱은 건너뜀) 그 방향으로 진입.
      큰봉 조건(K)과 같이 쓰거나 (K=0이면 큰봉 조건 없음), 신호 틱 다음 틱 체결, 청산 규칙은 위와 같음
      python -m backtests.big_bar_scalp runs [nospread]  -> results/big_bar_runs[_nospread].csv

사용법: python -m backtests.big_bar_scalp           # 실제 Bid/Ask
        python -m backtests.big_bar_scalp nospread  # 스프레드 0 가정 (Ask = Bid) -> results/big_bar_scalp_nospread.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.m5_ticks import first_hit

RES = Path(__file__).resolve().parent.parent / "results"
NOSPREAD = "nospread" in sys.argv[1:]
KS = (1.0, 1.5, 2.0)
TPS = (0.3, 0.5, 1.0, 2.0)
SLS = (0.0, 0.5, 1.0) if "sl0" not in sys.argv[1:] else (0.0,)
MAX_MIN = 30
BAR = 300_000


def triggers(t, bid, k):
    """봉마다 첫 신호 틱 (틱 idx, 방향, 60초 우세)"""
    bar = t // BAR
    start = np.flatnonzero(np.r_[True, bar[1:] != bar[:-1]])
    op = bid[start]
    hi, lo, cl = np.maximum.reduceat(bid, start), np.minimum.reduceat(bid, start), bid[np.r_[start[1:], len(bid)] - 1]
    pc = np.r_[np.nan, cl[:-1]]
    tr = np.nanmax(np.c_[hi - lo, np.abs(hi - pc), np.abs(lo - pc)], axis=1)
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().shift(1).values       # 직전 봉까지
    gid = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(bid)]))
    dev = bid - op[gid]
    hit = np.abs(dev) >= k * atr[gid]
    idx = np.flatnonzero(hit)
    first = idx[np.r_[True, gid[idx][1:] != gid[idx][:-1]]] if idx.size else idx
    d = np.sign(np.diff(bid))
    up, dn = np.r_[0, np.cumsum(d > 0)], np.r_[0, np.cumsum(d < 0)]
    k0 = np.searchsorted(t, t[first] - 60_000)
    u, n = up[first] - up[k0], (up[first] - up[k0]) + (dn[first] - dn[k0])
    r = np.where(n > 0, u / np.maximum(n, 1), 0.5)
    dom = np.where(r >= 0.55, 1, np.where(r <= 0.45, -1, 0))
    return first, np.sign(dev[first]).astype(int), dom


def run_triggers(t, bid, n, k):
    """Bid가 n틱 연속 같은 방향으로 바뀐 틱 (idx, 방향). k > 0이면 그 틱에 큰봉 조건(같은 방향)도"""
    ch = np.flatnonzero(np.diff(bid) != 0) + 1                 # 값이 바뀐 틱
    d = np.sign(bid[ch] - bid[ch - 1]).astype(int)
    brk = np.r_[True, d[1:] != d[:-1]]
    grp = np.cumsum(brk)
    pos = np.arange(len(d)) - np.flatnonzero(brk)[grp - 1] + 1  # 지금까지 연속 개수
    sel = pos == n
    idx, side = ch[sel], d[sel]
    if k > 0:
        bar = t // BAR
        start = np.flatnonzero(np.r_[True, bar[1:] != bar[:-1]])
        op = bid[start]
        hi, lo = np.maximum.reduceat(bid, start), np.minimum.reduceat(bid, start)
        cl = bid[np.r_[start[1:], len(bid)] - 1]
        pc = np.r_[np.nan, cl[:-1]]
        tr = np.nanmax(np.c_[hi - lo, np.abs(hi - pc), np.abs(lo - pc)], axis=1)
        atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().shift(1).values
        gid = np.searchsorted(start, idx, side="right") - 1
        dev = bid[idx] - op[gid]
        ok = (dev * side >= k * atr[gid])
        idx, side = idx[ok], side[ok]
    return idx, side, np.zeros(len(idx), int)


def runs_main():
    """거래를 모아두지 않고 (조건, 연도)별 합계만 쌓음 (거래 수가 많아 메모리 아끼기)"""
    acc = {}
    for y, m in lb.months():
        try:
            t, bid, ask = lb.load_month(y, m)
            if NOSPREAD:
                ask = bid
        except FileNotFoundError:
            continue
        for n in (3, 5, 8, 12):
            for k in (0.0, 1.0):
                idx, side, dom = run_triggers(t, bid, n, k)
                sigs = list(zip(idx, side, dom))
                for tp in (0.3, 0.5, 1.0):
                    for sl in (0.0, 0.5):
                        r = simulate(t, bid, ask, sigs, tp, sl)
                        oz = np.array([x[2] for x in r])
                        secs = np.array([x[3] for x in r])
                        c = acc.setdefault((n, k, tp, sl, y), [0, 0, 0.0, 0.0])
                        c[0] += len(oz)
                        c[1] += int((oz > 0).sum())
                        c[2] += float(oz.sum())
                        c[3] += float(secs.sum())
        del t, bid, ask
        print(f"  {y}-{m:02d}", flush=True)
    df = pd.DataFrame([k + tuple(v) for k, v in acc.items()], columns=["N", "K", "TP", "SL", "y", "n", "win", "oz", "secs"])
    keys = ["N", "K", "TP", "SL"]
    g = df.groupby(keys)[["n", "win", "oz", "secs"]].sum()
    res = pd.DataFrame({"건수": g.n, "승률": (g.win / g.n * 100).round(1), "oz합": g.oz.round(1),
                        "oz건": (g.oz / g.n).round(4), "평균초": (g.secs / g.n).round(1)})
    for yr in sorted(df.y.unique()):
        res[f"oz{yr % 100}"] = df[df.y == yr].set_index(keys).oz.round(1)
    res = res.reset_index()
    res.to_csv(RES / f"big_bar_runs{'_nospread' if NOSPREAD else ''}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 500)
    print(res.to_string(index=False))


def simulate(t, bid, ask, sigs, tp, sl):
    out, busy = [], -1
    for i, side, _ in sigs:
        k = i + 1
        if k >= len(t) or t[k] <= busy:
            continue
        lim = int(np.searchsorted(t, t[k] + MAX_MIN * 60_000))
        if side == 1:
            e = ask[k]
            lo = bid[k] - 1e-6 if sl == 0 else e - sl            # 손절 0: 진입 때 Bid보다 내려가야 (같은 값은 안 나감)
            j = first_hit(bid[:lim], k + 1, lo, e + tp)
            j = lim - 1 if j < 0 else j
            oz = bid[j] - e
        else:
            e = bid[k]
            hi = ask[k] + 1e-6 if sl == 0 else e + sl
            j = first_hit(ask[:lim], k + 1, e - tp, hi)
            j = lim - 1 if j < 0 else j
            oz = e - ask[j]
        busy = t[j]
        out.append((int(t[k]), side, oz, (t[j] - t[k]) / 1000, ask[k] - bid[k]))
    return out


def main():
    rows = []
    for y, m in lb.months():
        try:
            t, bid, ask = lb.load_month(y, m)
            if NOSPREAD:
                ask = bid
        except FileNotFoundError:
            continue
        for k in KS:
            first, side, dom = triggers(t, bid, k)
            for use_dom in (False, True):
                keep = (dom == side) if use_dom else np.ones(len(first), bool)
                sigs = list(zip(first[keep], side[keep], dom[keep]))
                for tp in TPS:
                    for sl in SLS:
                        for r in simulate(t, bid, ask, sigs, tp, sl):
                            rows.append((k, use_dom, tp, sl) + r)
        print(f"  {y}-{m:02d}", flush=True)
    df = pd.DataFrame(rows, columns=["K", "dom", "TP", "SL", "t", "side", "oz", "secs", "spread"])
    df["y"] = pd.to_datetime(df.t, unit="ms").dt.year
    g = df.groupby(["K", "dom", "TP", "SL"])
    res = g.agg(건수=("oz", "size"), 승률=("oz", lambda x: round((x > 0).mean() * 100, 1)), oz합=("oz", "sum"),
                oz건=("oz", "mean"), 초=("secs", "median"), 스프레드=("spread", "mean")).round(3)
    for yr in sorted(df.y.unique()):
        res[f"oz{yr % 100}"] = df[df.y == yr].groupby(["K", "dom", "TP", "SL"]).oz.sum().round(1)
    res = res.reset_index()
    res.to_csv(RES / (f"big_bar_scalp{'_nospread' if NOSPREAD else ''}{'_sl0' if 'sl0' in sys.argv[1:] else ''}.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 500)
    print(res.to_string(index=False))


if __name__ == "__main__":
    runs_main() if "runs" in sys.argv[1:] else main()
