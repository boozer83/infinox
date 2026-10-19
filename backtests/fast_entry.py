"""진입 신호를 더 빨리 받는 방법 비교 (실제 틱, 1,000달러 시작, 1,000달러당 0.05 고정 lot, TP 3 / SL 7.5)

  기준   : 5분봉 SAR(0.02) 전환 봉이 끝나면 진입 (지금 봇)
  SAR 0.03 / 0.04 : SAR 가속값을 올려서 점이 가격에 더 붙게 (봉 끝 진입)
  1분봉 SAR : 1분봉 SAR(0.02) 전환 + 1분봉 RSI 50 + H1 SAR (봉 끝 진입, ATR 조건은 마지막 완성 5분봉)
  틱 바로 : 5분봉 SAR 점을 가격(Bid)이 뚫는 첫 틱에 바로 진입 (봉 끝까지 안 기다림)
           RSI는 그 순간 가격을 종가로 놓고 계산, ATR은 직전 완성 5분봉
쉬기 규칙/승률 쉬기/금요일 청산은 지금 봇과 같음 (trail_exit.Acc의 기준 방식 그대로).

사용법: python -m backtests.fast_entry [시작 YYYY-MM-DD] [끝 YYYY-MM-DD]   (구간마다 1,000달러로 새로 시작)
"""
import sys

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.add_positions import ATR_REF_BARS, ATR_REF_MIN
from backtests.add_positions_ticks import load_m1_recent
from backtests.gold_rest_others import h1_flips_by_time, resample
from backtests.trail_exit import Acc, summary
from indicators import atr, psar, rsi

START, END = pd.Timestamp("2025-01-01"), pd.Timestamp("2100-01-01")
RSI_N = 14


def psar_trigger(h, l, step=0.02, mx=0.2):
    """indicators.psar와 같은 계산. (추세 배열, 각 봉에서 뒤집히려면 가격이 넘어야 하는 SAR 값)"""
    n = len(h)
    up, trig = np.zeros(n, bool), np.full(n, np.nan)
    up[0], sar, ep, af = True, l[0], h[0], step
    for i in range(1, n):
        s = sar + af * (ep - sar)
        if up[i-1]:
            s = min(s, l[i-1], l[i-2] if i > 1 else l[i-1])
            trig[i] = s
            if l[i] < s:
                up[i], s, ep, af = False, ep, l[i], step
            else:
                up[i] = True
                if h[i] > ep:
                    ep, af = h[i], min(af + step, mx)
        else:
            s = max(s, h[i-1], h[i-2] if i > 1 else h[i-1])
            trig[i] = s
            if h[i] > s:
                up[i], s, ep, af = True, ep, h[i], step
            else:
                up[i] = False
                if l[i] < ep:
                    ep, af = l[i], min(af + step, mx)
        sar = s
    return up, trig


def context(m5, h1):
    """시각별로 붙일 값: 완성 5분봉 ATR / 20일 중앙값, H1 SAR 방향, 그날 H1 전환 수"""
    a5 = atr(m5).values
    ctx5 = pd.DataFrame({"end": m5.time + pd.Timedelta(minutes=5), "atr": a5,
                         "ref": pd.Series(a5).rolling(ATR_REF_BARS, min_periods=ATR_REF_MIN).median().values})
    h = pd.DataFrame({"end": h1.time + pd.Timedelta(hours=1), "hdir": np.where(psar(h1.high.values, h1.low.values), 1, -1)})
    return ctx5, h, h1_flips_by_time(h1)[["end", "flips", "day"]]


def attach(rows, ctx):
    """rows: (진입 시각, 방향, RSI 조건 통과, ATR 또는 nan(nan이면 마지막 완성 5분봉 ATR))"""
    ctx5, h, fl = ctx
    sig = pd.DataFrame(rows, columns=["t", "side", "rsi_ok", "atr"]).sort_values("t").reset_index(drop=True)
    sig["t"] = sig.t.astype(ctx5.end.dtype)
    m = pd.merge_asof(sig[["t"]], ctx5, left_on="t", right_on="end", direction="backward")
    sig["atr"], sig["ref"] = sig.atr.fillna(pd.Series(m.atr.values)).values, m.ref.values
    sig["hdir"] = pd.merge_asof(sig[["t"]], h, left_on="t", right_on="end", direction="backward").hdir.values
    f = pd.merge_asof(sig[["t"]], fl, left_on="t", right_on="end", direction="backward")
    sig["flips"] = np.where(f.day.values == sig.t.dt.date.values, f.flips.fillna(0).values, 0)
    sig = sig[sig.rsi_ok & (sig.hdir == sig.side)]
    return sig[(sig.t >= START) & (sig.t < END)].reset_index(drop=True)


def bar_close_signals(bars, minutes, step, ctx, use_own_atr):
    """봉 끝 진입 신호. 5분봉이면 ATR은 그 봉 값(지금 봇과 같음), 1분봉이면 마지막 완성 5분봉 값"""
    up, _ = psar_trigger(bars.high.values, bars.low.values, step)
    r = rsi(bars.close).values
    end = bars.time + pd.Timedelta(minutes=minutes)
    i = np.nonzero(up[1:] != up[:-1])[0] + 1
    side = np.where(up[i], 1, -1)
    a = atr(bars).values[i] if use_own_atr else np.full(len(i), np.nan)
    return attach(list(zip(end.iloc[i], side, (r[i] - 50) * side > 0, a)), ctx)


def tick_candidates(m5):
    """5분봉 SAR 전환 봉마다 (봉 시작, 봉 끝, 방향, 넘어야 하는 값, 그 값을 종가로 놓은 RSI, 직전 봉 ATR)"""
    up, trig = psar_trigger(m5.high.values, m5.low.values)
    c = m5.close
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / RSI_N, adjust=False).mean().values
    lo = (-d.clip(upper=0)).ewm(alpha=1 / RSI_N, adjust=False).mean().values
    a5 = atr(m5).values
    i = np.nonzero(up[1:] != up[:-1])[0] + 1
    i = i[i >= 2]
    side = np.where(up[i], 1, -1)
    dd = trig[i] - c.values[i - 1]
    gn = g[i - 1] + (np.maximum(dd, 0) - g[i - 1]) / RSI_N
    ln = lo[i - 1] + (np.maximum(-dd, 0) - lo[i - 1]) / RSI_N
    r = 100 - 100 / (1 + gn / ln)
    st = m5.time.iloc[i].values.astype("datetime64[ms]").astype(np.int64)
    return pd.DataFrame({"s": st, "e": st + 300_000, "side": side, "trig": trig[i], "rsi_ok": (r - 50) * side > 0,
                         "atr": a5[i - 1]})


def tick_signals(cand, ctx):
    """후보 봉마다 Bid가 SAR 값을 처음 넘는 틱 시각을 찾아 신호로 (틱은 한 달씩 읽음)"""
    rows = []
    lo_ms = int(START.value // 1_000_000) - 86_400_000
    cand = cand[(cand.s >= lo_ms) & (cand.s < int(END.value // 1_000_000))]
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        sm, em = int(s.timestamp() * 1000), int(e.timestamp() * 1000)
        c = cand[(cand.s >= sm) & (cand.s < em)]
        if c.empty:
            continue
        t, bid, _ = lb.load_month(y, m)
        for r in c.itertuples(index=False):
            k0, k1 = np.searchsorted(t, r.s), np.searchsorted(t, r.e)
            seg = bid[k0:k1]
            k = np.flatnonzero(seg > r.trig if r.side == 1 else seg < r.trig)
            if k.size:
                rows.append((pd.Timestamp(int(t[k0 + k[0]]), unit="ms"), r.side, r.rsi_ok, r.atr))
        del t, bid
    return attach(rows, ctx)


def run(sig, label):
    acc = Acc(None)
    sig_ms = sig.t.values.astype("datetime64[ms]").astype(np.int64)
    ms = [(y, m) for y, m in lb.months() if (START.year, START.month) <= (y, m) <= (END.year, END.month)]
    for i, (y, m) in enumerate(ms):
        s, e = lb.month_range(y, m)
        lo, hi = np.searchsorted(sig_ms, int(s.timestamp() * 1000)), np.searchsorted(sig_ms, int(e.timestamp() * 1000))
        if lo == hi:
            continue
        t, bid, ask = lb.load_month(y, m)
        if i + 1 < len(ms):
            nt, nb, na = lb.load_month(*ms[i + 1])
            t, bid, ask = np.concatenate([t, nt]), np.concatenate([bid, nb]), np.concatenate([ask, na])
            del nt, nb, na
        for r in sig.iloc[lo:hi].itertuples(index=False):
            acc.on_signal(r, t, bid, ask, None)
        del t, bid, ask
    print(f"  {label} 완료", flush=True)
    return {"방식": label, "신호": len(sig), **summary(acc)}


def main():
    global START, END
    if len(sys.argv) > 1:
        START = pd.Timestamp(sys.argv[1])
    if len(sys.argv) > 2:
        END = pd.Timestamp(sys.argv[2])
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = load_m1_recent()
    m5, h1 = resample(m1, 5), resample(m1, 60)
    ctx = context(m5, h1)
    sigs = {
        "기준 (5분봉 SAR 0.02, 봉 끝)": bar_close_signals(m5, 5, 0.02, ctx, True),
        "5분봉 SAR 0.03": bar_close_signals(m5, 5, 0.03, ctx, True),
        "5분봉 SAR 0.04": bar_close_signals(m5, 5, 0.04, ctx, True),
        "1분봉 SAR 0.02": bar_close_signals(m1, 1, 0.02, ctx, False),
    }
    del m1
    sigs["틱 바로 (SAR 점 뚫는 순간)"] = tick_signals(tick_candidates(m5), ctx)
    res = pd.DataFrame([run(s, n) for n, s in sigs.items()])
    res.to_csv(f"fast_entry_{START.date()}_{END.date() if END.year < 2100 else 'now'}.csv", index=False,
               encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
