"""SAR 단타 진입 조건 그대로, 0.01 lot 고정, TP = SL = N달러, 포지션이 있어도 신호마다 계속 진입 (실제 틱)

진입: 5분봉 SAR 전환 + RSI 50 + H1 SAR 같은 방향, 쉬기 규칙(15~21시, 금요일 22:30 이후, H1 전환 3번, ATR 2.5 미만 / 20일 중앙값 2배 초과)
      승률 쉬기는 뺌 (TP 3 / SL 7.5 결과로 정하는 규칙이라)
청산: 진입가 ±N달러 먼저 닿는 쪽 (N = WIDTHS). 금요일 23:30 청산 있음 / 없음
동시 보유 제한: 1개 / 3개 / 5개 / 제한 없음 (열린 포지션이 제한만큼이면 신호 무시)

사용법: python -m backtests.sar_50 [시작 YYYY-MM-DD] [끝 YYYY-MM-DD]
"""
import heapq
import sys

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.long_period as lb
from backtests.add_positions_ticks import load_m1_recent
from backtests.gold_rest_others import (FRIDAY_NO_ENTRY, MAX_H1_FLIPS, MIN_ATR, NO_ENTRY_FROM, NO_ENTRY_UNTIL,
                                        resample)
from backtests.m5_ticks import first_hit

LOT, BAL0 = 0.01, 1000.0
WIDTHS = (20, 30, 50, 80)
CAPS = (1, 3, 5, None)
MAX_ATR_MULT = 2.0


def filtered(sig):
    mins = sig.t.dt.hour * 60 + sig.t.dt.minute
    bad = ((sig.t.dt.weekday == 4) & (mins >= FRIDAY_NO_ENTRY)) | ((mins >= NO_ENTRY_FROM) & (mins < NO_ENTRY_UNTIL)) \
        | (sig.flips >= MAX_H1_FLIPS) | (sig.atr < MIN_ATR) | ((sig.ref == sig.ref) & (sig.atr > MAX_ATR_MULT * sig.ref))
    return sig[~bad].reset_index(drop=True)


def trades(sig):
    """{(폭, 금요일 청산): DataFrame(open, close, side, ep, xp, why)} — 신호마다 모든 경우의 청산을 구함"""
    out = {(w, f): [] for w in WIDTHS for f in (True, False)}
    sig_ms = sig.t.values.astype("datetime64[ms]").astype(np.int64)
    ms = [(y, m) for y, m in lb.months() if (fe.START.year, fe.START.month) <= (y, m) <= (fe.END.year, fe.END.month)]
    for i, (y, m) in enumerate(ms):
        s, e = lb.month_range(y, m)
        lo, hi = np.searchsorted(sig_ms, int(s.timestamp() * 1000)), np.searchsorted(sig_ms, int(e.timestamp() * 1000))
        if lo == hi:
            continue
        parts = [lb.load_month(*x) for x in ms[i:i + 3]]     # 큰 폭은 며칠 걸릴 수 있어서 두 달 더 붙임
        t = np.concatenate([p[0] for p in parts])
        bid = np.concatenate([p[1] for p in parts])
        ask = np.concatenate([p[2] for p in parts])
        del parts
        for r in sig.iloc[lo:hi].itertuples(index=False):
            k = int(np.searchsorted(t, int(np.datetime64(r.t, "ms").astype(np.int64))))
            if k >= len(t) - 1:
                continue
            X = bid if r.side == 1 else ask
            ep = ask[k] if r.side == 1 else bid[k]
            d = r.t.normalize() + pd.Timedelta(days=(4 - r.t.weekday()) % 7, hours=23, minutes=30)
            fc = int(np.searchsorted(t, int(np.datetime64(d, "ms").astype(np.int64))))
            for w in WIDTHS:
                j = first_hit(X, k + 1, ep - w, ep + w)
                for fri in (True, False):
                    jj = j
                    why = "TP" if j >= 0 and (X[j] - ep) * r.side > 0 else "SL"
                    if fri and (j < 0 or j >= fc):
                        jj, why = fc, "금요일"
                    if jj < 0 or jj >= len(t):
                        out[w, fri].append((t[k], -1, r.side, ep, np.nan, "미청산"))
                    else:
                        out[w, fri].append((t[k], t[jj], r.side, ep, X[jj], why))
        del t, bid, ask
        print(f"  {y}-{m:02d} 완료", flush=True)
    return {k: pd.DataFrame(v, columns=["open", "close", "side", "ep", "xp", "why"]) for k, v in out.items()}


def limit(tr, cap):
    """열린 포지션이 cap개면 신호 무시 (None = 제한 없음)"""
    if cap is None:
        return tr
    keep, open_ = [], []
    for r in tr.itertuples():
        while open_ and open_[0] <= r.open:
            heapq.heappop(open_)
        if len(open_) < cap:
            keep.append(r.Index)
            heapq.heappush(open_, r.close if r.close >= 0 else np.iinfo(np.int64).max)
    return tr.loc[keep]


def summarize(tr, label, cap):
    tr = limit(tr.sort_values("open").reset_index(drop=True), cap)
    done = tr[tr.close >= 0].copy()
    done["pnl"] = (done.xp - done.ep) * done.side * LOT * 100
    ev = pd.concat([pd.Series(1, index=tr.open), pd.Series(-1, index=tr.close.where(tr.close >= 0, np.iinfo(np.int64).max))])
    conc = ev.groupby(level=0).sum().sort_index().cumsum()
    eq = done.sort_values("close").pnl.cumsum()
    dd = ((BAL0 + eq).cummax() - BAL0 - eq).max()
    gl = -done.pnl[done.pnl < 0].sum()
    month = pd.to_datetime(done.close, unit="ms").dt.to_period("M")
    mo = done.groupby(month).pnl.sum()
    res = {"방식": label, "거래": len(done), "익절": int((done.why == "TP").sum()), "손절": int((done.why == "SL").sum()),
           "금요일": int((done.why == "금요일").sum()), "승률%": round((done.pnl > 0).mean() * 100, 1),
           "손익$": round(done.pnl.sum()), "PF": round(done.pnl[done.pnl > 0].sum() / gl, 2) if gl else float("inf"),
           "최대동시": int(conc.max()), "최대낙폭$": round(dd), "손익/낙폭": round(done.pnl.sum() / dd, 2) if dd else float("inf"),
           "손실 달": f"{(mo < 0).sum()}/{len(mo)}"}
    return res, mo.round()


def main():
    fe.START = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-01-01")
    if len(sys.argv) > 2:
        fe.END = pd.Timestamp(sys.argv[2])
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = load_m1_recent()
    m5, h1 = resample(m1, 5), resample(m1, 60)
    del m1
    ctx = fe.context(m5, h1)
    sig = filtered(fe.bar_close_signals(m5, 5, 0.02, ctx, True))
    print(f"신호 {len(sig):,}개 ({sig.t.min()} ~ {sig.t.max()})", flush=True)
    tr = trades(sig)
    rows, months = [], {}
    for fri in (True, False):
        for w in WIDTHS:
            for cap in CAPS:
                label = f"±{w}$ 동시 {cap or '무제한'}{'' if fri else ' 금요일청산X'}"
                r, mo = summarize(tr[w, fri], label, cap)
                rows.append(r)
                months[label] = mo
    res = pd.DataFrame(rows)
    res.to_csv("sar_50_result.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    pd.DataFrame(months).fillna(0).to_csv("sar_50_monthly.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
