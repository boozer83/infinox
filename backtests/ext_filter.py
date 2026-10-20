"""과열 거르기 다듬기: 신호 봉이 얼마나 '이미 간' 자리인지 재는 방법 5가지 x 거르는 비율 5단계 (실제 틱, 지금 봇 기준)

  ext   : (종가 - 방금 뚫은 SAR 값) / ATR      (지금까지 쓴 것)
  risk  : (종가 - 새 SAR 점) / ATR             (전환 뒤 SAR = 직전 추세 극값. 반대 콩까지 거리)
  body  : 신호 봉 몸통 |종가 - 시가| / ATR
  range : 신호 봉 고저폭 / ATR
  ema   : (종가 - EMA20) / ATR (방향 기준)
기준값은 2025년 신호 분포의 60/70/80/90/95% 지점으로 정해서 data/ext_thresholds.json에 저장 -> 2026년은 그 값 그대로
(= 2025년 기준으로 위 40/30/20/10/5%를 거름). 2025로 고르고 2026으로 확인

사용법: python -m backtests.ext_filter 2025-01-01 2025-12-31   (먼저)
        python -m backtests.ext_filter 2026-01-01
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.long_period as lb
import backtests.trail_exit as te
from backtests.add_positions_ticks import load_m1_recent
from backtests.gold_rest_others import resample
from backtests.sar_exit import against_times
from backtests.winrate_ideas import Acc
from indicators import atr

TH_FILE = Path("data/ext_thresholds.json")
FEATS = ("ext", "risk", "body", "range", "ema")
QS = (0.6, 0.7, 0.8, 0.9, 0.95)


def sar_values(h, l, step=0.02, mx=0.2):
    """indicators.psar와 같은 계산. (추세, 각 봉의 SAR 값(봉 끝 기준), 그 봉에서 뚫어야 했던 값)"""
    n = len(h)
    up, sv, trig = np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan)
    up[0], sar, ep, af = True, l[0], h[0], step
    for i in range(1, n):
        s = sar + af * (ep - sar)
        if up[i - 1]:
            s = min(s, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            trig[i] = s
            if l[i] < s:
                up[i], s, ep, af = False, ep, l[i], step
            else:
                up[i] = True
                if h[i] > ep:
                    ep, af = h[i], min(af + step, mx)
        else:
            s = max(s, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            trig[i] = s
            if h[i] > s:
                up[i], s, ep, af = True, ep, h[i], step
            else:
                up[i] = False
                if l[i] < ep:
                    ep, af = l[i], min(af + step, mx)
        sar = s
        sv[i] = s
    return up, sv, trig


def features(m5):
    up, sv, trig = sar_values(m5.high.values, m5.low.values)
    a = atr(m5).values
    side = np.where(up, 1, -1)
    c, o = m5.close.values, m5.open.values
    ema = m5.close.ewm(span=20, adjust=False).mean().values
    return pd.DataFrame({"t": m5.time + pd.Timedelta(minutes=5),
                         "ext": (c - trig) * side / a, "risk": (c - sv) * side / a,
                         "body": np.abs(c - o) / a, "range": (m5.high.values - m5.low.values) / a,
                         "ema": (c - ema) * side / a})


def main():
    fe.START = pd.Timestamp(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] != "--combo":
        fe.END = pd.Timestamp(sys.argv[2])
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = load_m1_recent()
    m5, h1 = resample(m1, 5), resample(m1, 60)
    del m1
    gc.collect()
    against = against_times(m5, 5)
    sig = fe.bar_close_signals(m5, 5, 0.02, fe.context(m5, h1), True)
    sig = sig.merge(features(m5), on="t", how="left")
    del m5, h1
    gc.collect()
    if fe.START.year == 2025 and "--combo" not in sys.argv:
        th = {f: {str(q): float(sig[f].quantile(q)) for q in QS} for f in FEATS}
        TH_FILE.write_text(json.dumps(th, indent=1))
    th = json.loads(TH_FILE.read_text())
    variants = {"기준 (지금 봇)": {}}
    for f in FEATS:
        for q in QS:
            variants[f"{f} 위 {round((1 - q) * 100)}% 거름 (>{th[f][str(q)]:.2f})"] = {"feat": f, "max": th[f][str(q)]}
    if "--combo" in sys.argv:     # ext + risk 같이
        variants = {"기준 (지금 봇)": {}, "ext 0.8": {"feats": [("ext", 0.8)]}}
        for e_ in (0.5, 0.8):
            for r_ in (2.6, 2.8, 3.0):
                variants[f"ext {e_} + risk {r_}"] = {"feats": [("ext", e_), ("risk", r_)]}
    accs = {n: Acc(o, against, set()) for n, o in variants.items()}
    sig_ms = sig.t.values.astype("datetime64[ms]").astype(np.int64)
    ms = [(y, m) for y, m in lb.months() if (fe.START.year, fe.START.month) <= (y, m) <= (fe.END.year, fe.END.month)]
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
            for a in accs.values():
                a.on_signal(r, t, bid, ask, None)
        del t, bid, ask
        gc.collect()
        print(f"  {y}-{m:02d} 완료", flush=True)
    rows = [{"방식": n, **te.summary(a), "건너뜀": a.skipped} for n, a in accs.items()]
    res = pd.DataFrame(rows).drop(columns=["평균 이익$/oz", "3$넘게 번 비율%", "최대 이익$/oz", "이긴 거래 보유(분, 중간값)"])
    print(res.to_string(index=False))
    res.to_csv(f"ext_filter_{fe.START.year}{'_combo' if '--combo' in sys.argv else ''}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
