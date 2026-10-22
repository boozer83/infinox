"""옆으로 길 때(횡보) SAR 단타 쉬기 (실제 틱, 지금 봇 기준 = 반대 콩 청산 + 승률 쉬기 끔 + 과열 거르기)

횡보 판단 (신호 시점에 완성된 봉 기준, 값이 클수록 '횡보'가 되게 부호를 맞춤):
  폭 N시간   : 최근 N시간 5분봉 (최고 - 최저) / 1시간봉 ATR(14). 작으면 횡보
  효율 N시간 : |N시간 동안 움직인 거리| / 5분봉 움직임 합. 작으면 지그재그
  ADX 5분/1시간 : 낮으면 추세 없음
  1시간 SAR 전환 수 : 최근 12시간 동안 1시간봉 SAR이 뒤집힌 횟수. 많으면 횡보
각각 2025 신호 기준 가장 횡보인 10/20/30%를 거름 (기준값은 2025에서 정해 data/range_thresholds.json에 저장, 2026에 그대로 씀)

사용법: python -m backtests.range_filter 2025-01-01 2025-12-31   (기준값 정함)
        python -m backtests.range_filter 2026-01-01
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
from backtests.ext_filter import features
from backtests.gold_rest_others import resample
from backtests.sar_exit import against_times
from backtests.winrate_ideas import Acc
from indicators import adx, atr, psar

BASE = [("ext", 0.8), ("risk", 2.8)]
QS = (0.9, 0.8, 0.7)
TH_FILE = Path(__file__).resolve().parent.parent / "data" / "range_thresholds.json"
NAMES = {"w1": "폭 1시간", "w2": "폭 2시간", "w4": "폭 4시간", "e2": "효율 2시간", "e4": "효율 4시간",
         "adx5": "ADX 5분", "adx60": "ADX 1시간", "flip": "1시간 SAR 전환 수"}


def range_feats(m5, h1):
    end5 = m5.time + pd.Timedelta(minutes=5)
    h1a = pd.DataFrame({"end": h1.time + pd.Timedelta(hours=1), "a": atr(h1).values, "adx60": adx(h1).values,
                        "flip": pd.Series(np.diff(psar(h1.high.values, h1.low.values).astype(int), prepend=0) != 0)
                        .astype(int).rolling(12).sum().values})
    x = pd.merge_asof(pd.DataFrame({"t": end5}), h1a, left_on="t", right_on="end", direction="backward")
    out = pd.DataFrame({"t": end5})
    step = m5.close.diff().abs()
    for h in (1, 2, 4):
        n = 12 * h
        out[f"w{h}"] = -((m5.high.rolling(n).max() - m5.low.rolling(n).min()).values / x.a.values)
        if h > 1:
            out[f"e{h}"] = -((m5.close - m5.close.shift(n)).abs() / step.rolling(n).sum()).values
    out["adx5"] = -adx(m5).values
    out["adx60"] = -x.adx60.values
    out["flip"] = x.flip.values
    return out


def main():
    fe.START = pd.Timestamp(sys.argv[1])
    if len(sys.argv) > 2:
        fe.END = pd.Timestamp(sys.argv[2])
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = load_m1_recent()
    m5, h1 = resample(m1, 5), resample(m1, 60)
    del m1
    gc.collect()
    against = against_times(m5, 5)
    sig = fe.bar_close_signals(m5, 5, 0.02, fe.context(m5, h1), True)
    sig = sig.merge(features(m5), on="t", how="left").merge(range_feats(m5, h1), on="t", how="left")
    del m5, h1
    gc.collect()
    if fe.START.year == 2025:
        TH_FILE.write_text(json.dumps({f: {str(q): float(sig[f].quantile(q)) for q in QS} for f in NAMES}, indent=1))
    th = json.loads(TH_FILE.read_text())
    variants = {"기준 (지금 봇)": {"feats": BASE}}
    for f, nm in NAMES.items():
        for q in QS:
            variants[f"{nm} 횡보 {round((1 - q) * 100)}% 쉼 ({abs(th[f][str(q)]):.2f})"] = {"feats": BASE + [(f, th[f][str(q)])]}
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
    res.to_csv(f"range_filter_{fe.START.year}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
