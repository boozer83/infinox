"""일목균형표 구름 필터를 SAR 단타에 써 보기 (실제 틱, 지금 봇 기준 = 반대 콩 청산 + 승률 쉬기 끔 + 과열 거르기)

일목 (9, 26, 52): 전환선 = 9봉 (최고+최저)/2, 기준선 = 26봉, 선행1 = (전환+기준)/2, 선행2 = 52봉 (최고+최저)/2,
  구름 = 26봉 앞으로 민 선행1/선행2 (지금 봉 위에 그려진 구름). 미래 구름 = 밀지 않은 선행1/선행2
필터 (5분봉 / 1시간봉 각각, 1시간봉은 완성 봉 기준):
  구름 위/아래 : 매수는 종가가 구름 위, 매도는 구름 아래일 때만
  구름 안 X    : 종가가 구름 안이면 건너뜀
  구름 색      : 매수는 미래 구름이 양운(선행1 > 선행2), 매도는 음운일 때만
  전환 > 기준  : 매수는 전환선 > 기준선, 매도는 반대일 때만
2025로 고르고 2026으로 확인

사용법: python -m backtests.ichimoku_test 2025-01-01 2025-12-31
        python -m backtests.ichimoku_test 2026-01-01
"""
import gc
import sys

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

BASE = [("ext", 0.8), ("risk", 2.8)]


def ichimoku(df):
    mid = lambda n: (df.high.rolling(n).max() + df.low.rolling(n).min()) / 2
    tenkan, kijun, b = mid(9), mid(26), mid(52)
    a = (tenkan + kijun) / 2
    return pd.DataFrame({"close": df.close, "tk": tenkan - kijun, "fa": a, "fb": b,
                         "ca": a.shift(26), "cb": b.shift(26)})


def cloud_feats(ich, side, p):
    """side 방향 기준으로 '건너뛸 정도' (> 0 이면 건너뜀)"""
    top, bot = np.maximum(ich.ca, ich.cb), np.minimum(ich.ca, ich.cb)
    c = ich.close
    above = np.where(side == 1, top - c, c - bot)              # > 0: 구름 위/아래가 아님
    inside = np.where((c <= top) & (c >= bot), 1.0, -1.0)
    color = -(ich.fa - ich.fb) * side
    tk = -ich.tk * side
    return {f"{p}구름": above, f"{p}구름안": inside, f"{p}색": color, f"{p}전환": tk}


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
    ctx = fe.context(m5, h1)
    sig = fe.bar_close_signals(m5, 5, 0.02, ctx, True).merge(features(m5), on="t", how="left")
    i5 = ichimoku(m5)
    i5["t"] = m5.time + pd.Timedelta(minutes=5)
    x5 = sig[["t"]].merge(i5, on="t", how="left")
    ih = ichimoku(h1)
    ih["end"] = h1.time + pd.Timedelta(hours=1)
    ih = ih.drop(columns="close")
    xh = pd.merge_asof(sig[["t"]], ih, left_on="t", right_on="end", direction="backward")
    xh["close"] = x5.close.values          # 1시간봉 구름에 지금 5분봉 종가를 비교
    names = {}
    for p, x in (("5분 ", x5), ("1시간 ", xh)):
        for k, v in cloud_feats(x, sig.side.values, p).items():
            col = f"f{len(names)}"
            sig[col] = np.asarray(v, float)
            names[k] = col
    del m5, h1
    gc.collect()
    variants = {"기준 (지금 봇)": {"feats": BASE}}
    for k, col in names.items():
        variants[k + (" 위/아래만" if k.endswith("구름") else " X" if k.endswith("안") else
                      " 같은 방향만" if k.endswith("색") else " > 기준만")] = {"feats": BASE + [(col, 0)]}
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
    res.to_csv(f"ichimoku_test_{fe.START.year}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
