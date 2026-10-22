"""Smooth Price Oscillator [BigBeluga]를 SAR 단타에 써 보기 (실제 틱, 지금 봇 기준 = 반대 콩 청산 + 승률 쉬기 끔 + 과열 거르기)

지표 (트레이딩뷰 코드 그대로, 5분봉): 종가 SuperSmoother(20) - SuperSmoother(40) = osc,
  정규화 = HMA30(osc / 직전 49봉 stdev50(osc)의 최댓값), 밴드 = EMA250 ± 2 x stdev250, 기준선 ±1
필터 (기준 신호 중 건너뛸 것):
  같은 방향만   : 매수인데 osc < 0 (매도는 반대)
  반대 방향만   : 매수인데 osc > 0 (눌림에서만 진입)
  ±1 과열 X    : 매수인데 osc > 1 (매도는 < -1)
  밴드 밖 X     : 매수인데 osc > 위 밴드 (매도는 < 아래 밴드)
  기울기 같은 방향: 매수인데 osc가 직전 봉보다 내려감
단독 진입 (청산은 같음, 1시간봉 SAR 같은 방향일 때만):
  밴드 복귀 : osc가 아래 밴드를 위로 다시 넘으면 매수 / 위 밴드를 아래로 넘으면 매도
  원본 신호 : osc < -1 에서 위로 꺾이면 매수 / osc > 1 에서 아래로 꺾이면 매도 (signal_up / signal_dn)
  원본 강한 신호: 그중 밴드 밖에서 꺾인 것만 (Strong Mean Reversion)
2025로 고르고 2026으로 확인

사용법: python -m backtests.spo_test 2025-01-01 2025-12-31
        python -m backtests.spo_test 2026-01-01
"""
import gc
import math
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


def smoother(p, period):
    """BigBeluga smoother_F 그대로 (step 계산도 원본과 같게)"""
    step = 2.0 * math.pi / period
    a1 = math.exp(-math.sqrt(2) * math.pi / period)
    b1 = 2 * a1 * math.cos(math.sqrt(2) * step / period)
    c2, c3 = b1, -a1 * a1
    c1 = 1 - c2 - c3
    out = np.array(p, float)
    for i in range(4, len(p)):
        out[i] = c1 * (p[i] + p[i - 1]) / 2 + c2 * out[i - 1] + c3 * out[i - 2]
    return out


def wma(s, n):
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(s, n):
    return wma(2 * wma(s, n // 2) - wma(s, n), int(math.sqrt(n)))


def spo(close, n=20, len_std=50):
    c = close.values
    osc = pd.Series(smoother(c, n) - smoother(c, n * 2))
    sd = osc.rolling(len_std).std(ddof=0)
    mx = sd.shift(1).rolling(len_std - 1).max()       # 배열에서 맨 앞(현재 봉)을 빼고 남은 49개의 최댓값
    norm = hma(osc / mx, 30)
    basis = norm.ewm(span=250, adjust=False).mean()
    dev = 2 * norm.rolling(250).std(ddof=0)
    return norm.values, (basis + dev).values, (basis - dev).values


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
    norm, ub, lb_ = spo(m5.close)
    end = m5.time + pd.Timedelta(minutes=5)
    f = features(m5)
    f["osc"], f["ub"], f["lb"] = norm, ub, lb_
    f["prev"] = np.r_[np.nan, norm[:-1]]
    sig = fe.bar_close_signals(m5, 5, 0.02, ctx, True).merge(f, on="t", how="left")
    sd = sig.side
    sig["same_x"] = -sig.osc * sd                       # > 0 이면 osc가 반대 방향
    sig["opp_x"] = sig.osc * sd
    sig["hot_x"] = sig.osc * sd - 1
    sig["band_x"] = np.where(sd == 1, sig.osc - sig.ub, sig.lb - sig.osc)
    sig["slope_x"] = -(sig.osc - sig.prev) * sd
    # 단독 진입: 밴드 밖에서 안으로 다시 들어올 때
    a5 = fe.atr(m5).values
    up_x = np.flatnonzero((norm[1:] > lb_[1:]) & (norm[:-1] <= lb_[:-1])) + 1
    dn_x = np.flatnonzero((norm[1:] < ub[1:]) & (norm[:-1] >= ub[:-1])) + 1
    rows = [(end.iat[i], 1, True, a5[i]) for i in up_x] + [(end.iat[i], -1, True, a5[i]) for i in dn_x]
    solo = fe.attach(rows, ctx)
    turn_up = np.r_[False, False, (norm[2:] > norm[1:-1]) & (norm[1:-1] <= norm[:-2])]
    turn_dn = np.r_[False, False, (norm[2:] < norm[1:-1]) & (norm[1:-1] >= norm[:-2])]
    su, sd_ = np.flatnonzero(turn_up & (norm < -1)), np.flatnonzero(turn_dn & (norm > 1))
    orig = fe.attach([(end.iat[i], 1, True, a5[i]) for i in su] + [(end.iat[i], -1, True, a5[i]) for i in sd_], ctx)
    su2, sd2 = su[norm[su] < lb_[su]], sd_[norm[sd_] > ub[sd_]]
    strong = fe.attach([(end.iat[i], 1, True, a5[i]) for i in su2] + [(end.iat[i], -1, True, a5[i]) for i in sd2], ctx)
    del m5, h1
    gc.collect()
    variants = {"기준 (지금 봇)": (sig, {"feats": BASE}),
                "같은 방향만": (sig, {"feats": BASE + [("same_x", 0)]}),
                "반대 방향만": (sig, {"feats": BASE + [("opp_x", 0)]}),
                "±1 과열 X": (sig, {"feats": BASE + [("hot_x", 0)]}),
                "밴드 밖 X": (sig, {"feats": BASE + [("band_x", 0)]}),
                "기울기 같은 방향": (sig, {"feats": BASE + [("slope_x", 0)]}),
                "SPO 밴드 복귀 단독 진입": (solo, {}),
                "SPO 원본 신호 단독 진입": (orig, {}),
                "SPO 원본 강한 신호 단독 진입": (strong, {})}
    accs = {n: Acc(o, against, set()) for n, (_, o) in variants.items()}
    sms = {n: s.t.values.astype("datetime64[ms]").astype(np.int64) for n, (s, _) in variants.items()}
    ms = [(y, m) for y, m in lb.months() if (fe.START.year, fe.START.month) <= (y, m) <= (fe.END.year, fe.END.month)]
    for i, (y, m) in enumerate(ms):
        s, e = lb.month_range(y, m)
        t, bid, ask = lb.load_month(y, m)
        if i + 1 < len(ms):
            nt, nb, na = lb.load_month(*ms[i + 1])
            t, bid, ask = np.concatenate([t, nt]), np.concatenate([bid, nb]), np.concatenate([ask, na])
            del nt, nb, na
        for n, (sg, _) in variants.items():
            lo, hi = np.searchsorted(sms[n], int(s.timestamp() * 1000)), np.searchsorted(sms[n], int(e.timestamp() * 1000))
            for r in sg.iloc[lo:hi].itertuples(index=False):
                accs[n].on_signal(r, t, bid, ask, None)
        del t, bid, ask
        gc.collect()
        print(f"  {y}-{m:02d} 완료", flush=True)
    rows = [{"방식": n, **te.summary(a), "건너뜀": a.skipped} for n, a in accs.items()]
    res = pd.DataFrame(rows).drop(columns=["평균 이익$/oz", "3$넘게 번 비율%", "최대 이익$/oz", "이긴 거래 보유(분, 중간값)"])
    print(res.to_string(index=False))
    res.to_csv(f"spo_test_{fe.START.year}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
