"""기존 전략과 구조가 다른 금 단타 후보 (2026-10-31, 사용자가 다른 AI에게 받은 목록, 실제 틱 체결)

검증 방식은 idea_screen과 같음: 설정 8개 중 2022~23 PF 1등 (학습 30건 이상) -> 2024/25/26 모두 PF 1.1 이상이면 통과.
거래당 0.2$ 비용을 더 뺀 2024~26 PF도 표시. 5분봉 = 틱 Bid 봉, 진입은 신호 봉 다음 봉 첫 틱. 시간은 서버시간 (뉴욕 + 7)
ATR은 직전 완성 15분봉 ATR(14). 미국 중요 지표 앞뒤 15분은 진입 안 함 (8~11 모두)

 8. 돌파 실패 역방향 : 런던(서버 10:00) / 뉴욕(16:30) 개장 첫 15분 범위. 그 뒤 2시간 안에 5분봉 종가가 범위를 벗어났다가
                      N봉(1 / 3) 안에 종가가 범위 안으로 돌아오면 반대 방향. 손절 = 벗어났을 때 극값 + 0.1 ATR,
                      익절 = 범위 반대편 또는 1.5R, 개장 4시간 뒤 시간 청산, 세션마다 1번
 9. 유동성 스윕 실패 : 기준 = 전일 고가·저가 또는 직전 50봉 고가·저가. 5분봉 고가가 기준을 넘었는데 종가는 기준 아래면 매도
                      (저가 쪽은 반대). 확인 = 없음 / 다음 봉 종가가 스윕 봉 저가(고가) 넘음. 손절 = 스윕 극값 + 0.3 ATR,
                      익절 1.5R / 2R, 2시간 시간 청산. 서버 10:00~23:00만
10. 정수 가격대 거부 : 25$ / 50$ 간격 가격. 5분봉이 그 가격을 위로 찔렀다가 아래에서 마감하면 매도 (아래는 반대).
                      윗꼬리 >= 봉 길이 50% 조건 있음 / 없음. 손절 = 가격대 + 0.5 ATR, 익절 1R / 1.5R, 1시간 시간 청산. 10:00~23:00
11. 실현 반분산 반전 : 최근 W봉(6 / 24) 5분 수익률을 상승 / 하락 제곱합으로 나눠 하락 몫 >= q(0.7 / 0.8)인데
                      W봉 수익률이 -0.5 x sqrt(전체 제곱합)보다 덜 빠졌으면 매수 (상승 쪽은 반대). 조건이 처음 켜진 봉만.
                      손절 1.5 ATR, 익절 1.5R, 30 / 60분 시간 청산

사용법: python -m backtests.idea_screen3 [8 9 10 11]
"""
import itertools
import sys

import numpy as np
import pandas as pd

from backtests.idea_screen import LightStore, TRAIN_END
from backtests.idea_screen2 import usd_high_times
from backtests.scalp_ideas import m5_bars
from backtests.strategies import PASS_PF, YEARS, evaluate, execute, pf, resample, to_ms
from indicators import atr

MIN = 60_000


class Ctx:
    def __init__(self):
        self.m5 = m5_bars()
        m15 = resample(self.m5, "15min")
        a15, end15 = atr(m15).values, to_ms(m15.time) + 15 * MIN
        self.t = to_ms(self.m5.time)
        j = np.searchsorted(end15, self.t + 5 * MIN, side="right") - 1      # 5분봉 마감 때 끝난 15분봉
        self.a = np.where(j >= 0, a15[np.maximum(j, 0)], np.nan)
        self.news = to_ms(pd.Series(np.sort(usd_high_times().unique())))
        hm = self.m5.time.dt.hour * 60 + self.m5.time.dt.minute
        self.hm = hm.values
        self.day = self.m5.time.dt.normalize()

    def near_news(self, ms):
        k = np.searchsorted(self.news, ms)
        return (k < len(self.news) and self.news[k] - ms <= 15 * MIN) or (k > 0 and ms - self.news[k - 1] <= 15 * MIN)

    def spec(self, i, side, sl_abs, hold_min, tp_r=None, tp_abs=None):
        """i = 신호 봉. tp_abs면 R로 바꿔서 넣음 (진입가는 모르니 신호 봉 종가로 근사)"""
        if i + 1 >= len(self.t) or self.near_news(self.t[i]):
            return None
        if tp_abs is not None:
            c = self.m5.close.values[i]
            risk = (c - sl_abs) * side
            tp_r = (tp_abs - c) * side / risk if risk > 0 else None
            if not tp_r or tp_r <= 0:
                return None
        return {"entry": self.t[i + 1], "side": side, "exit": self.t[i + 1] + hold_min * MIN, "sl_abs": sl_abs, "tp_r": tp_r}


# ---------------------------------------------------------------- 8. 돌파 실패 역방향
def orb_fade(x, open_hm, confirm, tp_mode):
    m5, specs = x.m5, []
    h, lo, c = m5.high.values, m5.low.values, m5.close.values
    for d, idx in pd.Series(np.arange(len(m5))).groupby(x.day.values):
        idx = idx.values
        hm = x.hm[idx]
        r = idx[(hm >= open_hm) & (hm < open_hm + 15)]
        w = idx[(hm >= open_hm + 15) & (hm < open_hm + 135)]
        if len(r) < 3 or len(w) < 2:
            continue
        hi, lw = h[r].max(), lo[r].min()
        out = np.flatnonzero((c[w] > hi) | (c[w] < lw))
        if not out.size:
            continue
        b = w[out[0]]
        side = -1 if c[b] > hi else 1                      # 위로 벗어났다 돌아오면 매도
        for k in range(1, confirm + 1):
            i = b + k
            if i >= len(c) or x.hm[i] >= open_hm + 135 or x.day.values[i] != d:
                break
            if lw <= c[i] <= hi:
                ext = h[b:i + 1].max() if side == -1 else lo[b:i + 1].min()
                sl = ext - side * 0.1 * x.a[i]
                hold = open_hm + 240 - x.hm[i]
                s = x.spec(i, side, sl, hold, tp_r=1.5 if tp_mode == "1.5R" else None,
                           tp_abs=(lw if side == -1 else hi) if tp_mode == "범위 반대편" else None)
                if s:
                    specs.append(s)
                break
    return specs


# ---------------------------------------------------------------- 9. 유동성 스윕 실패
def sweep(x, level, confirm, tp_r):
    m5 = x.m5
    h, lo, c = m5.high.values, m5.low.values, m5.close.values
    if level == "전일":
        d = m5.groupby(x.day).agg(hi=("high", "max"), lo=("low", "min"))
        prev = d.shift(1)
        up = prev.hi.reindex(x.day).values
        dn = prev.lo.reindex(x.day).values
    else:
        up = m5.high.rolling(50).max().shift(1).values
        dn = m5.low.rolling(50).min().shift(1).values
    live = (x.hm >= 600) & (x.hm < 1380)
    sh = np.flatnonzero(live & (h > up) & (c < up))
    lg = np.flatnonzero(live & (lo < dn) & (c > dn))
    specs = []
    for i, side in sorted([(i, -1) for i in sh] + [(i, 1) for i in lg]):
        j = i
        if confirm:
            j = i + 1
            if j >= len(c) or not ((c[j] < lo[i]) if side == -1 else (c[j] > h[i])):
                continue
        ext = h[i:j + 1].max() if side == -1 else lo[i:j + 1].min()
        s = x.spec(j, side, ext - side * 0.3 * x.a[j], 120, tp_r=tp_r)
        if s:
            specs.append(s)
    return specs


# ---------------------------------------------------------------- 10. 정수 가격대 거부
def round_reject(x, step, wick, tp_r):
    m5 = x.m5
    o, h, lo, c = (m5[k].values for k in ("open", "high", "low", "close"))
    lv_up = np.floor(h / step) * step                    # 봉 고가 아래 가장 가까운 정수 가격 (위로 찌른 가격)
    lv_dn = np.ceil(lo / step) * step
    rng = np.maximum(h - lo, 1e-9)
    live = (x.hm >= 600) & (x.hm < 1380)
    short = live & (lv_up > np.maximum(o, c)) & (c < lv_up)
    long_ = live & (lv_dn < np.minimum(o, c)) & (c > lv_dn)
    if wick:
        short &= (h - np.maximum(o, c)) / rng >= 0.5
        long_ &= (np.minimum(o, c) - lo) / rng >= 0.5
    specs = []
    for i, side in sorted([(i, -1) for i in np.flatnonzero(short)] + [(i, 1) for i in np.flatnonzero(long_)]):
        lv = lv_up[i] if side == -1 else lv_dn[i]
        s = x.spec(i, side, lv - side * 0.5 * x.a[i], 60, tp_r=tp_r)
        if s:
            specs.append(s)
    return specs


# ---------------------------------------------------------------- 11. 실현 반분산 반전
def semivar(x, w, q, hold):
    c = pd.Series(x.m5.close.values)
    r = c.diff()
    up = (r.clip(lower=0) ** 2).rolling(w).sum()
    dn = (r.clip(upper=0) ** 2).rolling(w).sum()
    tot = up + dn
    ret = c - c.shift(w)
    share = dn / tot
    lg = ((share >= q) & (ret > -0.5 * np.sqrt(tot))).values
    sh = ((share <= 1 - q) & (ret < 0.5 * np.sqrt(tot))).values
    fresh = lambda m: m & ~np.roll(m, 1)
    specs = []
    for i, side in sorted([(i, 1) for i in np.flatnonzero(fresh(lg))] + [(i, -1) for i in np.flatnonzero(fresh(sh))]):
        if not x.a[i] > 0:
            continue
        s = x.spec(i, side, c.iat[i] - side * 1.5 * x.a[i], hold, tp_r=1.5)
        if s:
            specs.append(s)
    return specs


def main():
    pick = set(sys.argv[1:])
    x = Ctx()
    store = LightStore()
    P = itertools.product
    grids = {
        "8. 돌파 실패 역방향": [(f"{nm} 확인{cf}봉 {tp}", lambda o=o, cf=cf, tp=tp: orb_fade(x, o, cf, tp))
                             for (nm, o), cf, tp in P([("런던", 600), ("뉴욕", 990)], [1, 3], ["범위 반대편", "1.5R"])],
        "9. 유동성 스윕": [(f"{lv} {'확인' if cf else '바로'} {tp}R", lambda lv=lv, cf=cf, tp=tp: sweep(x, lv, cf, tp))
                         for lv, cf, tp in P(["전일", "50봉"], [False, True], [1.5, 2.0])],
        "10. 정수 가격대": [(f"{st}$ {'꼬리' if wk else '꼬리없이'} {tp}R", lambda st=st, wk=wk, tp=tp: round_reject(x, st, wk, tp))
                          for st, wk, tp in P([25, 50], [True, False], [1.0, 1.5])],
        "11. 반분산 반전": [(f"W{w} q{q} {h}분", lambda w=w, q=q, h=h: semivar(x, w, q, h))
                          for w, q, h in P([6, 24], [0.7, 0.8], [30, 60])],
    }
    rows, picks = [], []
    for fam, grid in grids.items():
        if pick and fam.split(".")[0] not in pick:
            continue
        best = None
        for name, make in grid:
            specs = [s for s in make() if s["exit"] > s["entry"]]
            tr = execute(specs, store)
            if tr.empty:
                print(f"  {fam} {name}: 거래 없음", flush=True)
                continue
            y = tr.close.dt.year
            row = {"전략": fam, "설정": name, **evaluate(tr), "oz합": round(tr.pnl_oz.sum(), 1),
                   "PF 24-26 비용0.2": pf(tr.pnl_oz[y >= TRAIN_END] - 0.2)}
            rows.append(row)
            if (y < TRAIN_END).sum() >= 30 and (best is None or row["PF 22-23"] > best["PF 22-23"]):
                best = row
            print(f"  {fam} {name}: {len(tr)}건 승률 {row['승률%']} PF22-23 {row['PF 22-23']} PF24-26 {row['PF 24-26']}", flush=True)
        if best:
            best["통과"] = all(best[f"PF {yr}"] >= PASS_PF for yr in YEARS if yr >= TRAIN_END)
            picks.append(best)
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n고른 설정")
    print(pd.DataFrame(picks).to_string(index=False))


if __name__ == "__main__":
    main()
