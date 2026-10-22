"""HK 마틴 재현 v2: 이기는 쪽 1$마다 불타기 (2026-10-22 캡처 243$ / 600$ / 45만$ 계좌에서 확인한 구조, 실제 틱)

캡처에서 본 흐름 (243$ 계좌 10/22):
  01:23 매수·매도 0.03 시작 -> 매도가 이기면서 1$ 내려갈 때마다 매도 0.03 추가 (4138.21 / 4137.17 / 4136.06)
  -> 01:31 5분봉 SAR이 봉 안에서 하락으로 넘어가는 순간 매도 묶음 익절 + 매도 0.06(2배)로 다시 시작, 또 1$마다 추가
  -> 지는 매수 0.03은 손절 없이 보유 -> 04:12 사이클 합계가 목표를 넘자 전부 청산
규칙:
  1) 사이클 시작: 지금 SAR 단타 신호(봉 마감 + RSI50 + 1시간 SAR + 시간/ATR/과열 필터, 뉴스 앞뒤 30분 제외)에
     매수·매도 묶음을 L0로 하나씩 시작. L0 = 시작 잔고 1,000달러당 LOT1000 (0.01 단위 내림, 최소 0.01)
  2) 묶음마다: 마지막 진입가에서 이기는 방향으로 GAP$ 가면 같은 lot 1개 추가 (묶음당 최대 MAXPOS개)
  3) 사이클 중 5분봉 SAR이 봉 안에서 d 방향으로 넘어가는 첫 틱 (HK 진입 시점, hk_style.hk_signals와 같음):
     d 방향 묶음이 있으면 그 묶음을 시장가로 정리(확정)하고 d 방향으로 다음 단계 lot(L0 x 2^단계) 묶음 새로 시작 (최대 MAXL단계)
     반대 방향 묶음(지는 쪽)은 손절 없이 그대로
  4) 사이클 합계(확정 + 평가)가 시작 잔고의 +TARGET면 전부 청산, -LIMIT면 전부 청산
매수 진입·매도 청산은 Ask, 매도 진입·매수 청산은 Bid. 주말 보유. 해마다 1만 달러로 새로 시작

사용법: python -m backtests.hedge_martin_v2 2026          (연도별)
        python -m backtests.hedge_martin_v2 2026 9         (그 해 9월부터)
"""
import gc
import itertools
import sys

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.hedge_martin_ticks as hm
import backtests.long_period as lb
from backtests.gold_rest_others import BARS, resample

BAL0 = 10_000.0
CONTRACT = 100.0
GAP, MAXPOS = 1.0, 3
FLIP_ONLY_WIN = False
REGIME = False   # True면 사이클 시작 때 장세로 분할 수·기본 lot 결정 (7월 영상: 추세 1개/Martin 1000, 일반 2개/1000, 횡보 3개/600)
SPLIT = {0: (1, 1000), 1: (2, 1000), 2: (3, 600)}   # 장세 -> (묶음당 최대 개수, 잔고 몇 $당 0.01 lot)   # True면 전환 방향 묶음이 이익일 때만 정리 + 다음 단계
END = pd.Timestamp("2026-10-22 06:00")
# (목표, 손실 한도, 최대 단계, 1,000달러당 L0)
GRID = list(itertools.product((0.01, 0.02), (0.30, 0.50), (3, 4, 5), (0.01, 0.03)))


class Sim:
    def __init__(self, target, limit, maxl, lot1000):
        self.target, self.limit, self.maxl, self.lot1000 = target, limit, maxl, lot1000
        self.bal = self.peak = BAL0
        self.mdd = 0.0
        self.g = {}                     # side -> [lot, [진입가...]]
        self.maxpos, self.per = MAXPOS, None
        self.level = -1
        self.real = 0.0
        self.start_bal = BAL0
        self.cycles = self.wins = self.limits = 0
        self.max_lot = 0.0
        self.dead = False

    def l0(self):
        if self.per:
            return max(0.01, np.floor(self.start_bal / self.per * self.lot1000 / 0.01) / 100)
        return max(0.01, np.floor(self.start_bal / 1000 * self.lot1000 * 100) / 100)

    def pnl_group(self, side, k, bid, ask):
        lot, es = self.g[side]
        px = bid[k] if side == 1 else ask[k]
        return sum(lot * CONTRACT * (px - e) * side for e in es)

    def start(self, k, bid, ask, regime=None):
        if self.dead or self.g:
            return
        if REGIME and regime is not None:
            self.maxpos, self.per = SPLIT[int(regime)]
        self.start_bal, self.real, self.level = self.bal, 0.0, 0
        l0 = self.l0()
        self.g = {1: [l0, [ask[k]]], -1: [l0, [bid[k]]]}
        self.cycles += 1
        self.max_lot = max(self.max_lot, l0)

    def flip(self, side, k, bid, ask):
        if not self.g or self.level >= self.maxl:
            return
        if side in self.g:
            pg = self.pnl_group(side, k, bid, ask)
            if FLIP_ONLY_WIN and pg <= 0:
                return                  # 그 방향 묶음이 손실 중이면 그대로 둠
            self.real += pg
        self.level += 1
        lot = round(self.l0() * 2 ** self.level, 2)
        self.g[side] = [lot, [ask[k] if side == 1 else bid[k]]]
        self.max_lot = max(self.max_lot, lot)

    def end(self, k, bid, ask, why):
        total = self.real + sum(self.pnl_group(s, k, bid, ask) for s in self.g)
        self.bal = self.start_bal + total
        self.g, self.real, self.level = {}, 0.0, -1
        if why == "limit":
            self.limits += 1
        elif total > 0:
            self.wins += 1
        if self.bal < 100:
            self.dead = True

    def run(self, k0, k1, bid, ask):
        k = k0
        while self.g and k < k1:
            n = min(k1, k + 200_000)
            b, a = bid[k:n], ask[k:n]
            eq = np.full(n - k, self.real)
            for s, (lot, es) in self.g.items():
                px = b if s == 1 else a
                eq += lot * CONTRACT * (len(es) * px - sum(es)) * s
            ev = np.flatnonzero((eq >= self.target * self.start_bal) | (eq <= -self.limit * self.start_bal))
            ek = k + ev[0] if ev.size else n
            add = n
            for s, (lot, es) in self.g.items():
                if len(es) < self.maxpos:
                    h = np.flatnonzero(a >= es[-1] + GAP) if s == 1 else np.flatnonzero(b <= es[-1] - GAP)
                    if h.size:
                        add = min(add, k + h[0])
            j = min(ek, add)
            low = self.start_bal + eq[: max(1, j - k + 1)].min()
            self.peak = max(self.peak, self.bal)
            self.mdd = max(self.mdd, (self.peak - low) / self.peak)
            if j >= n:
                k = n
                continue
            if j == ek:
                self.end(j, bid, ask, "limit" if eq[j - k] < 0 else "target")
                return
            for s, (lot, es) in self.g.items():
                if len(es) < self.maxpos and ((s == 1 and ask[j] >= es[-1] + GAP) or (s == -1 and bid[j] <= es[-1] - GAP)):
                    es.append(ask[j] if s == 1 else bid[j])
            k = j + 1


def regime_at(m5, h1, times):
    """0 추세 / 1 일반 / 2 횡보. 7월 영상 13장면 평균으로 정한 대략 기준: 1시간 ADX, 최근 2시간 5분봉 직진도"""
    from indicators import adx
    end5 = m5.time + pd.Timedelta(minutes=5)
    eff = ((m5.close - m5.close.shift(24)).abs() / m5.close.diff().abs().rolling(24).sum()).values
    a = pd.DataFrame({"end": (h1.time + pd.Timedelta(hours=1)).values, "adx": adx(h1).values})
    x = pd.merge_asof(pd.DataFrame({"t": times.values}), pd.DataFrame({"end": end5.values, "eff": eff}),
                      left_on="t", right_on="end", direction="backward")
    x = pd.merge_asof(x.drop(columns="end"), a, left_on="t", right_on="end", direction="backward")
    r = np.ones(len(x), int)
    r[(x.eff >= 0.35) | (x.adx >= 30)] = 0
    r[(x.eff <= 0.12) & (x.adx <= 20)] = 2
    return r


def events(year, from_month):
    """(틱 시각 ms, 종류, 방향): 종류 0 = 사이클 시작(SAR 단타 신호, 봉 마감), 1 = 봉 안 SAR 전환 첫 틱"""
    import zipfile
    with zipfile.ZipFile(BARS / f"XAUUSD_M1_{year}.csv.zip") as z:
        m1 = pd.read_csv(z.open(z.namelist()[0]), usecols=["time", "open", "high", "low", "close"])
    m1["time"] = pd.to_datetime(m1.time)
    m1 = m1.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if year == 2026:     # 10/17 이후 1분봉은 틱에서 만들어 붙임
        t, b, _ = lb.load_month(2026, 10)
        s = pd.Series(b, index=pd.to_datetime(t, unit="ms")).resample("1min").ohlc().dropna().reset_index()
        s = s.rename(columns={"index": "time"})
        m1 = pd.concat([m1, s[s.time > m1.time.iat[-1]]], ignore_index=True)
        del t, b
    m5, h1 = resample(m1, 5), resample(m1, 60)
    del m1
    fe.START, fe.END = pd.Timestamp(year, from_month, 1), END
    sig = fe.bar_close_signals(m5, 5, 0.02, fe.context(m5, h1), True)
    px = pd.merge_asof(sig[["t"]], pd.DataFrame({"t": m5.time + pd.Timedelta(minutes=5), "c": m5.close}), on="t").c.values
    sig["atr"], sig["ref"] = sig.atr * 4150 / px, sig.ref * 4150 / px
    from backtests.hk_style import our_filter
    from backtests.ext_filter import features
    sig = our_filter(sig).merge(features(m5), on="t", how="left")
    sig = sig[(sig.ext <= 0.8) & (sig.risk <= 2.8)]
    sig = sig[~hm.near_news(sig.t)]
    st = pd.DataFrame({"ms": sig.t.values.astype("datetime64[ms]").astype(np.int64), "kind": 0, "side": sig.side.values,
                       "regime": regime_at(m5, h1, sig.t)})
    cand = fe.tick_candidates(m5)
    import backtests.hk_style as hs
    fl = hs.hk_signals(cand)
    fl = pd.DataFrame({"ms": fl.t.values.astype("datetime64[ms]").astype(np.int64), "kind": 1, "side": fl.side.values, "regime": 1})
    ev = pd.concat([st, fl]).sort_values(["ms", "kind"]).reset_index(drop=True)
    return ev[ev.ms < int(END.value // 1_000_000)]


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    year = int(sys.argv[1])
    m0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ev = events(year, m0)
    gc.collect()
    print(f"  사이클 시작 신호 {int((ev.kind == 0).sum())}개, 봉 안 SAR 전환 {int((ev.kind == 1).sum())}개", flush=True)
    sims = {g: Sim(*g) for g in GRID}
    for y, m in [(y, m) for y, m in lb.months() if y == year and m >= m0 and (y, m) <= (END.year, END.month)]:
        s, e = lb.month_range(y, m)
        sm, em = int(s.timestamp() * 1000), int(e.timestamp() * 1000)
        t, bid, ask = lb.load_month(y, m)
        f = ev[(ev.ms >= sm) & (ev.ms < em)]
        ks = np.searchsorted(t, f.ms.values)
        for sim in sims.values():
            prev = 0
            for kind, side, k, rg in zip(f.kind.values, f.side.values, ks, f.regime.values):
                if k >= len(t):
                    break
                sim.run(prev, k, bid, ask)
                if kind == 0:
                    sim.start(k, bid, ask, rg)
                elif sim.g:
                    sim.flip(int(side), k, bid, ask)
                prev = k
            sim.run(prev, len(t), bid, ask)
        del t, bid, ask
        gc.collect()
        print(f"  {y}-{m:02d} 완료", flush=True)
    rows = [{"목표%": tg * 100, "한도%": lim * 100, "최대단계": ml, "L0/1000$": l1, "최종$": round(sim.bal),
             "수익%": round((sim.bal / BAL0 - 1) * 100, 1), "최대낙폭%": round(sim.mdd * 100, 1), "사이클": sim.cycles,
             "사이클승": sim.wins, "한도걸림": sim.limits, "최대lot": round(sim.max_lot, 2), "깡통": sim.dead,
             "진행중": bool(sim.g)}
            for (tg, lim, ml, l1), sim in sims.items()]
    res = pd.DataFrame(rows).sort_values("최종$", ascending=False)
    print(res.to_string(index=False))
    res.to_csv(f"hedge_martin_v2_{year}{'' if m0 == 1 else f'_{m0:02d}'}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
