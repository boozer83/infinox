"""HK 프로그램 설명(docs/hk_program.md)대로 단타·헤지마틴을 적용하고, 뉴스(필터 / 프리미엄 뉴스)를 두 전략 모두에 붙여 비교 (실제 틱, 연도별 1만 달러)

단타 (한 번에 1포지션, 금요일 23:30 청산)
  HK 홈런      : 5분봉 SAR 전환 봉 안에서 Bid가 SAR 값을 넘는 첫 틱 진입 (2-1), 그 순간 5분봉 RSI가 반대쪽
                 (매수 < 50-RSI_GAP, 매도 > 50+RSI_GAP, MT4 HK_HR 평균 37.6 / 60.8), 1시간·시간·ATR 필터 없음
                 청산 (2-3): 손절 7.5, 익절 없음, +3$ 닿으면 손절선 +2$로 올리고 최고점에서 1$ 되돌리면 청산
  우리 SAR 단타: 지금 봇 (봉 마감 + RSI50 + 1시간 SAR + 시간/ATR/과열 필터), 손절 7.5 / 익절 3 / 반대 콩 청산
  금액: 둘 다 HK lot (1,000달러당 0.13, 2-4)로 환산 -> 1$/oz = 시작 잔고의 1.3%
헤지마틴 (3-1, 3-2): hedge_martin_v2.Sim 그대로, 자동주행모드(장세별 분할·기본 lot) 켬, 1$ 간격 불타기,
  봉 안 SAR 전환 때 그 방향 묶음이 이익이면 익절 + 2배 단계 (손실 중이면 그대로 보유),
  사이클 +1% 청산 / 손실 한도 20%(구조대 동결)·30%·50%, 5마틴(단계 0~4)
  사이클 시작 신호 2가지: 우리 SAR 단타 신호 / HK 홈런 신호
뉴스 3가지 (MT5 캘린더 중요도 high USD·EUR·GBP, 미국 서머타임 아닌 날은 1시간 당김 = news_filter.py와 같음)
  없음
  필터     : 발표 앞뒤 NEWS_MIN분. 단타 = 진입 안 함 / 마틴 = 사이클 시작 안 함 + 단계 올리기(2배) 안 함
  프리미엄 : 마틴 마스터 프리미엄 뉴스 슬라이드 (4장, news_hk.one, 손절폭 2$). lot = 마틴 기본 lot의 10배 (1,000달러당 0.1)
             단타 = 단타와 따로 매매해 손익만 더함 (NEWS CYCLE P/L 별도)
             마틴 = 사이클 중이면 뉴스 손익을 사이클 합계에 넣음 (5마틴 구조대: 손실 상쇄), 사이클 밖이면 잔고에 더함
             뉴스 하나가 끝나기 전에 온 다음 발표는 건너뜀

사용법: python -m backtests.hk_apply 2025      (results/hk_apply_2025.csv)
"""
import gc
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.hedge_martin_v2 as hm2
import backtests.long_period as lb
import backtests.news_hk as nh
import backtests.trail_exit as te
from backtests.add_positions_ticks import hit
from backtests.ext_filter import features
from backtests.gold_rest_others import BARS, resample
from backtests.hk_style import our_filter
from indicators import psar, rsi

CAL = Path(os.environ["APPDATA"]) / "MetaQuotes/Terminal/Common/Files/calendar_all.csv"
NEWS_MIN = 30
RSI_GAP = 8
SL, TP = 7.5, 3.0
OZ_PCT = 0.13 * 100 / 1000 * 100          # 1,000달러당 0.13 lot이면 1$/oz = 1.3%
NEWS_PCT = 0.1 * 100 / 1000 * 100          # 프리미엄 뉴스 1,000달러당 0.1 lot이면 1$/oz = 1%
NEWS_CUT, NEWS_X = 2.0, 10                # 손절폭 $, 마틴 기본 lot 대비 배수
NEWS_WIN_STOP = os.environ.get("HK_NEWS_STOP") == "1"   # 남은 쪽 손절 (슬라이드에 없음, 1이면 켬 -> 결과 파일 _stop)
# (목표, 손실 한도, 최대 단계, 1,000달러당 L0, 구조대)
MARTIN = [(0.01, 0.20, 4, 0.01, False), (0.01, 0.30, 4, 0.01, False), (0.01, 0.50, 4, 0.01, False), (0.01, 0.50, 4, 0.01, True)]
# 구조대 (3-3): 5마틴 중 사이클이 -15%에 닿으면 10분 관찰, 그 안에 -20%까지 가면 매수·매도 lot을 맞춰 손익 동결
#   -> 동결 뒤 손익은 안 변하니 그 자리에서 손실 확정과 같게 봄 (스왑·증거금 비용, 홍콩 신호로 메우기는 뺌)
#   고정 한도 20%는 동결 조건 없이 -20%면 청산하는 비교용 (HK에 유리한 쪽)
RESCUE_ARM, RESCUE_HIT, RESCUE_MS = 0.15, 0.20, 600_000
OUT = Path(__file__).resolve().parent.parent / "results"


def to_ms(times):
    return pd.Series(times).values.astype("datetime64[ms]").astype(np.int64)


def news_ms():
    c = pd.read_csv(CAL, encoding="cp949")
    c = c[(c.importance == "high") & c.currency.isin(["USD", "EUR", "GBP"])]
    t = pd.to_datetime(c.time, format="%Y.%m.%d %H:%M")
    t = t[(t.dt.hour > 0) | (t.dt.minute > 0)]
    dst = t.dt.tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward").map(lambda x: bool(x.dst()))
    t = t - pd.to_timedelta((~dst.values).astype(int), unit="h")
    return np.unique(to_ms(t))


def near(ms, nt):
    """각 시각이 뉴스 앞뒤 NEWS_MIN분 안이면 True"""
    w = NEWS_MIN * 60_000
    j = np.searchsorted(nt, ms)
    before = np.where(j < len(nt), nt[np.minimum(j, len(nt) - 1)] - ms, w + 1)
    after = np.where(j > 0, ms - nt[np.maximum(j - 1, 0)], w + 1)
    return (before <= w) | (after <= w)


def news_trades(year):
    """프리미엄 뉴스 매매 (연도 안 발표마다) -> DataFrame(ms 진입, end 청산, oz 매수+매도 $/oz, kind)"""
    ev = nh.events()
    ev = ev[(ev.importance == "high") & (ev.t.dt.year == year)]
    ev_ms = to_ms(ev.t)
    rows, busy = [], 0
    for y, m in [(y, m) for y, m in lb.months() if y == year]:
        s, e = lb.month_range(y, m)
        f = ev_ms[(ev_ms >= int(s.timestamp() * 1000)) & (ev_ms < int(e.timestamp() * 1000))]
        if not len(f):
            continue
        t, bid, ask = lb.load_month(y, m)
        for ms in f:
            if ms - nh.PRE_MS < busy:
                continue
            r = nh.one(t, bid, ask, int(ms), NEWS_CUT, win_stop=NEWS_WIN_STOP)
            if r is None:
                continue
            busy = r[3]
            rows.append((int(ms) - nh.PRE_MS, r[3], r[0] + r[1], r[2]))
        del t, bid, ask
        gc.collect()
    return pd.DataFrame(rows, columns=["ms", "end", "oz", "kind"])


def load_year(year):
    """그 해 1분봉 + 지표 데우기용 전년 12월"""
    parts = []
    for y in (year - 1, year):
        f = BARS / f"XAUUSD_M1_{y}.csv.zip"
        if not f.exists():
            continue
        with zipfile.ZipFile(f) as z:
            m1 = pd.read_csv(z.open(z.namelist()[0]), usecols=["time", "open", "high", "low", "close"])
        m1["time"] = pd.to_datetime(m1.time)
        parts.append(m1[m1.time >= pd.Timestamp(year - 1, 12, 1)] if y < year else m1)
    m1 = pd.concat(parts).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return resample(m1, 5), resample(m1, 60)


def hk_homerun(m5, load=None):
    """봉 안 SAR 전환 첫 틱 전부(flips)와 그중 역RSI인 것(홈런). RSI는 SAR 값을 종가로 놓고 계산 (fast_entry.tick_candidates와 같음)"""
    up, trig = fe.psar_trigger(m5.high.values, m5.low.values)
    c = m5.close
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / fe.RSI_N, adjust=False).mean().values
    lo = (-d.clip(upper=0)).ewm(alpha=1 / fe.RSI_N, adjust=False).mean().values
    i = np.nonzero(up[1:] != up[:-1])[0] + 1
    i = i[i >= 2]
    dd = trig[i] - c.values[i - 1]
    gn = g[i - 1] + (np.maximum(dd, 0) - g[i - 1]) / fe.RSI_N
    ln = lo[i - 1] + (np.maximum(-dd, 0) - lo[i - 1]) / fe.RSI_N
    st = to_ms(m5.time.iloc[i])
    cand = pd.DataFrame({"s": st, "e": st + 300_000, "side": np.where(up[i], 1, -1), "trig": trig[i],
                         "r": 100 - 100 / (1 + gn / ln)})
    rows = []
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        cc = cand[(cand.s >= int(s.timestamp() * 1000)) & (cand.s < int(e.timestamp() * 1000))]
        if cc.empty:
            continue
        t, bid, _ = (load or lb.load_month)(y, m)     # load: 다른 시세(예: MT4식 가짜 틱)로 돌파 시각 찾기
        for r in cc.itertuples(index=False):
            k0, k1 = np.searchsorted(t, r.s), np.searchsorted(t, r.e)
            k = np.flatnonzero(bid[k0:k1] > r.trig if r.side == 1 else bid[k0:k1] < r.trig)
            if k.size:
                rows.append((int(t[k0 + k[0]]), r.side, (r.r - 50) * r.side < -RSI_GAP))
        del t, bid
        gc.collect()
    fl = pd.DataFrame(rows, columns=["ms", "side", "hr"])
    fl = fl[(fl.ms >= to_ms([fe.START])[0]) & (fl.ms < to_ms([fe.END])[0])].reset_index(drop=True)
    return fl, fl[fl.hr].reset_index(drop=True)


def our_signals(m5, h1):
    sig = fe.bar_close_signals(m5, 5, 0.02, fe.context(m5, h1), True)
    px = pd.merge_asof(sig[["t"]], pd.DataFrame({"t": m5.time + pd.Timedelta(minutes=5), "c": m5.close}), on="t").c.values
    sig["atr"], sig["ref"] = sig.atr * 4150 / px, sig.ref * 4150 / px      # ATR 기준은 금값 4150 때 값
    sig = our_filter(sig).merge(features(m5), on="t", how="left")
    sig = sig[(sig.ext <= 0.8) & (sig.risk <= 2.8)]
    return pd.DataFrame({"ms": to_ms(sig.t), "side": sig.side.values})


def friday_ms(ms):
    ts = pd.Timestamp(int(ms), unit="ms")
    return int(np.datetime64(ts.normalize() + pd.Timedelta(days=(4 - ts.weekday()) % 7, hours=23, minutes=30), "ms")
               .astype(np.int64))


def scalp_trade(t, bid, ask, k, side, kind, against):
    fri = friday_ms(t[k])
    if kind == "hk":
        return te.trade(t, bid, ask, k, side, (2.0, 1.0, None), fri, None)
    X = bid if side == 1 else ask
    e = ask[k] if side == 1 else bid[k]
    j = hit(X, k + 1, e - SL, e + TP) if side == 1 else hit(X, k + 1, e - TP, e + SL)
    ex = [j, int(np.searchsorted(t, fri))]
    a = against[side]
    i = np.searchsorted(a, t[k], side="right")
    if i < len(a):
        ex.append(int(np.searchsorted(t, a[i])))
    j = min(ex)
    return None if j >= len(t) else (j, (X[j] - e) * side)


def run_scalp(year, sigs, against):
    """sigs: {이름: (DataFrame(ms, side), 'hk' 또는 'our')} -> {이름: [(진입 ms, $/oz)]}"""
    res = {n: [] for n in sigs}
    busy = dict.fromkeys(sigs, 0)
    ms_list = [(y, m) for y, m in lb.months() if y == year]
    for y, m in ms_list:
        s, e = lb.month_range(y, m)
        sm, em = int(s.timestamp() * 1000), int(e.timestamp() * 1000)
        t, bid, ask = lb.load_month(y, m)
        nxt = (y + 1, 1) if m == 12 else (y, m + 1)     # 월말에 걸친 거래는 다음 달 틱까지 이어서
        if (lb.TICK_DIR / f"{nxt[0]}-{nxt[1]:02d}.npz").exists():
            nt, nb, na = lb.load_month(*nxt)
            t, bid, ask = np.concatenate([t, nt]), np.concatenate([bid, nb]), np.concatenate([ask, na])
            del nt, nb, na
        for n, (sig, kind) in sigs.items():
            f = sig[(sig.ms >= sm) & (sig.ms < em)]
            for ms, side in zip(f.ms.values, f.side.values):
                if ms < busy[n]:
                    continue
                k = int(np.searchsorted(t, ms))
                if k >= len(t) - 1 or t[k] >= friday_ms(t[k]):    # 금요일 23:30 이후 진입 안 함
                    continue
                out = scalp_trade(t, bid, ask, k, int(side), kind, against)
                if out is None:
                    continue
                busy[n] = int(t[out[0]])
                res[n].append((ms, out[1]))
        del t, bid, ask
        gc.collect()
    return res


class HKSim(hm2.Sim):
    """hm2.Sim + 사이클마다 몇 마틴까지 갔는지 기록 + 구조대 동결 (rescue=True)"""

    def __init__(self, target, limit, maxl, lot1000, rescue):
        super().__init__(target, limit, maxl, lot1000)
        self.rescue, self.freezes, self.levels, self.top, self.t = rescue, 0, [], -1, None
        self.t0, self.days, self.log = None, [], []      # log: (시작 ms, 끝 ms, 몇 마틴, 종료 이유, 손익 %)

    def start(self, k, bid, ask, regime=None):
        if not self.g and not self.dead:
            self.t0 = int(self.t[k])
        super().start(k, bid, ask, regime)
        self.top = max(self.top, self.level)

    def flip(self, side, k, bid, ask):
        super().flip(side, k, bid, ask)
        self.top = max(self.top, self.level)

    def end(self, k, bid, ask, why):
        self.levels.append(self.top + 1)
        self.days.append((int(self.t[k]) - self.t0) / 86_400_000)
        total = self.real + sum(self.pnl_group(s, k, bid, ask) for s in self.g)
        self.log.append((self.t0, int(self.t[k]), self.top + 1, why, total / self.start_bal * 100))
        self.top = -1
        super().end(k, bid, ask, why)

    def eq(self, k0, k1, bid, ask):
        e = np.full(k1 - k0, self.real)
        for s, (lot, es) in self.g.items():
            e += lot * hm2.CONTRACT * (len(es) * (bid[k0:k1] if s == 1 else ask[k0:k1]) - sum(es)) * s
        return e

    def freeze_at(self, k0, k1, bid, ask):
        """지금 포지션 그대로일 때 동결이 걸리는 첫 틱 (-15% 새로 닿은 뒤 10분 안에 -20%) 또는 None"""
        for c0 in range(k0, k1, 500_000):
            c1 = min(k1, c0 + 500_000)
            e = self.eq(c0, c1, bid, ask)
            below = e <= -RESCUE_ARM * self.start_bal
            prev = np.r_[False if c0 == k0 else bool(below_last), below[:-1]]
            below_last = below[-1]
            w_end = c0
            for i in np.flatnonzero(below & ~prev):
                if c0 + i < w_end:
                    continue
                w_end = min(c1, int(np.searchsorted(self.t, self.t[c0 + i] + RESCUE_MS)))
                j = np.flatnonzero(e[i:w_end - c0] <= -RESCUE_HIT * self.start_bal)
                if j.size:
                    return c0 + i + int(j[0])
        return None

    def run(self, k0, k1, bid, ask):
        while self.rescue and self.g and self.level >= self.maxl and k0 < k1:
            f = self.freeze_at(k0, k1, bid, ask)
            if f is None:
                break
            super().run(k0, f, bid, ask)
            if not self.g:
                return
            if self.eq(f, f + 1, bid, ask)[0] <= -RESCUE_HIT * self.start_bal:   # 그 사이 불타기가 있었으면 다시 확인
                self.freezes += 1
                self.end(f, bid, ask, "freeze")
                return
            k0 = f + 1
        super().run(k0, k1, bid, ask)


def add_news(sim, oz):
    """프리미엄 뉴스 손익: 사이클 중이면 사이클 합계에 (구조대), 아니면 잔고에"""
    if sim.dead:
        return
    lot = max(0.01, np.floor(sim.bal / 1000 * NEWS_X * 0.01 * 100) / 100)   # 잔고 1,000달러당 0.1 (장세 분할과 무관)
    usd = oz * lot * hm2.CONTRACT
    sim.news = getattr(sim, "news", 0.0) + usd
    if sim.g:
        sim.real += usd
    else:
        sim.bal += usd
        sim.mdd = max(sim.mdd, (sim.peak - sim.bal) / sim.peak)
        sim.peak = max(sim.peak, sim.bal)


def run_martin(year, starts, flips, news):
    """starts: {이름: DataFrame(ms, side, regime)}, flips: {이름: DataFrame(ms, side)}, news: {이름: 뉴스 매매 또는 None}
    -> {(이름, 조합): Sim}"""
    hm2.REGIME = True
    hm2.FLIP_ONLY_WIN = True     # 3-1 규칙 3·4: 이긴 묶음만 익절 후 2배, 지는 묶음은 그대로 보유
    sims = {(n, g): HKSim(*g) for n in starts for g in MARTIN}
    evs = {}
    for n in starts:
        a = starts[n].assign(kind=0, oz=0.0)
        b = flips[n].assign(kind=1, regime=1, oz=0.0)
        parts = [a, b]
        if news[n] is not None:     # 뉴스 손익은 청산 시각에 반영
            parts.append(pd.DataFrame({"ms": news[n].end.values, "side": 0, "regime": 1, "kind": 2, "oz": news[n].oz.values}))
        evs[n] = pd.concat(parts).sort_values(["ms", "kind"]).reset_index(drop=True)
    for y, m in [(y, m) for y, m in lb.months() if y == year]:
        s, e = lb.month_range(y, m)
        sm, em = int(s.timestamp() * 1000), int(e.timestamp() * 1000)
        t, bid, ask = lb.load_month(y, m)
        for (n, g), sim in sims.items():
            sim.t = t
            f = evs[n][(evs[n].ms >= sm) & (evs[n].ms < em)]
            ks = np.searchsorted(t, f.ms.values)
            prev = 0
            for kind, side, k, rg, oz in zip(f.kind.values, f.side.values, ks, f.regime.values, f.oz.values):
                if k >= len(t):
                    break
                sim.run(prev, k, bid, ask)
                if kind == 0:
                    sim.start(k, bid, ask, rg)
                elif kind == 2:
                    add_news(sim, oz)
                elif sim.g:
                    sim.flip(int(side), k, bid, ask)
                prev = k
            sim.run(prev, len(t), bid, ask)
        last = (bid[-1:], ask[-1:])
        del t, bid, ask
        gc.collect()
        print(f"  마틴 {y}-{m:02d} 완료", flush=True)
    for sim in sims.values():     # 끝날 때 열린 사이클은 평가손익까지 넣어 잔고 계산 (승패 집계에는 안 넣음)
        sim.final = sim.bal if not sim.g else \
            sim.start_bal + sim.real + sum(sim.pnl_group(s, 0, *last) for s in sim.g)
    return sims


def scalp_stats(name, tr, news=None):
    """단타 손익 (+ 프리미엄 뉴스 손익을 시각 순으로 합친 % 곡선)"""
    oz = np.array([x[1] for x in tr])
    if not len(oz):
        return {"전략": name, "거래": 0}
    w, l = oz[oz > 0], oz[oz <= 0]
    pct = pd.Series(oz * OZ_PCT, index=[x[0] for x in tr])
    nw = 0.0
    if news is not None and len(news):
        nw = news.oz.sum() * NEWS_PCT
        pct = pd.concat([pct, pd.Series(news.oz.values * NEWS_PCT, index=news.end.values)]).sort_index()
    eq = np.r_[0, pct.cumsum().values]
    dd = (np.maximum.accumulate(eq) - eq).max()
    return {"전략": name, "거래": len(oz), "승률%": round(len(w) / len(oz) * 100, 1),
            "PF": round(w.sum() / -l.sum(), 2) if l.sum() else np.inf, "합계$/oz": round(oz.sum(), 1),
            "뉴스%": round(nw, 1), "수익%": round(oz.sum() * OZ_PCT + nw, 1), "최대낙폭%": round(dd, 1)}


def main():
    year = int(sys.argv[1])
    has_news = CAL.exists()
    nt = news_ms() if has_news else None
    fe.START, fe.END = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
    m5, h1 = load_year(year)
    fl, hr = hk_homerun(m5)
    our = our_signals(m5, h1)
    against = {1: to_ms(m5.time[~psar(m5.high.values, m5.low.values)] + pd.Timedelta(minutes=5)),
               -1: to_ms(m5.time[psar(m5.high.values, m5.low.values)] + pd.Timedelta(minutes=5))}
    for d in (hr, our):
        d["regime"] = hm2.regime_at(m5, h1, pd.to_datetime(d.ms, unit="ms").astype(m5.time.dtype))
    del m5, h1
    gc.collect()
    print(f"{year}: 봉 안 SAR 전환 {len(fl)}, HK 홈런 {len(hr)}, 우리 신호 {len(our)}", flush=True)

    if not has_news:
        sys.exit("캘린더 파일 없음: MT5에서 ExportCalendar (Currency 빈칸) 실행")
    nw = news_trades(year)
    print(f"  프리미엄 뉴스 {len(nw)}번, 합계 {nw.oz.sum():.1f}$/oz", flush=True)
    keep = lambda d: d
    skip = lambda d: d[~near(d.ms.values, nt)].reset_index(drop=True)
    news_opts = [("뉴스 없음", keep, None), ("뉴스 필터", skip, None), ("프리미엄 뉴스", keep, nw)]

    sigs, starts, flips, news = {}, {}, {}, {}
    for nn, f, nwd in news_opts:
        for src, d, kind in (("HK 홈런", hr, "hk"), ("우리 SAR", our, "our")):
            sigs[f"단타 {src} / {nn}"] = (f(d), kind)
            k = f"마틴 시작={src} / {nn}"
            starts[k], flips[k], news[k] = f(d), f(fl[~fl.ms.isin(d.ms)]), nwd
    news_of = {f"단타 {src} / {nn}": nwd for nn, _, nwd in news_opts for src in ("HK 홈런", "우리 SAR")}

    sc = run_scalp(year, sigs, against)
    print("  단타 완료", flush=True)
    sims = run_martin(year, starts, flips, news)
    rows = [{"연도": year, **scalp_stats(n, tr, news_of[n])} for n, tr in sc.items()]
    for (n, (tg, lim, ml, l1, rescue)), sim in sims.items():
        lv = np.bincount(np.asarray(sim.levels, int), minlength=6)[1:6]
        rows.append({"연도": year, "전략": f"{n} / {'구조대 + ' if rescue else ''}한도 {int(lim * 100)}%", "거래": sim.cycles,
                     "마틴분포 1~5": "/".join(map(str, lv)), "동결": sim.freezes,
                     "사이클일수 중간": round(float(np.median(sim.days)), 2) if sim.days else np.nan,
                     "사이클일수 최대": round(max(sim.days), 1) if sim.days else np.nan,
                     "승률%": round(sim.wins / sim.cycles * 100, 1) if sim.cycles else np.nan,
                     "뉴스%": round(getattr(sim, "news", 0.0) / hm2.BAL0 * 100, 1),
                     "수익%": round((sim.final / hm2.BAL0 - 1) * 100, 1), "최대낙폭%": round(sim.mdd * 100, 1),
                     "한도걸림": sim.limits, "최대lot": round(sim.max_lot, 2), "깡통": sim.dead, "진행중": bool(sim.g)})
    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res["뉴스최대손실%(1만$)"] = round(nw.oz.min() * NEWS_PCT, 1) if len(nw) else np.nan
    res.to_csv(OUT / f"hk_apply_{year}{'_stop' if NEWS_WIN_STOP else ''}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
