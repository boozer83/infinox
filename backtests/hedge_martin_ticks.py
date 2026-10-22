"""HK 마틴(헤지 마틴) 구조 재현 백테스트 (실제 틱, 2022-01 ~ 2026-10-16, 1만 달러 시작)

캡처(9개 계좌)에서 본 구조를 단순화:
  1) 포지션이 없을 때 5분봉 SAR 전환 봉이 끝나면 매수·매도를 같은 lot(L0)으로 동시 진입 -> 사이클 시작
     L0 = 사이클 시작 잔고 1,000달러당 0.01 (패널 Martin 1000)
  2) 모든 포지션은 각자 진입가 ± X$ 에 익절. 손절 없음 (진 쪽은 들고 감)
  3) 사이클 중 다음 SAR 전환마다 전환 방향으로 새 포지션, lot은 직전 단계의 2배 (최대 단계 MAXL)
     단, 그 방향에 이미 열린 포지션이 있으면 새로 안 들어감
  4) 사이클 합계(확정 + 평가)가 시작 잔고의 +TARGET%면 전부 청산
  5) 사이클 합계가 -LIMIT%면 전부 청산 (Cycle Loss Limit)
  6) 포지션이 모두 익절로 사라져도 사이클 끝 (CONTINUE=True면 끝내지 않고, 다음 SAR 전환에 다음 단계 lot 1개로 이어감
     -> 10/20 0.05 계좌 캡처와 같은 흐름: 첫 쌍 둘 다 익절 -> 매수 0.10 -> 매도 0.20 -> 매수 0.40 -> 합계 목표에서 전부 청산)
청산 가격: 매수 Bid, 매도 Ask. 주말에도 보유. 수수료 없음 (스프레드만)
조합: X 2/3/5, 목표 1/2%, 손실 한도 30/50%, 최대 단계 4/5 (단계 0 = 첫 쌍)

사용법: python -m backtests.hedge_martin_ticks [연도]   (연도를 주면 그 해만 1만 달러로 새로 시작, 메모리 절약)
      python -m backtests.hedge_martin_ticks   (봉 기반 옛 버전: backtests/hedge_martin.py)
"""
import gc
import itertools
import sys
import zipfile

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.gold_rest_others import BARS, load_m1, resample
from indicators import psar

BAL0 = 10_000.0
END = pd.Timestamp("2026-10-16 23:00")
CONTRACT = 100.0
CONTINUE = False
START_SIG = False   # True면 사이클 시작은 우리 SAR 단타 신호(봉 마감 + 필터)일 때만, 뉴스 앞뒤 NEWS_MIN분은 시작 안 함
NEWS_MIN = 30
GRID = list(itertools.product((2.0, 3.0, 5.0), (0.01, 0.02), (0.30, 0.50), (4, 5)))


class Sim:
    def __init__(self, x, target, limit, maxl):
        self.x, self.target, self.limit, self.maxl = x, target, limit, maxl
        self.bal = self.peak = BAL0
        self.mdd = 0.0
        self.pos = []                   # [side, lot, entry]
        self.level = -1
        self.real = 0.0                 # 사이클 확정 손익
        self.start_bal = BAL0
        self.cycles = self.wins = self.limits = 0
        self.max_lot = 0.0
        self.dead = False
        self.monthly = {}

    def lot0(self):
        return max(0.01, np.floor(self.start_bal / 1000 * 0.01 * 100) / 100)

    def open(self, side, lot, px):
        self.pos.append([side, lot, px])
        self.max_lot = max(self.max_lot, lot)

    def on_flip(self, side, k, bid, ask, start_ok=True):
        if self.dead:
            return
        if not self.pos and self.level < 0 and not start_ok:
            return
        if not self.pos and self.level >= 0 and CONTINUE:
            if self.level < self.maxl:
                self.level += 1
                self.open(side, self.lot0() * 2 ** self.level, ask[k] if side == 1 else bid[k])
            return
        if not self.pos:
            self.start_bal, self.real, self.level = self.bal, 0.0, 0
            l0 = self.lot0()
            self.open(1, l0, ask[k])
            self.open(-1, l0, bid[k])
            self.cycles += 1
        elif self.level < self.maxl and not any(p[0] == side for p in self.pos):
            self.level += 1
            self.open(side, self.lot0() * 2 ** self.level, ask[k] if side == 1 else bid[k])

    def close_all(self, k, bid, ask, why):
        pnl = sum(p[1] * CONTRACT * ((bid[k] - p[2]) if p[0] == 1 else (p[2] - ask[k])) for p in self.pos)
        self.end_cycle(self.real + pnl, why)

    def end_cycle(self, total, why):
        self.bal = self.start_bal + total
        self.pos, self.real, self.level = [], 0.0, -1
        if why == "limit":
            self.limits += 1
        elif total > 0:
            self.wins += 1
        if self.bal < 100:
            self.dead = True

    def run(self, k0, k1, bid, ask):
        """틱 [k0, k1) 동안 익절 / 목표 / 손실 한도 처리"""
        k = k0
        while self.pos and k < k1:
            n = min(k1, k + 200_000)
            b, a = bid[k:n], ask[k:n]
            # 지금 포지션 그대로일 때 평가손익
            lb_ = sum(p[1] for p in self.pos if p[0] == 1)
            ls_ = sum(p[1] for p in self.pos if p[0] == -1)
            cb = sum(p[1] * p[2] for p in self.pos if p[0] == 1)
            cs = sum(p[1] * p[2] for p in self.pos if p[0] == -1)
            eq = self.real + CONTRACT * (lb_ * b - cb + cs - ls_ * a)
            # 첫 익절 틱
            tp = n
            for p in self.pos:
                hit = np.flatnonzero(b >= p[2] + self.x) if p[0] == 1 else np.flatnonzero(a <= p[2] - self.x)
                if hit.size:
                    tp = min(tp, k + hit[0])
            ev = np.flatnonzero((eq >= self.target * self.start_bal) | (eq <= -self.limit * self.start_bal))
            ek = k + ev[0] if ev.size else n
            j = min(tp, ek)
            # 그 구간 최저 평가금으로 낙폭
            low = self.start_bal + eq[: j - k + 1].min() if j > k else self.start_bal + eq[0]
            self.peak = max(self.peak, self.bal)
            self.mdd = max(self.mdd, (self.peak - low) / self.peak)
            if j >= n:
                k = n
                continue
            if j == ek and (j < tp or ev.size):
                e = eq[j - k]
                if e >= self.target * self.start_bal or e <= -self.limit * self.start_bal:
                    self.close_all(j, bid, ask, "limit" if e < 0 else "target")
                    return
            # 익절: 그 틱에 닿은 포지션 모두
            keep = []
            for p in self.pos:
                if (p[0] == 1 and bid[j] >= p[2] + self.x) or (p[0] == -1 and ask[j] <= p[2] - self.x):
                    self.real += p[1] * CONTRACT * self.x
                else:
                    keep.append(p)
            self.pos = keep
            if not self.pos:
                if CONTINUE and self.real < self.target * self.start_bal and self.level < self.maxl:
                    return                       # 다음 SAR 전환에 다음 단계로 이어감
                self.end_cycle(self.real, "tp")
                return
            k = j + 1


def near_news(times):
    """중요도 moderate 이상 USD/EUR/GBP 발표 앞뒤 NEWS_MIN분 (MT5 캘린더 calendar_all.csv, 서버 시간)"""
    import os
    f = os.path.join(os.environ["APPDATA"], "MetaQuotes/Terminal/Common/Files/calendar_all.csv")
    c = pd.read_csv(f, encoding="cp949")
    c = c[c.importance.isin(["moderate", "high"]) & c.currency.isin(["USD", "EUR", "GBP"])]
    nt = np.sort(pd.to_datetime(c.time, format="%Y.%m.%d %H:%M").values.astype("datetime64[ms]").astype(np.int64))
    tm = times.values.astype("datetime64[ms]").astype(np.int64)
    j = np.searchsorted(nt, tm)
    w = NEWS_MIN * 60_000
    before = np.where(j < len(nt), nt[np.minimum(j, len(nt) - 1)] - tm, np.inf)
    after = np.where(j > 0, tm - nt[np.maximum(j - 1, 0)], np.inf)
    return (before <= w) | (after <= w)


def flips(year=None):
    if year:
        with zipfile.ZipFile(BARS / f"XAUUSD_M1_{year}.csv.zip") as z:
            m1 = pd.read_csv(z.open(z.namelist()[0]), usecols=["time", "open", "high", "low", "close"])
        m1["time"] = pd.to_datetime(m1.time)
        m1 = m1.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    else:
        m1 = load_m1("XAUUSD")
    m5 = resample(m1, 5)
    h1 = resample(m1, 60)
    del m1
    up = psar(m5.high.values, m5.low.values)
    i = np.nonzero(up[1:] != up[:-1])[0] + 1
    end = (m5.time.iloc[i] + pd.Timedelta(minutes=5)).values.astype("datetime64[ms]").astype(np.int64)
    out = pd.DataFrame({"ms": end, "side": np.where(up[i], 1, -1), "start": True})
    if START_SIG:
        import backtests.fast_entry as fe
        from backtests.hk_style import our_filter
        fe.START, fe.END = m5.time.iat[0], END
        sig = fe.bar_close_signals(m5, 5, 0.02, fe.context(m5, h1), True)
        # ATR 최소값 2.5$는 지금 금값(약 4,150) 기준이라, 해마다 금값에 비례하게 바꿔서 거름
        px = pd.merge_asof(sig[["t"]], pd.DataFrame({"t": m5.time + pd.Timedelta(minutes=5), "c": m5.close}), on="t").c.values
        sig["atr"], sig["ref"] = sig.atr * 4150 / px, sig.ref * 4150 / px
        sig = our_filter(sig)
        sig = sig[~near_news(sig.t)]
        key = set(zip(sig.t.values.astype("datetime64[ms]").astype(np.int64), sig.side))
        out["start"] = [(a, b) in key for a, b in zip(out.ms, out.side)]
    return out[out.ms < int(END.value // 1_000_000)].reset_index(drop=True)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    year = int(sys.argv[1]) if len(sys.argv) > 1 else None
    fl = flips(year)
    gc.collect()
    sims = {g: Sim(*g) for g in GRID}
    months = [(y, m) for y, m in lb.months() if (y, m) <= (END.year, END.month) and (year is None or y == year)]
    for y, m in months:
        s, e = lb.month_range(y, m)
        sm, em = int(s.timestamp() * 1000), min(int(e.timestamp() * 1000), int(END.value // 1_000_000))
        t, bid, ask = lb.load_month(y, m)
        f = fl[(fl.ms >= sm) & (fl.ms < em)]
        ks = np.searchsorted(t, f.ms.values)
        bounds = list(ks) + [len(t)]
        for sim in sims.values():
            prev = 0
            for side, k, st in zip(f.side.values, ks, f.start.values):
                sim.run(prev, k, bid, ask)
                if k < len(t):
                    sim.on_flip(int(side), k, bid, ask, bool(st))
                prev = k
            sim.run(prev, len(t), bid, ask)
            sim.monthly[(y, m)] = sim.bal
        del t, bid, ask
        gc.collect()
        print(f"  {y}-{m:02d} 완료", flush=True)
    rows = []
    for (x, tg, lim, ml), sim in sims.items():
        mb = pd.Series(sim.monthly)
        yr = {yy: mb[[k for k in mb.index if k[0] == yy]] for yy in range(2022, 2027)}
        row = {"익절$": x, "목표%": tg * 100, "한도%": lim * 100, "최대단계": ml, "최종$": round(sim.bal),
               "최대낙폭%": round(sim.mdd * 100, 1), "사이클": sim.cycles, "한도걸림": sim.limits,
               "최대lot": round(sim.max_lot, 2), "깡통": sim.dead, "사이클승": sim.wins}
        prev = BAL0
        for yy, v in yr.items():
            if year and yy != year:
                continue
            if len(v):
                row[f"{yy}%"] = round((v.iloc[-1] / prev - 1) * 100, 1) if prev > 0 else 0
                prev = v.iloc[-1]
        rows.append(row)
    res = pd.DataFrame(rows).sort_values("최종$", ascending=False)
    print(res.to_string(index=False))
    res.to_csv(f"hedge_martin_ticks{year or ''}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
