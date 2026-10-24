"""우리 봇 6개를 금 + 다른 종목 11개에 같은 규칙으로 (실제 틱, 2022-01 ~ 2026-10)

봇 규칙은 그대로 두고, 금 기준 '달러' 값만 종목에 맞게 옮김:
  배율 k = 그 시점 가격 / 4,150 x (종목 5분봉 ATR% 중앙값 / 금 5분봉 ATR% 중앙값)   (gold_rest_others의 '변동성비례')
  -> 손절 7.5$ / 익절 3$ / 최소 ATR 2.5$ / 헤지 익절 5$ / 뉴스 되돌림 2$ / 최대 스프레드 2$ 는 x k,
     뉴스 0.05% 기준은 x (ATR% 비율), 수량은 금 1온스 = 그 종목 1/k 단위 (손절 한 번 손실 $가 금과 같음)
  XAUUSD   : 지금 봇 그대로 (k = 1)
  XAUUSD비례: 금도 같은 방식 (k = 금값 / 4,150, 2022~24는 금값이 낮아 거리·최소 ATR이 작아짐) - 다른 종목과 비교용
ATR·돌파·위험% 로 정해진 규칙(H4 추세, 아시아 돌파)은 배율 없이 그대로.

1. SAR 단타 (demo_trader)  : 5분봉 SAR 전환 + RSI50 + H1 SAR, 손절 7.5 / 익절 3, 반대 콩 청산, 과열 거르기(0.8 / 2.8),
                              미국장 15~21시 / H1 전환 3번 / ATR 2.5 미만 / ATR 20일 중앙값 2배 초과 쉬기,
                              금요일 22:30 진입 안 함·23:30 청산 (코인은 주말에도 거래하므로 금요일 규칙 없음)
                              1,000$ 시작, 그날 시작 잔고 1,000$당 0.05 lot (금 기준)
2. SAR 2마틴 (sar_martin_trader): 1과 같은 거래, 진 금액 / 익절로 회수하는 lot, 2마틴 + 2연속 손절 쉬기 + 매일 초기화
3. H4 추세 (trend_trader)  : 4시간봉 25봉 돌파, 12봉 반대 돌파 청산, 손절 3 ATR, 4시간봉 구름 방향 아니면 그림자(쉬기),
                              1만$ 시작, 거래당 2% 위험 (최소 lot 무시)
4. 아시아 돌파 v2 (asia_breakout_trader): 01:05~08:55 레인지, 09:00~17:55 첫 돌파, 5일 모멘텀, 익절 U / 손절 0.75U,
                              가중치 1.5/0.5, 마틴 3단계, 23:00 청산, 1만$ 시작, 1% x 가중치 / U (최소 lot 무시)
5. 헤지 마틴 (hedge_martin_trader): SAR 단타 신호(뉴스 앞뒤 30분 제외)에 매수·매도 L0 동시, 각자 익절 5$, 손절 없음,
                              SAR 전환마다 2배 (최대 4단계), 사이클 +1% / -30% 전부 청산, 1만$ 시작, L0 1,000$당 0.01
6. 뉴스 양방향 (news_trader): 미국 CPI·고용·FOMC 금리, 30초 전 양방향, 0.05% 움직이면 반대쪽 청산, 5분 무반응 정리,
                              남은 쪽 15분 또는 반대 2$ 되돌림에 정리, 1,000$ 시작, 1,000$당 0.05 lot (금 기준)
수수료·스왑 없음 (스프레드는 틱 그대로). 시간은 모두 서버시간.

사용법: python -m backtests.all_symbols BTCUSD     (종목 하나 -> results/all_symbols/BTCUSD.csv)
        python -m backtests.all_symbols summary    (모아서 results/all_symbols.csv)
"""
import gc
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.download_symbol as ds
import backtests.fast_entry as fe
import backtests.ichimoku_others as io
import backtests.long_period as lb
import backtests.sar_martin as sm
from backtests.add_positions_ticks import hit
from backtests.asia_breakout import ATR_MULT, ENTRY_UNTIL, EXIT_AT, MIN_RANGE_BARS, MOM_DAYS, RANGE_FROM, RANGE_UNTIL
from backtests.asia_v2 import tick_trade
from backtests.ext_filter import features
from backtests.gold_rest_others import resample
from backtests.news_hk import events
from backtests.sar_exit import against_times
from indicators import atr, psar

SYMS = ["XAUUSD", "XAUUSD비례", "XAGUSD", "XAUEUR", "BTCUSD", "XETUSD", "SOLUSD", "XRPUSD", "DOGUSD", "BNBUSD",
        "ADAUSD", "NAS100", "GER40", "JPN225"]
CRYPTO = {"BTCUSD", "XETUSD", "SOLUSD", "XRPUSD", "DOGUSD", "BNBUSD", "ADAUSD"}
REF_PRICE = 4150.0
OUT = Path(__file__).resolve().parent.parent / "results" / "all_symbols"
BARS = lb.TICK_DIR.parent / "bars"
YEARS = range(2022, 2027)
NEWS3 = ("CPI m/m", "실업률", "연방 금리 결정")


# ---------------------------------------------------------------- 데이터
def data_sym(sym):
    return "XAUUSD" if sym.startswith("XAUUSD") else sym


def month_list(sym):
    if data_sym(sym) == "XAUUSD":
        return [(y, m) for y, m in lb.months()
                if (lb.TICK_DIR / f"{y}-{m:02d}.npz").exists() or (lb.TICK_Z_DIR / f"{y}-{m:02d}.npz").exists()]
    return [tuple(map(int, f.stem.split("-"))) for f in sorted((ds.ROOT / sym).glob("*.npz"))]


def clean_mask(t, bid, ask):
    """브로커 틱 오류 거르기 (금은 오류 없어서 안 씀)
    1) 1분 중간값이 앞뒤 하루(1,441분) 중간값에서 40% 넘게 떨어진 분 통째로 (예: SOLUSD 2023-03-02 23시 BTC 가격 1시간)
    2) 스프레드 음수, 또는 앞뒤 약 2,000틱 스프레드 중간값의 10배를 넘으면서 가격의 1%도 넘는 틱
       (예: SOLUSD 2023-03-04 Ask가 Bid의 75배. XRPUSD는 실제 스프레드가 가격의 9%인 때도 있어서 비율만으로는 안 거름)
    3) 앞뒤 약 2,000틱 중간값에서 10% 넘게 튄 틱 (예: BTCUSD 2023-05-19 75,824, XRPUSD 2022-01 아래꼬리 15%)"""
    key = t // 60_000
    mm = pd.Series(bid).groupby(key).median()
    dev = (mm / mm.rolling(1441, center=True, min_periods=1).median() - 1).abs()
    ok = ~np.isin(key, mm.index[dev.values >= 0.4])
    ok &= ask >= bid
    idx = np.flatnonzero(ok)
    b, sp = bid[idx], ask[idx] - bid[idx]
    ref = np.repeat(pd.Series(b[::50]).rolling(41, center=True, min_periods=1).median().values, 50)[:len(b)]
    sp_ref = np.repeat(pd.Series(sp[::50]).rolling(41, center=True, min_periods=1).median().values, 50)[:len(b)]
    ok[idx[(np.abs(b / ref - 1) >= 0.10) | ((sp > 10 * sp_ref) & (sp > 0.01 * b))]] = False
    return ok


def load(sym, y, m):
    if data_sym(sym) == "XAUUSD":
        return lb.load_month(y, m)
    t, bid, ask = ds.load_month(sym, y, m)
    ok = clean_mask(t, bid, ask)
    return t[ok], bid[ok], ask[ok]


def load2(sym, ms, i):
    """i번째 달 + 다음 달 틱 (달을 넘기는 거래용)"""
    t, bid, ask = load(sym, *ms[i])
    if i + 1 < len(ms):
        nt, nb, na = load(sym, *ms[i + 1])
        t, bid, ask = np.concatenate([t, nt]), np.concatenate([bid, nb]), np.concatenate([ask, na])
    return t, bid, ask


def m1_bars(sym):
    """틱 Bid로 만든 1분봉 (data/bars/<종목>_M1_ticks.pkl에 저장해 두고 씀)"""
    sym = data_sym(sym)
    f = BARS / f"{sym}_M1_ticks{'' if sym == 'XAUUSD' else '_clean'}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    parts = []
    for y, m in month_list(sym):
        t, bid, _ = load(sym, y, m)
        key = t // 60_000
        st = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
        parts.append(pd.DataFrame({"k": key[st], "open": bid[st], "high": np.maximum.reduceat(bid, st),
                                   "low": np.minimum.reduceat(bid, st), "close": bid[np.r_[st[1:] - 1, len(bid) - 1]]}))
        del t, bid
    b = pd.concat(parts).groupby("k", sort=True).agg(open=("open", "first"), high=("high", "max"),
                                                      low=("low", "min"), close=("close", "last"))
    b["time"] = pd.to_datetime(b.index * 60, unit="s")
    b = b.reset_index(drop=True)
    BARS.mkdir(parents=True, exist_ok=True)
    b.to_pickle(f)
    return b


def atr_pct_median(m5):
    return float(np.nanmedian(atr(m5).values / m5.close.values))


class Scale:
    """시각 -> 배율 k. 금 그대로면 1"""

    def __init__(self, sym, m5):
        self.fixed = sym == "XAUUSD"
        if self.fixed:
            self.v = 1.0
            return
        gold5 = resample(m1_bars("XAUUSD"), 5)
        self.v = atr_pct_median(m5) / atr_pct_median(gold5)
        self.end = (m5.time + pd.Timedelta(minutes=5)).values
        self.close = m5.close.values

    def at(self, times):
        times = np.asarray(times, "datetime64[ns]")
        if self.fixed:
            return np.ones(len(times))
        i = np.clip(np.searchsorted(self.end, times, side="right") - 1, 0, len(self.close) - 1)
        return self.close[i] / REF_PRICE * self.v


def to_ms(x):
    return np.asarray(x, "datetime64[ms]").astype(np.int64)


def pf(x):
    gl = -x[x < 0].sum()
    return round(x[x > 0].sum() / gl, 2) if gl else float("inf")


def yearly(times, bal, bal0):
    """연도별 수익률 (그 해 마지막 잔고 / 전 해 마지막 잔고)"""
    s = pd.Series(np.asarray(bal), index=pd.DatetimeIndex(times)).groupby(lambda x: x.year).last()
    out, prev = {}, bal0
    for y in YEARS:
        if y in s.index:
            out[str(y)] = f"{(s[y] / prev - 1) * 100:+.0f}%" if prev > 0 else "깡통"
            prev = s[y]
    return out


def mdd(bal, bal0):
    b = np.r_[bal0, np.asarray(bal, float)]
    peak = np.maximum.accumulate(b)
    return round(float(((peak - b) / peak).max()) * 100, 1)


# ---------------------------------------------------------------- 1, 2. SAR 단타 / 2마틴
def scalp_signals(sym, m5, h1, sc):
    fe.START, fe.END = pd.Timestamp("2000-01-01"), pd.Timestamp("2100-01-01")
    sig = fe.bar_close_signals(m5, 5, 0.02, fe.context(m5, h1), True).merge(features(m5), on="t", how="left")
    sig["k"] = sc.at(sig.t.values)
    mins = sig.t.dt.hour * 60 + sig.t.dt.minute
    bad = ((mins >= 15 * 60) & (mins < 21 * 60)) | (sig.flips >= 3) | (sig.atr < 2.5 * sig.k) \
        | ((sig.ref == sig.ref) & (sig.atr > 2.0 * sig.ref)) | (sig.ext > 0.8) | (sig.risk > 2.8)
    if data_sym(sym) not in CRYPTO:
        bad |= (sig.t.dt.weekday == 4) & (mins >= 22 * 60 + 30)
    return sig[~bad].reset_index(drop=True)


def scalp_trades(sym, sig, against):
    rows, busy = [], -1
    sig_ms = to_ms(sig.t.values)
    ms = month_list(sym)
    for i, (y, m) in enumerate(ms):
        s, e = lb.month_range(y, m)
        lo, hi = np.searchsorted(sig_ms, int(s.timestamp() * 1000)), np.searchsorted(sig_ms, int(e.timestamp() * 1000))
        if lo == hi:
            continue
        t, bid, ask = load2(sym, ms, i)
        for r, sms in zip(sig.iloc[lo:hi].itertuples(index=False), sig_ms[lo:hi]):
            if sms < busy:
                continue
            k = int(np.searchsorted(t, sms))
            if k >= len(t) - 1 or t[k] - sms > 600_000:
                continue
            if data_sym(sym) in CRYPTO:
                fc = len(t)
            else:
                d = r.t.normalize() + pd.Timedelta(days=(4 - r.t.weekday()) % 7, hours=23, minutes=30)
                fc = int(np.searchsorted(t, to_ms(np.datetime64(d))))
            side = r.side
            X = bid if side == 1 else ask
            en = ask[k] if side == 1 else bid[k]
            sl_px, tp_px = en - side * 7.5 * r.k, en + side * 3.0 * r.k
            j = hit(X, k + 1, sl_px, tp_px) if side == 1 else hit(X, k + 1, tp_px, sl_px)
            a = against[side]
            ai = np.searchsorted(a, t[k], side="right")
            js = int(np.searchsorted(t, a[ai])) if ai < len(a) else len(t)
            j = min(j, js, fc)
            if j >= len(t):
                continue
            busy = int(t[j])
            oz = (X[j] - en) * side
            mae = float(((X[k:j + 1] - en) * side).min())
            rows.append((pd.Timestamp(int(t[k]), unit="ms"), r.t.date(), oz / r.k, mae / r.k, (ask[k] - bid[k]) / r.k))
        del t, bid, ask
        gc.collect()
    return pd.DataFrame(rows, columns=["open", "day", "oz", "mae", "spread"])


def scalp_rows(tr):
    out = []
    for name, case in (("1. SAR 단타", (0, 1.0, None, False)), ("2. SAR 2마틴", (2, 1.0, None, True, 2))):
        r = sm.sim(tr, *case)
        out.append({"전략": name, "거래": r["거래"], "승률%": round((tr.oz > 0).mean() * 100, 1), "PF": pf(tr.oz),
                    "시작$": 1000, "최종$": r["최종잔고$"], "최대낙폭%": r["최대낙폭%"], "최악의날%": r["최악의날%"],
                    **{y: r.get(y, "") for y in map(str, YEARS)},
                    "비고": f"평균 스프레드 {tr.spread.mean():.2f}$(금 환산)" + (f", 깡통 {r['깡통']}" if r["깡통"] else "")})
    return out


# ---------------------------------------------------------------- 3. H4 추세
def trend_row(sym, m1):
    h4 = resample(m1, 240)
    ent, exit_fn = io.trend_entries(h4)
    sl = (3.0 * atr(h4)).values
    idx = np.array([i for i, _ in ent])
    side = np.array([s for _, s in ent])
    x = io.attach((h4.time + pd.Timedelta(hours=4)).values[idx], h4, 240, h4.close.values[idx])
    keep = np.asarray(io.keep_mask(x, side, io.KINDS[0]))
    tb = to_ms(h4.time.values)
    specs = []
    for (i, s), kp in zip(ent, keep):
        j = exit_fn(i, s)
        if j is None or j + 1 >= len(h4) or i + 1 >= len(h4) or not sl[i] == sl[i]:
            continue
        specs.append((tb[i + 1], s, tb[j + 1], sl[i], kp))
    ms = month_list(sym)
    mi = {ym: n for n, ym in enumerate(ms)}
    cache, rows, busy = {}, [], -1

    def window(a, b):
        ya, yb = pd.Timestamp(a, unit="ms"), pd.Timestamp(b, unit="ms")
        n0 = mi.get((ya.year, ya.month))
        n1 = mi.get((yb.year, yb.month), len(ms) - 1)
        if n0 is None:
            return None
        n1 = min(max(n1, n0) + 1, len(ms) - 1)
        for n in list(cache):
            if n < n0:
                del cache[n]
        for n in range(n0, n1 + 1):
            if n not in cache:
                cache[n] = load(sym, *ms[n])
        parts = [cache[n] for n in range(n0, n1 + 1)]
        return tuple(np.concatenate([p[c] for p in parts]) for c in range(3))

    for en_ms, s, ex_ms, sld, kp in specs:
        if en_ms < busy:
            continue
        w = window(en_ms, ex_ms)
        if w is None:
            continue
        t, bid, ask = w
        k, xk = int(np.searchsorted(t, en_ms)), int(np.searchsorted(t, ex_ms))
        if k >= len(t) - 1 or xk >= len(t):
            continue
        en = ask[k] if s == 1 else bid[k]
        stop = en - s * sld
        price = bid if s == 1 else ask
        seg = price[k + 1:xk]
        h = np.flatnonzero(seg <= stop if s == 1 else seg >= stop)
        j = k + 1 + h[0] if h.size else xk
        busy = int(t[j])
        rows.append((pd.Timestamp(int(t[j]), unit="ms"), (price[j] - en) * s / sld, kp))
    tr = pd.DataFrame(rows, columns=["close", "r", "real"])
    real = tr[tr.real]
    bal = 10_000.0 * np.cumprod(1 + 0.02 * real.r.values)
    return {"전략": "3. H4 추세", "거래": len(real), "승률%": round((real.r > 0).mean() * 100, 1), "PF": pf(real.r),
            "시작$": 10_000, "최종$": round(bal[-1]) if len(bal) else 10_000, "최대낙폭%": mdd(bal, 10_000.0),
            "최악의날%": "", **yearly(real.close, bal, 10_000.0),
            "비고": f"그림자 {(~tr.real).sum()}건 평균 {tr.r[~tr.real].mean():+.2f}R"}


# ---------------------------------------------------------------- 4. 아시아 돌파 v2
def asia_signals(m1):
    m1 = m1.copy()
    m1["day"] = m1.time.dt.date
    d1 = m1.groupby("day").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    pc = d1.close.shift()
    tr = np.maximum(d1.high - d1.low, np.maximum((d1.high - pc).abs(), (d1.low - pc).abs()))
    tr.iat[0] = d1.high.iat[0] - d1.low.iat[0]
    a = tr.ewm(alpha=1 / 14, adjust=False).mean()
    feat = pd.DataFrame({"atr": a.shift(), "mom": np.sign(d1.close - d1.close.shift(MOM_DAYS)).shift(),
                         "ph": d1.high.shift(), "pl": d1.low.shift()})
    m5 = resample(m1, 5)
    m5["day"] = m5.time.dt.date
    m5["mins"] = m5.time.dt.hour * 60 + m5.time.dt.minute
    rows = []
    for day, g in m5.groupby("day"):
        if day not in feat.index or not np.isfinite(feat.atr[day]) or d1.index.get_loc(day) < 30:
            continue
        r = g[(g.mins >= RANGE_FROM) & (g.mins < RANGE_UNTIL)]
        if len(r) < MIN_RANGE_BARS:
            continue
        hi, lo = r.high.max(), r.low.min()
        w = g[(g.mins >= RANGE_UNTIL) & (g.mins < ENTRY_UNTIL)]
        b = w[(w.close > hi) | (w.close < lo)]
        if b.empty:
            continue
        bar = b.iloc[0]
        side = 1 if bar.close > hi else -1
        f = feat.loc[day]
        strong = bar.close > f.ph if side == 1 else bar.close < f.pl
        rows.append((pd.Timestamp(day), bar.time + pd.Timedelta(minutes=5), side, f.mom, f.atr, bool(strong)))
    sig = pd.DataFrame(rows, columns=["day", "t", "side", "mom", "atr", "strong"])
    return sig[sig.side == sig.mom].reset_index(drop=True)


def asia_row(sym, m1):
    sig = asia_signals(m1)
    res = []
    for y, m in month_list(sym):
        g = sig[(sig.day.dt.year == y) & (sig.day.dt.month == m)]
        if g.empty:
            continue
        t, bid, ask = load(sym, y, m)
        for r in g.itertuples():
            u = r.atr * ATR_MULT
            x = tick_trade(t, bid, ask, to_ms(np.datetime64(r.t)), to_ms(np.datetime64(r.day + pd.Timedelta(minutes=EXIT_AT))),
                           r.side, 0.75 * u, u)
            if x is not None:
                res.append((r.day, r.strong, (x[1] - x[0]) * r.side / u))
        del t, bid, ask
        gc.collect()
    tr = pd.DataFrame(res, columns=["day", "strong", "ru"])
    bal = peak = 10_000.0
    level, cyc, out, halted = 0, bal, [], ""
    for r in tr.itertuples():
        if level == 0:
            cyc = bal
        bal += cyc * 0.01 * (1.5 if r.strong else 0.5) * 2 ** level * r.ru
        peak = max(peak, bal)
        out.append((r.day, bal))
        if bal <= peak * 0.5:
            halted = str(r.day.date())
            break
        level = (level + 1 if level + 1 < 3 else 0) if r.ru <= 0 else 0
    d = pd.DataFrame(out, columns=["day", "bal"])
    return {"전략": "4. 아시아 돌파", "거래": len(d), "승률%": round((tr.ru > 0).mean() * 100, 1), "PF": pf(tr.ru),
            "시작$": 10_000, "최종$": round(bal), "최대낙폭%": mdd(d.bal, 10_000.0), "최악의날%": "",
            **yearly(d.day, d.bal, 10_000.0), "비고": f"평균 {tr.ru.mean():+.3f}U" + (f", 낙폭 50% 중단 {halted}" if halted else "")}


# ---------------------------------------------------------------- 5. 헤지 마틴
class Hedge:
    """hedge_martin_ticks.Sim (CONTINUE=True)과 같은 계산. 사이클 시작 때 배율 k로 익절폭 5k, 1 lot = 100/k 단위"""
    TP, TARGET, LIMIT, MAXL = 5.0, 0.01, 0.30, 4

    def __init__(self, sc):
        self.sc = sc
        self.bal = self.peak = 10_000.0
        self.mdd = 0.0
        self.pos, self.level, self.real, self.start_bal = [], -1, 0.0, self.bal
        self.x = self.c = 0.0
        self.cycles = self.limits = 0
        self.dead = None
        self.hist = []

    def lot0(self):
        return max(0.01, np.floor(self.start_bal / 1000 * 0.01 * 100) / 100)

    def on_flip(self, side, k, t, bid, ask, start_ok):
        if self.dead:
            return
        px = ask[k] if side == 1 else bid[k]
        if not self.pos and self.level >= 0:
            if self.level < self.MAXL:
                self.level += 1
                self.pos.append([side, self.lot0() * 2 ** self.level, px])
            return
        if not self.pos:
            if not start_ok:
                return
            kk = float(self.sc.at([np.datetime64(int(t[k]), "ms")])[0])
            self.x, self.c = self.TP * kk, 100.0 / kk
            self.start_bal, self.real, self.level = self.bal, 0.0, 0
            self.pos = [[1, self.lot0(), ask[k]], [-1, self.lot0(), bid[k]]]
            self.cycles += 1
        elif self.level < self.MAXL and not any(p[0] == side for p in self.pos):
            self.level += 1
            self.pos.append([side, self.lot0() * 2 ** self.level, px])

    def end_cycle(self, total, why, ms):
        self.bal = self.start_bal + total
        self.pos, self.real, self.level = [], 0.0, -1
        self.limits += why == "limit"
        self.hist.append((pd.Timestamp(int(ms), unit="ms"), self.bal))
        if self.bal < 100:
            self.dead = str(pd.Timestamp(int(ms), unit="ms").date())

    def run(self, k0, k1, t, bid, ask):
        k = k0
        while self.pos and k < k1:
            n = min(k1, k + 200_000)
            b, a = bid[k:n], ask[k:n]
            lb_ = sum(p[1] for p in self.pos if p[0] == 1)
            ls_ = sum(p[1] for p in self.pos if p[0] == -1)
            cb = sum(p[1] * p[2] for p in self.pos if p[0] == 1)
            cs = sum(p[1] * p[2] for p in self.pos if p[0] == -1)
            eq = self.real + self.c * (lb_ * b - cb + cs - ls_ * a)
            tp = n
            for p in self.pos:
                h = np.flatnonzero(b >= p[2] + self.x) if p[0] == 1 else np.flatnonzero(a <= p[2] - self.x)
                if h.size:
                    tp = min(tp, k + h[0])
            ev = np.flatnonzero((eq >= self.TARGET * self.start_bal) | (eq <= -self.LIMIT * self.start_bal))
            ek = k + ev[0] if ev.size else n
            j = min(tp, ek)
            low = self.start_bal + (eq[: j - k + 1].min() if j > k else eq[0])
            self.peak = max(self.peak, self.bal)
            self.mdd = max(self.mdd, (self.peak - low) / self.peak)
            if j >= n:
                k = n
                continue
            if j == ek and ev.size:
                e = eq[j - k]
                self.end_cycle(e, "limit" if e < 0 else "target", t[j])
                return
            keep = []
            for p in self.pos:
                if (p[0] == 1 and bid[j] >= p[2] + self.x) or (p[0] == -1 and ask[j] <= p[2] - self.x):
                    self.real += p[1] * self.c * self.x
                else:
                    keep.append(p)
            self.pos = keep
            if not self.pos:
                if self.real < self.TARGET * self.start_bal and self.level < self.MAXL:
                    return
                self.end_cycle(self.real, "tp", t[j])
                return
            k = j + 1


def near_news(times, minutes=30):
    ev = events()
    nt = np.sort(to_ms(ev[ev.currency.isin(["USD", "EUR", "GBP"])].t.values))
    tm = to_ms(times)
    j = np.searchsorted(nt, tm)
    w = minutes * 60_000
    before = np.where(j < len(nt), nt[np.minimum(j, len(nt) - 1)] - tm, np.inf)
    after = np.where(j > 0, tm - nt[np.maximum(j - 1, 0)], np.inf)
    return (before <= w) | (after <= w)


def hedge_row(sym, m5, sig, sc):
    up = psar(m5.high.values, m5.low.values)
    i = np.nonzero(up[1:] != up[:-1])[0] + 1
    fl = pd.DataFrame({"ms": to_ms((m5.time.iloc[i] + pd.Timedelta(minutes=5)).values), "side": np.where(up[i], 1, -1)})
    st = sig[~near_news(sig.t.values)]
    key = set(zip(to_ms(st.t.values), st.side))
    fl["start"] = [(a, b) in key for a, b in zip(fl.ms, fl.side)]
    h = Hedge(sc)
    for y, m in month_list(sym):
        s, e = lb.month_range(y, m)
        t, bid, ask = load(sym, y, m)
        f = fl[(fl.ms >= t[0]) & (fl.ms <= t[-1])]
        ks = np.searchsorted(t, f.ms.values)
        prev = 0
        for side, k, ok in zip(f.side.values, ks, f.start.values):
            h.run(prev, k, t, bid, ask)
            if k < len(t):
                h.on_flip(int(side), k, t, bid, ask, bool(ok))
            prev = k
        h.run(prev, len(t), t, bid, ask)
        del t, bid, ask
        gc.collect()
        if h.dead:
            break
    d = pd.DataFrame(h.hist, columns=["t", "bal"])
    wins = int((d.bal.diff().fillna(d.bal - 10_000) > 0).sum()) if len(d) else 0
    return {"전략": "5. 헤지 마틴", "거래": h.cycles, "승률%": round(wins / max(len(d), 1) * 100, 1), "PF": "",
            "시작$": 10_000, "최종$": round(h.bal), "최대낙폭%": round(h.mdd * 100, 1), "최악의날%": "",
            **(yearly(d.t, d.bal, 10_000.0) if len(d) else {}),
            "비고": f"거래 = 사이클 수, 승률 = 사이클 승률, 한도 -30% {h.limits}번" + (f", 깡통 {h.dead}" if h.dead else "")}


# ---------------------------------------------------------------- 6. 뉴스 양방향
def news_one(tt, b, a, ms, th_pct, revert):
    """news_cut.one의 '비율 0.05% / 15분'과 같은 계산 (되돌림 2$ 대신 revert). -> 매수+매도 합 (가격 단위)"""
    mid = (b + a) / 2
    r0 = int(np.searchsorted(tt, ms))
    if r0 >= len(tt) - 1:
        return None
    mr = mid[r0]
    eb, es = a[0], b[0]
    quiet = min(int(np.searchsorted(tt, ms + 300_000)), len(tt) - 1)
    d = mid - mr
    th = mr * th_pct / 100
    after = np.zeros(len(tt), bool)
    after[r0:quiet] = True
    ku, kd = np.flatnonzero(after & (d >= th)), np.flatnonzero(after & (d <= -th))
    ku, kd = (int(ku[0]) if ku.size else None), (int(kd[0]) if kd.size else None)
    if ku is None and kd is None:
        return (b[quiet] - eb) + (es - a[quiet])
    side = 1 if kd is None or (ku is not None and ku <= kd) else -1
    k = ku if side == 1 else kd
    kc = min(int(np.searchsorted(tt, tt[k] + 300)), len(tt) - 1)
    lose = (es - a[kc]) if side == 1 else (b[kc] - eb)
    x, e = (b, eb) if side == 1 else (a, es)
    pnl = (x - e) * side
    tend = min(int(np.searchsorted(tt, ms + 900_000)), len(tt) - 1)
    stop = np.flatnonzero((d[kc:tend + 1] * side) <= -revert)
    if stop.size:
        win = pnl[min(int(np.searchsorted(tt, tt[kc + stop[0]] + 300)), len(tt) - 1)]
    else:
        win = pnl[tend]
    return lose + win


def news_row(sym, sc):
    ev = events()
    key = ev.event.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    ev = ev[(ev.currency == "USD") & key.isin(NEWS3)].reset_index(drop=True)
    ev_ms = to_ms(ev.t.values)
    ks = sc.at(ev.t.values)
    res = []
    for y, m in month_list(sym):
        t, bid, ask = load(sym, y, m)
        for i in np.flatnonzero((ev_ms >= t[0] + 30_000) & (ev_ms < t[-1])):
            ms = int(ev_ms[i])
            k0, k1 = int(np.searchsorted(t, ms - 30_000)), int(np.searchsorted(t, ms + 1_000_000))
            if k0 >= len(t) or t[k0] - (ms - 30_000) > 20_000 or ask[k0] - bid[k0] > 2.0 * ks[i]:
                continue
            r = news_one(t[k0:k1], bid[k0:k1], ask[k0:k1], ms, 0.05 * sc.v, 2.0 * ks[i])
            if r is not None:
                res.append((ev.t.iat[i], r / ks[i]))
        del t, bid, ask
    tr = pd.DataFrame(res, columns=["t", "oz"]).sort_values("t")
    bal = 1000.0 * np.cumprod(1 + tr.oz.values * 0.005)
    return {"전략": "6. 뉴스 양방향", "거래": len(tr), "승률%": round((tr.oz > 0).mean() * 100, 1), "PF": pf(tr.oz),
            "시작$": 1000, "최종$": round(bal[-1]) if len(bal) else 1000, "최대낙폭%": mdd(bal, 1000.0),
            "최악의날%": round(tr.oz.min() * 0.5, 1) if len(tr) else "", **yearly(tr.t, bal, 1000.0),
            "비고": f"발표당 평균 {tr.oz.mean():+.2f}$/oz(금 환산)"}


# ---------------------------------------------------------------- 실행
def run(sym):
    OUT.mkdir(parents=True, exist_ok=True)
    m1 = m1_bars(sym)
    m5, h1 = resample(m1, 5), resample(m1, 60)
    sc = Scale(sym, m5)
    print(f"[{sym}] 1분봉 {len(m1):,}개 {m1.time.iat[0]:%Y-%m-%d} ~ {m1.time.iat[-1]:%Y-%m-%d}, ATR% 비율 {sc.v:.2f}", flush=True)
    rows = []
    sig = scalp_signals(sym, m5, h1, sc)
    tr = scalp_trades(sym, sig, against_times(m5, 5))
    tr.to_csv(OUT / f"{sym}_scalp_trades.csv", index=False, encoding="utf-8-sig")
    rows += scalp_rows(tr)
    print(f"[{sym}] SAR 단타 {len(tr)}건", flush=True)
    rows.append(trend_row(sym, m1))
    print(f"[{sym}] H4 추세 완료", flush=True)
    rows.append(asia_row(sym, m1))
    print(f"[{sym}] 아시아 완료", flush=True)
    rows.append(news_row(sym, sc))
    print(f"[{sym}] 뉴스 완료", flush=True)
    del m1, h1
    gc.collect()
    rows.append(hedge_row(sym, m5, sig, sc))
    print(f"[{sym}] 헤지 마틴 완료", flush=True)
    out = pd.DataFrame(rows)
    out.insert(0, "종목", sym)
    out.insert(1, "ATR%비율", round(sc.v, 2))
    out.to_csv(OUT / f"{sym}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    print(out.to_string(index=False))


def summary():
    d = pd.concat([pd.read_csv(OUT / f"{s}.csv") for s in SYMS if (OUT / f"{s}.csv").exists()])
    d = d.sort_values(["전략", "종목"], key=lambda c: c.map({s: i for i, s in enumerate(SYMS)}) if c.name == "종목" else c)
    d = d.fillna("")
    d.to_csv(OUT.parent / "all_symbols.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 320)
    pd.set_option("display.unicode.east_asian_width", True)
    print(d.drop(columns=["비고"]).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run(sys.argv[1])
