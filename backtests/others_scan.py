"""데모 전략을 다른 종목에 그대로 써 보기 (실제 틱, 2022-01 ~ 받은 달까지, 해마다 1,000달러 시작)

종목 틱: data/ticks_z/<종목>/YYYY-MM.npz (backtests.download_symbol), 금은 data/ticks_z/YYYY-MM.npz
달러로 정해진 값(SAR 손절 7.5 / 익절 3, ATR 2.5 기준, 뉴스 되돌림 2$)은 종목 배수 f를 곱해서 옮김
  f = 최근 12개월 5분봉 ATR 중앙값(그 종목) / 같은 기간 금 5분봉 ATR 중앙값   (금은 f = 1)
전략 (손익은 위험 대비로 계산해서 종목 계약 크기와 무관)
  H4 추세   : N25 돌파, 12봉 반대 돌파 청산, 손절 3 ATR, 4시간봉 구름 필터(그림자), 거래당 잔고 2% 위험
  아시아 v2 : 레인지 01:05~08:55, 09:00~17:55 첫 돌파, 5일 모멘텀 같은 방향, 손절 0.75U / 익절 1U (U = 0.5 일봉 ATR),
              가중치 1.5 / 0.5, 3단계 마틴(0 이하면 다음 단계), 사이클 시작 잔고 1%
  SAR 단타  : 5분봉 SAR 전환 + RSI 50 + 1시간봉 SAR 방향, 손절 7.5f / 익절 3f, 반대 콩 청산, 미국장(15~21시) 쉼,
              오늘 1시간봉 SAR 전환 3번 이상 쉼, 5분봉 ATR 2.5f 미만 쉼, 금요일 22:30 이후 진입 X / 23:30 청산 (코인 제외)
              손절 1번 = 잔고 3.75% (금 1,000달러당 0.05 lot). 실제 봇의 과열/승률 필터는 빠짐
  뉴스      : 미국 CPI / 실업률 / FOMC 금리. news_cut '비율 0.05% / 15분' 규칙, 2$ 되돌림은 2f, 금 1,000달러당 0.05 lot 비율
헤지 마틴은 금에서도 2022~24 손실이라 뺌

사용법: python -m backtests.others_scan XAUUSD XAGUSD        (결과 others_scan_<종목>.csv)
        python -m backtests.others_scan merge                (받은 결과 합쳐서 표)
"""
import gc
import glob
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.asia_v2 as av
import backtests.long_period as lb
import backtests.news_cut as nc
import backtests.news_hk as nh
from backtests.asia_breakout import ATR_MULT, EXIT_AT, MIN_RANGE_BARS, MOM_DAYS, RANGE_FROM, RANGE_UNTIL, ENTRY_UNTIL
from backtests.download_symbol import ROOT, load_month as sym_month
from backtests.ichimoku_others import attach, keep_mask, trend_entries
from backtests.strategies import bar_exit_specs, execute
from indicators import atr, psar, rsi

BAL0 = 1000.0
YEARS = range(2022, 2027)
CRYPTO = ("BTCUSD", "XETUSD", "XRPUSD")
SAR_SL, SAR_TP, SAR_MIN_ATR, SAR_RISK = 7.5, 3.0, 2.5, 0.0375
NO_FROM, NO_UNTIL, FRI_NO, FRI_CLOSE, MAX_FLIPS = 15 * 60, 21 * 60, 22 * 60 + 30, 23 * 60 + 30, 3


def months_of(sym):
    if sym == "XAUUSD":
        return list(lb.months())
    return sorted(tuple(int(x) for x in Path(f).stem.split("-")) for f in glob.glob(str(ROOT / sym / "*.npz")))


def load(sym, y, m):
    return lb.load_month(y, m) if sym == "XAUUSD" else sym_month(sym, y, m)


class Store:
    """strategies.TickStore와 같은 쓰임새 (종목 지정)"""

    def __init__(self, sym, keep=3):
        self.sym, self.months = sym, months_of(sym)
        self.cache, self.keep, self.last = OrderedDict(), keep, (None, None)

    def _month(self, i):
        if i not in self.cache:
            self.cache[i] = load(self.sym, *self.months[i])
            while len(self.cache) > self.keep:
                self.cache.popitem(last=False)
        self.cache.move_to_end(i)
        return self.cache[i]

    def index_of(self, ms):
        ts = pd.Timestamp(ms, unit="ms")
        key = (ts.year, ts.month)
        return self.months.index(key) if key in self.months else len(self.months) - 1

    def window(self, start_ms, end_ms):
        i0, i1 = self.index_of(start_ms), self.index_of(end_ms)
        i1 = min(max(i1, i0 + 1), len(self.months) - 1)
        if self.last[0] is not None and self.last[0][0] <= i0 and i1 <= self.last[0][1]:
            return self.last[1]
        parts = [self._month(i) for i in range(i0, i1 + 1)]
        arrs = tuple(np.concatenate([p[j] for p in parts]) for j in range(3))
        self.last = ((i0, i1), arrs)
        return arrs


def m1_bars(sym):
    """틱 Bid로 1분봉 (캐시: data/ticks_z/<종목>_M1.parquet 대신 npz)"""
    f = (ROOT / f"{sym}_M1.npz") if sym != "XAUUSD" else (ROOT / "XAUUSD_M1.npz")
    ms = months_of(sym)
    if f.exists():
        z = np.load(f, allow_pickle=False)
        if int(z["n_months"][0]) == len(ms):
            return pd.DataFrame({"time": pd.to_datetime(z["k"] * 60, unit="s"), "open": z["o"], "high": z["h"],
                                 "low": z["l"], "close": z["c"]})
    out = []
    for y, m in ms:
        t, bid, _ = load(sym, y, m)
        # pandas groupby는 한 달 틱(수천만 개)에서 메모리가 모자라서 numpy로 (틱은 시간순)
        k = t // 60_000
        s = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        e = np.r_[s[1:], len(k)] - 1
        out.append(pd.DataFrame({"open": bid[s], "high": np.maximum.reduceat(bid, s),
                                 "low": np.minimum.reduceat(bid, s), "close": bid[e]}, index=k[s]))
        del t, bid, k, s, e
        gc.collect()
    b = pd.concat(out)
    b = b[~b.index.duplicated()]
    np.savez(f, k=b.index.values, o=b.open.values, h=b.high.values, l=b.low.values, c=b.close.values,
             n_months=np.array([len(ms)]))
    return pd.DataFrame({"time": pd.to_datetime(b.index.values * 60, unit="s"), "open": b.open.values,
                         "high": b.high.values, "low": b.low.values, "close": b.close.values})


def resample(m1, minutes):
    g = m1.set_index("time").resample(f"{minutes}min", label="left", closed="left")
    b = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last()})
    return b.dropna().reset_index()


def to_ms(x):
    return np.asarray(x, dtype="datetime64[ms]").astype(np.int64)


def m5_atr_median(m1):
    m5 = resample(m1[m1.time >= m1.time.iat[-1] - pd.Timedelta(days=365)], 5)
    return float(atr(m5).median())


def year_stats(name, days, bal):
    """days: 거래 시각, bal: 거래 뒤 잔고 (그 해 1,000달러 시작)"""
    if not len(bal):
        return {"전략": name, "거래": 0, "수익률%": 0.0, "최대낙폭%": 0.0}
    eq = np.r_[BAL0, bal]
    peak = np.maximum.accumulate(eq)
    return {"전략": name, "거래": len(bal), "수익률%": round((eq[-1] / BAL0 - 1) * 100, 1),
            "최대낙폭%": round(((peak - eq) / peak).max() * 100, 1)}


def compound(r, risk):
    """r: 거래별 위험 대비 손익 배수. 거래마다 잔고 x risk 위험"""
    bal, out = BAL0, []
    for x in r:
        bal = max(bal + bal * risk * x, 0.0)
        out.append(bal)
    return np.array(out)


# ---------------------------------------------------------------- H4 추세
def trend_rows(sym, m1, store):
    h4 = resample(m1, 240)
    ent, exit_fn = trend_entries(h4)
    specs = bar_exit_specs(h4, ent, exit_fn, (3.0 * atr(h4)).values)
    tr = execute(specs, store)
    if not len(tr):
        return []
    spec_t = np.array([s["entry"] for s in specs])
    t_ms = to_ms(h4.time.values)
    k = np.searchsorted(spec_t, tr.open_ms.values, side="right") - 1
    sig_bar = np.searchsorted(t_ms, spec_t[k]) - 1
    t_sig = (h4.time + pd.Timedelta(hours=4)).values[sig_bar]
    ich = attach(t_sig, h4, 240, h4.close.values[sig_bar])
    tr = tr[np.asarray(keep_mask(ich, tr.side.values, "구름 위/아래만"))]
    rows = []
    for y in YEARS:
        g = tr[tr.close.dt.year == y]
        rows.append({"년": y, **year_stats("H4 추세", g.close, compound(g.r.values, 0.02))})
    return rows


# ---------------------------------------------------------------- 아시아 v2
def asia_rows(sym, m1, store):
    m1 = m1.assign(day=m1.time.dt.date)
    d1 = m1.groupby("day").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    pc = d1.close.shift()
    trr = np.maximum(d1.high - d1.low, np.maximum((d1.high - pc).abs(), (d1.low - pc).abs()))
    a = trr.ewm(alpha=1 / 14, adjust=False).mean()
    feat = pd.DataFrame({"atr": a.shift(), "mom": np.sign(d1.close - d1.close.shift(MOM_DAYS)).shift(),
                         "ph": d1.high.shift(), "pl": d1.low.shift()})
    m5 = resample(m1, 5)
    m5["day"], m5["mins"] = m5.time.dt.date, m5.time.dt.hour * 60 + m5.time.dt.minute
    sig = []
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
        if side != f.mom:
            continue
        strong = bar.close > f.ph if side == 1 else bar.close < f.pl
        sig.append((pd.Timestamp(day), bar.time + pd.Timedelta(minutes=5), side, f.atr, bool(strong)))
    sig = pd.DataFrame(sig, columns=["day", "t", "side", "atr", "strong"])
    res = []
    for r in sig.itertuples():
        ems = int(to_ms(r.t))
        xms = int(to_ms(r.day + pd.Timedelta(minutes=EXIT_AT)))
        u = r.atr * ATR_MULT
        t, bid, ask = store.window(ems, xms)
        x = av.tick_trade(t, bid, ask, ems, xms, r.side, 0.75 * u, u)
        res.append(None if x is None else (x[1] - x[0]) * r.side / u)
    sig["ru"] = res
    sig = sig.dropna(subset=["ru"])
    rows = []
    for y in YEARS:
        g = sig[sig.day.dt.year == y]
        bal = peak = BAL0
        level, cyc, out, halted = 0, BAL0, [], False
        for r in g.itertuples():
            if halted:
                break
            if level == 0:
                cyc = bal
            pnl = cyc * 0.01 * (1.5 if r.strong else 0.5) * 2 ** level * r.ru
            bal += pnl
            peak = max(peak, bal)
            out.append(bal)
            halted = bal <= peak * 0.5
            level = (level + 1 if level + 1 < 3 else 0) if pnl <= 0 else 0
        rows.append({"년": y, **year_stats("아시아 3마틴", g.day, np.array(out))})
    return rows


# ---------------------------------------------------------------- SAR 단타
def sar_rows(sym, m1, store, f):
    m5, h1 = resample(m1, 5), resample(m1, 60)
    up5 = psar(m5.high.values, m5.low.values)
    r5 = rsi(m5.close).values
    a5 = atr(m5).values
    end5 = to_ms((m5.time + pd.Timedelta(minutes=5)).values)
    uph = psar(h1.high.values, h1.low.values)
    h1_end = h1.time + pd.Timedelta(hours=1)
    flip = np.r_[False, uph[1:] != uph[:-1]]
    hf = pd.DataFrame({"end": h1_end, "hdir": np.where(uph, 1, -1),
                       "flips": pd.Series(flip).groupby(h1.time.dt.date.values).cumsum().values,
                       "day": h1.time.dt.date.values})
    mm = pd.merge_asof(pd.DataFrame({"end": m5.time + pd.Timedelta(minutes=5)}), hf, on="end", direction="backward")
    m5_day = (m5.time + pd.Timedelta(minutes=5)).dt.date.values
    flips_today = np.where(mm.day.values == m5_day, mm.flips.values, 0)
    flip_bars = np.flatnonzero(up5[1:] != up5[:-1]) + 1          # 이 봉에서 SAR 방향이 바뀜 (봉 끝에 확정)
    sl, tp, min_atr = SAR_SL * f, SAR_TP * f, SAR_MIN_ATR * f
    weekend_close = sym not in CRYPTO
    trades, busy = [], -1
    for i in flip_bars:
        s = 1 if up5[i] else -1
        t0 = int(end5[i])
        if t0 < busy or (r5[i] - 50) * s <= 0 or mm.hdir.values[i] != s:
            continue
        ts = pd.Timestamp(t0, unit="ms")
        mins = ts.hour * 60 + ts.minute
        if NO_FROM <= mins < NO_UNTIL or flips_today[i] >= MAX_FLIPS or not a5[i] >= min_atr:
            continue
        if weekend_close and ts.weekday() == 4 and mins >= FRI_NO:
            continue
        # 반대 콩 = 다음 SAR 전환 봉 끝, 금요일 23:30 (코인 제외)
        p = np.searchsorted(flip_bars, i + 1)
        x_ms = int(end5[flip_bars[p]]) if p < len(flip_bars) else t0 + 7 * 86_400_000
        if weekend_close:
            fri = ts.normalize() + pd.Timedelta(days=(4 - ts.weekday()) % 7, minutes=FRI_CLOSE)
            x_ms = min(x_ms, int(fri.value // 10**6))
        t, bid, ask = store.window(t0, x_ms)
        k = int(np.searchsorted(t, t0))
        if k >= len(t) or t[k] - t0 > 600_000:
            continue
        e = ask[k] if s == 1 else bid[k]
        X = bid if s == 1 else ask
        fc = min(int(np.searchsorted(t, x_ms)), len(t) - 1)
        mv = (X[k + 1:fc + 1] - e) * s
        hit = np.flatnonzero((mv >= tp) | (mv <= -sl))
        j = k + 1 + hit[0] if hit.size else fc
        trades.append((ts, (X[j] - e) * s / sl))
        busy = int(t[j])
    tr = pd.DataFrame(trades, columns=["t", "r"])
    rows = []
    for y in YEARS:
        g = tr[tr.t.dt.year == y]
        rows.append({"년": y, **year_stats("SAR 단타", g.t, compound(g.r.values, SAR_RISK))})
    return rows


# ---------------------------------------------------------------- 뉴스
def news_rows(sym, store, f):
    ev = nh.events()
    ev["key"] = ev.currency + " " + ev.event.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    ev = ev[ev.key.isin(nc.MAJOR3)].reset_index(drop=True)
    ms = set(months_of(sym))
    out = []
    for r in ev.itertuples():
        if (r.t.year, r.t.month) not in ms:
            continue
        e_ms = int(to_ms(r.t))
        t, bid, ask = store.window(e_ms - nc.PRE, e_ms + 1_000_000)
        k0, k1 = int(np.searchsorted(t, e_ms - nc.PRE)), int(np.searchsorted(t, e_ms + 1_000_000))
        if k1 - k0 < 10:
            continue
        x = news_one(t[k0:k1], bid[k0:k1], ask[k0:k1], e_ms, f)
        if x is not None:
            out.append((r.t, x / f))       # 금 기준 $/oz로 환산
    d = pd.DataFrame(out, columns=["t", "oz"])
    rows = []
    for y in YEARS:
        g = d[d.t.dt.year == y]
        bal, eq = BAL0, []
        for x in g.oz:
            bal = max(bal + bal / 1000 * 0.05 * 100 * x, 0.0)
            eq.append(bal)
        rows.append({"년": y, **year_stats("뉴스 양방향", g.t, np.array(eq))})
    return rows


def news_one(tt, b, a, ms, f):
    """news_cut.one ('pct' 0.05, '15분')과 같은 규칙, 되돌림 2$ -> 2f"""
    mid = (b + a) / 2
    r0 = int(np.searchsorted(tt, ms))
    if r0 >= len(tt) - 1:
        return None
    mr = mid[r0]
    eb, es = a[0], b[0]
    quiet = min(int(np.searchsorted(tt, ms + nc.QUIET)), len(tt) - 1)
    after = np.zeros(len(tt), bool)
    after[r0:quiet] = True
    d = mid - mr
    th = mr * 0.05 / 100
    ku, kd = nc.first(after & (d >= th)), nc.first(after & (d <= -th))
    if ku is None and kd is None:
        return (b[quiet] - eb) + (es - a[quiet])
    side = 1 if kd is None or (ku is not None and ku <= kd) else -1
    k = ku if side == 1 else kd
    kc = min(int(np.searchsorted(tt, tt[k] + nc.DELAY)), len(tt) - 1)
    lose = (es - a[kc]) if side == 1 else (b[kc] - eb)
    x, e = (b, eb) if side == 1 else (a, es)
    pnl = (x - e) * side
    tend = min(int(np.searchsorted(tt, ms + 900_000)), len(tt) - 1)
    stop = (d[kc:tend + 1] * side) <= -2.0 * f
    j = nc.first(stop)
    win = pnl[min(int(np.searchsorted(tt, tt[kc + j] + nc.DELAY)), len(tt) - 1)] if j is not None else pnl[tend]
    return lose + win


# ---------------------------------------------------------------- 실행
def run(sym, gold_atr):
    m1 = m1_bars(sym)
    f = 1.0 if sym == "XAUUSD" else m5_atr_median(m1) / gold_atr
    print(f"{sym}: {m1.time.iat[0]:%Y-%m} ~ {m1.time.iat[-1]:%Y-%m}, 배수 f = {f:.5g}", flush=True)
    store = Store(sym)
    rows = []
    for name, fn in (("H4", lambda: trend_rows(sym, m1, store)), ("아시아", lambda: asia_rows(sym, m1, store)),
                     ("SAR", lambda: sar_rows(sym, m1, store, f)), ("뉴스", lambda: news_rows(sym, store, f))):
        rows += fn()
        print(f"  {sym} {name} 완료", flush=True)
        gc.collect()
    d = pd.DataFrame(rows)
    d.insert(0, "종목", sym)
    d["f"] = round(f, 6)
    d.to_csv(f"others_scan_{sym}.csv", index=False, encoding="utf-8-sig")
    return d


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    if sys.argv[1:] == ["merge"]:
        d = pd.concat([pd.read_csv(x) for x in sorted(glob.glob("others_scan_*.csv"))], ignore_index=True)
        print(d.pivot_table(index=["전략", "종목"], columns="년", values="수익률%").round(1).to_string())
        return
    gold_atr = m5_atr_median(m1_bars("XAUUSD"))
    for sym in sys.argv[1:]:
        print(run(sym, gold_atr).pivot(index="전략", columns="년", values="수익률%").to_string(), flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
