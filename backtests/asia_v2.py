"""아시아 돌파 v2 백테스트 (받은 규칙 그대로, 실제 틱, 2022-01 ~ 현재, 1만 달러 시작)

v1(지금 봇)과 다른 점:
  - 신호 강도 가중치: 돌파 봉 종가가 전일 고가 위(매수) / 전일 저가 아래(매도)면 1.5, 아니면 0.5
  - 손절 = 0.75 x U (U = 0.5 x 일봉 ATR), 익절 = 1.0 x U
  - lot = 사이클 시작 잔고 x 1% x 가중치 x 2^(단계-1) / (U x 100), 0.01 내림 (최소 0.01)
  - 결과 0 이하면 다음 단계 (v1은 손절일 때만)
나머지(레인지 01:05~08:55, 09:00~17:55 첫 돌파, 5일 모멘텀, 23:00 청산, 마틴 3단계, lot 1.0 초과면 진입 안 함,
최고 잔고 대비 50% 줄면 중단)는 v1과 같음. 두 변경을 하나씩만 넣은 것도 같이 비교

사용법: python -m backtests.asia_v2
"""
import gc
import math

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.asia_breakout import ATR_MULT, EXIT_AT, MIN_RANGE_BARS, MOM_DAYS, RANGE_FROM, RANGE_UNTIL, ENTRY_UNTIL
from backtests.gold_rest_others import load_m1, resample
from backtests.m5_ticks import first_hit

CONTRACT, BAL0, MAX_LOT, STOP_PCT = 100, 10_000.0, 1.0, 50.0
# 이름: (가중치 사용, 손절 배수(U 대비), 0이면 다음 단계(True) / 손절일 때만(False))
CASES = {"v1 (지금 봇)": (False, 1.0, False), "가중치만": (True, 1.0, True), "손절 0.75U만": (False, 0.75, True),
         "v2 (둘 다)": (True, 0.75, True)}


def day_signals():
    m1 = load_m1("XAUUSD")
    m1["day"] = m1.time.dt.date
    d1 = m1.groupby("day").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    pc = d1.close.shift()
    tr = np.maximum(d1.high - d1.low, np.maximum((d1.high - pc).abs(), (d1.low - pc).abs()))
    tr.iat[0] = d1.high.iat[0] - d1.low.iat[0]
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    feat = pd.DataFrame({"atr": atr.shift(), "mom": np.sign(d1.close - d1.close.shift(MOM_DAYS)).shift(),
                         "ph": d1.high.shift(), "pl": d1.low.shift()})
    m5 = resample(m1, 5)
    del m1
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
    return pd.DataFrame(rows, columns=["day", "t", "side", "mom", "atr", "strong"])


def tick_trade(t, bid, ask, entry_ms, exit_ms, side, sl_usd, tp_usd):
    """-> (진입가, 청산가, 결과) 또는 None"""
    k = int(np.searchsorted(t, entry_ms))
    if k >= len(t) or t[k] - entry_ms > 600_000 or t[k] >= exit_ms:
        return None
    X = bid if side == 1 else ask
    e = ask[k] if side == 1 else bid[k]
    sl_px, tp_px = e - side * sl_usd, e + side * tp_usd
    fc = int(np.searchsorted(t, exit_ms))
    j = first_hit(X[:fc], k + 1, *((sl_px, tp_px) if side == 1 else (tp_px, sl_px)))
    if j < 0:
        if fc >= len(t):
            return None
        return e, X[fc], "TIME"
    px = X[j]
    return e, px, "TP" if (px - e) * side > 0 else "SL"


def outcomes(sig):
    """신호마다 손절 배수별 결과 ($/oz, 이유). 틱은 한 달씩"""
    out = []
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        g = sig[(sig.day.dt.year == y) & (sig.day.dt.month == m)]
        if g.empty:
            continue
        t, bid, ask = lb.load_month(y, m)
        for r in g.itertuples():
            ems = int(np.datetime64(r.t, "ms").astype(np.int64))
            xms = int(np.datetime64(r.day + pd.Timedelta(minutes=EXIT_AT), "ms").astype(np.int64))
            u = r.atr * ATR_MULT
            row = {"i": r.Index}
            for slm in (1.0, 0.75):
                res = tick_trade(t, bid, ask, ems, xms, r.side, slm * u, u)
                row[slm] = None if res is None else ((res[1] - res[0]) * r.side, res[2])
            out.append(row)
        del t, bid, ask
        gc.collect()
    return pd.DataFrame(out).set_index("i")


def simulate(sig, oc, use_w, slm, zero_next):
    bal = peak = BAL0
    level, cyc, halted, out = 0, BAL0, None, []
    for r in sig.itertuples():
        if halted:
            break
        res = oc.at[r.Index, slm]
        if res is None:
            continue
        oz, why = res
        u = r.atr * ATR_MULT
        if level == 0:
            cyc = bal
        w = (1.5 if r.strong else 0.5) if use_w else 1.0
        lot = max(0.01, math.floor(cyc * 0.01 * w * 2 ** level / (u * CONTRACT) * 100 + 1e-9) / 100)
        if lot > MAX_LOT:
            continue
        pnl = oz * lot * CONTRACT
        bal += pnl
        peak = max(peak, bal)
        out.append((r.day, r.strong, level + 1, lot, pnl, bal, why))
        if bal <= peak * (1 - STOP_PCT / 100):
            halted = r.day
        lose = pnl <= 0 if zero_next else why == "SL"
        level = (level + 1 if level + 1 < 3 else 0) if lose else 0
    return pd.DataFrame(out, columns=["day", "strong", "level", "lot", "pnl", "bal", "why"]), halted


def summarize(tr, label, halted):
    eq = tr.bal
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    mon = tr.groupby(tr.day.dt.to_period("M")).bal.last()
    mret = mon / mon.shift().fillna(BAL0) - 1
    win = tr.pnl > 0
    streak = (~win).astype(int).groupby(win.cumsum()).sum().max()
    yrs = (tr.day.iat[-1] - tr.day.iat[0]).days / 365.25
    res = {"방식": label, "최종$": round(eq.iat[-1]), "최대낙폭%": round(dd * 100, 1),
           "월평균%": round(mret.mean() * 100, 2), "최악의 달%": round(mret.min() * 100, 1),
           "거래/년": round(len(tr) / yrs), "승률%": round(win.mean() * 100, 1),
           "강한 승률%": round(win[tr.strong].mean() * 100, 1), "약한 승률%": round(win[~tr.strong].mean() * 100, 1),
           "최장 연패": int(streak), "최대 lot": tr.lot.max(), "중단": str(halted.date()) if halted is not None else ""}
    prev = BAL0
    for y, v in tr.groupby(tr.day.dt.year).bal.last().items():
        res[str(y)] = f"{(v / prev - 1) * 100:+.0f}%"
        prev = v
    return res


def main():
    pd.set_option("display.width", 320)
    pd.set_option("display.unicode.east_asian_width", True)
    sig = day_signals()
    gc.collect()
    print(f"돌파 있는 날 {len(sig)}일, 모멘텀 같은 방향 {(sig.side == sig.mom).sum()}일, "
          f"그중 강한 신호 {(sig.strong & (sig.side == sig.mom)).sum()}일", flush=True)
    sig = sig[sig.side == sig.mom].reset_index(drop=True)
    oc = outcomes(sig)
    rows = []
    for name, (use_w, slm, zero_next) in CASES.items():
        tr, halted = simulate(sig, oc, use_w, slm, zero_next)
        rows.append(summarize(tr, name, halted))
        if name.startswith("v2"):
            tr.to_csv("asia_v2_trades.csv", index=False, encoding="utf-8-sig")
            g = tr.groupby([tr.day.dt.year, "strong"]).pnl.agg(n="size", wr=lambda x: (x > 0).mean() * 100)
            print("v2 연도별 강한/약한 신호 승률")
            print(g.round(1).unstack().to_string())
    res = pd.DataFrame(rows)
    res.to_csv("asia_v2_result.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
