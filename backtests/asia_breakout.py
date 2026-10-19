"""asia_breakout_trader.py(텔레그램으로 받은 파일) 규칙 그대로 다시 백테스트 (실제 틱, 2022-01 ~ 현재)

규칙 (서버시간, 하루 최대 1거래):
  아시아 레인지 = 01:05~08:55 5분봉 최고/최저 (50봉 미만이면 거래 없음)
  09:00~17:55 5분봉 종가가 레인지 밖으로 처음 나간 봉 -> 그 봉 끝 첫 틱에 진입 (매수 Ask / 매도 Bid)
  5일 모멘텀: 전일 종가 vs 6거래일 전 종가 방향과 같을 때만 (다르면 그날 끝)
  손절 = 익절 = 0.5 x 일봉 ATR(14, Wilder, 전일까지). 23:00 첫 틱에 남은 포지션 청산
자금관리: 1단계 lot = 사이클 시작 잔고 1% / (손절폭 x 100), 0.01 내림(최소 0.01). 손절이면 x2, 최대 3단계.
  lot > 1.0이면 진입 안 함, 평가금 아닌 잔고가 최고 잔고의 50% 이하로 내려가면 중단.
봉은 data/bars 1분봉(Bid)으로 만듦. 일봉 = 서버 날짜별 1분봉 묶음.

사용법: python -m backtests.asia_breakout
"""
import math

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.gold_rest_others import load_m1, resample
from backtests.m5_ticks import first_hit

CONTRACT = 100
ATR_MULT, MOM_DAYS, MIN_RANGE_BARS = 0.5, 5, 50
RANGE_FROM, RANGE_UNTIL, ENTRY_UNTIL, EXIT_AT = 65, 540, 1080, 1380   # 분
RISK_PCT, MARTIN_STEPS, MAX_LOT, STOP_PCT = 1.0, 3, 1.0, 50.0


def day_signals():
    m1 = load_m1("XAUUSD")
    m1["day"] = m1.time.dt.date
    d1 = m1.groupby("day").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    pc = d1.close.shift()
    tr = np.maximum(d1.high - d1.low, np.maximum((d1.high - pc).abs(), (d1.low - pc).abs()))
    tr.iat[0] = d1.high.iat[0] - d1.low.iat[0]
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    feat = pd.DataFrame({"atr": atr.shift(), "mom": np.sign(d1.close - d1.close.shift(MOM_DAYS)).shift()})
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
        rows.append((pd.Timestamp(day), bar.time + pd.Timedelta(minutes=5), side, feat.mom[day], feat.atr[day],
                     hi - lo))
    return pd.DataFrame(rows, columns=["day", "t", "side", "mom", "atr", "range"])


def tick_trade(t, bid, ask, entry_ms, exit_ms, side, sl_usd):
    """-> (진입가, 청산가, 결과 TP/SL/TIME, 스프레드) 또는 None"""
    k = int(np.searchsorted(t, entry_ms))
    if k >= len(t) or t[k] - entry_ms > 600_000 or t[k] >= exit_ms:
        return None
    X = bid if side == 1 else ask
    e = ask[k] if side == 1 else bid[k]
    sl_px, tp_px = e - side * sl_usd, e + side * sl_usd
    fc = int(np.searchsorted(t, exit_ms))
    j = first_hit(X[:fc], k + 1, *((sl_px, tp_px) if side == 1 else (tp_px, sl_px)))
    if j < 0:
        if fc >= len(t):
            return None
        return e, X[fc], "TIME", ask[k] - bid[k]
    px = X[j]
    return e, px, "TP" if (px - e) * side > 0 else "SL", ask[k] - bid[k]


def simulate(tr, bal0):
    bal = peak = bal0
    level, base, mdd, halted, out = 0, 0.0, 0.0, None, []
    for r in tr.itertuples(index=False):
        if halted:
            break
        if level == 0:
            base = max(0.01, math.floor(bal * RISK_PCT / 100 / (r.sl_usd * CONTRACT) * 100 + 1e-9) / 100)
        lot = round(base * 2 ** level, 2)
        if lot > MAX_LOT:
            continue
        pnl = r.oz * lot * CONTRACT
        bal += pnl
        peak = max(peak, bal)
        mdd = max(mdd, (peak - bal) / peak)
        out.append((r.day, lot, pnl, bal, pnl / (bal - pnl) * 100))
        if bal <= peak * (1 - STOP_PCT / 100):
            halted = r.day
        if pnl > 0:
            level = 0
        else:
            level = level + 1 if level + 1 < MARTIN_STEPS else 0
    o = pd.DataFrame(out, columns=["day", "lot", "pnl", "bal", "pct"])
    y = o.groupby(o.day.dt.year).bal.last()
    res = {"시작$": bal0, "최종$": round(bal), "최대낙폭%": round(mdd * 100, 1), "최악 거래%": round(o.pct.min(), 1),
           "최대 lot": o.lot.max(), "중단": str(halted.date()) if halted is not None else ""}
    prev = bal0
    for yr, v in y.items():
        res[str(yr)] = f"{(v / prev - 1) * 100:+.0f}%"
        prev = v
    return res


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    sig = day_signals()
    print(f"돌파 있는 날 {len(sig)}일, 모멘텀 같은 방향 {(sig.side == sig.mom).sum()}일", flush=True)
    sig = sig[sig.side == sig.mom].reset_index(drop=True)
    rows = []
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        g = sig[(sig.day >= pd.Timestamp(s.date())) & (sig.day < pd.Timestamp(e.date()) + pd.Timedelta(days=1))
                & (sig.day.dt.month == m) & (sig.day.dt.year == y)]
        if g.empty:
            continue
        t, bid, ask = lb.load_month(y, m)
        for r in g.itertuples(index=False):
            ems = int(np.datetime64(r.t, "ms").astype(np.int64))
            xms = int(np.datetime64(r.day + pd.Timedelta(minutes=EXIT_AT), "ms").astype(np.int64))
            sl_usd = r.atr * ATR_MULT
            res = tick_trade(t, bid, ask, ems, xms, r.side, sl_usd)
            if res is None:
                continue
            e_, px, why, sp = res
            oz = (px - e_) * r.side
            rows.append((r.day, r.t, r.side, sl_usd, e_, px, why, sp, oz, oz / sl_usd))
        del t, bid, ask
    tr = pd.DataFrame(rows, columns=["day", "t", "side", "sl_usd", "entry", "exit", "why", "spread", "oz", "R"])
    tr.to_csv("asia_breakout_trades.csv", index=False, encoding="utf-8-sig")
    print(f"\n거래 {len(tr)}건, 승률 {(tr.oz > 0).mean() * 100:.1f}%, 평균 R {tr.R.mean():+.3f}, "
          f"평균 스프레드 {tr.spread.mean():.2f}$, 평균 손절폭 {tr.sl_usd.mean():.1f}$")
    print("결과별:", tr.why.value_counts().to_dict())
    by = tr.groupby(tr.day.dt.year).agg(건수=("R", "size"), 승률=("oz", lambda x: round((x > 0).mean() * 100, 1)),
                                        평균R=("R", lambda x: round(x.mean(), 3)),
                                        t값=("R", lambda x: round(x.mean() / x.std() * np.sqrt(len(x)), 2)),
                                        손절폭=("sl_usd", lambda x: round(x.mean(), 1)))
    print(by.to_string())
    for a, b in (("2022", "2025"), ("2025", "2026-07"), ("2026-07", "2100")):
        g = tr[(tr.day >= a) & (tr.day < b)]
        print(f"  {a} ~ {b}: {len(g)}건 평균 R {g.R.mean():+.3f}")
    print("\n마틴 3단계 (1% 리스크):")
    print(pd.DataFrame([simulate(tr, b) for b in (1000.0, 6000.0)]).to_string(index=False))
    tr25 = tr[tr.day >= "2025"]
    print("\n2025-01부터 시작:")
    print(pd.DataFrame([simulate(tr25, b) for b in (1000.0, 6000.0)]).to_string(index=False))


if __name__ == "__main__":
    main()
