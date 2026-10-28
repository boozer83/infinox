"""PA Master Grade-A (트레이딩뷰 Pine, pine/pa_master_grade_a.pine) 금 백테스트 (2026-10-28 사용자 요청, 실제 틱 체결)

신호: Pine 코드를 그대로 옮김 (장악형 / 핀바 / 샛별·석별 + EMA21·200 가치 구간 또는 BOS, 거래량 = MT5 틱 수)
  pivot은 오른쪽 5봉이 지나야 확정 (Pine ta.pivothigh와 같음, 미래 데이터 안 씀)
주문: 신호 봉 마감 -> 다음 봉 첫 틱 진입 (매수 Ask / 매도 Bid), 손절 = 패턴 끝, 익절 = 신호 봉 종가 ± R x (종가 - 패턴 끝)
  틱으로 먼저 닿는 쪽 (같은 봉 모호함 없음). 다음 봉 시가가 이미 손절 너머면 그 틱에 바로 청산. 한 번에 한 포지션
R = 손익 / (신호 봉 종가 - 패턴 끝). 잔고는 1만$, 거래당 1% 위험 (Pine은 자본 10% 진입이라 거래마다 위험이 다름)
봉: MT5 데모 서버 봉 (서버시간, tick_volume)

사용법: python -m backtests.pa_master
"""
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.strategies import TickStore, pf
from config import CONTRACT

DEMO = "C:/Program Files/Infinox MetaTrader 5 Terminal/terminal64.exe"
START = pd.Timestamp("2022-01-01")


def bars(tf, n):
    mt5.initialize(path=DEMO)
    r = pd.DataFrame(mt5.copy_rates_from_pos("XAUUSD", tf, 0, n))
    r["time"] = pd.to_datetime(r.time, unit="s")
    return r.rename(columns={"tick_volume": "volume"})[["time", "open", "high", "low", "close", "volume"]]


def rma(x, n):
    return pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().values


def pivots(h, l, n=5):
    """Pine ta.pivothigh/low(n, n): 봉 i에서 i-n 봉이 피벗이면 그 값 (아니면 nan)"""
    N = len(h)
    ph, pl = np.full(N, np.nan), np.full(N, np.nan)
    for i in range(2 * n, N):
        c = i - n
        if h[c] > h[c - n:c].max() and h[c] >= h[c + 1:i + 1].max():
            ph[i] = h[c]
        if l[c] < l[c - n:c].min() and l[c] <= l[c + 1:i + 1].min():
            pl[i] = l[c]
    return ph, pl


def signals(b, use_vol=True, vol_mult=1.1):
    o, h, l, c, v = (b[k].values.astype(float) for k in ("open", "high", "low", "close", "volume"))
    N = len(c)
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    a = rma(tr, 14)
    s = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]          # x[k]
    ema_f = b.close.ewm(span=21, adjust=False).mean().values
    ema_m = b.close.ewm(span=200, adjust=False).mean().values
    lo3, hi3 = b.low.rolling(3).min().values, b.high.rolling(3).max().values
    lo4, hi4 = b.low.rolling(4).min().values, b.high.rolling(4).max().values
    vsma = b.volume.rolling(20).mean().values
    has_vol = v > 0
    vol_base = np.where(use_vol & has_vol, v > vsma * vol_mult, True)

    rng = h - l
    body, body_p = np.abs(c - o), np.abs(s(c, 1) - s(o, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        bull_cs = np.where(rng > 0, (c - l) / rng, 0.0)
        bear_cs = np.where(rng > 0, (h - c) / rng, 0.0)

    # ── BOS (상태) ──
    ph, pl = pivots(h, l)
    last_ph = last_pl = np.nan
    trd, last_bos_bar, last_bos_dir, last_choch = 0, 0, 0, 0
    bos_up, bos_dn = np.zeros(N, bool), np.zeros(N, bool)
    x_ph_prev = x_pl_prev = 0.0
    for i in range(N):
        xph, xpl = (0.0 if np.isnan(last_ph) else last_ph), (0.0 if np.isnan(last_pl) else last_pl)
        cross_over = i > 0 and c[i] > xph and c[i - 1] <= x_ph_prev
        cross_under = i > 0 and c[i] < xpl and c[i - 1] >= x_pl_prev
        x_ph_prev, x_pl_prev = xph, xpl
        if not np.isnan(ph[i]):
            if not np.isnan(last_ph):
                trd = 1 if ph[i] > last_ph else -1
            last_ph = ph[i]
        if not np.isnan(pl[i]):
            last_pl = pl[i]
        bu = trd == 1 and not np.isnan(last_ph) and cross_over
        bd = trd == -1 and not np.isnan(last_pl) and cross_under
        cu = bu and (i - last_bos_bar >= 1) and (last_bos_dir != 1 or c[i] > last_ph * 1.001)
        cd = bd and (i - last_bos_bar >= 1) and (last_bos_dir != -1 or c[i] < last_pl * 0.999)
        chu = trd == -1 and not np.isnan(last_ph) and cross_over and (i - last_choch >= 1)
        chd = trd == 1 and not np.isnan(last_pl) and cross_under and (i - last_choch >= 1)
        if cu:
            last_bos_bar, last_bos_dir = i, 1
        if cd:
            last_bos_bar, last_bos_dir = i, -1
        if chu:
            last_choch, trd = i, 1
        if chd:
            last_choch, trd = i, -1
        bos_up[i], bos_dn[i] = cu, cd

    # ── 장악형 ──
    c1, o1 = s(c, 1), s(o, 1)
    bull_eng_raw = (c1 < o1) & (c > o) & (o <= c1) & (c > o1) & (body > body_p) & (body >= a * 0.6) & (body <= a * 2.8) & (bull_cs >= 0.65)
    bear_eng_raw = (c1 > o1) & (c < o) & (o >= c1) & (c < o1) & (body > body_p) & (body >= a * 0.6) & (body <= a * 2.8) & (bear_cs >= 0.65)
    prev_dn = (c1 < s(c, 5)) & (l <= lo3)
    prev_up = (c1 > s(c, 5)) & (h >= hi3)
    eng_vol = vol_base & np.where(use_vol & has_vol, v > s(v, 1), True)
    bull_eng, bear_eng = bull_eng_raw & prev_dn & eng_vol, bear_eng_raw & prev_up & eng_vol

    # ── 핀바 ──
    btop, bbot = np.maximum(o, c), np.minimum(o, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        lw = np.where(rng > 0, (bbot - l) / rng, 0.0)
        uw = np.where(rng > 0, (h - btop) / rng, 0.0)
        bull_pos = np.where(rng > 0, (h - bbot) / rng, 1.0)
        bear_pos = np.where(rng > 0, (btop - l) / rng, 1.0)
    pin_ok = (rng >= a * 0.8) & (rng <= a * 3.0)
    bull_pin = pin_ok & (lw >= 0.6) & (uw <= 0.15) & (bull_pos <= 0.35) & prev_dn & vol_base
    bear_pin = pin_ok & (uw >= 0.6) & (lw <= 0.15) & (bear_pos <= 0.35) & prev_up & vol_base

    # ── 샛별 / 석별 ──
    c2, o2 = s(c, 2), s(o, 2)
    b1, b2 = np.abs(c2 - o2), np.abs(c1 - o1)
    b2t, b2b = np.maximum(o1, c1), np.minimum(o1, c1)
    star_lo = np.minimum(s(l, 2), np.minimum(s(l, 1), l))
    star_hi = np.maximum(s(h, 2), np.maximum(s(h, 1), h))
    v1 = b1 >= s(a, 2) * 0.6
    vs = (b2 <= s(a, 1) * 0.45) & (b2 < b1 * 0.6)
    pos_bull = (b2b <= c2) & (b2t <= c2 + b1 * 0.5)
    pos_bear = (b2t >= c2) & (b2b >= c2 - b1 * 0.5)
    v3_bull = (c > o) & (c >= c2 + b1 * 0.5) & (c > b2t) & (bull_cs >= 0.60)
    v3_bear = (c < o) & (c <= c2 - b1 * 0.5) & (c < b2b) & (bear_cs >= 0.60)
    morning = (c2 < o2) & v1 & vs & pos_bull & v3_bull
    evening = (c2 > o2) & v1 & vs & pos_bear & v3_bear
    sprev_dn = (c2 < s(c, 7)) & (star_lo <= lo4)
    sprev_up = (c2 > s(c, 7)) & (star_hi >= hi4)
    bull_star, bear_star = morning & sprev_dn & vol_base, evening & sprev_up & vol_base

    # ── Grade-A ──
    any_bull = bull_eng | bull_pin | bull_star
    any_bear = bear_eng | bear_pin | bear_star
    pat_lo = np.where(bull_star, star_lo, np.minimum(l, s(l, 1)))
    pat_hi = np.where(bear_star, star_hi, np.maximum(h, s(h, 1)))
    zone_bull = ((pat_lo <= ema_f + a * 0.5) & (c >= ema_f - a * 0.25)) | ((pat_lo <= ema_m + a * 0.5) & (c >= ema_m - a * 0.25))
    zone_bear = ((pat_hi >= ema_f - a * 0.5) & (c <= ema_f + a * 0.25)) | ((pat_hi >= ema_m - a * 0.5) & (c <= ema_m + a * 0.25))
    g_bull = any_bull & ~any_bear & (((c > ema_m) & zone_bull) | bos_up)
    g_bear = any_bear & ~any_bull & (((c < ema_m) & zone_bear) | bos_dn)
    kind = np.where(bull_eng | bear_eng, "장악형", np.where(bull_pin | bear_pin, "핀바", "샛별"))
    out = []
    for i in np.flatnonzero(g_bull & (c > pat_lo)):
        out.append((i, 1, pat_lo[i], c[i], kind[i], bool(bos_up[i])))
    for i in np.flatnonzero(g_bear & (c < pat_hi) & ~(g_bull & (c > pat_lo))):
        out.append((i, -1, pat_hi[i], c[i], kind[i], bool(bos_dn[i])))
    return sorted(out)


def execute(b, sigs, r_mult, store, minutes):
    t_ms = (b.time.values.astype("datetime64[ms]").astype(np.int64))
    step = minutes * 60_000
    rows, busy = [], -1
    for i, side, sl, close, kind, bos in sigs:
        if i + 1 >= len(b) or b.time.iat[i] < START:
            continue
        ent = t_ms[i] + step                                   # 신호 봉 마감 = 다음 봉 시작
        if ent < busy:
            continue
        risk0 = (close - sl) * side
        tp = close + side * r_mult * risk0
        end = ent + 60 * 86_400_000
        try:
            t, bid, ask = store.window(ent, min(end, t_ms[-1] + step))
        except ValueError:
            continue
        k = np.searchsorted(t, ent)
        if k >= len(t) - 1:
            continue
        e = ask[k] if side == 1 else bid[k]
        px = bid if side == 1 else ask
        seg = px[k:]
        hit = (seg <= sl) | (seg >= tp) if side == 1 else (seg >= sl) | (seg <= tp)
        idx = np.flatnonzero(hit)
        if not idx.size:
            continue
        j = k + idx[0]
        pnl = (px[j] - e) * side
        rows.append((pd.Timestamp(int(t[k]), unit="ms"), side, kind, bos, e, pnl, pnl / risk0, risk0))
        busy = int(t[j])
    return pd.DataFrame(rows, columns=["open", "side", "kind", "bos", "entry", "pnl_oz", "r", "risk"])


def size_sim(tr, bal0=10_000.0, risk_pct=0.01):
    bal, peak, mdd = bal0, bal0, 0.0
    for r in tr.itertuples():
        lot = max(0.01, np.floor(bal * risk_pct / (r.risk * CONTRACT) * 100) / 100)
        bal += r.pnl_oz * lot * CONTRACT
        peak, mdd = max(peak, bal), max(mdd, (peak - bal) / peak)
    return bal, mdd


def max_streak(win):
    m = cur = 0
    for w in win:
        cur = 0 if w else cur + 1
        m = max(m, cur)
    return m


def summarize(name, tr):
    bal, mdd = size_sim(tr)
    y = tr.open.dt.year
    row = {"방식": name, "거래": len(tr), "승률%": round((tr.r > 0).mean() * 100, 1), "PF": pf(tr.pnl_oz),
           "평균R": round(tr.r.mean(), 3), "합계R": round(tr.r.sum(), 1), "최대연패": max_streak(tr.r > 0),
           "1만$→(1%위험)": round(bal), "낙폭%": round(mdd * 100, 1)}
    for yr in range(2022, 2027):
        row[f"R{yr % 100}"] = round(tr.r[y == yr].sum(), 1)
    return row


def main():
    store = TickStore()
    rows, keep = [], {}
    cases = [("H1 2R (원본)", mt5.TIMEFRAME_H1, 60, 2.0, True), ("H1 1.5R", mt5.TIMEFRAME_H1, 60, 1.5, True),
             ("H1 3R", mt5.TIMEFRAME_H1, 60, 3.0, True), ("H1 2R 거래량 조건 끔", mt5.TIMEFRAME_H1, 60, 2.0, False),
             ("H4 2R", mt5.TIMEFRAME_H4, 240, 2.0, True), ("M15 2R", mt5.TIMEFRAME_M15, 15, 2.0, True)]
    cache = {}
    for name, tf, mins, rm, vol in cases:
        if tf not in cache:
            cache[tf] = bars(tf, 99_000 if mins == 15 else 40_000)
        b = cache[tf]
        b = b[b.time >= START - pd.Timedelta(days=60)].reset_index(drop=True)
        sig = signals(b, use_vol=vol)
        tr = execute(b, sig, rm, store, mins)
        keep[name] = tr
        rows.append(summarize(name, tr))
        print(f"  {name}: 신호 {len(sig)} 거래 {len(tr)}", flush=True)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.to_string(index=False))
    res.to_csv("results/pa_master.csv", index=False, encoding="utf-8-sig")
    base = keep["H1 2R (원본)"]
    print("\nH1 2R 나눠 보기")
    for col, lab in (("side", {1: "매수", -1: "매도"}), ("kind", None), ("bos", {True: "BOS", False: "EMA 가치구간"})):
        for k, g in base.groupby(col):
            print(f"  {lab[k] if lab else k}: {len(g)}건 승률 {(g.r > 0).mean() * 100:.1f}% 평균R {g.r.mean():+.3f} 합계R {g.r.sum():+.1f}")
    base.to_csv("results/pa_master_trades_h1.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
