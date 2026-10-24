"""H4 추세 (trend_trader 규칙)를 여러 종목에 한 계좌로 돌리면? (실제 틱, 2022-01 ~ 2026-10)

규칙은 all_symbols.trend_row와 같음: 4시간봉 25봉 돌파, 12봉 반대 돌파 청산, 손절 3 ATR,
4시간봉 구름 위/아래 아니면 그림자 (그림자도 자리는 차지 -> 그동안 진입 안 함). 종목마다 한 번에 1포지션.

스왑: MT5에서 지금 값을 읽어 기간 내내 같은 비율로 가정 (results/trend_portfolio/swaps.csv)
  금속(포인트 방식)  : 지금 가격 대비 비율로 바꿔 그 시점 가격에 적용, 수요일 3배, 주말 없음
  코인(현재가 이자%) : 연 %/360 x 그 시점 가격, 금요일 3배, 토·일 없음 (주 7일치)
  지수(lot당 돈)     : 지금 가격 대비 비율, 금요일 3배
  -> 거래마다 스왑을 R로 바꿔 뺌 (스왑 R = 스왑 금액 / 손절 거리)

포트폴리오: 1만$ 시작, 진입 때 잔고(확정) x 위험% 를 손절 거리로 나눠 수량 (봇과 같음, 최소 lot 무시)
  총 위험 한도: 열린 포지션들 처음 위험 합 + 새 위험 > 한도 x 잔고면 그 진입 건너뜀
  평가: 매일 00시(서버) 1분봉 종가로 열린 포지션 평가 -> 최대낙폭·최악의 달·최장 낙폭 기간

사용법: python -m backtests.trend_portfolio trades XAUUSD   (종목 거래 -> results/trend_portfolio/trades_XAUUSD.csv)
        python -m backtests.trend_portfolio swaps            (MT5 스왑·계약 크기·증거금)
        python -m backtests.trend_portfolio report           (IS/OOS, 상관, 포트폴리오)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.ichimoku_others as io
from backtests.all_symbols import load, m1_bars, month_list, pf, to_ms
from backtests.gold_rest_others import resample
from indicators import atr

SYMS = ["XAUUSD", "XAGUSD", "XAUEUR", "BTCUSD", "XETUSD", "BNBUSD", "SOLUSD", "XRPUSD", "DOGUSD", "ADAUSD",
        "NAS100", "GER40", "JPN225"]
CRYPTO = {"BTCUSD", "XETUSD", "SOLUSD", "XRPUSD", "DOGUSD", "BNBUSD", "ADAUSD"}
CLUSTER = {"XAUUSD": "금속", "XAGUSD": "금속", "XAUEUR": "금속", "NAS100": "지수", "GER40": "지수", "JPN225": "지수"}
OUT = Path(__file__).resolve().parent.parent / "results" / "trend_portfolio"
OOS = pd.Timestamp("2025-01-01")
START = 10_000.0
YEARS = range(2022, 2027)


# ---------------------------------------------------------------- 1. 종목별 거래
def trades(sym):
    """all_symbols.trend_row와 같은 계산, 거래 기록 전부 저장"""
    m1 = m1_bars(sym)
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
        rows.append((sym, pd.Timestamp(int(t[k]), unit="ms"), pd.Timestamp(int(t[j]), unit="ms"), s, en, price[j], sld,
                     (price[j] - en) * s / sld, bool(kp), "손절" if h.size else "청산"))
    tr = pd.DataFrame(rows, columns=["sym", "open", "close", "side", "entry", "exit", "sld", "r", "real", "why"])
    OUT.mkdir(parents=True, exist_ok=True)
    tr.to_csv(OUT / f"trades_{sym}.csv", index=False, encoding="utf-8-sig")
    print(f"[{sym}] 거래 {tr.real.sum()}건 (그림자 {(~tr.real).sum()}), 합계 {tr.r[tr.real].sum():+.1f}R", flush=True)


# ---------------------------------------------------------------- 스왑·계약 (MT5)
def swaps():
    import MetaTrader5 as mt5
    if not mt5.initialize():
        print("MT5 연결 안 됨")
        return
    acc = mt5.account_info()
    rows = []
    for s in SYMS:
        i = mt5.symbol_info(s)
        if i is None:
            continue
        mt5.symbol_select(s, True)
        px = i.ask or i.bid
        if i.swap_mode == 1:   # 포인트: lot당 하룻밤 swap x point x 계약 = 가격 단위로 1단위당 swap x point
            lng, sht = i.swap_long * i.point / px, i.swap_short * i.point / px
        elif i.swap_mode == 5:  # 현재가 기준 연 %
            lng, sht = i.swap_long / 100 / 360, i.swap_short / 100 / 360
        elif i.swap_mode in (2, 3, 4):  # lot당 돈 (지수: 종목/증거금 통화 = 가격 통화로 봄)
            lng, sht = i.swap_long / i.trade_contract_size / px, i.swap_short / i.trade_contract_size / px
        else:
            lng = sht = 0.0
        m1 = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, s, 1.0, px)
        rows.append({"종목": s, "swap_mode": i.swap_mode, "swap_long": i.swap_long, "swap_short": i.swap_short,
                     "3배요일": i.swap_rollover3days, "매수 하룻밤%": lng * 100, "매도 하룻밤%": sht * 100,
                     "매수 연%(x360)": lng * 36000, "매도 연%(x360)": sht * 36000,
                     "계약크기": i.trade_contract_size, "최소lot": i.volume_min, "lot단위": i.volume_step,
                     "최대lot": i.volume_max, "가격": px, "1lot 증거금$": m1, "증거금 통화": i.currency_margin,
                     "계산방식": i.trade_calc_mode, "계좌 레버리지": acc.leverage})
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "swaps.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 320)
    print(d.to_string(index=False))


def rollovers(o, c, triple):
    """o~c 사이 서버 00시를 넘긴 횟수 (토·일 00시는 없음 = 금·토요일 밤, triple 요일 밤은 3번)
    MQL 요일 0=일 ... 5=금 -> 파이썬 weekday (월=0)"""
    d0, d1 = o.normalize(), c.normalize()
    if d1 <= d0:
        return 0
    days = pd.date_range(d0, d1 - pd.Timedelta(days=1), freq="D")   # 그 날 밤을 넘김
    wd = days.weekday.values
    py3 = (triple - 1) % 7
    n = np.where(wd >= 5, 0, 1)   # 토·일 밤은 없음
    n = np.where(wd == py3, 3, n)
    return int(n.sum())


def add_swap(tr, sw):
    """sw: 종목 -> (매수 하룻밤 비율, 매도 하룻밤 비율, 3배요일)"""
    lng, sht, tri = sw[tr.sym.iat[0]]
    nights = np.array([rollovers(o, c, tri) for o, c in zip(tr.open, tr.close)])
    rate = np.where(tr.side == 1, lng, sht)
    tr["nights"] = nights
    tr["swap_r"] = nights * rate * tr.entry / tr.sld
    tr["rn"] = tr.r + tr.swap_r
    return tr


def load_trades(sw):
    out = {}
    for s in SYMS:
        f = OUT / f"trades_{s}.csv"
        if f.exists():
            tr = pd.read_csv(f, parse_dates=["open", "close"])
            out[s] = add_swap(tr, sw)
    return out


def swap_table():
    f = OUT / "swaps.csv"
    if f.exists():
        d = pd.read_csv(f)
        return {r.종목: (r["매수 하룻밤%"] / 100, r["매도 하룻밤%"] / 100, int(r["3배요일"])) for _, r in d.iterrows()}
    print("swaps.csv 없음 -> 가정: 금속 매수 -7%/년 매도 +2%/년, 코인 매수 -20%/년 매도 -10%/년, 지수 매수 -5% 매도 +1%")
    sw = {}
    for s in SYMS:
        if s in CRYPTO:
            sw[s] = (-0.20 / 360, -0.10 / 360, 5)
        elif s in CLUSTER and CLUSTER[s] == "금속":
            sw[s] = (-0.07 / 360, 0.02 / 360, 3)
        else:
            sw[s] = (-0.05 / 360, 0.01 / 360, 5)
    return sw


# ---------------------------------------------------------------- 2. 종목별 IS / OOS
def stats(r):
    r = np.asarray(r, float)
    if not len(r):
        return {"거래": 0, "승률%": "", "PF": "", "평균R": "", "합계R": 0}
    return {"거래": len(r), "승률%": round((r > 0).mean() * 100, 1), "PF": pf(r), "평균R": round(r.mean(), 3),
            "합계R": round(r.sum(), 1)}


def is_oos(T):
    rows = []
    for s, tr in T.items():
        re = tr[tr.real]
        for name, part in (("IS 2022-24", re[re.close < OOS]), ("OOS 2025-26.10", re[re.close >= OOS])):
            st = stats(part.rn)
            rows.append({"종목": s, "구간": name, **st, "스왑 없이 평균R": round(part.r.mean(), 3) if len(part) else "",
                         "스왑R/거래": round(part.swap_r.mean(), 3) if len(part) else "",
                         "평균 보유일": round(((part.close - part.open).dt.total_seconds() / 86400).mean(), 1) if len(part) else "",
                         **{str(y): round(part.rn[part.close.dt.year == y].sum(), 1) for y in YEARS
                            if (name.startswith("IS") and y < 2025) or (name.startswith("OOS") and y >= 2025)}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. 일별 R (평가 포함, 1R 고정)
def daily_close(sym):
    m1 = m1_bars(sym)
    s = m1.set_index("time").close
    return s.groupby(s.index.normalize()).last()


def daily_r(tr, dc):
    """1R 고정 크기로 그 날 손익(R, 평가 포함, 스왑은 청산일에)"""
    days = pd.date_range("2022-01-03", dc.index[-1], freq="D")
    dc = dc.reindex(days).ffill()
    val = pd.Series(0.0, index=days)   # 그 날 끝 누적 R
    for x in tr[tr.real].itertuples():
        d0, d1 = x.open.normalize(), x.close.normalize()
        span = days[(days >= d0) & (days < d1)]
        u = (dc[span].values - x.entry) * x.side / x.sld
        v = pd.Series(0.0, index=days)
        v[span] = u
        v[days >= d1] = x.rn
        val += v
    return val.diff().fillna(val.iloc[0])


def correlations(T):
    D = {}
    for s, tr in T.items():
        D[s] = daily_r(tr, daily_close(s))
    D = pd.DataFrame(D).fillna(0.0)
    W = D.resample("W").sum()
    return D, D.corr().round(2), W.corr().round(2)


# ---------------------------------------------------------------- 4. 포트폴리오
def portfolio(T, dcs, syms, risk, cap=None, start=None, end=None):
    """잔고 기준 위험%. cap: 열린 처음 위험 합 한도 (잔고 대비). 매일 평가"""
    tr = pd.concat([T[s][T[s].real] for s in syms]).sort_values("open")
    if start is not None:
        tr = tr[tr.open >= start]
    if end is not None:
        tr = tr[tr.open < end]
    ev = [(o, 1, i) for i, o in enumerate(tr.open)] + [(c, 0, i) for i, c in enumerate(tr.close)]
    ev.sort()
    tr = tr.reset_index(drop=True)
    d0 = (start or pd.Timestamp("2022-01-03")).normalize()
    d_end = min(dcs[s].index[-1] for s in syms) if end is None else end
    days = pd.date_range(d0, d_end, freq="D")
    dcr = {s: dcs[s].reindex(days).ffill() for s in syms}
    bal, opn, skipped, n, max_open = START, {}, 0, 0, 0.0
    eq = np.empty(len(days))
    e = 0
    for di, day in enumerate(days):
        nxt = day + pd.Timedelta(days=1)
        while e < len(ev) and ev[e][0] < nxt:
            t, kind, i = ev[e]
            e += 1
            if kind == 1:
                rk = risk * bal
                if cap is not None and sum(opn.values()) + rk > cap * bal + 1e-9:
                    skipped += 1
                    continue
                opn[i] = rk
                n += 1
                max_open = max(max_open, sum(opn.values()) / bal)
            elif i in opn:
                bal += opn.pop(i) * tr.rn.iat[i]
        u = 0.0
        for i, rk in opn.items():
            x = tr.iloc[i]
            u += rk * (dcr[x.sym].iat[di] - x.entry) * x.side / x.sld
        eq[di] = bal + u
    s = pd.Series(eq, index=days)
    return s, n, skipped, max_open


def metrics(s, n, skipped, max_open):
    v = s.values
    peak = np.maximum.accumulate(np.r_[START, v])[1:]
    dd = (peak - v) / peak
    # 최장 낙폭 기간 (고점 -> 회복, 끝까지 회복 못 하면 끝까지)
    longest, cur_start = 0, None
    for i in range(len(v)):
        if v[i] < peak[i] - 1e-9:
            if cur_start is None:
                cur_start = i
            longest = max(longest, i - cur_start + 1)
        else:
            cur_start = None
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    mon = s.resample("ME").last()
    mret = mon / mon.shift().fillna(START) - 1
    yr = s.groupby(s.index.year).last()
    out = {"거래": n, "건너뜀": skipped, "최종$": round(v[-1]), "배": round(v[-1] / START, 2),
           "CAGR%": round(((v[-1] / START) ** (1 / yrs) - 1) * 100, 1), "MDD%": round(dd.max() * 100, 1),
           "최악의 달%": round(mret.min() * 100, 1), "최장 낙폭(일)": longest, "최대 열린위험%": round(max_open * 100, 1)}
    prev = START
    for y in YEARS:
        if y in yr.index:
            out[str(y)] = f"{(yr[y] / prev - 1) * 100:+.0f}%"
            prev = yr[y]
    return out


COMBOS = {
    "a 금": ["XAUUSD"],
    "b 금+ETH+BTC": ["XAUUSD", "XETUSD", "BTCUSD"],
    "c 금+ETH+BTC+BNB+XAG": ["XAUUSD", "XETUSD", "BTCUSD", "BNBUSD", "XAGUSD"],
    "d1 금+BTC (묶음당 1)": ["XAUUSD", "BTCUSD"],
    "d2 금+ETH (묶음당 1)": ["XAUUSD", "XETUSD"],
    "d3 금+BTC+NAS100 (묶음당 1)": ["XAUUSD", "BTCUSD", "NAS100"],
    "e 전부 13종목 (선택 없음)": SYMS,
    "e2 XRP 뺀 12종목 (XRP 22-23 틱 이상)": [x for x in SYMS if x != "XRPUSD"],
    "f IS(22-24) PF>1.4로 고른 6종목": ["XAUUSD", "XAUEUR", "BTCUSD", "XETUSD", "BNBUSD", "DOGUSD"],
}
CASES = [(0.02, None), (0.01, None), (0.02, 0.06), (0.01, 0.03), (0.01, 0.04)]


def feasibility(T):
    """1만$ 계좌에서 lot·증거금 (지금 가격·증거금, 손절 거리는 OOS 중간값 % 그대로). XAUEUR은 유로 가격 (약 12% 오차)"""
    f = OUT / "swaps.csv"
    if not f.exists():
        return pd.DataFrame()
    sw = pd.read_csv(f).set_index("종목")
    rows = []
    for s, tr in T.items():
        if s not in sw.index or s in ("JPN225", "GER40"):   # 엔·유로 가격 지수는 환산 안 함 (후보 아님)
            continue
        x = sw.loc[s]
        o = tr[tr.real & (tr.open >= OOS)]
        sl_pct = float((o.sld / o.entry).median())
        px, cs = x["가격"], x["계약크기"]
        mrate = x["1lot 증거금$"] / (cs * px)
        row = {"종목": s, "손절거리 중간값%": round(sl_pct * 100, 2), "계약크기": cs, "가격": px,
               "증거금률%": round(mrate * 100, 2), "최소lot 위험%": round(0.01 * cs * px * sl_pct / 1e4 * 100, 3)}
        for r in (0.01, 0.02):
            lot = r * 1e4 / (sl_pct * px * cs)
            row[f"{r:.0%} lot"] = round(lot, 3)
            row[f"{r:.0%} 명목$"] = round(r * 1e4 / sl_pct)
            row[f"{r:.0%} 증거금$"] = round(r * 1e4 / sl_pct * mrate)
            row[f"{r:.0%} 내림 위험오차%"] = round((np.floor(lot * 100 + 1e-9) / 100 / lot - 1) * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def report():
    pd.set_option("display.width", 340)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.unicode.east_asian_width", True)
    sw = swap_table()
    T = load_trades(sw)
    print(f"종목: {list(T)}")

    t1 = is_oos(T)
    t1.to_csv(OUT / "is_oos.csv", index=False, encoding="utf-8-sig")
    print("\n=== 1. 종목별 IS(2022-24) / OOS(2025-01~2026-10), 스왑 포함 R ===")
    print(t1.to_string(index=False))

    D, cd, cw = correlations(T)
    D.to_csv(OUT / "daily_r.csv", encoding="utf-8-sig")
    cd.to_csv(OUT / "corr_daily.csv", encoding="utf-8-sig")
    cw.to_csv(OUT / "corr_weekly.csv", encoding="utf-8-sig")
    print("\n=== 2. 일별 R 상관 ===")
    print(cd.to_string())
    print("\n=== 2. 주별 R 상관 ===")
    print(cw.to_string())

    fz = feasibility(T)
    fz.to_csv(OUT / "margin_lots.csv", index=False, encoding="utf-8-sig")
    print("\n=== 4. 1만$ 계좌 lot·증거금 (지금 증거금률, 레버리지 계좌 기준) ===")
    print(fz.to_string(index=False))

    dcs = {s: daily_close(s) for s in T}
    rows, curves = [], {}
    for name, syms in COMBOS.items():
        syms = [s for s in syms if s in T]
        cases = CASES + ([(0.005, 0.03)] if len(syms) > 5 else [])
        for risk, cap in cases:
            s, n, sk, mo = portfolio(T, dcs, syms, risk, cap)
            curves[f"{name} | {risk:.0%} | {cap if cap is None else f'{cap:.0%}'}"] = s
            m = metrics(s, n, sk, mo)
            # OOS만 따로 (2025-01 1만$ 새로 시작)
            so, no, sko, moo = portfolio(T, dcs, syms, risk, cap, start=OOS)
            mo_ = metrics(so, no, sko, moo)
            rows.append({"조합": name, "위험%": f"{risk:.1%}", "한도": "없음" if cap is None else f"{cap:.0%}", **m,
                         "OOS 배": mo_["배"], "OOS MDD%": mo_["MDD%"]})
            print(f"  {name} {risk:.1%} {cap} 완료", flush=True)
    t3 = pd.DataFrame(rows)
    t3.to_csv(OUT / "portfolio.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(OUT / "equity_daily.csv", encoding="utf-8-sig")
    print("\n=== 3. 포트폴리오 (1만$, 매일 평가, 스왑 포함) ===")
    print(t3.to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "trades":
        for s in sys.argv[2:]:
            trades(s)
    elif cmd == "swaps":
        swaps()
    else:
        report()
