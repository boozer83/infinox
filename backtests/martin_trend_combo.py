"""금 마틴(demo_trader.py) + 금 4시간봉 추세추종(trend_trader.py)을 한 계좌에서 같이 돌리면 (2022-01 ~ 현재, 1분봉 체결)

마틴   : gold_rest_others.run_gold와 같음 (쉬기 규칙 전부, 승률 쉬기 중 가상 진입은 돈 없음)
         lot = 그날 시작 잔고 1,000달러당 0.03 lot x 2^단계 (시작 lot의 5배까지, 0.01 단위 내림, 최소 0.01),
         lot이 MAX_LOT(1.0)을 넘으면 그날 신규 진입 정지 (demo_trader.py 안전장치)
추세추종: 4시간봉 종가가 직전 25봉 최고가 위 -> 매수 / 최저가 아래 -> 매도, 12봉 반대 돌파 청산 (다음 봉 시가),
         손절 3 ATR(14), lot = 잔고 x 2% / (손절폭 x 100) (0.01 단위 내림, 최소 0.01, 최대 5)
두 전략은 매직 넘버가 달라서 동시에 포지션을 가질 수 있음. lot은 진입 시점의 잔고(청산된 손익만)로 계산.

스왑: 지금 MT5 값(매수 -78.33 / 매도 +26.13 달러 per lot per 밤, 수요일 3배)을 전 기간에 적용 (과거 금리는 달랐음)
      서버 자정을 넘길 때마다 부과 (토/일 밤 제외)

결과: 마틴만 / 추세만 / 둘 다, 2022-01과 2025-01에서 1,000달러로 시작
      연도별 수익률, 최종 잔고, 최대 낙폭(청산 잔고 기준), 최악의 달, 손실 월 수, 두 전략 월 손익 상관계수
사용법: python -m backtests.martin_trend_combo   (data/bars/XAUUSD_M1_*.csv.zip 필요)
"""
import heapq

import numpy as np
import pandas as pd

from backtests.gold_rest_others import Market, load_m1, resample, run_gold
from backtests.strategies import trend

CONTRACT = 100
SL_USD = 7.5
LOT_PER_1000, MARTIN_MULT, MAX_LOT_MULT, MARTIN_MAX_LOT = 0.03, 2, 5, 1.0
TREND_N, TREND_SL_ATR, TREND_RISK, TREND_MAX_LOT = 25, 3.0, 0.02, 5.0
SWAP_LONG, SWAP_SHORT, TRIPLE_WEEKDAY = -78.33, 26.13, 2      # 달러 / lot / 밤, 수요일(2) 3배
STARTS = (pd.Timestamp("2022-01-01"), pd.Timestamp("2025-01-01"))
BAL0 = 1000.0


def floor_lot(x):
    return max(0.01, np.floor(x * 100 + 1e-9) / 100)


def swap_nights(o, c):
    """o~c 사이에 넘긴 서버 자정 수 (그 밤의 요일이 월~금만, 수요일은 3)"""
    days = pd.date_range(pd.Timestamp(o).normalize(), pd.Timestamp(c).normalize() - pd.Timedelta(days=1))
    return int(sum(3 if d.weekday() == TRIPLE_WEEKDAY else (1 if d.weekday() < 5 else 0) for d in days))


def trend_trades(m1, flat=None):
    """추세추종 거래: (진입, 청산, 방향, 손익 $/oz, 손절폭 $/oz, 스왑 밤 수, 재진입 횟수)

    flat=(청산 시각 분, 재진입 대기 분, 재진입당 추가 비용 $/oz)이면 스왑을 피하려고 매일 그 시각에 청산하고,
    다음 날 장이 열린 뒤 대기 분이 지나면 같은 방향/같은 손절 가격으로 다시 진입 (lot은 처음과 같음).
    그사이 청산 신호가 났거나 재진입 가격이 이미 손절선을 넘었으면 다시 들어가지 않음."""
    h4 = resample(m1, 240)
    mk = Market("XAUUSD", m1)
    t, o, h, l, sp = mk.t, mk.o, mk.h, mk.l, mk.sp
    if flat:
        tt = pd.Series(t)
        cut = ((tt.dt.hour * 60 + tt.dt.minute) >= flat[0]).values
    rows, busy = [], np.datetime64("1970-01-01")
    for s in trend(h4, TREND_N, TREND_SL_ATR):
        te, tx = np.datetime64(int(s["entry"]), "ms"), np.datetime64(int(s["exit"]), "ms")
        if te < busy or tx <= te:
            continue
        k, x = int(np.searchsorted(t, te)), int(np.searchsorted(t, tx))
        if x >= len(t) or k >= x:
            continue
        side = s["side"]
        e = o[k] + (sp[k] if side == 1 else 0.0)
        sl = e - side * s["sl_dist"]
        k0, pnl, rolls = k, 0.0, 0
        while True:
            end = x
            if flat:
                c = np.flatnonzero(cut[k + 1:x])        # 진입 봉 다음부터 청산 시각 봉
                end = k + 1 + c[0] if c.size else x
            hit = (l[k:end] <= sl) if side == 1 else (h[k:end] + sp[k:end] >= sl)
            idx = np.flatnonzero(hit)
            if idx.size:
                j = k + idx[0]
                op = o[j] + (0.0 if side == 1 else sp[j])
                pnl += ((min(sl, op) if side == 1 else max(sl, op)) - e) * side     # 갭이면 시가
                break
            j = end
            pnl += (o[j] + (0.0 if side == 1 else sp[j]) - e) * side
            if j == x:
                break
            # 자정 청산 -> 다음 날 첫 봉 + 대기 분에 재진입
            nxt = (pd.Timestamp(t[j]).normalize() + pd.Timedelta(days=1)).to_datetime64()
            r0 = int(np.searchsorted(t, nxt))
            if r0 >= x:
                break
            r = int(np.searchsorted(t, t[r0] + np.timedelta64(flat[1], "m")))
            if r >= x:
                break
            e = o[r] + (sp[r] if side == 1 else 0.0)
            if (e - sl) * side <= 0:                   # 이미 손절선 너머
                break
            pnl -= flat[2]
            k, rolls = r, rolls + 1
        nights = 0 if flat else swap_nights(t[k0], t[j])
        rows.append((t[k0], t[j], side, pnl, s["sl_dist"], nights, rolls))
        busy = t[j]
    return pd.DataFrame(rows, columns=["open", "close", "side", "pnl_oz", "risk", "nights", "rolls"])


def run(martin, trends, start, use_martin, use_trend):
    """진입/청산 이벤트를 시간순으로 처리하는 한 계좌 시뮬레이션 -> 청산 기록"""
    ev = []
    if use_martin:
        for r in martin[martin.open >= start].itertuples():
            ev.append((r.open, 0, "m", r))
    if use_trend:
        for r in trends[trends.open >= start].itertuples():
            ev.append((r.open, 0, "t", r))
    # 청산 이벤트는 진입 때 lot이 정해진 뒤 추가 (같은 시각이면 청산 먼저: 우선순위 -1)
    heap = [(np.datetime64(a), b, i, c, d) for i, (a, b, c, d) in enumerate(ev)]
    heapq.heapify(heap)
    seq = len(heap)
    bal, out = BAL0, []
    day, level, day_bal, stopped = None, 0, bal, False
    while heap:
        tm, pri, _, kind, r = heapq.heappop(heap)
        if pri == 0:        # 진입
            if kind == "m":
                d = pd.Timestamp(tm).date()
                if d != day:
                    day, level, day_bal, stopped = d, 0, bal, False
                base = floor_lot(day_bal / 1000 * LOT_PER_1000)
                lot = round(min(base * MARTIN_MULT ** level, base * MAX_LOT_MULT), 2)
                if stopped or lot > MARTIN_MAX_LOT:
                    stopped = True
                    continue
                pnl = r.r * SL_USD * lot * CONTRACT
            else:
                lot = min(TREND_MAX_LOT, floor_lot(bal * TREND_RISK / (r.risk * CONTRACT)))
                pnl = r.pnl_oz * lot * CONTRACT
            nights = r.nights if kind == "t" else swap_nights(r.open, r.close)
            pnl += nights * (SWAP_LONG if r.side == 1 else SWAP_SHORT) * lot
            heapq.heappush(heap, (np.datetime64(r.close), -1, seq, kind, (lot, pnl, r)))
            seq += 1
        else:               # 청산
            lot, pnl, rr = r
            bal += pnl
            out.append((tm, kind, lot, pnl, bal))
            if kind == "m":
                if pnl > 0:
                    level = 0
                elif level < 4:
                    level += 1
    return pd.DataFrame(out, columns=["close", "kind", "lot", "pnl", "bal"])


def report(name, d, start):
    if d.empty:
        return {"전략": name}
    eq = np.r_[BAL0, d.bal.values]
    peak = np.maximum.accumulate(eq)
    mdd = ((peak - eq) / peak).max() * 100
    d = d.assign(y=pd.to_datetime(d.close).dt.year, m=pd.to_datetime(d.close).dt.to_period("M"))
    yr, prev = {}, BAL0
    for y, g in d.groupby("y"):
        yr[str(y)] = f"{(g.bal.iat[-1] / prev - 1) * 100:+.0f}%"
        prev = g.bal.iat[-1]
    mon = d.groupby("m").agg(p=("pnl", "sum"), b=("bal", "last"))
    mon_ret = mon.p / (mon.b - mon.p) * 100
    return {"전략": name, **yr, "최종$": f"{d.bal.iat[-1]:,.0f}", "최대낙폭%": round(mdd, 1),
            "최악의달%": round(mon_ret.min(), 1), "손실월": f"{(mon.p < 0).sum()}/{len(mon)}",
            "최저잔고$": f"{eq.min():,.0f}"}


FLAT_CUT = 23 * 60 + 50     # 자정 전 청산 시각 (서버시간)
FLAT_VARIANTS = [(0, 0.0), (15, 0.0), (30, 0.0), (60, 0.0), (0, 0.5), (0, 1.0)]   # (재진입 대기 분, 재진입당 추가 비용)


def compare_flat(m1, martin, hold):
    """추세추종: 자정 넘겨 보유(스왑) vs 매일 23:50 청산 후 다음 날 재진입 (스왑 없음)"""
    pf = lambda x: round(x[x > 0].sum() / -x[x < 0].sum(), 2)
    st = STARTS[0]
    rows = []
    for name, tr in [("보유 (스왑 냄)", hold)] + [
            (f"23:50 청산, 대기 {d}분" + (f", 재진입 비용 +{c}$" if c else ""), trend_trades(m1, (FLAT_CUT, d, c)))
            for d, c in FLAT_VARIANTS]:
        net = tr.pnl_oz + tr.nights * np.where(tr.side == 1, SWAP_LONG, SWAP_SHORT) / CONTRACT
        y = pd.to_datetime(tr.close).dt.year
        row = {"방식": name, "거래": len(tr), "재진입": int(tr.rolls.sum()),
               **{str(k): pf(net[y == k]) for k in sorted(y.unique())}, "전체PF": pf(net)}
        for label, um in (("추세만", False), ("둘 다", True)):
            d = run(martin, tr, st, um, True)
            eq = np.r_[BAL0, d.bal.values]
            row[f"{label} 최종$"] = f"{d.bal.iat[-1]:,.0f}"
            row[f"{label} 낙폭%"] = round(((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100, 1)
        rows.append(row)
    print(f"===== 추세추종 스왑 피하기: 연도별 PF (스왑 포함), {st:%Y-%m}부터 {BAL0:,.0f}$ 계좌 =====")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print("마틴 시뮬레이션...", flush=True)
    gold_tr = run_gold()[0]
    martin = gold_tr[~gold_tr.virtual].reset_index(drop=True)
    print("추세추종 시뮬레이션...", flush=True)
    m1 = load_m1("XAUUSD")
    trends = trend_trades(m1)
    tw = trends.pnl_oz > 0
    print(f"마틴 실제 거래 {len(martin):,}건, 추세추종 {len(trends):,}건 (승률 {tw.mean() * 100:.0f}%, "
          f"평균 보유 {(trends.close - trends.open).mean() / pd.Timedelta(days=1):.1f}일, "
          f"PF(스왑 전) {trends.pnl_oz[tw].sum() / -trends.pnl_oz[~tw].sum():.2f})\n")
    for st in STARTS:
        res = [report("마틴만", run(martin, trends, st, True, False), st),
               report("추세추종만", run(martin, trends, st, False, True), st)]
        both = run(martin, trends, st, True, True)
        res.append(report("둘 다", both, st))
        m = both.assign(m=pd.to_datetime(both.close).dt.to_period("M")).pivot_table(
            index="m", columns="kind", values="pnl", aggfunc="sum").fillna(0)
        corr = m["m"].corr(m["t"]) if {"m", "t"} <= set(m.columns) else np.nan
        print(f"===== {st:%Y-%m}부터 {BAL0:,.0f}$ =====")
        print(pd.DataFrame(res).fillna("").to_string(index=False))
        print(f"(둘 다) 월 손익 상관계수 마틴 vs 추세추종: {corr:.2f}, "
              f"마틴 손실 월 중 추세추종이 수익인 달: {((m['m'] < 0) & (m['t'] > 0)).sum()}/{(m['m'] < 0).sum()}\n")
    compare_flat(m1, martin, trends)


if __name__ == "__main__":
    main()
