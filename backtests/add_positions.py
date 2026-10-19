"""마틴 신호에 추가 진입(물타기 / 불타기)을 붙이면 나아지는가 (2022-01 ~ 현재, 1분봉 체결)

기준: demo_trader.py 현재 규칙 (gold_rest_others.run_gold와 같은 쉬기 규칙 + 급변동 쉬기:
      5분봉 ATR이 직전 20일(5,520봉) ATR 중앙값의 2배를 넘으면 진입 안 함)
첫 진입은 기준과 같은 신호/같은 lot. 거기에 같은 lot을 추가로 진입:
  물타기: 첫 진입가보다 STEP달러 불리해질 때마다 1개 추가 (최대 UNITS개).
          전체 평균가 + TP달러에서 모두 익절, 첫 진입가 - SL달러에서 모두 손절
  불타기: 첫 진입가보다 STEP달러 유리해질 때마다 1개 추가 (최대 UNITS개).
          첫 진입가 + TP달러에서 모두 익절, 첫 진입가 - SL달러에서 모두 손절
승률 쉬기 기준은 그 방식의 손절/익절로 계산한 손익분기 승률 (손절 / (손절 + 익절)). 1개 기준이라 대략적인 값.
묶음 전체 손익이 +면 익절로 보고 마틴 단계/승률 기록에 사용 (마틴은 묶음 단위: 손절하면 다음 묶음 lot x2)

체결: 추가 진입은 그 가격에 바로 체결 (매수 Ask, 매도 Bid). 1분봉 하나 안에서는
      손절 먼저 확인 -> 추가 진입 -> 익절 순서. 물타기는 추가 진입한 봉에서는 새 평균가 익절을 보지 않음 (보수적)
      금요일 23:30 보유분은 시가 청산. 스프레드는 1분봉 기록값, 수수료/스왑 없음.
lot: 그날 시작 잔고 1,000달러당 0.03 lot x 2^단계 (시작 lot의 5배까지), 1개당 lot이 1.0을 넘으면 진입 안 함.

사용법: python -m backtests.add_positions   (data/bars/XAUUSD_M1_*.csv.zip 필요)
"""
import numpy as np
import pandas as pd

from backtests.gold_rest_others import (FRIDAY_NO_ENTRY, MAX_H1_FLIPS, MIN_ATR, NO_ENTRY_FROM, NO_ENTRY_UNTIL,
                                        WINRATE_MIN, WINRATE_N, Market, h1_flips_by_time, load_m1, signal_table)
from indicators import atr, rsi

CONTRACT = 100
SL, TP = 7.5, 3.0
LOT_PER_1000, MAX_LEVEL, MAX_LOT_MULT, MAX_LOT = 0.03, 4, 5, 1.0
MAX_ATR_MULT, ATR_REF_BARS, ATR_REF_MIN = 2.0, 20 * 276, 2000
STARTS = (pd.Timestamp("2022-01-01"), pd.Timestamp("2025-01-01"))
BAL0 = 1000.0

# 이름: (방식, 추가 간격 $, 최대 개수, 손절 $(첫 진입가 기준), 익절 $)
# 이름: (방식, 추가 간격 $, 최대 개수, 손절 $(첫 진입가 기준), 익절 $, 불타기 RSI 조건)
# RSI 조건 r: 신호 봉 RSI가 매수 50+r 이상 / 매도 50-r 이하일 때만 추가 진입 (아니면 1개만)
VARIANTS = {
    "기준 (지금 규칙, 1개)": ("none", 0, 1, SL, TP, 0),
    "불타기 1$마다 3개 (조건 없음)": ("pyr", 1.0, 3, SL, TP, 0),
    "불타기 + RSI 60/40": ("pyr", 1.0, 3, SL, TP, 10),
    "불타기 + RSI 65/35": ("pyr", 1.0, 3, SL, TP, 15),
    "불타기 + RSI 70/30": ("pyr", 1.0, 3, SL, TP, 20),
}


def basket(mk, k, side, mode, step, units, sl, tp):
    """k번째 1분봉 시가에 첫 진입 -> (청산 idx, 진입가 목록, 청산가). 손익 $/oz(1개당 lot 기준) = sum((px-e)*side)"""
    o, h, l, sp, n = mk.o, mk.h, mk.l, mk.sp, len(mk.t)
    first = o[k] + (sp[k] if side == 1 else 0.0)
    entries = [first]
    sl_px = first - side * sl
    for j in range(k, n):
        if j > k and mk.fri_close[j]:
            return j, entries, o[j] + (0.0 if side == 1 else sp[j])
        # 이 봉에서 매수 청산은 Bid(l~h), 매도 청산은 Ask(l+sp~h+sp)
        lo_x, hi_x = (l[j], h[j]) if side == 1 else (l[j] + sp[j], h[j] + sp[j])
        # 추가 진입 가격대: 매수는 Ask, 매도는 Bid
        lo_e, hi_e = (l[j] + sp[j], h[j] + sp[j]) if side == 1 else (l[j], h[j])
        bad_x = lo_x if side == 1 else hi_x          # 이 봉에서 가장 불리한 청산 가격
        good_x = hi_x if side == 1 else lo_x
        if (bad_x - sl_px) * side <= 0:
            op = o[j] + (0.0 if side == 1 else sp[j])
            px = min(sl_px, op) if side == 1 else max(sl_px, op)
            return j, entries, px
        added = False
        while len(entries) < units:
            lv = first + side * step * len(entries) * (-1 if mode == "avg" else 1)
            reach = (lo_e <= lv) if (side == 1) == (mode == "avg") else (hi_e >= lv)
            if not reach:
                break
            entries.append(lv)
            added = True
        if mode == "avg":
            if added:
                continue
            tp_px = np.mean(entries) + side * tp
        else:
            tp_px = first + side * tp
        if (good_x - tp_px) * side >= 0:
            return j, entries, tp_px
    return None


def signals_with_rest(m1):
    sig, m5, h1 = signal_table(m1)
    a5 = pd.Series(atr(m5).values)
    ref = a5.rolling(ATR_REF_BARS, min_periods=ATR_REF_MIN).median()
    end = m5.time + pd.Timedelta(minutes=5)
    r = pd.merge_asof(sig[["t"]], pd.DataFrame({"end": end, "ref": ref.values}), left_on="t", right_on="end",
                      direction="backward")
    sig["ref"] = r.ref.values
    rs = pd.merge_asof(sig[["t"]], pd.DataFrame({"end": end, "rsi": rsi(m5.close).values}), left_on="t", right_on="end",
                       direction="backward")
    sig["rsi"] = rs.rsi.values
    fl = pd.merge_asof(sig[["t"]], h1_flips_by_time(h1)[["end", "flips", "day"]], left_on="t", right_on="end",
                       direction="backward")
    sig["flips"] = np.where(fl.day.values == sig.t.dt.date.values, fl.flips.fillna(0).values, 0)
    return sig


def floor_lot(x):
    return max(0.01, np.floor(x * 100 + 1e-9) / 100)


def run(mk, sig, var, start):
    mode, step, units, sl, tp, rsi_min = var
    wr_min = WINRATE_MIN if mode == "none" else 100 * sl / (sl + tp)     # 승률 쉬기 기준 = 1개 기준 손익분기 승률
    rows, results = [], []
    busy = np.datetime64("1970-01-01")
    bal, peak, mdd = BAL0, BAL0, 0.0
    day, level, stopped, base = None, 0, False, 0.0
    for t, side, a, ref, nfl, r in sig[["t", "side", "atr", "ref", "flips", "rsi"]].itertuples(index=False):
        if t < start:
            continue
        t64 = np.datetime64(t)
        if t64 < busy:
            continue
        if t.date() != day:
            day, level, stopped = t.date(), 0, False
            base = floor_lot(bal / 1000 * LOT_PER_1000)
        mins = t.hour * 60 + t.minute
        if stopped or (t.weekday() == 4 and mins >= FRIDAY_NO_ENTRY) or NO_ENTRY_FROM <= mins < NO_ENTRY_UNTIL \
                or nfl >= MAX_H1_FLIPS or a < MIN_ATR or (ref == ref and a > MAX_ATR_MULT * ref):
            continue
        last = results[-WINRATE_N:]
        paused = len(last) >= WINRATE_N and np.mean(last) * 100 < wr_min
        lot = round(min(base * 2 ** level, base * MAX_LOT_MULT), 2)
        if not paused and lot > MAX_LOT:
            continue
        k = int(np.searchsorted(mk.t, t64))
        if k >= len(mk.t) or mk.t[k] - t64 > np.timedelta64(10, "m"):
            continue
        n_max = units if (r - 50) * side >= rsi_min else 1
        res = basket(mk, k, side, mode, step, n_max, sl, tp)
        if res is None:
            break
        j, entries, px = res
        busy = mk.t[j]
        oz = sum((px - e) * side for e in entries)
        win = oz > 0
        results.append(win)
        if paused:
            continue
        profit = oz * lot * CONTRACT
        bal += profit
        peak, mdd = max(peak, bal), max(mdd, (peak - bal) / peak)
        rows.append((mk.t[k], mk.t[j], day, side, len(entries), level, lot, oz, profit, bal))
        if bal <= 0:
            break
        if win:
            level = 0
        elif level >= MAX_LEVEL:
            stopped = True
        else:
            level += 1
    tr = pd.DataFrame(rows, columns=["open", "close", "day", "side", "units", "level", "lot", "oz", "profit", "bal"])
    return tr, mdd


def summary(tr, mdd):
    daily = tr.groupby("day").profit.sum()
    gp, gl = tr.profit[tr.profit > 0].sum(), -tr.profit[tr.profit < 0].sum()
    yr = pd.to_datetime(tr.close).dt.year
    out, prev = {}, BAL0
    for y, g in tr.groupby(yr):
        out[f"{y}"] = f"{(g.bal.iat[-1] / prev - 1) * 100:+.0f}%"
        prev = g.bal.iat[-1]
    return {"거래(묶음)": len(tr), "승률%": round((tr.oz > 0).mean() * 100, 1),
            "평균 개수": round(tr.units.mean(), 2), "PF": round(gp / gl, 2) if gl else np.inf,
            "최종잔고$": round(tr.bal.iat[-1]), "최대낙폭%": round(mdd * 100, 1),
            "최악의날$": round(daily.min()), "최악의날%": round((daily / (tr.groupby("day").bal.last() - daily)).min() * 100, 1),
            **out}


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = load_m1("XAUUSD")
    mk = Market("XAUUSD", m1)
    sig = signals_with_rest(m1)
    del m1
    rows = []
    for st in STARTS:
        for name, var in VARIANTS.items():
            tr, mdd = run(mk, sig, var, st)
            rows.append({"시작": f"{st:%Y-%m}", "방식": name, **summary(tr, mdd)})
            print(f"완료: {st:%Y-%m} {name}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("add_positions.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
