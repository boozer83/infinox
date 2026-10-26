"""SAR 단타 손절·익절을 ATR에 맞추기 (2026-10-27 사용자 요청: 변동성 큰 장에서 손절이 잦음 -> 쉬지 말고 넓히기)

진입·반대 콩 청산·필터는 지금 봇 그대로 (backtests.sar_martin.trades), 손절·익절만 진입 때 5분봉 ATR로 바꿈
  고정 7.5 / 3          : 지금
  ATR a배 / a/2.5배     : 손익비 2.5:1 유지 (2025~26 ATR 중앙값 약 4.1 -> 7.5는 약 1.8 ATR)
  넓히기만 (기준 ATR r)  : ATR이 r보다 크면 7.5 / 3에 ATR/r을 곱함 (작을 땐 그대로)
  손익비 바꾸기          : 손절 a ATR / 익절 b ATR
위험을 같게 보려고 "같은 위험 $/oz" = 결과 x 7.5 / 그 거래 손절 (손절이 넓으면 lot을 줄였다고 보는 값)도 같이

사용법: python -m backtests.sar_atr_stops 2025 2026     (결과 results/sar_atr_stops.csv)
"""
import sys

import numpy as np
import pandas as pd

import backtests.sar_martin as sm
import backtests.trail_exit as te
import backtests.winrate_ideas as wi
from backtests.gold_rest_others import REF_PRICE
from backtests.strategies import pf

SL0, TP0 = 7.5, 3.0
CUR = {}
SLS = []

VARIANTS = {"고정 7.5 / 3": ("fixed",)}
VARIANTS.update({f"ATR {a}배 / {a / 2.5:.2f}배": ("atr", a, a / 2.5) for a in (1.4, 1.8, 2.2, 2.6, 3.0)})
VARIANTS.update({f"넓히기만 기준 ATR {r}": ("widen", r) for r in (3.0, 4.0, 5.0)})
VARIANTS.update({f"손절 {a} / 익절 {b} ATR": ("atr", a, b) for a, b in ((1.8, 1.0), (1.8, 0.5), (2.5, 0.7), (1.2, 0.7))})


def stops(v, atr):
    if v[0] == "fixed" or not atr == atr:
        return SL0, TP0
    if v[0] == "atr":
        return v[1] * atr, v[2] * atr
    if v[0] == "floor":                               # ATR 배수지만 지금(7.5 / 3)보다 좁히지는 않음
        return max(SL0, v[1] * atr), max(TP0, v[2] * atr)
    k = max(1.0, atr / v[1])
    return SL0 * k, TP0 * k


def run(year, v):
    orig_on, orig_trade = wi.Acc.on_signal, sm.TRADE

    def on_signal(self, s, *a):
        CUR["atr"] = s.atr
        return orig_on(self, s, *a)

    scaled = year < 2025                              # 2022~24: 쉬기 필터용 ATR이 금값 비례로 바뀌어 있음 -> 손절엔 실제 ATR
    def trade(t, bid, ask, k, side, fc, against, o, sig_px):
        a = CUR.get("atr", np.nan)
        te.SL, te.TP = stops(v, a * bid[k] / REF_PRICE if scaled else a)
        res = orig_trade(t, bid, ask, k, side, fc, against, o, sig_px)
        if res is not None:
            SLS.append(te.SL)
        return res

    wi.Acc.on_signal, sm.TRADE = on_signal, trade
    sm.MAE.clear()
    SLS.clear()
    try:
        sm.trades(year, scaled)
    finally:
        wi.Acc.on_signal, sm.TRADE = orig_on, orig_trade
        te.SL, te.TP = SL0, TP0
    tr = pd.read_csv(f"sar_martin_{'scaled' if scaled else 'trades'}_{year}.csv")
    tr["sl"] = SLS
    return tr


def main():
    global VARIANTS
    args = sys.argv[1:]
    out = "sar_atr_stops"
    if args[0] == "pick":                              # pick 2022 2023 2024 -> 고정 + 고른 몇 개
        VARIANTS = {"고정 7.5 / 3": ("fixed",), **{f"손절 {a} / 익절 {b} ATR": ("atr", a, b)
                    for a, b in ((1.4, 1.1), (1.6, 1.0), (1.6, 1.2), (1.8, 1.0), (1.8, 1.2))}}
        out = f"sar_atr_pick_{'_'.join(args[1:])}"
        args = args[1:]
    elif args[0] == "floor":                           # floor 2025 2026 -> max(지금, ATR 배수)
        VARIANTS = {"고정 7.5 / 3": ("fixed",), **{f"최소 지금, 손절 {a} / 익절 {b} ATR": ("floor", a, b)
                    for a, b in ((1.4, 1.1), (1.6, 1.0), (1.6, 1.2), (1.8, 1.0), (1.8, 1.2), (1.6, 0.9), (1.4, 0.9))}}
        out = f"sar_atr_floor_{'_'.join(args[1:])}"
        args = args[1:]
    elif args[0] == "grid":                              # grid 1.6,1.8 2025 2026 -> 손절 a x 익절 0.8~1.4 ATR
        sls = [float(x) for x in args[1].split(",")]
        VARIANTS = {f"손절 {a} / 익절 {b} ATR": ("atr", a, b) for a in sls for b in (0.8, 0.9, 1.0, 1.1, 1.2, 1.4)}
        out = f"sar_atr_grid_{args[1].replace(',', '_')}"
        args = args[2:]
    years = [int(y) for y in args]
    rows, all_tr = [], []
    for name, v in VARIANTS.items():
        trs = []
        for y in years:
            tr = run(y, v)
            tr["year"], tr["rule"] = y, name
            trs.append(tr)
        tr = pd.concat(trs, ignore_index=True)
        all_tr.append(tr)
        tr["norm"] = tr.oz * SL0 / tr.sl
        row = {"규칙": name, "건수": len(tr), "승률%": round((tr.oz > 0).mean() * 100, 1),
               "손절평균": round(tr.sl.mean(), 2), "oz합": round(tr.oz.sum(), 1), "PF": pf(tr.oz),
               "같은위험 합": round(tr.norm.sum(), 1)}
        eq = tr.norm.cumsum()
        row["같은위험 낙폭"] = round((eq.cummax() - eq).max(), 1)
        half = pd.to_datetime(tr.open).dt.to_period("Q").astype(str)
        row["분기 플러스"] = f"{(tr.groupby(half).norm.sum() > 0).sum()}/{half.nunique()}"
        for y in years:
            g = tr[tr.year == y]
            row[f"oz {y}"] = round(g.oz.sum(), 1)
            row[f"같은위험 {y}"] = round(g.norm.sum(), 1)
        rows.append(row)
        print("완료", name, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"results/{out}.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_tr).to_csv(f"results/{out}_trades.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
