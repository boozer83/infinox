"""H4 추세 청산을 더 빨리 하면? (2026-10-28 사용자 질문: 추세 익절이 적다, 고점 근처에서 청산 못 하나)

진입은 지금 봇 그대로 (N25 돌파, 4시간봉 구름 방향만, 손절 3 ATR). 청산만 바꿔 봄:
  기준      : 종가가 직전 12봉 최저가 아래 (매도는 최고가 위)
  N봉 청산  : 12봉 대신 8 / 6봉, 더 기다리면 16 / 20봉
  이익 반납 : 진입 뒤 가장 좋았던 종가에서 m x ATR 되돌리면 청산 (12봉 청산과 먼저 닿는 쪽)
  중간 익절 : 진입가 ± R배 손절폭에 익절 (12봉 청산과 같이)
  GPT 제안 (2026-10-28): 2R에서 절반 청산 + 나머지 12봉 / 종가가 2R 넘은 뒤부터 6봉 청산으로 조이기
1만 달러 2% 위험, 실제 틱 체결 (그림자 포지션은 빼고 구름 반대 신호는 건너뜀 - 비교용)

사용법: python -m backtests.trend_exit
"""
import numpy as np
import pandas as pd

from backtests.gold_rest_others import load_m1, resample
from backtests.ichimoku_others import attach, keep_mask
from backtests.strategies import TickStore, bar_exit_specs, execute, pf
from backtests.trend_deep import size_sim
from indicators import atr

N = 25


def channel_exit(h4, nx):
    xh = h4.high.rolling(nx).max().shift(1).values
    xl = h4.low.rolling(nx).min().shift(1).values
    c = h4.close.values
    ex_long, ex_short = np.flatnonzero(c < xl), np.flatnonzero(c > xh)

    def f(i, side):
        arr = ex_long if side == 1 else ex_short
        p = np.searchsorted(arr, i + 1)
        return arr[p] if p < len(arr) else None
    return f


def giveback_exit(h4, base, m):
    c, a = h4.close.values, atr(h4).values

    def f(i, side):
        j0 = base(i, side)
        end = j0 if j0 is not None else len(c) - 1
        best = c[i]
        for j in range(i + 1, end + 1):
            best = max(best, c[j]) if side == 1 else min(best, c[j])
            if (best - c[j]) * side >= m * a[j]:
                return j
        return j0
    return f


def tighten_exit(h4, base, short, sl, r=2.0):
    c = h4.close.values

    def f(i, side):
        j0 = base(i, side)
        end = j0 if j0 is not None else len(c) - 1
        on = False
        for j in range(i + 1, end + 1):
            if on and short(j - 1, side) == j:
                return j
            if (c[j] - c[i]) * side >= r * sl[i]:
                on = True
        return j0
    return f


def half_tp(specs, store, r=2.0):
    """기준 청산 거래마다 절반은 rR 익절, 절반은 그대로 -> 한 거래로 합침"""
    base = execute(specs, store)
    rows = []
    for b in base.itertuples():
        s = [x for x in specs if x["entry"] <= b.open_ms][-1]
        tp = execute([dict(s, tp_r=r)], store).iloc[0]
        pnl = 0.5 * b.pnl_oz + 0.5 * tp.pnl_oz
        rows.append((b.open_ms, b.close_ms, b.side, pnl, pnl / b.risk, b.risk))
    tr = pd.DataFrame(rows, columns=["open_ms", "close_ms", "side", "pnl_oz", "r", "risk"])
    tr["close"] = pd.to_datetime(tr.close_ms, unit="ms")
    return tr


def main():
    m1 = load_m1("XAUUSD")
    h4 = resample(m1, 240)
    del m1
    hh = h4.high.rolling(N).max().shift(1).values
    ll = h4.low.rolling(N).min().shift(1).values
    c = h4.close.values
    ent = sorted([(i, 1) for i in np.flatnonzero(c > hh)] + [(i, -1) for i in np.flatnonzero(c < ll)])
    idx = np.array([i for i, _ in ent]); side = np.array([s for _, s in ent])
    x = attach((h4.time + pd.Timedelta(hours=4)).values[idx], h4, 240, c[idx])
    ent = [e for e, k in zip(ent, keep_mask(x, side, "구름 위/아래만")) if k]
    sl = (3.0 * atr(h4)).values
    base = channel_exit(h4, 12)
    cases = {"기준 12봉 청산 (지금 봇)": (base, None), "8봉 청산": (channel_exit(h4, 8), None),
             "6봉 청산": (channel_exit(h4, 6), None), "16봉 청산 (더 기다림)": (channel_exit(h4, 16), None),
             "20봉 청산 (더 기다림)": (channel_exit(h4, 20), None)}
    for m in (2.0, 3.0, 4.0):
        cases[f"이익 반납 {m:g} ATR"] = (giveback_exit(h4, base, m), None)
    for r in (1.5, 2.0, 3.0):
        cases[f"중간 익절 {r:g}R"] = (base, r)
    cases["2R에서 절반 청산 + 나머지 12봉"] = (base, "half")
    cases["2R 넘으면 6봉 청산으로 조이기"] = (tighten_exit(h4, base, channel_exit(h4, 6), sl), None)
    store = TickStore()
    rows = []
    for name, (fn, tp) in cases.items():
        specs = bar_exit_specs(h4, ent, fn, sl)
        if tp == "half":
            tr = half_tp(specs, store)
        else:
            for s in specs:
                if tp:
                    s["tp_r"] = tp
            tr = execute(specs, store)
        df, mdd, _ = size_sim(tr, 10_000.0, 0.02)
        y = tr.close.dt.year
        row = {"청산": name, "거래": len(tr), "승률%": round((tr.pnl_oz > 0).mean() * 100, 1), "PF": pf(tr.pnl_oz),
               "평균R": round(tr.r.mean(), 2), "최종$": round(df.bal.iat[-1]), "최대낙폭%": round(mdd * 100, 1)}
        for yr in range(2022, 2027):
            row[f"PF{yr % 100}"] = pf(tr.pnl_oz[y == yr])
        rows.append(row)
        print(f"  {name} 완료", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("results/trend_exit.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250); pd.set_option("display.unicode.east_asian_width", True)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
