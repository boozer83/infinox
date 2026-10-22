"""일목 구름 필터를 H4 추세 / 아시아 3마틴(v2)에 써 보기 (실제 틱, 2022-01 ~ 현재)

일목 (9, 26, 52), 신호가 나온 시점에 완성된 봉 기준:
  구름 위/아래 : 매수는 종가가 구름 위, 매도는 아래일 때만
  구름 안 X    : 종가가 구름 안이면 건너뜀
  구름 색      : 매수는 미래 구름 양운(선행1 > 선행2), 매도는 음운일 때만
  전환 > 기준  : 매수는 전환선 > 기준선, 매도는 반대일 때만
H4 추세: 4시간봉 / 일봉 일목. 지금 봇 규칙 (N25 진입, 12봉 청산, 손절 3 ATR), 1만 달러 2% 위험
아시아 3마틴 v2: 1시간봉 / 일봉 일목. backtests.asia_v2와 같은 규칙, 1만 달러

사용법: python -m backtests.ichimoku_others
"""
import gc

import numpy as np
import pandas as pd

import backtests.asia_v2 as av
from backtests.gold_rest_others import load_m1, resample
from backtests.ichimoku_test import ichimoku
from backtests.strategies import TickStore, bar_exit_specs, execute, pf
from backtests.trend_deep import size_sim
from indicators import atr

KINDS = ("구름 위/아래만", "구름 안 X", "구름 색 같은 방향만", "전환 > 기준만")


def keep_mask(ich, side, kind):
    """ich: 신호마다 붙인 일목 값(close, tk, fa, fb, ca, cb). True면 진입"""
    top, bot = np.maximum(ich.ca, ich.cb), np.minimum(ich.ca, ich.cb)
    c = ich.close
    if kind == KINDS[0]:
        return np.where(side == 1, c > top, c < bot)
    if kind == KINDS[1]:
        return ~((c <= top) & (c >= bot))
    if kind == KINDS[2]:
        return (ich.fa - ich.fb) * side > 0
    return ich.tk * side > 0


def attach(times, bars, minutes, close):
    """times 시점에 완성된 bars의 일목 값 + 비교할 종가"""
    ich = ichimoku(bars).drop(columns="close")
    ich["end"] = bars.time + pd.Timedelta(minutes=minutes)
    x = pd.merge_asof(pd.DataFrame({"t": times}), ich, left_on="t", right_on="end", direction="backward")
    x["close"] = np.asarray(close, float)
    return x


# ---------------------------------------------------------------- H4 추세
def trend_entries(h4, n=25):
    hh = h4.high.rolling(n).max().shift(1).values
    ll = h4.low.rolling(n).min().shift(1).values
    xh = h4.high.rolling(n // 2).max().shift(1).values
    xl = h4.low.rolling(n // 2).min().shift(1).values
    c = h4.close.values
    ent = sorted([(i, 1) for i in np.flatnonzero(c > hh)] + [(i, -1) for i in np.flatnonzero(c < ll)])
    ex_long, ex_short = np.flatnonzero(c < xl), np.flatnonzero(c > xh)

    def exit_fn(i, side):
        arr = ex_long if side == 1 else ex_short
        p = np.searchsorted(arr, i + 1)
        return arr[p] if p < len(arr) else None
    return ent, exit_fn


def trend_part(m1):
    h4, d1 = resample(m1, 240), resample(m1, 1440)
    ent, exit_fn = trend_entries(h4)
    sl = (3.0 * atr(h4)).values
    idx = np.array([i for i, _ in ent])
    side = np.array([s for _, s in ent])
    t_sig = (h4.time + pd.Timedelta(hours=4)).values[idx]
    feats = {"4시간봉": attach(t_sig, h4, 240, h4.close.values[idx]),
             "일봉": attach(t_sig, d1, 1440, h4.close.values[idx])}
    store = TickStore()
    rows = []
    cases = [("기준 (지금 봇)", np.ones(len(ent), bool))]
    cases += [(f"{tf} {k}", keep_mask(x, side, k)) for tf, x in feats.items() for k in KINDS]
    for name, mask in cases:
        tr = execute(bar_exit_specs(h4, [e for e, m in zip(ent, mask) if m], exit_fn, sl), store)
        df, mdd, _ = size_sim(tr, 10_000.0, 0.02)
        y = tr.close.dt.year
        row = {"방식": name, "거래": len(tr), "승률%": round((tr.pnl_oz > 0).mean() * 100, 1), "PF": pf(tr.pnl_oz),
               "최종$": round(df.bal.iat[-1]), "최대낙폭%": round(mdd * 100, 1)}
        for yr in range(2022, 2027):
            row[f"PF {yr}"] = pf(tr.pnl_oz[y == yr])
        rows.append(row)
        print(f"  H4 추세 {name} 완료", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 아시아 3마틴 v2
def asia_part(m1):
    sig = av.day_signals()
    sig = sig[sig.side == sig.mom]
    h1, d1 = resample(m1, 60), resample(m1, 1440)
    m5 = resample(m1, 5)
    close = pd.merge_asof(sig[["t"]], pd.DataFrame({"end": m5.time + pd.Timedelta(minutes=5), "c": m5.close}),
                          left_on="t", right_on="end", direction="backward").c.values
    del m5
    feats = {"1시간봉": attach(sig.t.values, h1, 60, close), "일봉": attach(sig.t.values, d1, 1440, close)}
    oc = av.outcomes(sig)
    rows = []
    cases = [("기준 v2 (지금 봇)", np.ones(len(sig), bool))]
    cases += [(f"{tf} {k}", keep_mask(x, sig.side.values, k)) for tf, x in feats.items() for k in KINDS]
    for name, mask in cases:
        tr, halted = av.simulate(sig[np.asarray(mask)], oc, True, 0.75, True)
        rows.append(av.summarize(tr, name, halted))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 320)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = load_m1("XAUUSD")
    tr = trend_part(m1)
    gc.collect()
    print("\n=== H4 추세 (1만 달러, 거래당 2% 위험) ===")
    print(tr.to_string(index=False))
    tr.to_csv("ichimoku_trend.csv", index=False, encoding="utf-8-sig")
    asia = asia_part(m1)
    print("\n=== 아시아 3마틴 v2 (1만 달러) ===")
    print(asia.drop(columns=["최대 lot", "중단"]).to_string(index=False))
    asia.to_csv("ichimoku_asia.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
