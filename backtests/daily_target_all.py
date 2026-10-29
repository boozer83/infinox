"""전략 합산 하루 목표: 그날 실현 손익 합이 목표%에 닿으면 모든 전략 신규 진입 쉼, 들고 있는 건 그대로 (2026-10-29 사용자 아이디어)

대상: 지금 실계좌에서 도는 핵심 3개 (거래 목록은 각 백테스트 그대로, 2022-03 ~ 2026-10)
  SAR 단타   : results/sar_us_now_YYYY.csv (진입 시각, 보유 분, $/oz), 1,000$당 0.05 lot -> 손익% = oz x 0.5%
  아시아 v2  : asia_rest.day_signals + 틱 체결 (손절 0.75U / 익절 1U / 23:00), 1% x 가중 1.5/0.5 (마틴 뺌)
               -> 손익% = (oz / U) x 1% x 가중
  추세 4시간 : strategies.trend H4 N25 3ATR, 1분봉 체결 (idea_deep.execute_m1), 2% 위험 -> 손익% = R x 2%
  (구름 필터·마틴은 빼서 실제 봇과 조금 다름. 복리 없이 % 단순 합)
규칙: 서버 날짜마다 진입 시각 전에 청산된 거래의 손익% 합이 목표 이상이면 그 진입은 건너뜀
  (건너뛴 거래 대신 다른 신호가 생기는 효과는 무시 -> 단타는 실제로는 조금 다름)
목표: 0.5 / 1 / 1.5 / 2 / 3 / 5%
결과 results/daily_target_all.csv

사용법: python -m backtests.daily_target_all
"""
import gc
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.asia_breakout import ATR_MULT, EXIT_AT
from backtests.asia_rest import day_signals
from backtests.idea_deep import execute_m1
from backtests.m5_ticks import first_hit
from backtests.scalp_ideas import m5_bars
from backtests.strategies import resample, trend

RES = Path(__file__).resolve().parent.parent / "results"
START = pd.Timestamp("2022-03-01")
TARGETS = [None, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def sar_trades():
    tr = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES / "sar_us_now_*.csv")))])
    o = pd.to_datetime(tr.open)
    return pd.DataFrame({"bot": "SAR", "open": o, "close": o + pd.to_timedelta(tr.mins, unit="m"), "pct": tr.oz * 0.5})


def asia_trades():
    sig = day_signals()
    sig = sig[sig.side == sig.mom].reset_index(drop=True)
    gc.collect()
    rows = []
    for y, m in lb.months():
        g = sig[(sig.day.dt.year == y) & (sig.day.dt.month == m)]
        if g.empty:
            continue
        t, bid, ask = lb.load_month(y, m)
        for r in g.itertuples():
            ems = int(np.datetime64(r.t, "ms").astype(np.int64))
            xms = int(np.datetime64(r.day + pd.Timedelta(minutes=EXIT_AT), "ms").astype(np.int64))
            k = int(np.searchsorted(t, ems))
            if k >= len(t) or t[k] - ems > 600_000 or t[k] >= xms:
                continue
            u = r.atr * ATR_MULT
            X = bid if r.side == 1 else ask
            e = ask[k] if r.side == 1 else bid[k]
            sl, tp = e - r.side * 0.75 * u, e + r.side * u
            fc = int(np.searchsorted(t, xms))
            if fc >= len(t):
                continue
            j = first_hit(X[:fc], k + 1, *((sl, tp) if r.side == 1 else (tp, sl)))
            j = fc if j < 0 else j
            w = 1.5 if r.strong else 0.5
            rows.append(("아시아", pd.Timestamp(int(t[k]), unit="ms"), pd.Timestamp(int(t[j]), unit="ms"),
                         (X[j] - e) * r.side / u * 1.0 * w))
        del t, bid, ask
        gc.collect()
    return pd.DataFrame(rows, columns=["bot", "open", "close", "pct"])


def trend_trades():
    tr = execute_m1(trend(resample(m5_bars(), "4h"), 25, 3.0))
    return pd.DataFrame({"bot": "추세", "open": pd.to_datetime(tr.open_ms, unit="ms"), "close": tr.close, "pct": tr.r * 2.0})


def apply(tr, target):
    """목표에 닿은 날 그 뒤 진입을 뺀 거래 목록"""
    if target is None:
        return tr
    keep = np.ones(len(tr), bool)
    tr = tr.sort_values("open").reset_index(drop=True)
    for day, idx in tr.groupby(tr.open.dt.normalize()).groups.items():
        done = []                                     # (청산 시각, 손익%) 지금까지 들어간 거래
        for i in idx:
            o = tr.at[i, "open"]
            # 그날 청산된 손익: 그날 진입 + 전날 진입이 그날 청산된 것도 포함
            prev = tr[(tr.close >= day) & (tr.close < o) & (tr.open < day) & keep[tr.index]]
            real = sum(p for c, p in done if c < o) + prev.pct.sum()
            if real >= target:
                keep[i] = False
            else:
                done.append((tr.at[i, "close"], tr.at[i, "pct"]))
    return tr[keep]


def summary(tr, name):
    d = tr.groupby(tr.close.dt.normalize()).pct.sum()
    eq = d.cumsum()
    row = {"규칙": name, "거래": len(tr), "승률%": round((tr.pct > 0).mean() * 100, 1), "합계%": round(tr.pct.sum(), 1),
           "최대낙폭%p": round((eq.cummax() - eq).max(), 1), "손실 날": int((d < 0).sum()), "이익 날": int((d > 0).sum()),
           "최악의 날%": round(d.min(), 1)}
    m = tr.groupby(tr.close.dt.to_period("M")).pct.sum()
    row["손실 달"] = f"{int((m < 0).sum())}/{len(m)}"
    for y in range(2022, 2027):
        row[str(y)] = round(tr.pct[tr.close.dt.year == y].sum(), 1)
    for b in ("SAR", "아시아", "추세"):
        row[f"{b} 건수"] = int((tr.bot == b).sum())
    return row


def main():
    allt = pd.concat([sar_trades(), asia_trades(), trend_trades()], ignore_index=True)
    allt = allt[allt.open >= START].sort_values("open").reset_index(drop=True)
    rows = []
    for tg in TARGETS:
        rows.append(summary(apply(allt, tg), "목표 없음 (지금)" if tg is None else f"하루 목표 {tg:g}%"))
        print("완료", tg, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "daily_target_all.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 320)
    pd.set_option("display.unicode.east_asian_width", True)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
