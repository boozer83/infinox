"""SAR 콩 개수·기울기로 반대 방향 진입 (2026-10-27 사용자 아이디어, 실제 틱 2022-01 ~ 현재)

  콩 N개 역진입 : 5분봉 SAR이 같은 쪽에 N개째 찍힌 봉이 끝나면 SAR 반대 방향으로 다음 봉 첫 틱 진입
                  (SAR이 아래(상승)에 N개면 매도). 비교용으로 같은 시점 순방향도
  기울기 역진입 : 콩 기울기 = |이번 SAR - 직전 SAR| / ATR(14). 같은 추세 안에서 처음 k 이상이 된 봉에서 반대 방향
                  (콩 간격이 벌어짐 = 추세가 가속해 많이 감 -> 되돌림 노림)
체결은 counter_trend와 같음: SL 7.5 / TP 3, 한 번에 1포지션, 미국장(서버 15~21시) 진입 안 함

사용법: python -m backtests.sar_count_reverse     (결과 results/sar_count_reverse.csv, 거래는 data/sar_count_reverse/)
"""
import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.martin_scaled import NO_ENTRY_HOURS, simulate
from backtests.scalp_ideas import m5_bars
from backtests.strategies import pf
from indicators import atr, psar_levels

CACHE_DIR = lb.TICK_DIR.parent / "sar_count_reverse"
YEARS = [2022, 2023, 2024, 2025, 2026]


def signals(m5, count=None, slope=None, reverse=True):
    up, sar, _ = psar_levels(m5.high.values, m5.low.values)
    d = np.where(up, 1, -1)
    run = np.ones(len(d), int)                       # 같은 쪽 몇 번째 콩인지 (뒤집힌 봉 = 1)
    for i in range(1, len(d)):
        run[i] = run[i - 1] + 1 if d[i] == d[i - 1] else 1
    if count is not None:
        hit = run == count
    else:
        sl = np.abs(np.diff(sar, prepend=sar[0])) / atr(m5).values
        sl[run == 1] = 0                              # 뒤집힌 봉의 점프는 기울기가 아님
        over = sl >= slope
        first = np.zeros(len(d), bool)
        seen = False
        for i in range(len(d)):
            if run[i] == 1:
                seen = False
            if over[i] and not seen:
                first[i], seen = True, True
        hit = first
    t = m5.time.values.astype("datetime64[ms]").astype(np.int64)
    out = []
    for i in np.flatnonzero(hit)[:-1]:
        if pd.Timestamp(t[i + 1], unit="ms").hour in NO_ENTRY_HOURS:
            continue
        out.append((t[i + 1], -d[i] if reverse else d[i]))   # 다음 봉 시작 = 이 봉 끝
    return out


def main():
    m5 = m5_bars()
    variants = {f"콩 {n}개 역진입": dict(count=n) for n in (3, 4, 5, 6, 8)}
    variants["콩 4개 순방향 (비교)"] = dict(count=4, reverse=False)
    variants.update({f"기울기 {k} ATR 역진입": dict(slope=k) for k in (0.3, 0.5, 0.8)})
    CACHE_DIR.mkdir(exist_ok=True)
    rows = []
    for i, (name, kw) in enumerate(variants.items()):
        cache = CACHE_DIR / f"trades_{i}.csv"
        if cache.exists():
            tr = pd.read_csv(cache)
            tr["open"] = pd.to_datetime(tr.open_ms, unit="ms")
        else:
            tr = simulate(signals(m5, **kw))
            tr.drop(columns=["open"]).to_csv(cache, index=False)
        y = tr.open.dt.year
        row = {"규칙": name, "거래수": len(tr), "승률%": round((tr.pnl_oz > 0).mean() * 100, 1),
               "oz합": round(tr.pnl_oz.sum(), 1), "PF": pf(tr.pnl_oz)}
        for yr in YEARS:
            row[f"oz {yr}"] = round(tr.pnl_oz[y == yr].sum(), 1)
        rows.append(row)
        print(f"완료: {name}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(lb.TICK_DIR.parent.parent / "results" / "sar_count_reverse.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
