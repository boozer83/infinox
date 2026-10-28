"""단타가 지는 조건에서는 반대로 (2026-10-27 사용자 아이디어, 실제 틱)

1) 미국장(서버 15~21시)에만 콩 개수·기울기 역진입 (backtests.sar_count_reverse 신호, 그 시간만)
2) 지금 단타 거래(results/sar_us_now_YYYY.csv, sar_us_hours가 만든 것)를 진입 때 조건별로 나눠 승률·$/oz
   조건: 서버 시각, 요일, ATR(14, 5분봉), RSI(14), 과열(ext), 콩까지 거리(risk), 신호봉 크기(range),
         직전 SAR 추세 길이(뒤집히기 전 콩 수), 1시간 SAR 추세 길이, 직전 거래 결과
3) 같은 신호 시각에 반대 방향으로 들어갔으면 (SL 7.5 / TP 3) -> 조건 구간마다 단타 vs 반대
과최적화 확인: 2022~24에서 고른 구간이 2025~26에도 맞는지 나란히 표시

사용법: python -m backtests.sar_regime us | feat
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.long_period as lb
import backtests.sar_count_reverse as cr
from backtests.ext_filter import features
from backtests.m5_ticks import first_hit
from backtests.martin_scaled import NO_ENTRY_HOURS, simulate
from backtests.scalp_ideas import m5_bars
from backtests.strategies import pf, resample
from indicators import atr, psar_levels, rsi

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CACHE = lb.TICK_DIR.parent / "sar_regime"


def us_only():
    m5 = m5_bars()
    orig = cr.NO_ENTRY_HOURS
    cr.NO_ENTRY_HOURS = [h for h in range(24) if h not in NO_ENTRY_HOURS]   # 미국장 아닌 시간을 거름 = 미국장만
    variants = {f"미국장만 콩 {n}개 역진입": dict(count=n) for n in (2, 3, 4, 5, 6, 8, 10)}
    variants["미국장만 콩 4개 순방향 (비교)"] = dict(count=4, reverse=False)
    variants.update({f"미국장만 기울기 {k} 역진입": dict(slope=k) for k in (0.2, 0.3, 0.5, 0.8)})
    rows = []
    for name, kw in variants.items():
        tr = simulate(cr.signals(m5, **kw))
        y = tr.open.dt.year
        row = {"규칙": name, "거래수": len(tr), "승률%": round((tr.pnl_oz > 0).mean() * 100, 1),
               "oz합": round(tr.pnl_oz.sum(), 1), "PF": pf(tr.pnl_oz)}
        row.update({f"oz {yr}": round(tr.pnl_oz[y == yr].sum(), 1) for yr in cr.YEARS})
        rows.append(row)
        print("완료", name, flush=True)
    cr.NO_ENTRY_HOURS = orig
    show(pd.DataFrame(rows), "sar_regime_us.csv")


def show(df, name):
    df.to_csv(RES / name, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.max_rows", 500)
    print(df.to_string(index=False))


def bot_trades(m5):
    """지금 단타 거래 + 진입 때 조건 + 같은 시각 반대 진입 결과"""
    tr = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("sar_us_now_*.csv"))], ignore_index=True)
    tr["open"] = pd.to_datetime(tr.open)
    up, sar, _ = psar_levels(m5.high.values, m5.low.values)
    d = np.where(up, 1, -1)
    run = np.ones(len(d), int)
    for i in range(1, len(d)):
        run[i] = run[i - 1] + 1 if d[i] == d[i - 1] else 1
    prev_run = np.zeros(len(d), int)                 # 뒤집힌 봉: 직전 추세가 몇 개였는지
    for i in range(1, len(d)):
        prev_run[i] = run[i - 1] if run[i] == 1 else prev_run[i - 1]
    h1 = resample(m5, "1h")
    hup = psar_levels(h1.high.values, h1.low.values)[0]
    hrun = np.ones(len(hup), int)
    for i in range(1, len(hup)):
        hrun[i] = hrun[i - 1] + 1 if hup[i] == hup[i - 1] else 1
    hh = pd.DataFrame({"t": h1.time + pd.Timedelta(hours=1), "h1_run": hrun})
    f = features(m5)
    f["side"], f["prev_run"] = d, prev_run
    f["atr"], f["rsi"] = atr(m5).values, rsi(m5.close).values
    f = pd.merge_asof(f.sort_values("t"), hh, on="t")
    tr["t"] = tr.open.dt.floor("5min")
    tr = tr.merge(f, on="t", how="left")
    tr["rsi_dir"] = (tr.rsi - 50) * tr.side
    tr["hour"], tr["dow"] = tr.open.dt.hour, tr.open.dt.dayofweek
    tr["prev_win"] = (tr.oz.shift() > 0).astype(int)
    cache = CACHE / "reverse.csv"
    if cache.exists():
        rv = pd.read_csv(cache)
    else:
        CACHE.mkdir(exist_ok=True)
        ms = tr.open.values.astype("datetime64[ms]").astype(np.int64)
        rv = rev_each(ms, tr.side.values)
        rv.to_csv(cache, index=False)
    tr["rev"] = rv.rev.values
    return tr


def rev_each(ms, side):
    """신호마다 따로 반대 진입 (앞 거래가 안 끝나도 막지 않음) -> rev $/oz"""
    out = np.full(len(ms), np.nan)
    months = list(lb.months())
    for idx, (y, m) in enumerate(months):
        s, e = lb.month_range(y, m)
        sel = np.flatnonzero((ms >= s.timestamp() * 1000) & (ms < e.timestamp() * 1000))
        if not len(sel):
            continue
        t, bid, ask = lb.load_month(y, m)
        if idx + 1 < len(months):
            nt, nb, na = lb.load_month(*months[idx + 1])
            t, bid, ask = np.concatenate([t, nt]), np.concatenate([bid, nb]), np.concatenate([ask, na])
        for i in sel:
            sd = -side[i]
            k = int(np.searchsorted(t, ms[i]))
            if k >= len(t) - 1:
                continue
            if sd == 1:
                ep = ask[k]
                j = first_hit(bid, k + 1, ep - lb.SL, ep + lb.TP)
                out[i] = bid[j] - ep if j >= 0 else np.nan
            else:
                ep = bid[k]
                j = first_hit(ask, k + 1, ep - lb.TP, ep + lb.SL)
                out[i] = ep - ask[j] if j >= 0 else np.nan
        print(f"  {y}-{m:02d}", flush=True)
    return pd.DataFrame({"rev": out})


def feat():
    m5 = m5_bars()
    tr = bot_trades(m5)
    tr["per"] = np.where(tr.open.dt.year <= 2024, "22~24", "25~26")
    tr.to_csv(RES / "sar_regime_trades.csv", index=False, encoding="utf-8-sig")
    cols = {"hour": None, "dow": None, "prev_win": None, "atr": 5, "rsi_dir": 5, "ext": 5, "risk": 5,
            "range": 5, "prev_run": [0, 3, 5, 8, 12, 1000], "h1_run": [0, 2, 4, 8, 16, 1000]}
    rows = []
    for c, q in cols.items():
        if q is None:
            b = tr[c]
        elif isinstance(q, list):
            b = pd.cut(tr[c], q)
        else:
            b = pd.qcut(tr[c], q, duplicates="drop")
        for key, g in tr.groupby([b, "per"], observed=True):
            rows.append({"조건": c, "구간": str(key[0]), "기간": key[1], "건수": len(g),
                         "단타 승률%": round((g.oz > 0).mean() * 100, 1), "단타 oz/건": round(g.oz.mean(), 2),
                         "반대 승률%": round((g.rev > 0).mean() * 100, 1), "반대 oz/건": round(g.rev.mean(), 2)})
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["조건", "구간"], columns="기간",
                          values=["건수", "단타 oz/건", "반대 oz/건", "단타 승률%"], sort=False)
    wide.columns = [f"{a} {b}" for a, b in wide.columns]
    show(wide.reset_index(), "sar_regime_feat.csv")
    tot = tr.groupby("per").agg(건수=("oz", "size"), 단타=("oz", "mean"), 반대=("rev", "mean"))
    print(tot.round(3))


if __name__ == "__main__":
    {"us": us_only, "feat": feat}[sys.argv[1]]()
