"""금/은 비율 되돌림에 스왑 넣기 (2026-10-31, gsr_trader는 스왑 미반영으로 관찰 중)

대표 H1 N100 z2.5 + 이웃 N100/150 x z1.5~3 8개, 실제 틱 체결 (idea_screen.gsr 그대로), 은 = data/bars/XAGUSD_M1_ticks_clean.pkl
스왑 $/oz 하룻밤, 수요일 밤 3배 (trend_box_filter와 같은 지금 계좌 값). 과거 스왑 기록이 없어서 3가지로 봄
  고정   : 매수 -0.789 / 매도 +0.2557 그대로
  금값비례: 고정 값 x 진입가 / 4150 (스왑이 금값 %로 움직인다고 가정)
  보수   : 매수 -0.789 / 매도 0 (매도 스왑을 못 받는다고 가정)
PF는 스왑 뺀 oz로, 추가비용은 거래당 0.2$

사용법: python -m backtests.gsr_swap
"""
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.idea_screen as ids
from backtests.scalp_ideas import m5_bars
from backtests.strategies import TickStore, execute, pf, resample
from backtests.trend_box_filter import SWAP_LONG, SWAP_SHORT, TRIPLE
from backtests.trend_portfolio import rollovers

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
REF_PRICE = 4150.0
YEARS = [2022, 2023, 2024, 2025, 2026]
GRID = [(n, z) for n in (100, 150) for z in (1.5, 2.0, 2.5, 3.0)]


def add_swaps(tr):
    o, c = pd.to_datetime(tr.open_ms, unit="ms"), pd.to_datetime(tr.close_ms, unit="ms")
    tr = tr.assign(open=o, close=c)
    tr["nights"] = [rollovers(a, b, TRIPLE) for a, b in zip(o, c)]
    long = tr.side == 1
    tr["sw_fixed"] = tr.nights * np.where(long, SWAP_LONG, SWAP_SHORT)
    tr["sw_cons"] = tr.nights * np.where(long, SWAP_LONG, 0.0)
    return tr


def row(name, tr, col=None, cost=0.0):
    x = tr.pnl_oz + (tr[col] if col else 0) - cost
    y = tr.close.dt.year
    eq = x.cumsum()
    r = {"설정": name, "거래": len(tr), "승률": round((x > 0).mean() * 100, 1), "oz": round(x.sum(), 1),
         "낙폭": round((eq.cummax() - eq).max(), 1)}
    for yr in YEARS:
        r[f"PF{yr % 100}"] = pf(x[y == yr])
    r["매수PF"], r["매도PF"] = pf(x[tr.side == 1]), pf(x[tr.side == -1])
    return r


def main():
    m5 = m5_bars()
    h1 = resample(m5, "1h")
    ag = pd.read_pickle(ROOT / "data" / "bars" / "XAGUSD_M1_ticks_clean.pkl")
    ids._other[("XAGUSD", "1h")] = resample(ag, "1h").set_index("time").close
    last = ag.time.max()
    print("봉", h1.time.min(), h1.time.max(), "은 끝", last, flush=True)
    store = TickStore()
    rows = []
    for n, z in GRID:
        specs = [s for s in ids.gsr(h1, "1h", n, z) if s["exit"] <= int(last.value // 1_000_000)]
        tr = add_swaps(execute(specs, store))
        px = h1.set_index("time").close.reindex(tr.open.dt.floor("h")).ffill().values
        tr["sw_price"] = tr.sw_fixed * px / REF_PRICE
        name = f"N{n} z{z}"
        rows += [{**row(name + " 스왑 없음", tr)}, {**row(name + " 고정", tr, "sw_fixed")},
                 {**row(name + " 금값비례", tr, "sw_price")}, {**row(name + " 보수", tr, "sw_cons")},
                 {**row(name + " 보수+비용0.2", tr, "sw_cons", 0.2)}]
        print(name, len(tr), flush=True)
        if (n, z) == (100, 2.5):
            tr.to_csv(RES / "gsr_swap_trades.csv", index=False)
            top = tr.pnl_oz.nlargest(5).sum() / tr.pnl_oz.sum() * 100
            print(f"밤 평균 {tr.nights.mean():.1f} 중간값 {tr.nights.median():.0f} 최대 {tr.nights.max()}, 매수 {(tr.side == 1).sum()} 매도 {(tr.side == -1).sum()}")
            print(f"스왑 합 고정 {tr.sw_fixed.sum():+.1f} (매수 {tr.sw_fixed[tr.side == 1].sum():+.1f} 매도 {tr.sw_fixed[tr.side == -1].sum():+.1f}), 금값비례 {tr.sw_price.sum():+.1f}, 보수 {tr.sw_cons.sum():+.1f}, 상위 5건 {top:.0f}%")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "gsr_swap.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.max_rows", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
