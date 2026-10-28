"""PA Master Grade-A 포트폴리오: 금 + 은 + 나스닥 + 일본 (1시간봉, 익절 3R, 실제 틱 체결, 2022-01 ~ 2026-10)

종목마다 신호는 tv_batch2.pa_master (종목별 1분 틱 수 거래량), 체결은 tv_pa_ticks.tick_trades.
거래량 조건: 4종목 모두 원본(켬) / 4종목 모두 끔, 두 가지 (종목별로 골라 쓰지 않음)
크기: 거래마다 고정 위험 1R (손절 한 번 = 1R). 종목끼리는 동시에 보유 가능
결과: 종목별·합계 R, 연도별 R, R 기준 최대 낙폭, 같은 달 손실, 지금 H4 추세 봇(구름)과 월 손익 상관
스왑: 금만 (다른 종목 값 없음)

사용법: python -m backtests.tv_pa_portfolio
"""
import numpy as np
import pandas as pd

import backtests.tickvol_m1 as tv
import backtests.tv_overlap as tov
import backtests.tv_single as ts
from backtests.tv_batch2 import bars_vol, pa_master
from backtests.tv_pa_ticks import tick_trades
from backtests.tv_symbols import GOLD_SWAP, load_sym, spread_table

SYMS = ["XAUUSD", "XAGUSD", "NAS100", "JPN225"]
RR = 3.0


def r_stats(r, when):
    eq = r.cumsum()
    yr = pd.to_datetime(when).dt.year
    out = {"거래": len(r), "승률%": round((r > 0).mean() * 100, 1), "PF(R)": round(ts.pf(r), 2),
           "합계R": round(r.sum(), 1), "거래당R": round(r.mean(), 3), "최대낙폭R": round((eq.cummax() - eq).max(), 1)}
    for y in ts.YEARS:
        out[f"{y}R"] = round(r[yr == y].sum(), 1)
    return out


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    sp_tab = spread_table()
    allt = {True: [], False: []}
    rows = []
    for s in SYMS:
        m1 = load_sym(s, sp_tab)
        ts.SWAP_LONG, ts.SWAP_SHORT = GOLD_SWAP if s == "XAUUSD" else (0.0, 0.0)
        d = bars_vol(m1, tv.load(s), 60)
        del m1
        for use_vol in (True, False):
            side, stop, _, _ = pa_master(d, use_vol=use_vol)
            tr = tick_trades(d, side, stop, RR, s)
            tr["sym"] = s
            allt[use_vol].append(tr)
            rows.append({"거래량": "켬" if use_vol else "끔", "종목": s, **r_stats(tr.r, tr.close),
                         "매수PF": round(ts.pf(tr.r[tr.side == 1]), 2), "매도PF": round(ts.pf(tr.r[tr.side == -1]), 2)})
        print(f"{s} 끝", flush=True)

    ts.SWAP_LONG, ts.SWAP_SHORT = GOLD_SWAP
    h4 = ts.bars(ts.load_m1("XAUUSD"), 240)
    trend = tov.trend_bot(h4, True)[1]
    tm = trend.groupby(pd.to_datetime(trend.close).dt.to_period("M")).pct.sum()

    for use_vol in (True, False):
        p = pd.concat(allt[use_vol]).sort_values("close").reset_index(drop=True)
        p.to_csv(f"results/tv_pa_portfolio_{'vol' if use_vol else 'novol'}.csv", index=False, encoding="utf-8-sig")
        rows.append({"거래량": "켬" if use_vol else "끔", "종목": "합계 4종목", **r_stats(p.r, p.close),
                     "매수PF": round(ts.pf(p.r[p.side == 1]), 2), "매도PF": round(ts.pf(p.r[p.side == -1]), 2)})
        mon = p.groupby(pd.to_datetime(p.close).dt.to_period("M")).r.sum()
        m = pd.concat([mon, tm], axis=1).fillna(0)
        print(f"\n[거래량 {'켬' if use_vol else '끔'}] 손실 달 {(mon < 0).sum()} / {len(mon)}, 최악의 달 {mon.min():.1f}R, "
              f"추세 봇과 월 손익 상관 {m.corr().iat[0, 1]:.2f}, 둘 다 손실인 달 {((m.iloc[:, 0] < 0) & (m.iloc[:, 1] < 0)).sum()}")
        sm = pd.DataFrame({s: p[p.sym == s].groupby(pd.to_datetime(p[p.sym == s].close).dt.to_period("M")).r.sum()
                           for s in SYMS}).fillna(0)
        print("  종목끼리 월 손익 상관:\n" + sm.corr().round(2).to_string())
    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
