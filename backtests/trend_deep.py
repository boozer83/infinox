"""추세추종(돈치안 돌파) 심층 검증 (2022-01 ~ 현재, 실제 틱 체결)

규칙 (strategies.py의 3번과 동일):
  종가가 직전 N봉 최고가 위면 매수, 최저가 아래면 매도 (다음 봉 첫 틱 진입)
  종가가 직전 N/2봉 최저가 아래(매도는 최고가 위)면 다음 봉 첫 틱에 청산
  손절 = 진입가 ± k x ATR(14), 한 번에 1포지션

검증 항목:
  1. 주변 설정: 봉(H1/H2/H4/D1) x N(10/15/20/25/30) x k(2/3/4) = 60개, 연도별 PF
  2. 워크포워드: 직전 2년 PF가 가장 좋은 설정(학습 거래 60건 이상)을 다음 1년에 적용 (2024, 2025, 2026)
  3. 기준 설정(H4 N20 k3): 매수/매도 분리, 비용 스트레스, 손절폭 분포, 월별 수익률, 연속 손실
  4. 수량: 거래당 계좌의 1% / 2% 위험 (1,000달러와 10,000달러 시작), 최소 0.01 lot 제약 반영

사용법: python -m backtests.trend_deep   (long_period를 먼저 한 번 실행해 틱이 저장돼 있어야 함)
"""
import itertools

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.strategies import TickStore, execute, pf, resample, trend
from config import CONTRACT

TFS = {"H1": "1h", "H2": "2h", "H4": "4h", "D1": "1D"}
NS = [10, 15, 20, 25, 30]
KS = [2.0, 3.0, 4.0]
YEARS = [2022, 2023, 2024, 2025, 2026]
BASE = ("H4", 20, 3.0)


def yearly_pf(tr):
    y = tr.close.dt.year
    return {yr: pf(tr.pnl_oz[y == yr]) for yr in YEARS}


def size_sim(tr, bal0, risk_pct, max_lot=50.0):
    """거래마다 잔고의 risk_pct만큼 잃도록 수량 계산 (0.01 단위 내림, 최소 0.01)"""
    bal, peak, mdd, rows, forced = bal0, bal0, 0.0, [], 0
    for r in tr.itertuples():
        lot = np.floor(bal * risk_pct / (r.risk * CONTRACT) * 100) / 100
        if lot < 0.01:
            lot, forced = 0.01, forced + 1          # 최소 수량이라 목표보다 큰 위험
        lot = min(lot, max_lot)
        pnl = r.pnl_oz * lot * CONTRACT
        bal += pnl
        peak, mdd = max(peak, bal), max(mdd, (peak - bal) / peak)
        rows.append((r.close, pnl, bal))
        if bal <= 0:
            break
    df = pd.DataFrame(rows, columns=["close", "pnl", "bal"])
    return df, mdd, forced


def monthly_table(df, bal0):
    m = df.set_index("close").resample("ME").agg({"pnl": "sum", "bal": "last"})
    m["bal"] = m.bal.ffill().fillna(bal0)
    start = m.bal.shift().fillna(bal0)
    m["pct"] = m.pnl / start * 100
    p = pd.DataFrame({"y": m.index.year, "mo": m.index.month, "pct": m.pct.values}).pivot(
        index="y", columns="mo", values="pct").round(1)
    yr_end = m.groupby(m.index.year).bal.last()
    yr_start = yr_end.shift().fillna(bal0)
    p["연간%"] = ((yr_end / yr_start - 1) * 100).round(1).values
    return p


def main():
    print("봉 만드는 중...", flush=True)
    m5 = lb.bars_from_ticks(5)
    bars = {k: resample(m5, v) for k, v in TFS.items()}
    store = TickStore()

    # 1. 주변 설정
    trades, rows = {}, []
    for tf, n, k in itertools.product(TFS, NS, KS):
        tr = execute(trend(bars[tf], n, k), store)
        trades[(tf, n, k)] = tr
        yp = yearly_pf(tr)
        rows.append({"봉": tf, "N": n, "k": k, "거래수": len(tr), "승률%": round((tr.pnl_oz > 0).mean() * 100, 1),
                     "전체PF": pf(tr.pnl_oz), **{f"PF {y}": v for y, v in yp.items()},
                     "최저연PF": min(yp.values())})
        print(f"  {tf} N{n} k{k:g}: PF {pf(tr.pnl_oz)} / 최저연 {min(yp.values())}", flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv("trend_grid.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n=== 1. 주변 설정 (전체 PF) ===")
    for tf in TFS:
        g = grid[grid.봉 == tf].pivot(index="N", columns="k", values="전체PF")
        mn = grid[grid.봉 == tf].pivot(index="N", columns="k", values="최저연PF")
        print(f"[{tf}] 전체 PF")
        print(g.to_string())
        print(f"[{tf}] 가장 나쁜 해의 PF")
        print(mn.to_string())
    print(f"\n60개 중 전체 PF > 1: {(grid.전체PF > 1).sum()}개, 모든 해 PF > 1: {(grid.최저연PF > 1).sum()}개, "
          f"모든 해 PF >= 1.1: {(grid.최저연PF >= 1.1).sum()}개")

    # 2. 워크포워드
    print("\n=== 2. 워크포워드 (직전 2년으로 고르고 다음 1년 적용) ===")
    oos = []
    for test_y in (2024, 2025, 2026):
        best, best_pf = None, -1
        for key, tr in trades.items():
            y = tr.close.dt.year
            tr_train = tr[(y >= test_y - 2) & (y < test_y)]
            if len(tr_train) < 60:
                continue
            p = pf(tr_train.pnl_oz)
            if p > best_pf:
                best, best_pf = key, p
        t_test = trades[best][trades[best].close.dt.year == test_y]
        oos.append(t_test)
        print(f"  {test_y}: 고른 설정 {best[0]} N{best[1]} k{best[2]:g} (학습 PF {best_pf}) -> {test_y} PF {pf(t_test.pnl_oz)}, 거래 {len(t_test)}")
    oos = pd.concat(oos)
    print(f"  워크포워드 합계 (2024~2026): PF {pf(oos.pnl_oz)}, 거래 {len(oos)}")

    # 3. 기준 설정 상세
    tr = trades[BASE]
    print(f"\n=== 3. 기준 설정 {BASE[0]} N{BASE[1]} k{BASE[2]:g} ===")
    y = tr.close.dt.year
    side = pd.DataFrame({yr: {"매수 PF": pf(tr.pnl_oz[(y == yr) & (tr.side == 1)]),
                              "매도 PF": pf(tr.pnl_oz[(y == yr) & (tr.side == -1)]),
                              "매수 건": int(((y == yr) & (tr.side == 1)).sum()),
                              "매도 건": int(((y == yr) & (tr.side == -1)).sum())} for yr in YEARS})
    print(side.to_string())
    for c in (0.2, 0.5, 1.0):
        print(f"  거래당 추가 비용 {c}달러: PF {pf(tr.pnl_oz - c)}")
    print("  손절폭(달러/온스) 연도별 중앙값: " +
          ", ".join(f"{yr} {tr.risk[y == yr].median():.1f}" for yr in YEARS))
    hold_h = (tr.close_ms - tr.open_ms) / 3_600_000
    print(f"  보유 시간 중앙값 {hold_h.median():.0f}시간, 승률 {(tr.pnl_oz > 0).mean():.1%}, "
          f"평균 이익 {tr.pnl_oz[tr.pnl_oz > 0].mean():.1f} / 평균 손실 {tr.pnl_oz[tr.pnl_oz < 0].mean():.1f}")
    streak = mx = 0
    for p in tr.pnl_oz:
        streak = streak + 1 if p < 0 else 0
        mx = max(mx, streak)
    print(f"  최대 연속 손실 {mx}번")

    # 4. 수량
    print("\n=== 4. 수량별 결과 (2022-01 시작) ===")
    for bal0, rp in ((1000, 0.01), (1000, 0.02), (10000, 0.01), (10000, 0.02)):
        df, mdd, forced = size_sim(tr, bal0, rp)
        print(f"  시작 {bal0:,}$, 거래당 {rp:.0%} 위험: 최종 {df.bal.iat[-1]:,.0f}$, 최대낙폭 {mdd:.0%}, "
              f"최소 수량 때문에 목표보다 큰 위험 {forced}/{len(df)}건")
    df, mdd, _ = size_sim(tr, 10000, 0.01)
    print("\n[시작 10,000$, 거래당 1% 위험] 월별 수익률 %")
    print(monthly_table(df, 10000).to_string())


if __name__ == "__main__":
    main()
