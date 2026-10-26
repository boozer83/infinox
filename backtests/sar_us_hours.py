"""SAR 단타(지금 봇)에서 미국장 진입 금지(서버 15:00~21:00)를 뺐을 때 비교 (실제 틱, 2026-10-27 사용자 요청)

backtests/sar_martin.trades와 같은 거래 목록을 두 번 뽑음: 지금 규칙 / 미국장도 진입
  sar_us_<now|all>_YYYY.csv (open, day, mins, oz, mae)
2022~2024는 ATR 기준 가격비례(sar_martin scaled). 그 다음 연도별 건수·승률·$/oz 합·1,000$ 시작 고정 lot(1,000$당 0.05) 잔고와 최저점을 표로

사용법: python -m backtests.sar_us_hours trades 2025      (한 해, 둘 다)
        python -m backtests.sar_us_hours sum              (저장된 것 합쳐서 표)
"""
import glob
import sys

import pandas as pd

import backtests.sar_martin as sm
import backtests.winrate_ideas as wi

LOT_PER_1000 = 0.05


def run(year):
    import os
    for name, off in (("now", False), ("all", True)):
        if off:
            wi.NO_ENTRY_FROM = wi.NO_ENTRY_UNTIL = 0          # 빈 구간 = 미국장도 진입
        sm.MAE.clear()                                        # 거래마다 쌓는 전역 목록 (두 번째 실행 때 길이가 안 맞음)
        scaled = year < 2025                                  # 금값 낮던 해는 쉬기 필터(ATR 2.5$)를 가격비례로 (sar_martin과 같음)
        sm.trades(year, scaled)
        os.replace(f"sar_martin_{'scaled' if scaled else 'trades'}_{year}.csv", f"sar_us_{name}_{year}.csv")


def stats(tr):
    bal, low, peak, mdd = 1000.0, 1000.0, 1000.0, 0.0
    for oz in tr.oz:
        lot = max(0.01, int(bal / 1000 * LOT_PER_1000 * 100) / 100)
        bal += oz * lot * 100
        peak, low = max(peak, bal), min(low, bal)
        mdd = max(mdd, (peak - bal) / peak)
    us = tr[(tr.mins >= 15 * 60) & (tr.mins < 21 * 60)] if "mins" in tr else tr.iloc[0:0]
    return {"건수": len(tr), "승률%": round((tr.oz > 0).mean() * 100, 1), "oz합": round(tr.oz.sum(), 1),
            "잔고(1000시작)": round(bal), "최대낙폭%": round(mdd * 100, 1),
            "미국장 건수": len(us), "미국장 oz": round(us.oz.sum(), 1), "미국장 승률%": round((us.oz > 0).mean() * 100, 1) if len(us) else 0}


def summary():
    rows = []
    for f in sorted(glob.glob("sar_us_*_*.csv")):
        name, year = f[:-4].split("_")[2:]
        tr = pd.read_csv(f)
        tr["open"] = pd.to_datetime(tr.open)
        tr["mins"] = tr.open.dt.hour * 60 + tr.open.dt.minute
        rows.append({"규칙": "지금" if name == "now" else "미국장도", "연도": int(year), **stats(tr)})
    df = pd.DataFrame(rows).sort_values(["연도", "규칙"], ascending=[True, False])
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "trades":
        run(int(sys.argv[2]))
    else:
        summary()
