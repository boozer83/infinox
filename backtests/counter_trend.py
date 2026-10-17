"""역추세 재현: 홍콩 일반 EA처럼 RSI가 낮을 때 매수 / 높을 때 매도, 익절 3 / 손절 7.5 (2022-01 ~ 현재, 실제 틱)

홍콩 일반 실계좌 내역(2026-09-11 ~ 10-16, 112건)을 진입 직전 5분봉에 맞춰 본 결과:
  매수 RSI(14) 중앙값 37.8 (24~47), 매도 61.5 (50~77), 5분봉 SAR과 반대 방향 진입 65%,
  평균 이익 +3.05 / 평균 손실 -7.60 달러/온스, 승률 60.7%
진입 트리거(봉 중간에 들어감)는 알 수 없어서, 5분봉이 끝났을 때 RSI가 기준선을 새로 넘어선 봉에서 진입하는 것으로 근사.

규칙: 5분봉 RSI(14)가 X 아래로 내려간 봉 -> 매수, 100-X 위로 올라간 봉 -> 매도 (다음 봉 첫 틱 진입)
      SL 7.5 / TP 3 (long_period와 같은 체결), 한 번에 1포지션
비교: X = 30 / 35 / 40, 1시간봉 SAR 방향 필터 (같은 방향 = 추세 속 눌림 / 반대 방향 = 순수 역추세), 미국장 제외

결과: 연도별 PF, 홍콩 일반 실계좌 기간(2026-09-11 ~ 10-16) 승률/PF (근사가 맞는지 확인용)
사용법: python -m backtests.counter_trend   (long_period를 먼저 한 번 실행해 틱이 저장돼 있어야 함)
"""
import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.martin_scaled import NO_ENTRY_HOURS, simulate
from backtests.scalp_ideas import m5_bars
from backtests.strategies import pf, resample
from indicators import psar, rsi

CACHE_DIR = lb.TICK_DIR.parent / "counter_trend"
YEARS = [2022, 2023, 2024, 2025, 2026]
LIVE_FROM, LIVE_TO = pd.Timestamp("2026-09-11"), pd.Timestamp("2026-10-17")


def signals(m5, h1, x, h1_filter=None, skip_us=False):
    """RSI 기준선 돌파 신호: [(진입 ms, 방향)]. h1_filter: None / 'with' (같은 방향만) / 'against' (반대 방향만)"""
    r = rsi(m5.close).values
    buy = (r < x) & (np.r_[np.nan, r[:-1]] >= x)
    sell = (r > 100 - x) & (np.r_[np.nan, r[:-1]] <= 100 - x)
    h = pd.DataFrame({"close_time": h1.time + pd.Timedelta(hours=1),
                      "hdir": np.where(psar(h1.high.values, h1.low.values), 1, -1)})
    mm = pd.merge_asof(pd.DataFrame({"close_time": m5.time + pd.Timedelta(minutes=5)}), h,
                       on="close_time", direction="backward")
    hdir = mm.hdir.values
    t = m5.time.values.astype("datetime64[ms]").astype(np.int64)
    hours = m5.time.dt.hour.values
    out = []
    for i in np.flatnonzero(buy | sell)[:-1]:
        side = 1 if buy[i] else -1
        if h1_filter == "with" and hdir[i] != side:
            continue
        if h1_filter == "against" and hdir[i] == side:
            continue
        if skip_us and pd.Timestamp(t[i + 1], unit="ms").hour in NO_ENTRY_HOURS:
            continue
        out.append((t[i + 1], side))
    return out


def main():
    m5 = m5_bars()
    h1 = resample(m5, "1h")
    variants = {
        "RSI 30/70": dict(x=30),
        "RSI 35/65": dict(x=35),
        "RSI 40/60": dict(x=40),
        "RSI 35/65 + H1 같은 방향(눌림)": dict(x=35, h1_filter="with"),
        "RSI 35/65 + H1 반대 방향(순수 역추세)": dict(x=35, h1_filter="against"),
        "RSI 35/65 + 미국장 제외": dict(x=35, skip_us=True),
    }
    CACHE_DIR.mkdir(exist_ok=True)
    rows = []
    for i, (name, kw) in enumerate(variants.items()):
        cache = CACHE_DIR / f"trades_{i}.csv"
        if cache.exists():
            tr = pd.read_csv(cache)
            tr["open"] = pd.to_datetime(tr.open_ms, unit="ms")
        else:
            tr = simulate(signals(m5, h1, **kw))
            tr.drop(columns=["open"]).to_csv(cache, index=False)
        y = tr.open.dt.year
        row = {"규칙": name, "거래수": len(tr), "승률%": round((tr.pnl_oz > 0).mean() * 100, 1), "PF 전체": pf(tr.pnl_oz)}
        for yr in YEARS:
            row[f"PF {yr}"] = pf(tr.pnl_oz[y == yr])
            row[f"승률 {yr}"] = round((tr.pnl_oz[y == yr] > 0).mean() * 100, 1)
        live = tr[(tr.open >= LIVE_FROM) & (tr.open < LIVE_TO)]
        row.update({"실계좌기간 거래": len(live), "실계좌기간 승률%": round((live.pnl_oz > 0).mean() * 100, 1),
                    "실계좌기간 PF": pf(live.pnl_oz)})
        rows.append(row)
        print(f"완료: {name}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("counter_trend.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
