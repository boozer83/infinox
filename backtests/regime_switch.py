"""장중 시장 판단으로 마틴을 멈추거나 줄이면 나아지는가 (2022-01 ~ 현재, 실제 틱)

기준: demo_trader.py 현재 규칙 (RSI50 + H1 SAR, SL 7.5 / TP 3, 일일 마틴 5단계, 미국장 15~21시 제외)
판단에 쓰는 값은 모두 진입 시점까지 완성된 1시간봉만 사용 (미래 정보 없음):
  - 오늘 SAR 전환 수 : 오늘(서버 날짜) 완성된 1시간봉 중 SAR 방향이 바뀐 봉의 수 (횡보일수록 많음)
  - ADX              : 마지막 완성 1시간봉의 ADX(14)

비교:
  미국장 제외 (현재)
  + 오늘 SAR 전환 3번 이상이면 그날 정지
  + 오늘 SAR 전환 2번 이상이면 그날 정지
  + 오늘 SAR 전환 2번 이상이면 lot 절반 (수비)
  + H1 ADX 20 미만이면 진입 안 함
정지/진입 안 함은 신호를 미리 거른 뒤 다시 시뮬레이션 (쉬는 동안 잡을 수 있게 된 다음 신호도 반영).
lot 절반은 같은 거래에 수량만 바꿈.

결과: 연도별 PF, 2025-01부터 10,000$ / 시작 0.30 lot 고정 일일 마틴
사용법: python -m backtests.regime_switch   (long_period를 먼저 한 번 실행해 틱이 저장돼 있어야 함)
"""
import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.martin_scaled import NO_ENTRY_HOURS, simulate
from backtests.scalp_ideas import m5_bars
from backtests.strategies import pf, resample
from config import CONTRACT
from indicators import adx, psar

CACHE_DIR = lb.TICK_DIR.parent / "regime_switch"
YEARS = [2022, 2023, 2024, 2025, 2026]
MONEY_START = pd.Timestamp("2025-01-01")
BAL0, BASE = 10_000.0, 0.30


def h1_state(h1):
    """1시간봉별 완성 시각, 오늘 누적 SAR 전환 수, ADX"""
    up = psar(h1.high.values, h1.low.values)
    flip = np.r_[False, up[1:] != up[:-1]]
    day = h1.time.dt.date
    return pd.DataFrame({"done": h1.time + pd.Timedelta(hours=1), "day": day,
                         "flips": pd.Series(flip).groupby(day.values).cumsum().values, "adx": adx(h1).values})


def signal_features(st, h1s):
    """각 신호 시각에 이미 완성된 마지막 1시간봉 기준 (같은 날이 아니면 전환 수 0)"""
    sig = pd.DataFrame({"t": pd.to_datetime(st, unit="ms").astype("datetime64[ms]")})
    h1s = h1s.assign(done=h1s.done.astype("datetime64[ms]"))
    m = pd.merge_asof(sig, h1s, left_on="t", right_on="done", direction="backward")
    same_day = m.day.values == sig.t.dt.date.values
    flips = np.where(same_day, m.flips.fillna(0).values, 0)
    return flips.astype(int), m.adx.values


def money(tr, mult):
    """일일 마틴 (martin_scaled.martin_money 고정 lot과 같은 규칙) + 거래별 수량 배수"""
    bal, peak, mdd, level, cur, stopped = BAL0, BAL0, 0.0, 0, None, None
    rows = []
    for r, k in zip(tr.itertuples(), mult):
        day = r.open.date()
        if day != cur:
            cur, level = day, 0
        if stopped == day:
            continue
        profit = r.pnl_oz * round(BASE * k * 2 ** level, 2) * CONTRACT
        bal += profit
        peak, mdd = max(peak, bal), max(mdd, (peak - bal) / peak)
        rows.append((day, profit))
        if r.pnl_oz < 0:
            if level >= 5:
                stopped = day
            else:
                level += 1
        else:
            level = 0
    daily = pd.DataFrame(rows, columns=["day", "profit"]).groupby("day").profit.sum()
    return {"2025~ 최종잔고": round(bal), "최대낙폭%": round(mdd * 100, 1), "최악의날": round(daily.min()),
            "손실일": f"{(daily < 0).sum()}/{len(daily)}", "큰 손실일(-1000$ 이하)": int((daily <= -1000).sum())}


def main():
    m5 = m5_bars()
    h1 = resample(m5, "1h")
    sigs = lb.signals(m5, h1)
    del m5
    st = np.array([s[0] for s in sigs], dtype=np.int64)
    flips, adx_v = signal_features(st, h1_state(h1))
    us = np.isin(pd.to_datetime(st, unit="ms").hour, list(NO_ENTRY_HOURS))

    variants = {
        "미국장 제외 (현재)": us,
        "+ SAR 전환 3번 이상 정지": us | (flips >= 3),
        "+ SAR 전환 2번 이상 정지": us | (flips >= 2),
        "+ H1 ADX 20 미만 진입 안 함": us | (adx_v < 20),
    }
    CACHE_DIR.mkdir(exist_ok=True)
    trades = {}
    for i, (name, skip) in enumerate(variants.items()):
        cache = CACHE_DIR / f"trades_{i}.csv"
        if cache.exists():
            tr = pd.read_csv(cache)
            tr["open"] = pd.to_datetime(tr.open_ms, unit="ms")
        else:
            tr = simulate([s for s, k in zip(sigs, skip) if not k])
            tr.drop(columns=["open"]).to_csv(cache, index=False)
        trades[name] = (tr, np.ones(len(tr)), int(skip.sum()))
        print(f"완료: {name}", flush=True)

    # 수비: 미국장 제외 거래 그대로, 진입 시점 SAR 전환 2번 이상이면 lot 절반
    base_tr = trades["미국장 제외 (현재)"][0]
    f_tr, _ = signal_features(base_tr.open_ms.values, h1_state(h1))
    trades["+ SAR 전환 2번 이상 lot 절반"] = (base_tr, np.where(f_tr >= 2, 0.5, 1.0), int(us.sum()))

    rows = []
    for name, (tr, mult, skipped) in trades.items():
        y = tr.open.dt.year
        w = tr.pnl_oz * mult
        row = {"규칙": name, "무시한 신호": skipped, "거래수": len(tr), "PF 전체": pf(w)}
        for yr in YEARS:
            row[f"PF {yr}"] = pf(w[y == yr])
        sel = (tr.open >= MONEY_START).values
        row.update(money(tr[sel], mult[sel]))
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv("regime_switch.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
