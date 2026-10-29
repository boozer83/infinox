"""아시아 돌파 v2에 '쉬는 시간/조건'을 넣으면 (2026-10-29 사용자 아이디어, 실제 틱)

backtests/asia_v2.py와 같은 신호·체결 (레인지 01:05~08:55, 09:00~17:55 첫 돌파, 5일 모멘텀 같은 방향만,
손절 0.75U / 익절 1U, 23:00 청산, 가중 1.5/0.5, 마틴 3단계, 결과 0 이하면 다음 단계)
쉬기 후보 (그날 신호를 건너뜀, 마틴 단계는 그대로 유지):
  돌파 시각: 서버 HH시 이후 돌파는 쉼 (12 / 13 / 14 / 15시 = 한국 18~21시), 또는 미국장 겹침(15~18시)만 쉼
  요일: 월요일 / 금요일 쉼
  약한 신호 쉼 (전일 고·저를 못 넘은 돌파)
  레인지 폭: U 대비 너무 좁음(< 0.3U) / 너무 넓음(> 1.2U) 쉼
  연패 쉼: 이 전략이 직전 2번 연속 졌으면 1번 쉼
평가: 승률, 고정 lot 거래당 R (R = 0.75U), v2 마틴 1만$ 최종$·최대낙폭, 2022~24 / 2025~26 나눠서
결과 results/asia_rest.csv

사용법: python -m backtests.asia_rest
"""
import gc

import numpy as np
import pandas as pd

import backtests.asia_v2 as av
from backtests.asia_breakout import ATR_MULT, ENTRY_UNTIL, MIN_RANGE_BARS, MOM_DAYS, RANGE_FROM, RANGE_UNTIL
from backtests.gold_rest_others import load_m1, resample
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"


def day_signals():
    """asia_v2.day_signals와 같고 레인지 고/저, 돌파 봉 시각을 더 남김"""
    m1 = load_m1("XAUUSD")
    m1["day"] = m1.time.dt.date
    d1 = m1.groupby("day").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    pc = d1.close.shift()
    tr = np.maximum(d1.high - d1.low, np.maximum((d1.high - pc).abs(), (d1.low - pc).abs()))
    tr.iat[0] = d1.high.iat[0] - d1.low.iat[0]
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    feat = pd.DataFrame({"atr": atr.shift(), "mom": np.sign(d1.close - d1.close.shift(MOM_DAYS)).shift(),
                         "ph": d1.high.shift(), "pl": d1.low.shift()})
    m5 = resample(m1, 5)
    del m1
    m5["day"] = m5.time.dt.date
    m5["mins"] = m5.time.dt.hour * 60 + m5.time.dt.minute
    rows = []
    for day, g in m5.groupby("day"):
        if day not in feat.index or not np.isfinite(feat.atr[day]) or d1.index.get_loc(day) < 30:
            continue
        r = g[(g.mins >= RANGE_FROM) & (g.mins < RANGE_UNTIL)]
        if len(r) < MIN_RANGE_BARS:
            continue
        hi, lo = r.high.max(), r.low.min()
        w = g[(g.mins >= RANGE_UNTIL) & (g.mins < ENTRY_UNTIL)]
        b = w[(w.close > hi) | (w.close < lo)]
        if b.empty:
            continue
        bar = b.iloc[0]
        side = 1 if bar.close > hi else -1
        f = feat.loc[day]
        strong = bar.close > f.ph if side == 1 else bar.close < f.pl
        rows.append((pd.Timestamp(day), bar.time + pd.Timedelta(minutes=5), side, f.mom, f.atr, bool(strong), hi - lo))
    return pd.DataFrame(rows, columns=["day", "t", "side", "mom", "atr", "strong", "width"])


def stats(sig, oc, keep):
    s = sig[keep]
    o = [oc.at[i, 0.75] for i in s.index]
    ok = [x is not None for x in o]
    s = s[ok]
    oz = np.array([x[0] for x in o if x is not None])
    r = oz / (0.75 * s.atr.values * ATR_MULT)
    per = np.where(s.day.dt.year <= 2024, "22~24", "25~26")
    tr, halted = av.simulate(s.reset_index(drop=True), oc.loc[s.index].reset_index(drop=True), True, 0.75, True)
    eq = tr.bal
    row = {"건수": len(s), "승률%": round((oz > 0).mean() * 100, 1), "R/건": round(r.mean(), 3),
           "승률 22~24": round((oz[per == "22~24"] > 0).mean() * 100, 1), "R 22~24": round(r[per == "22~24"].mean(), 3),
           "승률 25~26": round((oz[per == "25~26"] > 0).mean() * 100, 1), "R 25~26": round(r[per == "25~26"].mean(), 3),
           "마틴 최종$": round(eq.iat[-1]), "낙폭%": round(((eq.cummax() - eq) / eq.cummax()).max() * 100, 1)}
    for y in range(2022, 2027):
        row[f"R{y % 100}"] = round(r[s.day.dt.year.values == y].sum(), 1)
    return row


def main():
    sig = day_signals()
    gc.collect()
    sig = sig[sig.side == sig.mom].reset_index(drop=True)
    oc = av.outcomes(sig)
    h = sig.t.dt.hour + sig.t.dt.minute / 60
    u = sig.atr * ATR_MULT
    wd = sig.day.dt.dayofweek
    # 연패 쉼: 고정 결과 기준 직전 2번 연속 패배면 쉼 (쉬는 날 결과는 연패 계산에 안 넣음)
    res = [oc.at[i, 0.75] for i in sig.index]
    streak_keep, lose = [], 0
    for x in res:
        if lose >= 2:
            streak_keep.append(False)
            lose = 0
            continue
        streak_keep.append(True)
        if x is not None:
            lose = lose + 1 if x[0] <= 0 else 0
    rules = {"지금 (v2)": pd.Series(True, index=sig.index)}
    for hh in (12, 13, 14, 15):
        rules[f"서버 {hh}시 이후 돌파 쉼 (한국 {hh + 6}시~)"] = h < hh
    rules["미국장 겹침 15~18시 돌파 쉼"] = ~((h >= 15) & (h < 18))
    rules["월요일 쉼"] = wd != 0
    rules["금요일 쉼"] = wd != 4
    rules["약한 신호 쉼"] = sig.strong
    rules["레인지 < 0.3U 쉼"] = sig.width >= 0.3 * u
    rules["레인지 > 1.2U 쉼"] = sig.width <= 1.2 * u
    rules["2연패 뒤 1번 쉼"] = pd.Series(streak_keep, index=sig.index)
    rows = [{"규칙": k, **stats(sig, oc, v)} for k, v in rules.items()]
    df = pd.DataFrame(rows)
    df.to_csv(RES / "asia_rest.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 320)
    pd.set_option("display.unicode.east_asian_width", True)
    print(df.to_string(index=False))
    print("\n돌파 시각(서버)별", sig.assign(h=sig.t.dt.hour).groupby("h").size().to_dict())


if __name__ == "__main__":
    main()
