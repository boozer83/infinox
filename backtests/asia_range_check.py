"""아시아 돌파: 진입 직전 2시간 범위 / 일봉 ATR이 넓은 날 돌파가 더 좋은가 (2026-10-28, 다른 AI 분석 확인)

거래: backtests.asia_xaueur outcomes (지금 봇 설정 pick: 레인지 끝 540, 손절 0.75U / 익절 1.0U, 모멘텀 방향만)
값: 진입 시각 전에 완성된 5분봉 24개 (최고 - 최저) / 일봉 ATR(14, 어제까지), 5분봉은 1분봉에서 만듦
자금: ax.martin (각 0.6%, 강한 1.5 / 약한 0.5), 넓은 날(>= 0.4)만 lot x1.5 가중 비교

사용법: python -m backtests.asia_range_check
"""
import numpy as np
import pandas as pd

import backtests.all_symbols as al
import backtests.asia_xaueur as ax
from backtests.gold_rest_others import resample

TH = 0.4
END = pd.Timestamp("2026-10-17")


def ratio(sym, en):
    m5 = resample(al.m1_bars(sym), 5)
    d = m5.set_index("time").resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    pc = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    datr = m5.time.dt.normalize().map(tr.rolling(14).mean().shift(1)).values
    r2 = (m5.high.rolling(24).max() - m5.low.rolling(24).min()).values / datr
    tclose = (m5.time + pd.Timedelta(minutes=5)).values
    k = np.searchsorted(tclose, en.values, side="right") - 1
    return r2[k]


def martin_w(trs, wide_mult):
    """ax.martin과 같은데 넓은 날 lot x wide_mult. -> (최종, 낙폭%, 해마다 %)"""
    ev = pd.concat([x.assign(key=k) for k, x in trs.items()]).sort_values("en_ms")
    bal, peak, mdd = 10_000.0, 10_000.0, 0.0
    lv, cyc, yr = {k: 0 for k in trs}, {k: 10_000.0 for k in trs}, {}
    for r in ev.itertuples():
        if lv[r.key] == 0:
            cyc[r.key] = bal
        w = wide_mult if r.wide else 1.0
        bal += cyc[r.key] * 0.006 * (1.5 if r.strong else 0.5) * w * 2 ** lv[r.key] * r.ru
        lv[r.key] = 0 if r.ru > 0 or lv[r.key] >= 2 else lv[r.key] + 1
        peak = max(peak, bal)
        mdd = max(mdd, (peak - bal) / peak)
        yr[pd.Timestamp(r.en_ms, unit="ms").year] = bal
    prev, out = 10_000.0, []
    for y in sorted(yr):
        out.append(f"{(yr[y] / prev - 1) * 100:.0f}")
        prev = yr[y]
    return round(bal), round(mdd * 100, 1), "/".join(out)


def main():
    d = ax.get_outcomes(False)
    trs = {}
    rows = []
    for k, sym in (("g", "XAUUSD"), ("e", "XAUEUR")):
        x = ax.pick(d, sym)
        x = x[pd.to_datetime(x.en_ms, unit="ms") < END].reset_index(drop=True)
        en = pd.to_datetime(x.en_ms, unit="ms")
        x["r2"] = ratio(sym, en)
        x["wide"] = x.r2 >= TH
        x["y"] = en.dt.year
        trs[k] = x
        for (strong, wide), g in x.groupby(["strong", "wide"]):
            rows.append({"종목": sym, "전일고저": "넘음(강)" if strong else "안넘음(약)", "2시간범위": "넓음" if wide else "좁음",
                         "건수": len(g), "승률%": round((g.ru > 0).mean() * 100, 1), "평균R": round(g.ru.mean(), 3),
                         "22~24 R": round(g.ru[g.y <= 2024].mean(), 3), "25~26 R": round(g.ru[g.y >= 2025].mean(), 3)})
        for wide, g in x.groupby("wide"):
            rows.append({"종목": sym, "전일고저": "전체", "2시간범위": "넓음" if wide else "좁음", "건수": len(g),
                         "승률%": round((g.ru > 0).mean() * 100, 1), "평균R": round(g.ru.mean(), 3),
                         "22~24 R": round(g.ru[g.y <= 2024].mean(), 3), "25~26 R": round(g.ru[g.y >= 2025].mean(), 3)})
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print(pd.DataFrame(rows).to_string(index=False))
    # 마틴 단계 규칙 확인: ax.martin과 같은 결과가 나와야 함 (가중 1.0)
    base = ax.martin(trs, bal0=10_000.0, risk=0.006)
    print("ax.martin 최종", round(base.bal.iat[-1]))
    for wm in (1.0, 1.5):
        for sel in ("금+EUR", "금만"):
            t = trs if sel == "금+EUR" else {"g": trs["g"], "e": trs["e"].assign(wide=False)}
            print(f"넓은 날 x{wm} ({sel})", martin_w(t, wm))


if __name__ == "__main__":
    main()
