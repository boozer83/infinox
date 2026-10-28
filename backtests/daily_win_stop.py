"""SAR 단타 하루 N승이면 그날 운행 종료 (2026-10-28 사용자 요청, 2022-01 ~ 2026-10-16)

SAR 단타 거래: results/sar_us_now_YYYY.csv (미국장 제외 켬 = 지금 봇) / sar_us_all_YYYY.csv (미국장도 진입)
  하루(서버 날짜) 이긴 거래가 N번 나오면 그 뒤 신호는 안 들어감 (한 번에 1개라 앞 거래는 그대로, 뒤만 잘라내면 정확)
같이 종료: 그날 N승째 청산 시각 뒤에 진입하는 아시아 돌파(금 + XAUEUR) / 지수(NAS100, GER40)도 안 들어감 (추세는 그대로)
  아시아: backtests.asia_xaueur outcomes, 거른 뒤 마틴 다시 계산 (각 0.6%, 1만$에서 연속)
  지수: results/index_ticks.csv, 거래당 0.25% 위험 (live_withdraw와 같은 근사). 진입 시각은 GER 서버 10:30, NAS 17:00으로 둠 (가장 이른 진입)
단위: SAR은 1,000$당 0.05 lot라 1$/oz = 잔고 0.5%. 모두 '잔고 %' 단순합 (복리 아님, 아시아만 마틴 잔고 기준 연수익)

사용법: python -m backtests.daily_win_stop
"""
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.asia_xaueur as ax

RES = Path(__file__).resolve().parent.parent / "results"
YEARS = range(2022, 2027)
END = pd.Timestamp("2026-10-17")


def cut(tr, n):
    """(남은 거래, 날짜 -> 그날 n승째 청산 시각)"""
    tr = tr.sort_values("open").reset_index(drop=True)
    if not n:
        return tr, {}
    win = (tr.oz > 0).astype(int)
    before = win.groupby(tr.day).cumsum() - win          # 이 거래 전까지 그날 이긴 수
    kept = tr[before < n]
    nth = kept[(kept.oz > 0) & (before[kept.index] == n - 1)]
    return kept, dict(zip(nth.day, nth.close))


def asia_year(stop):
    d = ax.get_outcomes(False)
    trs = {}
    for k, sym in (("g", "XAUUSD"), ("e", "XAUEUR")):
        x = ax.pick(d, sym)
        x = x[pd.to_datetime(x.en_ms, unit="ms") < END]
        if stop:
            en = pd.to_datetime(x.en_ms, unit="ms")
            st = pd.to_datetime(en.dt.strftime("%Y-%m-%d").map(stop))
            x = x[~(st.notna() & (en > st))]
        trs[k] = x.reset_index(drop=True)
    b = ax.martin(trs, bal0=10_000.0, risk=0.006)
    yb = b.groupby(b.t.dt.year).bal.last()
    prev = pd.Series([10_000.0] + list(yb.values[:-1]), index=yb.index)
    return ((yb / prev - 1) * 100).round(1), len(trs["g"]) + len(trs["e"])


def index_year(stop):
    d = pd.read_csv(RES / "index_ticks.csv", parse_dates=["날짜"])
    out = []
    for (sym, y), g in d.groupby(["종목", "y"]):
        risk = g.atr if sym == "NAS100" else g.pts[g["손절"]].abs().median()
        en = g["날짜"] + pd.Timedelta(hours=17 if sym == "NAS100" else 10.5)
        out.append(pd.DataFrame({"en": en, "pct": 0.25 * g.pts / risk}))
    x = pd.concat(out)
    if stop:
        st = x.en.dt.normalize().dt.strftime("%Y-%m-%d").map(stop)
        x = x[~(pd.to_datetime(st).notna() & (x.en > pd.to_datetime(st)))]
    return x.groupby(x.en.dt.year).pct.sum().round(1), len(x)


def main():
    rows = []
    for name, pat in (("미국장 제외 켬(지금)", "sar_us_now_{}.csv"), ("미국장도 진입", "sar_us_all_{}.csv")):
        sar = pd.concat([pd.read_csv(RES / pat.format(y), parse_dates=["open"]) for y in YEARS])
        sar["close"] = sar.open + pd.to_timedelta(sar.mins, unit="min")
        sar["day"] = sar.day.astype(str)
        for n in (0, 3, 4):
            kept, stop = cut(sar, n)
            yr = kept.open.dt.year
            row = {"단타": name, "규칙": f"{n}승이면 종료" if n else "계속", "단타 거래": len(kept),
                   "승률%": round((kept.oz > 0).mean() * 100, 1), "단타 oz": round(kept.oz.sum(), 1)}
            row.update({f"단타 {y}": round(kept.oz[yr == y].sum(), 1) for y in YEARS})
            eq = kept.oz.cumsum()
            row["단타 낙폭oz"] = round((eq.cummax() - eq).max(), 1)
            row["종료한 날"] = len(stop)
            for tag, s in (("그대로", {}), ("같이 종료", stop)):
                if n == 0 and tag == "같이 종료":
                    continue
                a, na = asia_year(s)
                ix, ni = index_year(s)
                row[f"아시아 {tag} 거래"] = na
                row[f"아시아 {tag} 연%"] = "/".join(f"{a.get(y, 0):g}" for y in YEARS)
                row[f"지수 {tag} 거래"] = ni
                row[f"지수 {tag} %합"] = round(ix.sum(), 1)
            rows.append(row)
            print("완료", name, n, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RES / "daily_win_stop.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.T.to_string())


if __name__ == "__main__":
    main()
