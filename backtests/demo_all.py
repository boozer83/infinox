"""데모에서 도는 봇 전체를 연도별 1,000달러 시작으로 (실제 틱, 2022-01 ~ 2026-10)

  SAR 단타 / 단타 2마틴 : backtests/sar_martin.py 거래 목록 (sar_martin_trades_YYYY.csv, 지금 봇 그대로 ATR 2.5$ 고정)
  H4 추세              : 지금 봇 (N25 진입, 12봉 청산, 손절 3 ATR, 2% 위험, 최소 0.01) + 4시간봉 구름 필터 + 그림자
                         (구름 반대 신호도 들고 있는 것처럼 시간을 차지 = 필터 없는 거래 순서에서 구름 맞는 것만 남김)
  아시아 3마틴 v2      : backtests.asia_v2 v2 규칙, BAL0 = 1,000
  뉴스 양방향          : backtests.news_cut '비율 0.05% / 15분', CPI·고용·FOMC 금리 결정, lot 1,000달러당 0.05
  헤지 마틴            : 따로 안 돌림 (hedge_martin_trader.py 머리말의 연도별 결과, lot이 1,000달러당 0.01이라 % 같음)
해마다 1,000달러로 새로 시작 (복리는 그 해 안에서만)

사용법: python -m backtests.demo_all            (전부, 결과 demo_all.csv)
        python -m backtests.demo_all trend      (하나만: sar / news / trend / asia / hedge, 결과 demo_all_<이름>.csv)
        python -m backtests.demo_all merge      (demo_all_*.csv 합쳐서 표)
"""
import gc
import math

import numpy as np
import pandas as pd

import backtests.asia_v2 as av
import backtests.long_period as lb
import backtests.news_cut as nc
import backtests.news_hk as nh
import backtests.sar_martin as sm
from backtests.gold_rest_others import load_m1, resample
from backtests.ichimoku_others import attach, keep_mask, trend_entries
from backtests.strategies import TickStore, bar_exit_specs, execute
from indicators import atr

BAL0 = 1000.0
YEARS = range(2022, 2027)


def year_row(name, d, col_bal="bal"):
    """d: 그 해 거래 (bal = 거래 뒤 잔고). 1,000달러 시작 기준 요약"""
    if not len(d):
        return {"봇": name, "거래": 0, "연말 잔고$": BAL0, "수익률%": 0.0, "최대낙폭%": 0.0}
    eq = pd.concat([pd.Series([BAL0]), d[col_bal].reset_index(drop=True)])
    dd = ((eq.cummax() - eq) / eq.cummax()).max()
    return {"봇": name, "거래": len(d), "연말 잔고$": round(eq.iat[-1]), "수익률%": round((eq.iat[-1] / BAL0 - 1) * 100, 1),
            "최대낙폭%": round(dd * 100, 1)}


def sar_rows():
    rows = []
    for y in YEARS:
        try:
            tr = pd.read_csv(f"sar_martin_trades_{y}.csv", parse_dates=["open"])
        except (FileNotFoundError, pd.errors.EmptyDataError):
            tr = pd.DataFrame(columns=["open", "day", "mins", "oz", "mae"])
        tr["day"] = pd.to_datetime(tr.day).dt.date if len(tr) else tr.day
        for name, case in (("SAR 단타", (0, 1.0, None, False)), ("단타 2마틴", (2, 1.0, None, True, 2))):
            if not len(tr):
                rows.append({"년": y, **year_row(name, tr)})
                continue
            r = sm.sim(tr, *case)
            rows.append({"년": y, "봇": name, "거래": r["거래"], "연말 잔고$": r["최종잔고$"],
                         "수익률%": round((r["최종잔고$"] / BAL0 - 1) * 100, 1), "최대낙폭%": r["최대낙폭%"]})
    return rows


def size_year(tr, risk_pct=0.02):
    bal, out = BAL0, []
    for r in tr.itertuples():
        lot = max(0.01, math.floor(bal * risk_pct / (r.risk * 100) * 100 + 1e-9) / 100)
        bal += r.pnl_oz * lot * 100
        out.append(bal)
    return tr.assign(bal=out)


def trend_rows():
    m1 = load_m1("XAUUSD")
    h4 = resample(m1, 240)
    del m1
    gc.collect()
    ent, exit_fn = trend_entries(h4)
    sl = (3.0 * atr(h4)).values
    specs = bar_exit_specs(h4, ent, exit_fn, sl)
    tr = execute(specs, TickStore())          # 필터 없는 순서 (한 번에 1개) = 그림자 포함 시간 차지
    spec_t = np.array([s["entry"] for s in specs])
    t_ms = h4.time.values.astype("datetime64[ms]").astype(np.int64)
    k = np.searchsorted(spec_t, tr.open_ms.values, side="right") - 1
    sig_bar = np.searchsorted(t_ms, spec_t[k]) - 1              # 진입 = 신호 봉 다음 봉 시가
    side = tr.side.values
    t_sig = (h4.time + pd.Timedelta(hours=4)).values[sig_bar]
    ich = attach(t_sig, h4, 240, h4.close.values[sig_bar])
    keep = keep_mask(ich, side, "구름 위/아래만")
    tr = tr[np.asarray(keep)].reset_index(drop=True)
    print(f"  H4: 필터 없는 거래 {len(keep)} -> 구름 맞는 거래 {len(tr)}", flush=True)
    rows = []
    for y in YEARS:
        rows.append({"년": y, **year_row("H4 추세", size_year(tr[tr.close.dt.year == y]))})
    return rows


def asia_rows():
    av.BAL0 = BAL0
    sig = av.day_signals()
    sig = sig[sig.side == sig.mom]
    oc = av.outcomes(sig)
    rows = []
    for y in YEARS:
        g = sig[sig.day.dt.year == y]
        tr, halted = av.simulate(g, oc, True, 0.75, True)
        rows.append({"년": y, **year_row("아시아 3마틴", tr), **({"비고": f"{halted.date()} 중단"} if halted is not None else {})})
    return rows


def news_rows():
    ev = nh.events()
    ev["key"] = ev.currency + " " + ev.event.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    ev = ev[ev.key.isin(nc.MAJOR3)].reset_index(drop=True)
    ev_ms = ev.t.values.astype("datetime64[ms]").astype(np.int64)
    oz = np.full(len(ev), np.nan)
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        idx = np.flatnonzero((ev_ms >= int(s.timestamp() * 1000)) & (ev_ms < int(e.timestamp() * 1000)))
        if not len(idx):
            continue
        t, bid, ask = lb.load_month(y, m)
        for i in idx:
            ms = int(ev_ms[i])
            k0, k1 = int(np.searchsorted(t, ms - nc.PRE)), int(np.searchsorted(t, ms + 1_000_000))
            r = nc.one(t[k0:k1], bid[k0:k1], ask[k0:k1], ms, ("pct", 0.05), "15분", 1.0)
            if r is not None:
                oz[i] = r[0]
        del t, bid, ask
        gc.collect()
    ev["oz"] = oz
    ev = ev.dropna(subset=["oz"])
    rows = []
    for y in YEARS:
        g = ev[ev.t.dt.year == y]
        bal, out = BAL0, []
        for x in g.oz:
            lot = max(0.01, math.floor(bal / 1000 * 0.05 * 100 + 1e-9) / 100)
            bal += x * lot * 100
            out.append(bal)
        rows.append({"년": y, **year_row("뉴스 양방향", g.assign(bal=out))})
    return rows


def hedge_rows():
    hm = {2022: -81, 2023: -32, 2024: -34, 2025: 55, 2026: 33}
    return [{"년": y, "봇": "헤지 마틴", "거래": np.nan, "연말 잔고$": round(BAL0 * (1 + p / 100)), "수익률%": p,
             "비고": "1만 달러 결과 비율"} for y, p in hm.items()]


PARTS = {"sar": sar_rows, "news": news_rows, "trend": trend_rows, "asia": asia_rows, "hedge": hedge_rows}


def main():
    import glob
    import sys
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    if len(sys.argv) > 1 and sys.argv[1] in PARTS:
        d = pd.DataFrame(PARTS[sys.argv[1]]())
        d.to_csv(f"demo_all_{sys.argv[1]}.csv", index=False, encoding="utf-8-sig")
        print(d.to_string(index=False))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("demo_all_*.csv"))], ignore_index=True)
        print(d.pivot(index="봇", columns="년", values="수익률%").to_string())
        print(d.to_string(index=False))
        return
    rows = sar_rows()
    print("  SAR 완료", flush=True)
    rows += news_rows()
    print("  뉴스 완료", flush=True)
    rows += trend_rows()
    gc.collect()
    print("  H4 완료", flush=True)
    rows += asia_rows()
    print("  아시아 완료", flush=True)
    hm = {2022: -81, 2023: -32, 2024: -34, 2025: 55, 2026: 33}
    rows += [{"년": y, "봇": "헤지 마틴", "거래": np.nan, "연말 잔고$": round(BAL0 * (1 + p / 100)), "수익률%": p,
              "비고": "1만 달러 결과 비율"} for y, p in hm.items()]
    d = pd.DataFrame(rows)
    d.to_csv("demo_all.csv", index=False, encoding="utf-8-sig")
    print(d.pivot(index="봇", columns="년", values="수익률%").to_string())
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
