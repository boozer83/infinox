"""idea_screen에서 남은 2개 더 확인 (2026-10-29, 실제 틱): 5. 1시간봉 EMA 교차 / 8. 금/은 비율 되돌림

1) 이웃 설정 촘촘히: EMA 빠른 3~8 x 느린 10~30 x 손절 2/2.5/3 ATR, GSR 1시간봉 N 50~200 x z 1.5~3 (+ 30분봉·2시간봉 N100)
   해마다 PF, 거래당 0.2$ 비용을 더 뺀 PF (미끄러짐·넓은 스프레드 가정), 둘 다 해마다 1.0 넘는 설정 비율
2) 지금 봇들과 일별 손익 상관 (같은 lot 기준 $/oz): SAR 단타 (results/sar_us_now_*), 추세 4시간 (strategies.trend H4 N25 3ATR),
   PA 1시간 (results/pa_master_trades_h1_novol_3r.csv)
3) 대표 설정 거래 목록 저장 -> GPT 교차 검증용 (results/idea_ema_h1_5_15.csv, idea_gsr_h1_100_25.csv)

체결: 틱 대신 1분봉 (gpt_data/XAUUSD/M1.csv.gz, Bid OHLC + 그 봉 평균 스프레드) -> 메모리 8GB PC에서 봇과 같이 돌리려고
  진입 = 진입 시각 이후 첫 1분봉 시가 (매수 + 스프레드), 손절·익절 = 1분봉 고가·저가로 처음 닿는 봉
  (매도는 고가·저가 + 스프레드), 같은 봉에서 둘 다 닿으면 손절로 (보수적), 시가가 이미 넘었으면 시가에 체결
  청산 = 청산 시각 이후 첫 1분봉 시가. 틱 결과(idea_screen)와 대표 설정을 나란히 비교 (check)

사용법: python -m backtests.idea_deep check | grid | corr
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.idea_screen import GPT, LightStore, ema_cross, gsr
from backtests.scalp_ideas import m5_bars
from backtests.strategies import execute, pf, resample, trend

_m1 = {}


def m1():
    if not _m1:
        d = pd.read_csv(GPT / "XAUUSD" / "M1.csv.gz", usecols=["time", "open", "high", "low", "spread"], parse_dates=["time"])
        _m1.update(t=d.time.values.astype("datetime64[ms]").astype(np.int64), o=d.open.values, h=d.high.values,
                   l=d.low.values, sp=d.spread.values)
    return _m1


def execute_m1(specs, store=None):
    """strategies.execute와 같은 입력·출력, 체결만 1분봉으로"""
    b = m1()
    t, o, h, l, sp = b["t"], b["o"], b["h"], b["l"], b["sp"]
    out, busy = [], -1
    for s in specs:
        if s["entry"] < busy or s["exit"] <= s["entry"]:
            continue
        k, x = np.searchsorted(t, s["entry"]), np.searchsorted(t, s["exit"])
        if k >= len(t) - 1 or x >= len(t) or x <= k:
            continue
        side = s["side"]
        e = o[k] + sp[k] if side == 1 else o[k]
        sl = s["sl_abs"] if "sl_abs" in s else e - side * s["sl_dist"]
        risk = (e - sl) * side
        if risk <= 0:
            continue
        tp = e + side * s["tp_r"] * risk if s.get("tp_r") else None
        if side == 1:
            lo, hi, op = l[k:x], h[k:x], o[k:x]
            hs = lo <= sl
            ht = hi >= tp if tp else np.zeros(len(lo), bool)
        else:
            lo, hi, op = l[k:x] + sp[k:x], h[k:x] + sp[k:x], o[k:x] + sp[k:x]
            hs = hi >= sl
            ht = lo <= tp if tp else np.zeros(len(lo), bool)
        js = np.flatnonzero(hs)
        jt = np.flatnonzero(ht)
        j1 = js[0] if js.size else None
        j2 = jt[0] if jt.size else None
        if j1 is not None and (j2 is None or j1 <= j2):
            px = op[j1] if (op[j1] - sl) * side < 0 else sl            # 시가가 이미 손절 너머면 시가
            j = k + j1
        elif j2 is not None:
            px = op[j2] if (op[j2] - tp) * side > 0 else tp
            j = k + j2
        else:
            j = x
            px = o[x] if side == 1 else o[x] + sp[x]
        pnl = (px - e) * side
        out.append((t[k], t[j], side, pnl, pnl / risk, risk))
        busy = t[j]
    tr = pd.DataFrame(out, columns=["open_ms", "close_ms", "side", "pnl_oz", "r", "risk"])
    tr["close"] = pd.to_datetime(tr.close_ms, unit="ms")
    return tr


execute = execute_m1                                   # 이 파일 안의 execute는 전부 1분봉 체결

RES = Path(__file__).resolve().parent.parent / "results"
YEARS = [2022, 2023, 2024, 2025, 2026]
COST = 0.2


def row(name, tr):
    y = tr.close.dt.year
    r = {"설정": name, "거래수": len(tr), "승률%": round((tr.pnl_oz > 0).mean() * 100, 1), "oz합": round(tr.pnl_oz.sum(), 1)}
    for yr in YEARS:
        r[f"PF{yr % 100}"] = pf(tr.pnl_oz[y == yr])
    for yr in YEARS:
        r[f"비용PF{yr % 100}"] = pf(tr.pnl_oz[y == yr] - COST)
    r["해마다>1"] = all(r[f"PF{yr % 100}"] > 1 for yr in YEARS)
    r["비용후 해마다>1"] = all(r[f"비용PF{yr % 100}"] > 1 for yr in YEARS)
    return r


def grid():
    m5 = m5_bars()
    bars = {"M30": resample(m5, "30min"), "H1": resample(m5, "1h"), "H2": resample(m5, "2h")}
    rule = {"M30": "30min", "H1": "1h", "H2": "2h"}
    store = None
    rows = []
    for f, s, k in itertools.product([3, 5, 8], [10, 15, 21, 30], [2.0, 2.5, 3.0]):
        if f >= s:
            continue
        specs = ema_cross(bars["H1"], f, s, False)
        if k != 2.5:                                   # ema_cross는 손절 2.5 ATR 고정 -> 비율로 바꿈
            for x in specs:
                x["sl_dist"] *= k / 2.5
        tr = execute(specs, store)
        rows.append({"전략": "EMA H1", **row(f"{f}/{s} 손절{k}", tr)})
        if (f, s, k) == (5, 15, 2.5):
            tr.to_csv(RES / "idea_ema_h1_5_15.csv", index=False)
        print("EMA", f, s, k, flush=True)
    for tf, n, z in [("H1", n, z) for n, z in itertools.product([50, 75, 100, 150, 200], [1.5, 2.0, 2.5, 3.0])] + \
                    [("M30", 100, 2.0), ("M30", 100, 2.5), ("H2", 100, 2.0), ("H2", 100, 2.5)]:
        tr = execute(gsr(bars[tf], rule[tf], n, z), store)
        rows.append({"전략": "GSR", **row(f"{tf} N{n} z{z}", tr)})
        if (tf, n, z) == ("H1", 100, 2.5):
            tr.to_csv(RES / "idea_gsr_h1_100_25.csv", index=False)
        print("GSR", tf, n, z, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "idea_deep_grid.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.max_rows", 200)
    print(df.to_string(index=False))
    print(df.groupby("전략")[["해마다>1", "비용후 해마다>1"]].mean().round(2))


def daily(tr, col="pnl_oz", tcol="close"):
    return tr.groupby(pd.to_datetime(tr[tcol]).dt.normalize())[col].sum()


def corr():
    ema_tr = pd.read_csv(RES / "idea_ema_h1_5_15.csv", parse_dates=["close"])
    gsr_tr = pd.read_csv(RES / "idea_gsr_h1_100_25.csv", parse_dates=["close"])
    sar = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("sar_us_now_*.csv"))])
    sar["close"] = pd.to_datetime(sar.open)
    pa = pd.read_csv(RES / "pa_master_trades_h1_novol_3r.csv")
    pa_t = "close" if "close" in pa else "open"
    pa_c = "pnl_oz" if "pnl_oz" in pa else "oz"
    m5 = m5_bars()
    tt = execute(trend(resample(m5, "4h"), 25, 3.0))
    s = pd.DataFrame({"EMA H1": daily(ema_tr), "GSR H1": daily(gsr_tr), "SAR 단타": daily(sar, "oz"),
                      "추세 4시간": daily(tt), "PA 1시간": daily(pa, pa_c, pa_t)}).fillna(0)
    pd.set_option("display.width", 200)
    print("일별 손익 상관\n", s.corr().round(2))
    mo = s.groupby(s.index.to_period("M")).sum()
    print("월별 손익 상관\n", mo.corr().round(2))
    s.to_csv(RES / "idea_deep_daily.csv", encoding="utf-8-sig")


def check():
    """틱 결과(results/idea_screen_all.csv)와 같은 설정을 1분봉으로"""
    m5 = m5_bars()
    h1 = resample(m5, "1h")
    tick = pd.read_csv(RES / "idea_screen_all.csv")
    for name, specs, key in [("EMA H1 5/15", ema_cross(h1, 5, 15, False), "H1 5/15"),
                             ("EMA H1 9/21", ema_cross(h1, 9, 21, False), "H1 9/21"),
                             ("GSR H1 N100 z2.5", gsr(h1, "1h", 100, 2.5), "H1 N100 z2.5"),
                             ("GSR H4 N50 z2.0", gsr(resample(m5, "4h"), "4h", 50, 2.0), "H4 N50 z2.0")]:
        r = row(name + " (1분봉)", execute_m1(specs))
        t = tick[tick.설정 == key].iloc[0]
        print(r["설정"], r["거래수"], r["승률%"], r["oz합"], [r[f"PF{y % 100}"] for y in YEARS])
        print(f"  틱: {t['거래수']} {t['승률%']} {t['oz합']}", [t[f"PF {y}"] for y in YEARS], flush=True)


if __name__ == "__main__":
    {"check": check, "grid": grid, "corr": corr}[sys.argv[1]]()
