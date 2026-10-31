"""strategies.execute 허점 수정 전/후 비교 (2026-10-31, 10/29 기록의 청산 틱 ms 차이로 반대 진입을 건너뛰던 문제)

같은 틱, 같은 신호로 busy만 다르게: 전 = 청산 틱 시각까지 / 후 = 시간 청산이면 신호 시각까지
idea_screen과 같은 판정: 8개 설정 중 2022~23 PF 1등 (학습 30건 이상), 2024/25/26 포함 전 연도 PF 1.1 이상이면 통과

사용법: python -m backtests.execute_fix_check 6 7 trend
        python -m backtests.execute_fix_check 9    (data/bars/{EURUSD,USDJPY,GBPUSD}_M1_*.csv.zip 필요)
"""
import itertools
import sys
import zipfile
from pathlib import Path

import pandas as pd

import backtests.idea_screen as ids
from backtests.idea_screen import LightStore
from backtests.scalp_ideas import m5_bars
from backtests.strategies import PASS_PF, YEARS, evaluate, execute, resample, trend

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def run(specs, store, old):
    """execute를 거래 하나씩 불러서 busy를 여기서 관리"""
    rows, busy = [], -1
    for s in specs:
        if s["entry"] < busy or s["exit"] <= s["entry"]:
            continue
        r = execute([s], store)
        if r.empty:
            continue
        rows.append(r)
        c = int(r.close_ms.iat[0])
        busy = c if old or c < s["exit"] else s["exit"]
    return pd.concat(rows, ignore_index=True)


def fx_close(sym, rule):
    parts = []
    for f in sorted((ROOT / "data" / "bars").glob(f"{sym}_M1_*.csv.zip")):
        with zipfile.ZipFile(f) as z:
            parts.append(pd.read_csv(z.open(z.namelist()[0]), usecols=["time", "open", "high", "low", "close"], parse_dates=["time"]))
    d = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time")
    return resample(d, rule).set_index("time").close


def main():
    want = sys.argv[1:]
    m5 = m5_bars()
    bars = {"M15": resample(m5, "15min"), "H1": resample(m5, "1h"), "H4": resample(m5, "4h")}
    rule = {"H1": "1h", "H4": "4h"}
    if "9" in want:
        for sym in ("EURUSD", "USDJPY", "GBPUSD"):
            for r in rule.values():
                ids._other[(sym, r)] = fx_close(sym, r)
    P = itertools.product
    fams = {
        "6": ("6. MACD+EMA정렬", [(f"{tf} 손절 {k}ATR 청산 {e}", lambda tf=tf, k=k, e=e: ids.macd_align(bars[tf], k, e))
                                 for tf, k, e in P(["M15", "H1"], [2.0, 3.0], ["히스토그램", "정렬"])]),
        "7": ("7. SuperTrend", [(f"{tf} ({n},{mu})", lambda tf=tf, n=n, mu=mu: ids.st_follow(bars[tf], n, mu))
                                for tf, (n, mu) in [("M15", (10, 3)), ("M15", (10, 2)), ("H1", (10, 3)), ("H1", (10, 2)),
                                                    ("H1", (14, 3)), ("H4", (10, 3)), ("H4", (10, 2)), ("H4", (14, 3))]]),
        "9": ("9. 달러 역상관", [(f"{tf} EMA{f}/{s} 손절 {k}ATR", lambda tf=tf, f=f, s=s, k=k: ids.dxy(bars[tf], rule[tf], f, s, k))
                               for tf, (f, s), k in P(["H1", "H4"], [(9, 21), (20, 50)], [2.0, 3.0])]),
        "trend": ("H4 추세 N25 3ATR", [("H4 N25 3ATR", lambda: trend(bars["H4"], 25, 3.0))]),
    }
    store = LightStore()
    rows = []
    for key in want:
        fam, grid = fams[key]
        for name, make in grid:
            specs = make()
            for label, old in (("전", True), ("후", False)):
                tr = run(specs, store, old)
                n_train = int((tr.close.dt.year < 2024).sum())
                rows.append({"전략": fam, "설정": name, "execute": label, **evaluate(tr), "oz합": round(tr.pnl_oz.sum(), 1), "학습건": n_train})
                print(fam, name, label, len(tr), rows[-1]["oz합"], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"execute_fix_{'_'.join(want)}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.max_rows", 200)
    print(df.to_string(index=False))
    pf_cols = [f"PF {y}" for y in YEARS]
    for (fam, lab), g in df.groupby(["전략", "execute"], sort=False):
        g = g[g.학습건 >= 30]
        if g.empty:
            continue
        b = g.loc[g["PF 22-23"].idxmax()]
        print(f"{fam} {lab}: {b.설정} PF {[b[c] for c in pf_cols]} -> {'통과' if b[pf_cols].min() >= PASS_PF else '탈락'}")


if __name__ == "__main__":
    main()
