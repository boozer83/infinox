"""서프라이즈 매매를 지표 종류별로 고르기 (실제 틱, news_surprise_exits와 같은 진입·청산)

대상: |S| >= 1.5, 엇갈림 없음. 진입 발표 L초 뒤 +0.3초 (L = 1, 1.5), 청산 큰이익추적 5/2.5/15분 / 30분 보유, 체결마다 0.1$
지표 종류 = 그 발표의 대표 지표 (news_moves2의 key: 같은 시각 high 중 첫 번째, 없으면 moderate 첫 번째)
고르는 방법 (모두 고를 때 2025~2026은 안 봄)
  변동성 X%    : 2022~2024 그 지표 발표 전부(서프라이즈 무관, 매매 결과 안 씀)에서 15분 0.2% 넘게 움직인 비율 >= X%
  중간 움직임 X%: 2022~2024 그 지표 발표 15분 움직임 크기 중간값 >= X%
  매매 성적    : 2022~2024 그 지표로 한 모의 매매 N번 이상, 평균 > 0
  워크포워드   : 해마다 그 해 전까지 발표 전부로 변동성 비율을 다시 계산 (2023년부터, 2022는 빈칸)
결과: results/news_surprise_types.csv, 발표별 매매 results/news_surprise_trades.csv

사용법: python -m backtests.news_surprise_types
"""
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.long_period as lb
from common.surprise import outcome

ROOT = Path(__file__).resolve().parents[1]
LAGS = (1.0, 1.5)
SLIP = 0.1
EXITS = ("큰이익추적 5/2.5/15분", "30분 보유")


def trades():
    c = pd.read_csv(ROOT / "results" / "news_surprise_events.csv", parse_dates=["time"])
    c = c[(c.S.abs() >= 1.5) & (c.npos.fillna(0) * c.nneg.fillna(0) == 0)].reset_index(drop=True)
    ms = c.time.values.astype("datetime64[ms]").astype(np.int64)
    rows = []
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        idx = np.flatnonzero((ms >= int(s.timestamp() * 1000)) & (ms < int(e.timestamp() * 1000)))
        if not len(idx):
            continue
        t, bid, ask = lb.load_month(y, m)
        for i in idx:
            m0, side = int(ms[i]), int(np.sign(c.S[i]))
            k0, k1 = np.searchsorted(t, [m0 - 60_000, m0 + 1_800_000])
            if k1 - k0 < 10:
                continue
            row = {"time": c.time[i], "key": c.key[i], "S": c.S[i], "y": c.y[i]}
            for lag in LAGS:
                r = outcome(t[k0:k1], bid[k0:k1], ask[k0:k1], m0, side, m0 + int(lag * 1000) + 300)
                for ex in EXITS:
                    row[f"{ex} {lag:g}초"] = r[ex] - 2 * SLIP if r else np.nan
            rows.append(row)
        print(f"  {y}-{m:02d} 완료", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results" / "news_surprise_trades.csv", index=False, encoding="utf-8-sig")
    return d


def vol_table(mv, years):
    x = mv[mv.y.isin(years)]
    return x.groupby("key").agg(n=("m15", "size"), big=("m15", lambda v: (v.abs() >= 0.2).mean()),
                                med=("m15", lambda v: v.abs().median()))


def main():
    mv = pd.read_csv(ROOT / "results" / "news_moves2.csv", parse_dates=["time"])
    mv["y"] = mv.time.dt.year
    mv["m15"] = mv.d15m / mv.ref * 100
    tr = trades()
    vt = vol_table(mv, range(2022, 2025))
    vt = vt[vt.n >= 8]
    sels = {"전부": np.ones(len(tr), bool)}
    for x in (0.1, 0.2, 0.3, 0.4):
        sels[f"변동성 {x:.0%} 이상"] = tr.key.isin(vt.index[vt.big >= x]).values
    for x in (0.05, 0.08, 0.12):
        sels[f"중간 움직임 {x}% 이상"] = tr.key.isin(vt.index[vt.med >= x]).values
    wf = np.zeros(len(tr), bool)
    for y in range(2023, 2027):
        v = vol_table(mv, range(2022, y))
        good = v.index[(v.n >= 8) & (v.big >= 0.2)]
        wf |= (tr.y.values == y) & tr.key.isin(good).values
    sels["워크포워드 변동성 20%"] = wf
    rows = []
    for col in [c for c in tr.columns if c.endswith("초")]:
        ins = tr[tr.y <= 2024].groupby("key")[col].agg(["size", "mean"])
        sel_perf = {f"매매 성적 {n}번 이상 평균+": tr.key.isin(ins.index[(ins["size"] >= n) & (ins["mean"] > 0)]).values for n in (3, 5)}
        for thr in (1.5, 2.0):
            for name, sel in {**sels, **sel_perf}.items():
                k = sel & (tr.S.abs().values >= thr) & tr[col].notna().values
                v, yy = tr[col].values[k], tr.y.values[k]
                if not len(v):
                    continue
                row = {"청산·지연": col, "|S|": thr, "고르기": name, "매매 수": len(v), "승률%": round((v > 0).mean() * 100, 1)}
                for yr in range(2022, 2027):
                    row[str(yr)] = round(v[yy == yr].sum(), 1)
                row["2025~26 매매"] = int((yy >= 2025).sum())
                row["2025~26 승률%"] = round((v[yy >= 2025] > 0).mean() * 100, 1) if (yy >= 2025).any() else np.nan
                row["2025~26"] = round(v[yy >= 2025].sum(), 1)
                row["5년"] = round(v.sum(), 1)
                row["최악"] = round(v.min(), 1)
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "news_surprise_types.csv", index=False, encoding="utf-8-sig")
    vt.sort_values("big", ascending=False).round(3).to_csv(ROOT / "results" / "news_key_volatility.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    for col in ("큰이익추적 5/2.5/15분 1.5초", "30분 보유 1.5초"):
        for thr in (1.5, 2.0):
            print(f"\n=== {col}, |S| >= {thr}")
            print(out[(out["청산·지연"] == col) & (out["|S|"] == thr)].drop(columns=["청산·지연", "|S|"]).to_string(index=False))


if __name__ == "__main__":
    main()
