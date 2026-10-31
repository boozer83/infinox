"""SAR 머신러닝 거르기 2: 표본 늘리기 (필터 전 SAR 전환 전부 + 다른 종목) (2026-10-29 사용자 요청)

표본: 종목마다 5분봉 SAR 전환 전부 (RSI·1시간봉·과열·시간대 필터 전). 2022-01 ~ 2026-10-22
  데이터 gpt_data/<종목>/M1.csv.gz (Bid OHLC + 봉 평균 스프레드) -> 5분봉·1시간봉으로 묶음
  거래: 전환 봉 끝 다음 1분봉 시가 진입 (매수 + 스프레드), 익절 3$ / 손절 7.5$ x 종목 배율, 다음 전환(반대 콩)에 청산
        손절·익절은 1분봉 고가·저가 (매도는 + 스프레드), 같은 봉 둘 다면 손절
        종목 배율 = 그 종목 5분봉 ATR 중간값 / 금 5분봉 ATR 중간값 (같은 위험 크기로 맞춤)
  특징 (진입 순간 값, 종목과 무관하게 ATR·% 단위): ext, risk, body, range, ema(EMA20 거리), side, prev_run,
        atr%(ATR/가격), atr_rel(ATR / 20일 평균 ATR), rsi_dir, h1_agree(1시간 SAR 같은 방향), h1_run, h1_flips(오늘 1시간 전환 수),
        시각 sin/cos, 요일
  금 '봇 신호' = 지금 봇 필터 통과 (RSI 같은 쪽, 1시간 같은 방향, ext <= 0.8, risk <= 2.8, ATR >= 2.5$, ATR <= 20일 x2,
        서버 15~21시 아님, 금요일 22:30 이후 아님, 1시간 전환 < 3. 2022~24는 ATR 기준을 금값 비례로 = sar_us_now와 같음)
검증 (시간순): 시험 해 2024 / 2025 / 2026, 학습 = 그 전 해 전부
  학습 묶음: A 금 봇 신호만 (전 실험과 같음) / B 금 전환 전부 / C 모든 종목 전환 전부 (최대 15만 건 무작위)
  시험 = 그해 금 봇 신호. 점수 하위 q% 쉼 (학습 기간 금 봇 신호 점수 기준 / 최근 200개 기준)
결과 results/sar_ml_multi.csv, 종목별 표본 data/sar_ml/<종목>.pkl

사용법: python -m backtests.sar_ml_multi data     (표본 만들기, 종목 하나씩)
        python -m backtests.sar_ml_multi fit
"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.sar_ml_filter import Boost, Logit
from indicators import atr, ema, psar, psar_levels, rsi

ROOT = Path(__file__).resolve().parent.parent
GPT = ROOT.parent / "gpt_data"
OUT = ROOT / "data" / "sar_ml"
RES = ROOT / "results"
SYMS = ["XAUUSD", "XAGUSD", "XAUEUR", "XPTUSD", "NAS100", "US30", "SPX500", "GER40", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD"]
FEATS = ["ext", "risk", "body", "range", "ema", "side", "prev_run", "atr_pct", "atr_rel", "rsi_dir", "h1_agree", "h1_run",
         "h1_flips", "hs", "hc", "dow"]
REF_PRICE = 4150.0
GOLD_ATR = None


def bars(m1, rule):
    b = m1.set_index("time").resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return b.reset_index()


def build(sym, gold_atr_med):
    m1 = pd.read_csv(GPT / sym / "M1.csv.gz", usecols=["time", "open", "high", "low", "spread"], parse_dates=["time"])
    m1c = pd.read_csv(GPT / sym / "M1.csv.gz", usecols=["close"])
    m1["close"] = m1c.close.values
    del m1c
    m5 = bars(m1, "5min")
    h, l, c, o = m5.high.values, m5.low.values, m5.close.values, m5.open.values
    up, sar, trig = psar_levels(h, l)
    a = atr(m5).values
    r = rsi(m5.close).values
    e20 = ema(m5.close, 20).values
    end = (m5.time + pd.Timedelta(minutes=5)).values
    a_rel = a / pd.Series(a).rolling(20 * 276, min_periods=2000).mean().values
    # 1시간봉 (그 5분봉 끝까지 완성된 것만)
    h1 = bars(m5, "1h")
    hup = psar(h1.high.values, h1.low.values)
    hrun = np.ones(len(hup), int)
    for i in range(1, len(hup)):
        hrun[i] = hrun[i - 1] + 1 if hup[i] == hup[i - 1] else 1
    hflip = np.r_[0, (hup[1:] != hup[:-1]).astype(int)]
    hday = h1.time.dt.normalize()
    hfl_today = pd.Series(hflip).groupby(hday.values).cumsum().values
    hq = pd.DataFrame({"end": h1.time + pd.Timedelta(hours=1), "hdir": np.where(hup, 1, -1), "h1_run": hrun,
                       "h1_flips": hfl_today, "hday": hday})
    flips = np.flatnonzero(up[1:] != up[:-1]) + 1
    run = np.ones(len(up), int)
    for i in range(1, len(up)):
        run[i] = run[i - 1] + 1 if up[i] == up[i - 1] else 1
    side = np.where(up[flips], 1, -1)
    f = pd.DataFrame({"t": pd.to_datetime(end[flips]), "side": side})
    f["ext"] = (c[flips] - trig[flips]) * side / a[flips]
    f["risk"] = (c[flips] - sar[flips]) * side / a[flips]
    f["body"] = np.abs(c[flips] - o[flips]) / a[flips]
    f["range"] = (h[flips] - l[flips]) / a[flips]
    f["ema"] = (c[flips] - e20[flips]) * side / a[flips]
    f["prev_run"] = run[flips - 1]
    f["atr"] = a[flips]
    f["atr_pct"] = a[flips] / c[flips] * 100
    f["atr_rel"] = a_rel[flips]
    f["rsi_dir"] = (r[flips] - 50) * side
    f["close"] = c[flips]
    j = pd.merge_asof(f[["t"]], hq, left_on="t", right_on="end", direction="backward")
    f["h1_agree"] = (j.hdir.values == side).astype(int)
    f["h1_run"] = j.h1_run.values
    f["h1_flips"] = np.where(j.hday.values == f.t.dt.normalize().values, j.h1_flips.values, 0)
    hr = f.t.dt.hour + f.t.dt.minute / 60
    f["hs"], f["hc"] = np.sin(hr / 24 * 2 * np.pi), np.cos(hr / 24 * 2 * np.pi)
    f["dow"] = f.t.dt.dayofweek
    # 거래 (1분봉)
    k_sym = np.nanmedian(a) / gold_atr_med
    tp, sl = 3.0 * k_sym, 7.5 * k_sym
    t1 = m1.time.values.astype("datetime64[ms]").astype(np.int64)
    O, H, L, S = m1.open.values, m1.high.values, m1.low.values, m1.spread.values
    ems = f.t.values.astype("datetime64[ms]").astype(np.int64)
    oz = np.full(len(f), np.nan)
    for n in range(len(f) - 1):
        k, x = np.searchsorted(t1, ems[n]), np.searchsorted(t1, ems[n + 1])
        if k >= len(t1) - 1 or x >= len(t1) or x <= k or (t1[k] - ems[n]) > 600_000:
            continue
        sd = side[n]
        if sd == 1:
            e = O[k] + S[k]
            hs_, ht = L[k:x] <= e - sl, H[k:x] >= e + tp
        else:
            e = O[k]
            hs_, ht = H[k:x] + S[k:x] >= e + sl, L[k:x] + S[k:x] <= e - tp
        js, jt = np.flatnonzero(hs_), np.flatnonzero(ht)
        if js.size and (not jt.size or js[0] <= jt[0]):
            oz[n] = -sl
        elif jt.size:
            oz[n] = tp
        else:
            px = O[x] if sd == 1 else O[x] + S[x]
            oz[n] = (px - e) * sd
    f["oz"] = oz / k_sym                                  # 금 1온스 단위로 맞춤
    f["win"] = (f.oz > 0).astype(float)
    f["sym"] = sym
    f["y"] = f.t.dt.year
    if sym == "XAUUSD":
        scale = np.where(f.y < 2025, REF_PRICE / f.close, 1.0)
        a_s = f.atr * scale
        mins = f.t.dt.hour * 60 + f.t.dt.minute
        f["bot"] = ((f.rsi_dir > 0) & (f.h1_agree == 1) & (f.ext <= 0.8) & (f.risk <= 2.8) & (a_s >= 2.5) & (f.atr_rel <= 2.0)
                    & ~((mins >= 15 * 60) & (mins < 21 * 60)) & ~((f.dow == 4) & (mins >= 22 * 60 + 30)) & (f.h1_flips < 3))
    else:
        f["bot"] = False
    del m1, m5
    gc.collect()
    return f.dropna(subset=FEATS + ["oz"]), np.nanmedian(a)


def data():
    OUT.mkdir(parents=True, exist_ok=True)
    gold_med = None
    for sym in SYMS:
        if gold_med is None:
            m1 = pd.read_csv(GPT / "XAUUSD" / "M1.csv.gz", usecols=["time", "open", "high", "low", "close"], parse_dates=["time"])
            gold_med = float(np.nanmedian(atr(bars(m1, "5min")).values))
            del m1
        f, _ = build(sym, gold_med)
        f.to_pickle(OUT / f"{sym}.pkl")
        g = f[f.bot] if sym == "XAUUSD" else f
        print(sym, len(f), "건, 승률", round(f.win.mean() * 100, 1), "%, oz/건", round(f.oz.mean(), 3),
              ("| 봇 신호 " + str(len(g)) + "건 " + str(g.groupby("y").agg(n=("oz", "size"), wr=("win", "mean"), oz=("oz", "sum")).round(2).to_dict()))
              if sym == "XAUUSD" else "", flush=True)
        gc.collect()


def fit():
    d = pd.concat([pd.read_pickle(p) for p in sorted(OUT.glob("*.pkl"))], ignore_index=True)
    gold = d[d.sym == "XAUUSD"]
    rows = []
    rng = np.random.default_rng(0)
    for ty in (2024, 2025, 2026):
        te = gold[gold.bot & (gold.y == ty)]
        rows.append({"시험 해": ty, "학습": "-", "모델": "-", "거르기": "없음", "학습 건수": 0, "건수": len(te),
                     "승률%": round(te.win.mean() * 100, 1), "oz": round(te.oz.sum(), 1)})
        ref = gold[gold.bot & (gold.y < ty)]
        sets = {"A 금 봇 신호": ref, "B 금 전환 전부": gold[gold.y < ty], "C 모든 종목 전환": d[d.y < ty]}
        for sname, tr in sets.items():
            if len(tr) > 150_000:
                tr = tr.iloc[rng.choice(len(tr), 150_000, replace=False)]
            X, yv = tr[FEATS].values.astype(float), tr.win.values
            for mname, model in (("로지스틱", Logit().fit(X, yv)), ("나무 부스팅", Boost().fit(X, yv))):
                s_ref, s_te = model.score(ref[FEATS].values.astype(float)), model.score(te[FEATS].values.astype(float))
                w = te.win.values
                auc = (pd.Series(s_te).rank().values[w == 1].mean() - (w.sum() + 1) / 2) / max((1 - w).sum(), 1)
                allsc = np.concatenate([s_ref, s_te])
                n0 = len(s_ref)
                for q in (10, 20, 30):
                    for how, keep in (("학습 기준", s_te >= np.percentile(s_ref, q)),
                                      ("최근 200개", np.array([s_te[i] >= np.percentile(allsc[n0 + i - 200:n0 + i], q)
                                                               for i in range(len(s_te))]))):
                        rows.append({"시험 해": ty, "학습": sname, "모델": mname, "거르기": f"{how} 하위 {q}%",
                                     "학습 건수": len(tr), "건수": int(keep.sum()), "승률%": round(w[keep].mean() * 100, 1),
                                     "oz": round(te.oz.values[keep].sum(), 1), "AUC": round(auc, 3),
                                     "쉰 승률%": round(w[~keep].mean() * 100, 1) if (~keep).any() else None})
                print(ty, sname, mname, "AUC", round(auc, 3), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "sar_ml_multi.csv", index=False, encoding="utf-8-sig")
    base = df[df.모델 == "-"]
    print(base.to_string(index=False))
    m = df[df.모델 != "-"]
    agg = m.groupby(["학습", "모델", "거르기"]).agg(건수=("건수", "sum"), oz=("oz", "sum"), AUC=("AUC", "mean")).round(3)
    wr = m.assign(wins=m["승률%"] * m["건수"] / 100).groupby(["학습", "모델", "거르기"]).wins.sum()
    agg["승률%"] = (wr / agg.건수 * 100).round(1)
    bw = (base["승률%"] * base["건수"] / 100).sum() / base["건수"].sum() * 100
    print(f"\n기준 2024~26: {int(base.건수.sum())}건 승률 {bw:.1f}% {base.oz.sum():.1f}oz")
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.max_rows", 200)
    print(agg.to_string())


if __name__ == "__main__":
    {"data": data, "fit": fit}[sys.argv[1]]()
