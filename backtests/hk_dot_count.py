"""HK 프로그램 공격·수비 = 진입 때 보는 콩 개수 (2026-10-27 사용자가 들은 설명, 실제 틱 2022-01 ~ 2026-10-16)

  5분봉 SAR이 뒤집힐 때, 뒤집히기 전 같은 쪽 콩이 N개 이상이었으면 새 방향으로 진입
  공격: 번트 1 (전환마다 바로) / 안타 4 / 홈런 8,  수비: 2 / 8 / 16
1. 헤지 마틴 (backtests.hedge_martin_ticks, 지금 봇 설정: 익절 5 / 목표 1% / 한도 30% / 4단계 / 이어가기, 해마다 1만$)
     시작만 : 사이클 시작은 콩 N개 이상 전환에서만, 추가(2배)는 전환마다
     전부   : 시작·추가 모두 콩 N개 이상 전환에서만
     우리 필터: 지금 봇처럼 시작은 SAR 단타 신호(필터·뉴스) + 콩 N개 이상, 추가는 전환마다
2. 단타 (backtests.martin_scaled.simulate: SL 7.5 / TP 3, 한 번에 1개, 미국장 진입 안 함): 콩 N개 이상 전환에서 순방향

사용법: python -m backtests.hk_dot_count hedge 2025     (해마다 따로, 결과 results/hk_dot_hedge_YYYY.csv)
        python -m backtests.hk_dot_count scalp           (결과 results/hk_dot_scalp.csv)
"""
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.hedge_martin_ticks as hm
import backtests.long_period as lb
from indicators import psar

RES = Path(__file__).resolve().parent.parent / "results"
COUNTS = (1, 2, 4, 8, 16)


def prev_run(up):
    """i번째 봉이 전환 봉이면 그 전 같은 쪽 콩 개수, 아니면 0"""
    run = np.ones(len(up), int)
    for i in range(1, len(up)):
        run[i] = run[i - 1] + 1 if up[i] == up[i - 1] else 1
    out = np.zeros(len(up), int)
    f = np.flatnonzero(up[1:] != up[:-1]) + 1
    out[f] = run[f - 1]
    return out


def hedge(year):
    from backtests.gold_rest_others import resample
    import zipfile
    hm.CONTINUE = True
    with zipfile.ZipFile(hm.BARS / f"XAUUSD_M1_{year}.csv.zip") as z:
        m1 = pd.read_csv(z.open(z.namelist()[0]), usecols=["time", "open", "high", "low", "close"])
    m1["time"] = pd.to_datetime(m1.time)
    m5 = resample(m1.drop_duplicates("time").sort_values("time").reset_index(drop=True), 5)
    del m1
    up = psar(m5.high.values, m5.low.values)
    pr = prev_run(up)
    i = np.flatnonzero(pr > 0)
    ms_ = (m5.time.iloc[i] + pd.Timedelta(minutes=5)).values.astype("datetime64[ms]").astype(np.int64)
    base = pd.DataFrame({"ms": ms_, "side": np.where(up[i], 1, -1), "run": pr[i]})
    hm.START_SIG = True
    ours = hm.flips(year)                                   # 같은 전환 목록 + 지금 봇 시작 신호
    base = base.merge(ours[["ms", "side", "start"]].rename(columns={"start": "ours"}), on=["ms", "side"], how="left")
    base["ours"] = base.ours.fillna(False).astype(bool)
    base = base[base.ms < int(hm.END.value // 1_000_000)].reset_index(drop=True)
    variants = {}
    for n in COUNTS:
        variants[f"시작만 콩{n}"] = (base, base.run >= n)
        variants[f"전부 콩{n}"] = (base[base.run >= n].reset_index(drop=True), np.ones((base.run >= n).sum(), bool))
        variants[f"우리 필터 + 콩{n}"] = (base, base.ours & (base.run >= n))
    sims = {k: hm.Sim(5.0, 0.01, 0.30, 4) for k in variants}
    for y, m in [(year, m) for m in range(1, 13) if (year, m) <= (hm.END.year, hm.END.month)]:
        s, e = lb.month_range(y, m)
        sm_, em = int(s.timestamp() * 1000), min(int(e.timestamp() * 1000), int(hm.END.value // 1_000_000))
        t, bid, ask = lb.load_month(y, m)
        for k, (fl, st) in variants.items():
            sel = ((fl.ms >= sm_) & (fl.ms < em)).values
            f, stt = fl[sel], np.asarray(st)[sel]
            ks = np.searchsorted(t, f.ms.values)
            sim, prev = sims[k], 0
            for side, kk, s_ in zip(f.side.values, ks, stt):
                sim.run(prev, kk, bid, ask)
                if kk < len(t):
                    sim.on_flip(int(side), kk, bid, ask, bool(s_))
                prev = kk
            sim.run(prev, len(t), bid, ask)
        del t, bid, ask
        gc.collect()
        print(f"  {y}-{m:02d}", flush=True)
    rows = [{"규칙": k, "사이클": s.cycles, "사이클승": s.wins, "한도걸림": s.limits, "연말$": round(s.bal),
             "수익%": round((s.bal / hm.BAL0 - 1) * 100, 1), "최대낙폭%": round(s.mdd * 100, 1), "최대lot": round(s.max_lot, 2)}
            for k, s in sims.items()]
    return pd.DataFrame(rows)


def scalp():
    from backtests.martin_scaled import NO_ENTRY_HOURS, simulate
    from backtests.scalp_ideas import m5_bars
    from backtests.strategies import pf
    m5 = m5_bars()
    up = psar(m5.high.values, m5.low.values)
    pr = prev_run(up)
    t = (m5.time + pd.Timedelta(minutes=5)).values.astype("datetime64[ms]").astype(np.int64)
    rows = []
    for n in COUNTS:
        sig = [(t[i], 1 if up[i] else -1) for i in np.flatnonzero(pr >= n)
               if pd.Timestamp(t[i], unit="ms").hour not in NO_ENTRY_HOURS]
        tr = simulate(sig)
        y = tr.open.dt.year
        row = {"규칙": f"콩 {n}개 이상 전환", "거래": len(tr), "승률%": round((tr.pnl_oz > 0).mean() * 100, 1),
               "oz합": round(tr.pnl_oz.sum(), 1), "PF": pf(tr.pnl_oz)}
        row.update({f"oz {yr}": round(tr.pnl_oz[y == yr].sum(), 1) for yr in range(2022, 2027)})
        rows.append(row)
        print("완료", n, flush=True)
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    if sys.argv[1] == "hedge":
        y = int(sys.argv[2])
        res = hedge(y)
        res.to_csv(RES / f"hk_dot_hedge_{y}.csv", index=False, encoding="utf-8-sig")
    else:
        res = scalp()
        res.to_csv(RES / "hk_dot_scalp.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
