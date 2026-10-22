"""SAR 단타(지금 봇)에 마틴: 지면 다음 진입 lot을 '지금까지 진 금액을 익절 3$로 딱 되찾는 크기'로 (실제 틱)

진입·청산은 지금 봇과 같음 (반대 콩 청산 + 과열 거르기, 손절 7.5 / 익절 3). 한 번에 1포지션이라 lot만 바뀜.
  1) 년도별로 지금 봇 거래 목록(진입 시각, $/oz)을 뽑아 sar_martin_trades_YYYY.csv 에 저장
       python -m backtests.sar_martin trades 2025
       python -m backtests.sar_martin trades 2022 scaled   (ATR 기준을 가격비례로: 금값 낮던 해에 쉬기 필터가 거의 다 막아서)
  2) 저장한 거래를 이어 붙여 마틴 방식별로 잔고 계산
       python -m backtests.sar_martin sim          (2022~24는 지금 봇 그대로)
       python -m backtests.sar_martin sim scaled   (2022~24는 ATR 가격비례 거래 목록 사용)
마틴 lot (단계 n): max(기본 lot, 이번 묶음 누적 손실$ / (TP_OZ x 100)) -> 익절 한 번이면 누적 손실 회수 (더 벌진 않음)
묶음 합계가 0 이상이 되면 기본 lot으로. MAX_STEP 단계까지 지면 손실 확정하고 기본 lot으로.
기본 lot = 그날 시작 잔고 1,000달러당 0.05 (지금 봇)
구조대 (HK 3-3 흉내): 묶음 손실(확정 + 보유 중 평가손실)이 묶음 시작 잔고의 FREEZE%에 닿으면 그 자리에서 동결
  (반대 방향 같은 lot = 그 자리 손실 확정과 같음, 스왑 뺌). 동결 뒤에는 마틴을 쉬고 기본 lot 단타만,
  동결 뒤 단타 이익 합이 동결 손실 이상이 되면 마틴 다시 시작. 평가손실은 거래마다 저장한 최대 역행(mae, $/oz)으로 판단
"""
import gc
import glob
import math
import sys
import zipfile

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.long_period as lb
import backtests.trail_exit as te
from backtests.add_positions_ticks import load_m1_recent
from backtests.ext_filter import features
from backtests.gold_rest_others import BARS, POINT, REF_PRICE, resample
from backtests.sar_exit import against_times
import backtests.winrate_ideas as wi
from backtests.winrate_ideas import Acc

BASE = [("ext", 0.8), ("risk", 2.8)]
TP_OZ = 3.0
# 이름: (마틴 단계 수, 회수 목표 배수 (1 = 진 만큼만))
# 이름: (마틴 단계 수, 회수 목표 배수 (1 = 진 만큼만), 구조대 동결 % (None = 없음), 매일 초기화)
# 매일 초기화: 날짜(서버)가 바뀌면 진행 중인 묶음 손실을 확정하고 기본 lot으로
CASES = {"지금 봇 (마틴 없음)": (0, 1.0, None, False), "3마틴": (3, 1.0, None, False), "5마틴": (5, 1.0, None, False),
         "5마틴 + 구조대 15%": (5, 1.0, 0.15, False), "5마틴 + 구조대 30%": (5, 1.0, 0.30, False),
         "3마틴 매일 초기화": (3, 1.0, None, True), "4마틴 매일 초기화": (4, 1.0, None, True),
         "5마틴 매일 초기화": (5, 1.0, None, True), "9마틴 매일 초기화": (9, 1.0, None, True),
         "5마틴 + 구조대 30% 매일 초기화": (5, 1.0, 0.30, True),
         "마틴 없음 + 2연속 손절 쉬기": (0, 1.0, None, False, 2), "2마틴": (2, 1.0, None, False),
         "2마틴 + 2연속 손절 쉬기": (2, 1.0, None, False, 2), "2마틴 + 2연속 손절 쉬기 + 매일 초기화": (2, 1.0, None, True, 2),
         "2마틴 + 3연속 손절 쉬기": (2, 1.0, None, False, 3), "3마틴 + 3연속 손절 쉬기": (3, 1.0, None, False, 3),
         "3마틴 + 2연속 손절 쉬기": (3, 1.0, None, False, 2)}
# 5번째 값: 그날 N연속 손절이면 그날 남은 신호는 쉼 (없으면 안 쉼)
MAE = []


def trade_mae(t, bid, ask, k, side, fc, against, o, sig_px):
    res = TRADE(t, bid, ask, k, side, fc, against, o, sig_px)
    if res is not None:
        k, j, _ = res
        e = ask[k] if side == 1 else bid[k]
        X = bid[k:j + 1] if side == 1 else ask[k:j + 1]
        MAE.append(float(((X - e) * side).min()))
    return res


TRADE = wi.trade


def load_year(year):
    if year >= 2025:
        return load_m1_recent()
    with zipfile.ZipFile(BARS / f"XAUUSD_M1_{year}.csv.zip") as z:
        m1 = pd.read_csv(z.open(z.namelist()[0]))
    m1["time"] = pd.to_datetime(m1.time)
    m1["sp"] = m1.spread * POINT["XAUUSD"]
    return m1.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def trades(year, scaled=False):
    fe.START, fe.END = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31 23:59")
    m1 = load_year(year)
    m5, h1 = resample(m1, 5), resample(m1, 60)
    del m1
    gc.collect()
    against = against_times(m5, 5)
    sig = fe.bar_close_signals(m5, 5, 0.02, fe.context(m5, h1), True).merge(features(m5), on="t", how="left")
    if scaled:
        px = pd.merge_asof(sig[["t"]], pd.DataFrame({"t": m5.time + pd.Timedelta(minutes=5), "c": m5.close}), on="t").c.values
        sig["atr"], sig["ref"] = sig.atr * REF_PRICE / px, sig.ref * REF_PRICE / px
    del m5, h1
    gc.collect()
    wi.trade = trade_mae
    acc = Acc({"feats": BASE}, against, set())
    sig_ms = sig.t.values.astype("datetime64[ms]").astype(np.int64)
    ms = [(y, m) for y, m in lb.months() if y == year]
    for i, (y, m) in enumerate(ms):
        s, e = lb.month_range(y, m)
        lo, hi = np.searchsorted(sig_ms, int(s.timestamp() * 1000)), np.searchsorted(sig_ms, int(e.timestamp() * 1000))
        if lo == hi:
            continue
        t, bid, ask = lb.load_month(y, m)
        if i + 1 < len(ms):
            nt, nb, na = lb.load_month(*ms[i + 1])
            t, bid, ask = np.concatenate([t, nt]), np.concatenate([bid, nb]), np.concatenate([ask, na])
            del nt, nb, na
        for r in sig.iloc[lo:hi].itertuples(index=False):
            acc.on_signal(r, t, bid, ask, None)
        del t, bid, ask
        gc.collect()
        print(f"  {y}-{m:02d} 완료", flush=True)
    tr = pd.DataFrame(acc.rows, columns=["open", "day", "mins", "oz", "profit", "bal"])[["open", "day", "mins", "oz"]]
    tr["mae"] = MAE
    tr.to_csv(f"sar_martin_{'scaled' if scaled else 'trades'}_{year}.csv", index=False)
    print(f"{year}: {len(tr)}건 저장")


def sim(tr, steps, mult, freeze=None, daily=False, stop_n=0):
    bal = peak = te.BAL0
    mdd, day, base, chain, step = 0.0, None, 0.0, 0.0, 0
    max_lot, busted, rows = 0.0, None, []
    ref, frozen, n_freeze, freeze_days = bal, 0.0, 0, []
    cyc_win = cyc_loss = 0
    step_hits = [0] * (steps + 1)
    for r in tr.itertuples(index=False):
        if r.day != day:
            if daily and chain < 0:
                cyc_loss += 1
                chain, step = 0.0, 0
            streak = 0
            day, base = r.day, te.floor_lot(bal / 1000 * te.LOT_PER_1000)
        if base <= 0:
            busted = busted or r.open
            break
        if stop_n and streak >= stop_n:
            continue
        if step == 0:
            ref = bal
        lot = base
        if step > 0 and frozen <= 0:
            lot = max(base, math.ceil(-chain * mult / (TP_OZ * te.CONTRACT) * 100) / 100)
        step_hits[step] += 1
        max_lot = max(max_lot, lot / base)
        p = r.oz * lot * te.CONTRACT
        froze = False
        if freeze and frozen <= 0 and chain + r.mae * lot * te.CONTRACT <= -freeze * ref:
            p, froze = -freeze * ref - chain, True       # 그 자리 동결 = 손실 확정
            n_freeze += 1
            freeze_days.append(str(r.day))
        bal += p
        streak = streak + 1 if p < 0 else 0
        rows.append((r.open, r.day, p, bal, step))
        if bal <= 0:
            busted = r.open
            break
        peak = max(peak, bal)
        mdd = max(mdd, (peak - bal) / peak)
        if frozen > 0:
            frozen -= p
            continue
        chain += p
        if froze:
            frozen, chain, step = -chain, 0.0, 0
            cyc_loss += 1
        elif chain >= 0 or step >= steps:
            cyc_win, cyc_loss = cyc_win + (chain >= 0), cyc_loss + (chain < 0)
            chain, step = 0.0, 0
        else:
            step += 1
    d = pd.DataFrame(rows, columns=["open", "day", "profit", "bal", "step"])
    daily = d.groupby("day").profit.sum()
    day_start = d.groupby("day").bal.last() - daily
    out = {"거래": len(d), "사이클": cyc_win + cyc_loss,
           "사이클 승률%": round(cyc_win / max(cyc_win + cyc_loss, 1) * 100, 1), "최종잔고$": round(bal), "최대낙폭%": round(mdd * 100, 1),
           "최악의날%": round((daily / day_start).min() * 100, 1), "최대 lot(기본의 배)": round(max_lot, 1),
           "단계별 진입": "/".join(map(str, step_hits)), "깡통": str(busted.date()) if busted is not None else "",
           "동결 횟수": n_freeze, "동결 날짜": " ".join(freeze_days[:6])}
    prev = te.BAL0
    for y, g in d.groupby(d.open.dt.year):
        out[str(y)] = f"{(g.bal.iat[-1] / prev - 1) * 100:+.0f}%"
        prev = g.bal.iat[-1]
    return out


def main():
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    if sys.argv[1] == "trades":
        trades(int(sys.argv[2]), len(sys.argv) > 3 and sys.argv[3] == "scaled")
        return
    scaled = len(sys.argv) > 2 and sys.argv[2] == "scaled"
    files = sorted(glob.glob("sar_martin_trades_*.csv"))
    if scaled:
        files = sorted(glob.glob("sar_martin_scaled_*.csv")) + [f for f in files if f[-8:-4] >= "2025"]
    tr = pd.concat([pd.read_csv(f, parse_dates=["open"]) for f in files]).sort_values("open")
    tr["day"] = pd.to_datetime(tr.day).dt.date
    first = tr.open.min().year
    starts = [(f"{first}-01-01", f"{first}~ 이어서")] + [(f"{y}-01-01", f"{y}년만") for y in range(2022, 2027)]
    rows = []
    for s, name in starts:
        part = tr[tr.open >= s] if name.endswith("이어서") else tr[tr.open.dt.year == int(s[:4])]
        if not len(part) or (name.endswith("이어서") and tr.open.dt.year.nunique() < 2):
            continue
        for n, c in CASES.items():
            r = sim(part, *c)
            rows.append({"구간": name, "방식": n, **{c: r[c] for c in r if not c[:2].isdigit()},
                         **({c: r[c] for c in r if c[:2].isdigit()} if name.endswith("이어서") else {})})
    out = pd.DataFrame(rows).fillna("")
    print(out.to_string(index=False))
    out.to_csv(f"sar_martin_summary{'_scaled' if scaled else ''}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
