"""add_positions.py(물타기 / 불타기)를 실제 틱으로 다시 확인 (2026-01 ~ 현재, 1,000달러 시작)

규칙은 add_positions.py와 같음. 체결만 틱 단위:
  진입/추가 진입은 그 틱의 실제 호가 (매수 Ask, 매도 Bid), 청산은 매수 Bid / 매도 Ask가 처음 닿는 틱의 가격.
  금요일 23:30 이후 첫 틱에 남은 포지션 청산.
불타기 조건 (추가):
  RSI r : 신호 봉 RSI가 매수 50+r 이상 / 매도 50-r 이하일 때만 추가 진입
  속도 s: 첫 진입 후 s초 안에 첫 추가 가격(+1달러)에 닿았을 때만 추가 진입 (늦게 닿으면 1개로 끝까지)
승률 쉬기는 2026-01부터 새로 셈 (처음 200건은 쉬지 않음).
틱은 한 달씩 올리고, 다음 달 틱을 붙여서 달을 넘기는 거래도 처리 (메모리 약 0.5GB).

사용법: python -m backtests.add_positions_ticks   (data/ticks/<연도>-<월>.npz, data/bars/XAUUSD_M1_2025~2026 필요)
"""
import zipfile

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.add_positions import (ATR_REF_BARS, ATR_REF_MIN, BAL0, CONTRACT, LOT_PER_1000, MAX_ATR_MULT,
                                     MAX_LEVEL, MAX_LOT, MAX_LOT_MULT, SL, TP, floor_lot)
from backtests.gold_rest_others import (BARS, FRIDAY_NO_ENTRY, MAX_H1_FLIPS, MIN_ATR, NO_ENTRY_FROM, NO_ENTRY_UNTIL,
                                        POINT, WINRATE_MIN, WINRATE_N, h1_flips_by_time, signal_table)
from backtests.m5_ticks import first_hit
from indicators import atr, rsi

START = pd.Timestamp("2026-01-01")
INF = np.inf

# 이름: (방식, 추가 간격 $, 최대 개수, 손절 $, 익절 $, RSI 조건, 속도 조건 초)
VARIANTS = {
    "기준 (지금 규칙, 1개)": ("none", 0, 1, SL, TP, 0, 0),
    "물타기 2.5$마다 3개, 손절 7.5$": ("avg", 2.5, 3, SL, TP, 0, 0),
    "물타기 5$마다 3개, 손절 15$": ("avg", 5.0, 3, 15.0, TP, 0, 0),
    "물타기 5$마다 4개, 손절 25$": ("avg", 5.0, 4, 25.0, TP, 0, 0),
    "불타기 1$마다 3개 (조건 없음)": ("pyr", 1.0, 3, SL, TP, 0, 0),
    "불타기 + RSI 60/40": ("pyr", 1.0, 3, SL, TP, 10, 0),
    "불타기 + RSI 65/35": ("pyr", 1.0, 3, SL, TP, 15, 0),
    "불타기 + RSI 70/30": ("pyr", 1.0, 3, SL, TP, 20, 0),
    "불타기 + 속도 30초": ("pyr", 1.0, 3, SL, TP, 0, 30),
    "불타기 + 속도 60초": ("pyr", 1.0, 3, SL, TP, 0, 60),
}


def load_m1_recent():
    parts = []
    for y in (2025, 2026):
        f = BARS / f"XAUUSD_M1_{y}.csv.zip"
        with zipfile.ZipFile(f) as z:
            parts.append(pd.read_csv(z.open(z.namelist()[0])))
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df.time)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df["sp"] = df.spread * POINT["XAUUSD"]
    return df


def signals():
    sig, m5, h1 = signal_table(load_m1_recent())
    end = m5.time + pd.Timedelta(minutes=5)
    a5 = pd.Series(atr(m5).values)
    extra = pd.DataFrame({"end": end, "ref": a5.rolling(ATR_REF_BARS, min_periods=ATR_REF_MIN).median().values,
                          "rsi": rsi(m5.close).values})
    m = pd.merge_asof(sig[["t"]], extra, left_on="t", right_on="end", direction="backward")
    sig["ref"], sig["rsi"] = m.ref.values, m.rsi.values
    fl = pd.merge_asof(sig[["t"]], h1_flips_by_time(h1)[["end", "flips", "day"]], left_on="t", right_on="end",
                       direction="backward")
    sig["flips"] = np.where(fl.day.values == sig.t.dt.date.values, fl.flips.fillna(0).values, 0)
    return sig[sig.t >= START].reset_index(drop=True)


def hit(arr, pos, lo, hi):
    j = first_hit(arr, pos, lo, hi)
    return len(arr) if j < 0 else j


def tick_basket(t, bid, ask, k, side, var, n_max, fri_ms):
    """k번째 틱에 첫 진입 -> (청산 idx, 진입가 목록, 청산가) 또는 None(데이터 끝)"""
    mode, step, _, sl, tp, _, speed = var
    X, E = (bid, ask) if side == 1 else (ask, bid)       # 청산 가격 / 추가 진입 가격
    first = E[k]
    entries = [first]
    sl_px = first - side * sl
    fc = int(np.searchsorted(t, fri_ms))
    pos = k + 1
    while True:
        tp_px = (np.mean(entries) if mode == "avg" else first) + side * tp
        jx = hit(X, pos, sl_px, tp_px) if side == 1 else hit(X, pos, tp_px, sl_px)
        ja = len(t)
        if len(entries) < n_max:
            lv = first + side * step * len(entries) * (-1 if mode == "avg" else 1)
            up = (side == 1) != (mode == "avg")              # 추가 가격이 위쪽인가
            ja = hit(E, pos, -INF, lv) if up else hit(E, pos, lv, INF)
            if speed and len(entries) == 1 and ja < len(t) and t[ja] - t[k] > speed * 1000:
                n_max, ja = 1, len(t)                       # 늦게 닿음 -> 추가 안 함
        if ja < jx and ja < fc:
            entries.append(E[ja])
            pos = ja + 1
            continue
        j = min(jx, fc)
        if j >= len(t):
            return None
        return j, entries, X[j]


class Account:
    def __init__(self, var):
        self.var = var
        self.wr_min = WINRATE_MIN if var[0] == "none" else 100 * var[3] / (var[3] + var[4])
        self.results, self.rows = [], []
        self.busy = -1
        self.bal = self.peak = BAL0
        self.mdd = 0.0
        self.day, self.level, self.stopped, self.base = None, 0, False, 0.0

    def on_signal(self, s, t, bid, ask):
        ms = int(np.datetime64(s.t, "ms").astype(np.int64))
        if ms < self.busy:
            return
        if s.t.date() != self.day:
            self.day, self.level, self.stopped = s.t.date(), 0, False
            self.base = floor_lot(self.bal / 1000 * LOT_PER_1000)
        mins = s.t.hour * 60 + s.t.minute
        if self.stopped or (s.t.weekday() == 4 and mins >= FRIDAY_NO_ENTRY) or NO_ENTRY_FROM <= mins < NO_ENTRY_UNTIL \
                or s.flips >= MAX_H1_FLIPS or s.atr < MIN_ATR or (s.ref == s.ref and s.atr > MAX_ATR_MULT * s.ref):
            return
        last = self.results[-WINRATE_N:]
        paused = len(last) >= WINRATE_N and np.mean(last) * 100 < self.wr_min
        lot = round(min(self.base * 2 ** self.level, self.base * MAX_LOT_MULT), 2)
        if not paused and lot > MAX_LOT:
            return
        k = int(np.searchsorted(t, ms))
        if k >= len(t) - 1 or t[k] - ms > 600_000:
            return
        mode, _, units, _, _, rsi_min, _ = self.var
        n_max = units if (s.rsi - 50) * s.side >= rsi_min else 1
        d = s.t.normalize() + pd.Timedelta(days=(4 - s.t.weekday()) % 7, hours=23, minutes=30)
        res = tick_basket(t, bid, ask, k, s.side, self.var, n_max, int(np.datetime64(d, "ms").astype(np.int64)))
        if res is None:
            return
        j, entries, px = res
        self.busy = int(t[j])
        oz = float(sum((px - e) * s.side for e in entries))
        win = oz > 0
        self.results.append(win)
        if paused:
            return
        profit = oz * lot * CONTRACT
        self.bal += profit
        self.peak = max(self.peak, self.bal)
        self.mdd = max(self.mdd, (self.peak - self.bal) / self.peak)
        self.rows.append((pd.Timestamp(int(t[k]), unit="ms"), pd.Timestamp(int(t[j]), unit="ms"), self.day, s.side,
                          len(entries), self.level, lot, oz, profit, self.bal))
        if win:
            self.level = 0
        elif self.level >= MAX_LEVEL:
            self.stopped = True
        else:
            self.level += 1


def summary(acc):
    tr = pd.DataFrame(acc.rows, columns=["open", "close", "day", "side", "units", "level", "lot", "oz", "profit", "bal"])
    daily = tr.groupby("day").profit.sum()
    day_start = tr.groupby("day").bal.last() - daily
    gp, gl = tr.profit[tr.profit > 0].sum(), -tr.profit[tr.profit < 0].sum()
    return {"거래(묶음)": len(tr), "승률%": round((tr.oz > 0).mean() * 100, 1), "평균 개수": round(tr.units.mean(), 2),
            "PF": round(gp / gl, 2) if gl else np.inf, "최종잔고$": round(tr.bal.iat[-1]),
            "수익률%": round((tr.bal.iat[-1] / BAL0 - 1) * 100), "최대낙폭%": round(acc.mdd * 100, 1),
            "최악의날$": round(daily.min()), "최악의날%": round((daily / day_start).min() * 100, 1)}


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    sig = signals()
    print(f"2026 신호 {len(sig):,}개", flush=True)
    accs = {name: Account(v) for name, v in VARIANTS.items()}
    ms = [(y, m) for y, m in lb.months() if (y, m) >= (START.year, START.month)]
    sig_ms = sig.t.values.astype("datetime64[ms]").astype(np.int64)
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
        for srow in sig.iloc[lo:hi].itertuples(index=False):
            for acc in accs.values():
                acc.on_signal(srow, t, bid, ask)
        del t, bid, ask
        print(f"  {y}-{m:02d} 완료", flush=True)
    res = pd.DataFrame([{"방식": n, **summary(a)} for n, a in accs.items()])
    res.to_csv("add_positions_ticks_2026.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
