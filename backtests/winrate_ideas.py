"""SAR 단타 승률 올리기 아이디어 비교 (실제 틱, 지금 봇 기준 = 반대 콩 청산 + 승률 쉬기 끔)

  기준        : 지금 봇
  시간대 거르기: 기준 2025년 거래에서 시간대(서버 시) 평균 손익이 마이너스(15건 이상)인 시간은 진입 안 함
                (반대 콩 청산은 작은 손실이 많아서 승률 71.4% 기준은 안 맞음)
                (2025년으로 골라 data/bad_hours.json에 저장 -> 2026년은 그 시간 그대로 적용)
  15분봉 SAR  : 완성 15분봉 SAR도 같은 방향일 때만
  RSI 55      : 매수 RSI > 55 / 매도 < 45
  눌림 X$     : 신호 후 15분 안에 X$ 유리한 가격(매수 Ask <= 신호가 - X)이 오면 그 가격에 진입, 안 오면 건너뜀
  과열 X      : 신호 봉 종가가 뚫은 SAR 값에서 ATR x X 넘게 떨어져 있으면 건너뜀
  스프레드     : 진입 틱 스프레드가 X$ 넘으면 건너뜀
  본전 X$     : +X$ 가면 손절을 진입가로
나머지(쉬기 규칙, 금요일 청산, 1,000달러 시작, 1,000달러당 0.05 lot)는 trail_exit 기준과 같음.
메모리: 틱은 한 달(+다음 달)씩만 읽고 지움. 연도별로 따로 실행 권장

사용법: python -m backtests.winrate_ideas 2025-01-01 2025-12-31   (먼저 2025 -> 시간대 저장)
        python -m backtests.winrate_ideas 2026-01-01
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.long_period as lb
import backtests.trail_exit as te
from backtests.add_positions_ticks import hit, load_m1_recent
from backtests.gold_rest_others import (FRIDAY_NO_ENTRY, MAX_H1_FLIPS, MIN_ATR, NO_ENTRY_FROM, NO_ENTRY_UNTIL,
                                        resample)
from backtests.sar_exit import against_times
from indicators import atr, psar, rsi

BAD_HOURS = Path("data/bad_hours.json")
VARIANTS = {
    "기준 (지금 봇)": {},
    "시간대 거르기": {"hours": True},
    "15분봉 SAR": {"m15": True},
    "RSI 55": {"rsi": 55},
    "눌림 0.5$": {"pull": 0.5},
    "눌림 1$": {"pull": 1.0},
    "과열 1.0 ATR": {"ext": 1.0},
    "과열 0.5 ATR": {"ext": 0.5},
    "스프레드 0.35$": {"spread": 0.35},
    "본전 2$": {"be": 2.0},
    "본전 1.5$": {"be": 1.5},
}


def extras(m5, m15):
    """5분봉 끝 시각별 RSI 값, 과열(종가 - 뚫은 SAR 값)/ATR, 15분봉 SAR 방향"""
    up, trig = fe.psar_trigger(m5.high.values, m5.low.values)
    a = atr(m5).values
    end = m5.time + pd.Timedelta(minutes=5)
    side = np.where(up, 1, -1)
    df = pd.DataFrame({"t": end, "rsi": rsi(m5.close).values,
                       "ext": (m5.close.values - trig) * side / a})
    u15 = psar(m15.high.values, m15.low.values)
    q = pd.DataFrame({"end": m15.time + pd.Timedelta(minutes=15), "m15": np.where(u15, 1, -1)})
    df["m15"] = pd.merge_asof(df[["t"]], q, left_on="t", right_on="end", direction="backward").m15.values
    return df


def trade(t, bid, ask, k, side, fc, against, o, sig_px):
    """-> (진입 idx, 청산 idx, $/oz) 또는 None"""
    X = bid if side == 1 else ask
    E = ask if side == 1 else bid
    if o.get("spread") and ask[k] - bid[k] > o["spread"]:
        return None
    e = E[k]
    if o.get("pull"):
        lim = e - side * o["pull"]
        w = int(np.searchsorted(t, t[k] + 15 * 60_000))
        seg = E[k:w]
        f = np.flatnonzero(seg <= lim if side == 1 else seg >= lim)
        if not f.size:
            return None
        k, e = k + int(f[0]), lim
    sl_px, tp_px = e - side * te.SL, e + side * te.TP
    j = hit(X, k + 1, sl_px, tp_px) if side == 1 else hit(X, k + 1, tp_px, sl_px)
    if o.get("be"):
        b_px = e + side * o["be"]
        jb = hit(X, k + 1, -np.inf, b_px) if side == 1 else hit(X, k + 1, b_px, np.inf)
        if jb < j and jb < len(t):
            j2 = hit(X, jb + 1, e, tp_px) if side == 1 else hit(X, jb + 1, tp_px, e)
            j = j2
    a = against[side]
    i = np.searchsorted(a, t[k], side="right")
    js = int(np.searchsorted(t, a[i])) if i < len(a) else len(t)
    j = min(j, js, fc)
    if j >= len(t):
        return None
    return k, j, (X[j] - e) * side


class Acc(te.Acc):
    def __init__(self, opts, against, bad):
        super().__init__(None)
        self.o, self.against, self.bad = opts, against, bad
        self.skipped = 0

    def on_signal(self, s, t, bid, ask, _):
        ms = int(np.datetime64(s.t, "ms").astype(np.int64))
        if ms < self.busy:
            return
        if s.t.date() != self.day:
            self.day, self.base = s.t.date(), te.floor_lot(self.bal / 1000 * te.LOT_PER_1000)
        mins = s.t.hour * 60 + s.t.minute
        if (s.t.weekday() == 4 and mins >= FRIDAY_NO_ENTRY) or NO_ENTRY_FROM <= mins < NO_ENTRY_UNTIL \
                or s.flips >= MAX_H1_FLIPS or s.atr < MIN_ATR or (s.ref == s.ref and s.atr > 2.0 * s.ref):
            return
        if self.base > te.MAX_LOT:
            return
        o = self.o
        if (o.get("hours") and s.t.hour in self.bad) or (o.get("m15") and s.m15 != s.side) \
                or (o.get("rsi") and (s.rsi - 50) * s.side <= o["rsi"] - 50) or (o.get("ext") and s.ext > o["ext"])                 or (o.get("feat") and getattr(s, o["feat"]) > o["max"])                 or any(getattr(s, f) > v for f, v in o.get("feats", ())):
            self.skipped += 1
            return
        k = int(np.searchsorted(t, ms))
        if k >= len(t) - 1 or t[k] - ms > 600_000:
            return
        d = s.t.normalize() + pd.Timedelta(days=(4 - s.t.weekday()) % 7, hours=23, minutes=30)
        fc = int(np.searchsorted(t, int(np.datetime64(d, "ms").astype(np.int64))))
        res = trade(t, bid, ask, k, s.side, fc, self.against, o, None)
        if res is None:
            self.skipped += 1
            return
        k, j, oz = res
        self.busy = int(t[j])
        self.results.append(oz > 0)
        profit = oz * self.base * te.CONTRACT
        self.bal += profit
        self.peak = max(self.peak, self.bal)
        self.mdd = max(self.mdd, (self.peak - self.bal) / self.peak)
        self.rows.append((pd.Timestamp(int(t[k]), unit="ms"), self.day, (t[j] - t[k]) / 60000, oz, profit, self.bal))


def main():
    if len(sys.argv) > 1:
        fe.START = pd.Timestamp(sys.argv[1])
    if len(sys.argv) > 2:
        fe.END = pd.Timestamp(sys.argv[2])
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = load_m1_recent()
    m5, m15, h1 = resample(m1, 5), resample(m1, 15), resample(m1, 60)
    del m1
    gc.collect()
    against = against_times(m5, 5)
    sig = fe.bar_close_signals(m5, 5, 0.02, fe.context(m5, h1), True)
    sig = sig.merge(extras(m5, m15)[["t", "rsi", "ext", "m15"]], on="t", how="left")
    del m5, m15, h1
    gc.collect()
    selecting = fe.START.year == 2025
    bad = set(json.loads(BAD_HOURS.read_text())) if BAD_HOURS.exists() and not selecting else set()
    accs = {n: Acc(o, against, bad) for n, o in VARIANTS.items() if not (o.get("hours") and selecting)}
    sig_ms = sig.t.values.astype("datetime64[ms]").astype(np.int64)
    ms = [(y, m) for y, m in lb.months() if (fe.START.year, fe.START.month) <= (y, m) <= (fe.END.year, fe.END.month)]
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
            for a in accs.values():
                a.on_signal(r, t, bid, ask, None)
        del t, bid, ask
        gc.collect()
        print(f"  {y}-{m:02d} 완료", flush=True)
    if selecting:
        tr = pd.DataFrame(accs["기준 (지금 봇)"].rows, columns=["open", "day", "mins", "oz", "profit", "bal"])
        h = tr.groupby(tr.open.dt.hour).oz.agg(n="size", wr=lambda x: (x > 0).mean() * 100, avg="mean")
        print(h.round(2).T.to_string())
        bad = sorted(int(x) for x in h[(h.n >= 15) & (h.avg < 0)].index)
        BAD_HOURS.write_text(json.dumps(bad))
        print(f"쉬는 시간대 (서버시간, 2025 기준): {bad}")
    rows = [{"방식": n, **te.summary(a), "건너뜀": a.skipped} for n, a in accs.items()]
    res = pd.DataFrame(rows).drop(columns=["평균 이익$/oz", "3$넘게 번 비율%", "최대 이익$/oz", "이긴 거래 보유(분, 중간값)"])
    print(res.to_string(index=False))
    res.to_csv(f"winrate_ideas_{fe.START.year}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
