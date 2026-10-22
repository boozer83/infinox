"""HK 신규 기능 2개를 SAR 단타(지금 봇: 반대 콩 청산 + 과열 거르기)에 적용해 보기 (실제 틱)

10/22 생방송 슬라이드:
  슈퍼쉴드필터 : 15분봉 + 1시간봉 콩(SAR) 방향이 신호 방향과 같을 때만 진입
                 -> 우리는 이미 1시간봉 SAR 일치를 씀. 15분봉 SAR 일치(마지막 완성 15분봉)를 더함
  원웨이 모드  : 추세 중 반대 콩이 4개 이하로 잠깐 나왔다가 다시 원래 방향 콩이 나오는 순간 추세 방향 진입
                 -> 5분봉 SAR이 d로 전환됐고 그 직전 반대 방향 구간이 4봉 이하, 1시간봉 SAR = d
변형:
  기준 (지금 봇)
  + 15분봉 SAR 일치
  + 원웨이 추가 : 원웨이 전환은 RSI50 / 과열 거르기 없이 추가 진입 (시간·ATR 필터는 그대로)
  원웨이만     : 원웨이 전환만 진입 (RSI50·과열 거르기 없이)
  기준 중 원웨이 : 지금 봇 신호 중 원웨이 조건인 것만 (참고)

사용법: python -m backtests.oneway_test 2025-01-01 2025-12-31
        python -m backtests.oneway_test 2026-01-01
"""
import gc
import sys

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.long_period as lb
import backtests.trail_exit as te
from backtests.add_positions_ticks import load_m1_recent
from backtests.ext_filter import features
from backtests.gold_rest_others import resample
from backtests.sar_exit import against_times
from backtests.winrate_ideas import Acc
from indicators import psar, rsi

BASE = [("ext", 0.8), ("risk", 2.8)]
RUN_MAX = 4
VARIANTS = {
    "기준 (지금 봇)": {"src": "base"},
    "+ 15분봉 SAR 일치": {"src": "base", "m15": True},
    "+ 원웨이 추가": {"src": "base+ow"},
    "+ 원웨이 추가 + 15분봉": {"src": "base+ow", "m15": True},
    "원웨이만": {"src": "ow"},
    "기준 중 원웨이만": {"src": "base&ow"},
    "3분봉 일치면 2배 (슈퍼쉴드 비중)": {"src": "base", "size15": 2.0},
    "3분봉 일치면 3배": {"src": "base", "size15": 3.0},
}


class Acc2(Acc):
    def on_signal(self, s, t, bid, ask, _):
        src = self.o["src"]
        ok = {"base": s.base, "ow": s.ow, "base+ow": s.base or s.ow, "base&ow": s.base and s.ow}[src]
        if not ok:
            return
        if not s.base:                      # 원웨이로만 들어온 신호는 과열 거르기 면제
            s = s._replace(ext=-9.0, risk=-9.0)
        k = self.o.get("size15")
        if k and s.m15 == s.side:           # 5분 + 15분 + 1시간 SAR 같은 방향이면 비중 키움
            if s.t.date() != self.day:
                self.day, self.base = s.t.date(), te.floor_lot(self.bal / 1000 * te.LOT_PER_1000)
            b0 = self.base
            self.base = b0 * k
            super().on_signal(s, t, bid, ask, None)
            self.base = b0
            return
        super().on_signal(s, t, bid, ask, None)


def signals(m5, m15, h1):
    ctx = fe.context(m5, h1)
    up, _ = fe.psar_trigger(m5.high.values, m5.low.values)
    r = rsi(m5.close).values
    end = m5.time + pd.Timedelta(minutes=5)
    i = np.nonzero(up[1:] != up[:-1])[0] + 1
    run = np.diff(np.r_[0, i])                     # 직전 반대 방향 구간 길이 (봉 수)
    side = np.where(up[i], 1, -1)
    a = fe.atr(m5).values[i] if hasattr(fe, "atr") else None
    from indicators import atr
    a = atr(m5).values[i]
    rows = list(zip(end.iloc[i], side, np.ones(len(i), bool), a))
    sig = fe.attach(rows, ctx)                     # RSI 조건 없이 (1시간 SAR 일치만)
    meta = pd.DataFrame({"t": end.iloc[i].values, "rsi_pass": (r[i] - 50) * side > 0, "run": run})
    sig = sig.merge(meta, on="t", how="left").merge(features(m5), on="t", how="left")
    u15 = psar(m15.high.values, m15.low.values)
    q = pd.DataFrame({"end": m15.time + pd.Timedelta(minutes=15), "m15": np.where(u15, 1, -1)})
    sig["m15"] = pd.merge_asof(sig[["t"]], q, left_on="t", right_on="end", direction="backward").m15.values
    sig["base"] = sig.rsi_pass
    sig["ow"] = sig.run <= RUN_MAX
    sig["rsi"] = 50 + sig.side * 10.0              # winrate_ideas Acc의 rsi 옵션은 안 씀
    return sig[sig.base | sig.ow].reset_index(drop=True)


def main():
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
    sig = signals(m5, m15, h1)
    del m5, m15, h1
    gc.collect()
    print(f"  신호: 기준 {int(sig.base.sum())}, 원웨이 {int(sig.ow.sum())}, 겹침 {int((sig.base & sig.ow).sum())}", flush=True)
    accs = {n: Acc2({**o, "feats": BASE}, against, set()) for n, o in VARIANTS.items()}
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
    rows = [{"방식": n, **te.summary(a), "건너뜀": a.skipped} for n, a in accs.items()]
    res = pd.DataFrame(rows).drop(columns=["평균 이익$/oz", "3$넘게 번 비율%", "최대 이익$/oz", "이긴 거래 보유(분, 중간값)"])
    print(res.to_string(index=False))
    res.to_csv(f"oneway_test_{fe.START.year}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
