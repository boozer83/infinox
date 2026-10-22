"""HK 방송 백테스트 구간 (2026-03-01 ~ 03-09) 재현: 1만 달러 -> 3/5 04:57에 33번째 거래 후 15,321$ (+53%), 3/9 23,197$ (+132%)

방송 MT4 테스터 (10/22 1부, 2026.03.01~03.31, M5, 모든 틱(1분봉 보간), 스프레드 35): 1만 달러에 1.00 lot, 2.3만 달러에 2.31 lot
진입: HK 홈런 (hk_apply와 같음) / 봉 안 SAR 전환 전부 (안타·홈런 구분 없이) / 우리 SAR 단타
청산: HK식 (손절 7.5, +3 닿으면 +2 잠금 후 1$ 추적) / 손절 7.5 + 익절 3 /
      빠른 추적 (+1 닿으면 +0.3 잠금, 최고점에서 0.5 되돌리면 청산; 방송 거래 목록처럼 손절선을 바짝 올림)
시세: 실제 틱 / 스프레드 0.35$ 고정 (Ask = Bid + 0.35) /
      MT4 테스터 "모든 틱 (1분봉 보간)" 흉내: 1분봉마다 시가 -> 먼저 닿은 쪽 극값 -> 반대 극값 -> 종가를 5초 간격 12틱으로 이은 가짜 틱,
      스프레드 0.35 고정 (양봉은 저가 먼저, 음봉은 고가 먼저)
lot = 진입 때 잔고 1,000달러당 0.10 (0.01 단위 내림, 방송 거래 목록 기준), 한 번에 1포지션
2026-10-22 확인: 돌파 시각을 실제 틱에서 잡고 가짜 틱으로 체결하면 시점이 어긋나 결과가 크게 부풀려짐 -> 가짜 틱이면 돌파도 가짜 틱에서

사용법: python -m backtests.hk_march [시작 YYYY-MM-DD] [끝 YYYY-MM-DD]
"""
import sys

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.hk_apply as ha
import backtests.long_period as lb
import backtests.trail_exit as te
from indicators import psar

LOT1000, BAL0 = 0.10, 10_000.0      # 방송 테스터 거래 목록: 1만 달러에 1.00 lot, 2.3만 달러에 2.31 lot


def mt4_ticks(year, month):
    """1분봉으로 만든 가짜 틱 (t ms, bid, ask)"""
    import zipfile
    from backtests.gold_rest_others import BARS
    with zipfile.ZipFile(BARS / f"XAUUSD_M1_{year}.csv.zip") as z:
        m1 = pd.read_csv(z.open(z.namelist()[0]), usecols=["time", "open", "high", "low", "close"])
    m1["time"] = pd.to_datetime(m1.time)
    m1 = m1[(m1.time.dt.month == month)].reset_index(drop=True)
    o, h, l, c = (m1[k].values for k in ("open", "high", "low", "close"))
    up = c >= o
    a, b = np.where(up, l, h), np.where(up, h, l)            # 먼저 닿는 극값, 나중 극값
    w = np.linspace(0, 1, 5)[:-1]
    path = np.concatenate([o[:, None] + (a - o)[:, None] * w, a[:, None] + (b - a)[:, None] * w,
                           b[:, None] + (c - b)[:, None] * w], axis=1)          # 12점
    t0 = m1.time.values.astype("datetime64[ms]").astype(np.int64)
    t = (t0[:, None] + np.arange(12) * 5000).ravel()
    bid = np.round(path.ravel(), 2)
    return t, bid, bid + 0.35


def run(sig, kind, exit_case, t, bid, ask, against):
    bal, busy, rows = BAL0, 0, []
    for ms, side in zip(sig.ms.values, sig.side.values):
        if ms < busy:
            continue
        k = int(np.searchsorted(t, ms))
        if k >= len(t) - 1 or t[k] >= ha.friday_ms(t[k]):
            continue
        fri = ha.friday_ms(t[k])
        if kind == "our" and exit_case is None:
            out = ha.scalp_trade(t, bid, ask, k, int(side), "our", against)
        else:
            te.TP = exit_case[3] if exit_case and len(exit_case) > 3 else 3.0     # 잠그기 시작하는 이익
            out = te.trade(t, bid, ask, k, int(side), exit_case[:3] if exit_case else None, fri, None)
            te.TP = 3.0
        if out is None:
            continue
        j, oz = out
        lot = np.floor(bal / 1000 * LOT1000 * 100) / 100
        bal += oz * lot * 100
        busy = int(t[j])
        rows.append((pd.Timestamp(int(ms), unit="ms"), side, lot, round(oz, 2), round(bal)))
    return pd.DataFrame(rows, columns=["진입", "방향", "lot", "$/oz", "잔고"])


def main():
    s = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-03-01")
    e = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2026-03-06")
    fe.START, fe.END = pd.Timestamp(s.year, 1, 1), pd.Timestamp(s.year + 1, 1, 1)
    m5, h1 = ha.load_year(s.year)
    fl, hr = ha.hk_homerun(m5)
    our = ha.our_signals(m5, h1)
    up = psar(m5.high.values, m5.low.values)
    end5 = m5.time + pd.Timedelta(minutes=5)
    against = {1: ha.to_ms(end5[~up]), -1: ha.to_ms(end5[up])}
    sm, em = ha.to_ms([s])[0], ha.to_ms([e])[0]
    t, bid, ask = lb.load_month(s.year, s.month)
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    rows = []
    mt, mb, ma = mt4_ticks(s.year, s.month)
    fl_f, hr_f = ha.hk_homerun(m5, load=lambda y, m: mt4_ticks(y, m) if (y, m) == (s.year, s.month) else lb.load_month(y, m))
    for feed, (t, bid, a) in (("실제 틱", (t, bid, ask)), ("스프레드 0.35 고정", (t, bid, bid + 0.35)),
                              ("MT4식 가짜 틱", (mt, mb, ma))):
        f_hr, f_fl = (hr_f, fl_f) if feed == "MT4식 가짜 틱" else (hr, fl)    # 가짜 틱이면 돌파 시각도 가짜 틱에서
        for name, sig, kind in (("HK 홈런", f_hr, "hk"), ("봉 안 SAR 전환 전부", f_fl, "hk"), ("우리 SAR 단타", our, "our")):
            sig = sig[(sig.ms >= sm) & (sig.ms < em)]
            for xn, case in (("HK식 추적", (2.0, 1.0, None)), ("익절 3", None),
                             ("빠른 추적 (+1 -> 0.3 잠금, 0.5 추적)", (0.3, 0.5, None, 1.0))):
                tr = run(sig, kind, case, t, bid, a, against)
                n33 = tr.잔고.iat[min(32, len(tr) - 1)] if len(tr) else BAL0
                rows.append({"시세": feed, "진입": name, "청산": xn, "거래": len(tr),
                             "승률%": round((tr["$/oz"] > 0).mean() * 100, 1) if len(tr) else np.nan,
                             "33번째 거래 후": round(n33), "최종 잔고": round(tr.잔고.iat[-1]) if len(tr) else BAL0,
                             "최저 잔고": round(min(BAL0, tr.잔고.min())) if len(tr) else BAL0})

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(ha.OUT / f"hk_march_{s:%Y%m%d}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
