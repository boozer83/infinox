"""청산 방식 비교: 고정 익절 vs SAR 반대 전환 청산 vs 트레일링 스탑 (2022-01 ~ 현재, 실제 틱)

진입은 모두 같음: long_period.py의 RSI50 + H1 SAR 신호, 다음 5분봉 첫 틱 체결, 한 번에 1포지션.
모든 방식에 손절 7.5달러를 서버 SL처럼 유지.

청산 방식:
  고정 TP          : 진입가 +TP 달러에서 익절 (현재 demo_trader 방식)
  SAR 반대 청산    : 보유 중 5분봉 SAR이 반대로 전환된 봉이 끝나면 다음 봉 첫 틱에 청산
  트레일링 A/D     : 수익이 A달러 이상 나면, 그때부터 최고가(매도는 최저가)에서 D달러 되돌리면 청산

결과: 연도별 PF, 평균 익절/손절, 2025-01부터 1,000달러로 시작했을 때 (0.03 lot 고정 / 일일 마틴 5단계)
사용법: python -m backtests.exits   (long_period를 먼저 한 번 실행해 틱이 저장돼 있어야 함)
"""
import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.m5_ticks import first_hit
from indicators import psar

SL = 7.5
CHUNK = 20_000


def fixed_tp(tp):
    def f(side, k, t, bid, ask, e, ctx):
        if side == 1:
            j = first_hit(bid, k + 1, e - SL, e + tp)
            return (j, bid[j] - e) if j >= 0 else None
        j = first_hit(ask, k + 1, e - tp, e + SL)
        return (j, e - ask[j]) if j >= 0 else None
    f.name = f"고정 TP {tp:g}"
    return f


def sar_exit():
    def f(side, k, t, bid, ask, e, ctx):
        t5, nxt_dn, nxt_up = ctx
        b = np.searchsorted(t5, t[k], side="right") - 1          # 진입한 5분봉
        i = (nxt_dn if side == 1 else nxt_up)[b]                # 반대 SAR이 나온 첫 봉
        if i < 0 or i + 1 >= len(t5):
            return None
        x = np.searchsorted(t, t5[i + 1])                       # 그 다음 봉 첫 틱
        if x >= len(t):
            return None
        price = bid if side == 1 else ask
        seg = price[k + 1:x]
        hit = np.flatnonzero(seg <= e - SL) if side == 1 else np.flatnonzero(seg >= e + SL)
        if hit.size:
            j = k + 1 + hit[0]
        else:
            j = x
        return j, (price[j] - e) if side == 1 else (e - price[j])
    f.name = "SAR 반대 청산"
    return f


def trailing(act, dist):
    def f(side, k, t, bid, ask, e, ctx):
        price = bid if side == 1 else ask
        best = e if side == 1 else e
        s = k + 1
        while s < len(price):
            seg = price[s:s + CHUNK]
            if side == 1:
                run = np.maximum.accumulate(np.r_[best, seg])[1:]
                stop = np.where(run - e >= act, run - dist, e - SL)
                hit = np.flatnonzero(seg <= stop)
            else:
                run = np.minimum.accumulate(np.r_[best, seg])[1:]
                stop = np.where(e - run >= act, run + dist, e + SL)
                hit = np.flatnonzero(seg >= stop)
            if hit.size:
                j = s + hit[0]
                return j, (price[j] - e) if side == 1 else (e - price[j])
            best = run[-1]
            s += CHUNK
        return None
    f.name = f"트레일링 {act:g}/{dist:g}"
    return f


def simulate(sigs, exit_fn, ctx):
    """long_period.simulate와 같은 방식(이번 달 + 다음 달 틱)으로, 청산만 exit_fn으로"""
    ms = list(lb.months())
    trades, busy_until, si = [], -1, 0
    for idx, (y, m) in enumerate(ms):
        s, e = lb.month_range(y, m)
        s_ms, e_ms = int(s.timestamp() * 1000), int(e.timestamp() * 1000)
        cur = [lb.load_month(y, m)] + ([lb.load_month(*ms[idx + 1])] if idx + 1 < len(ms) else [])
        t = np.concatenate([c[0] for c in cur])
        bid = np.concatenate([c[1] for c in cur])
        ask = np.concatenate([c[2] for c in cur])
        while si < len(sigs) and sigs[si][0] < e_ms:
            st, side = sigs[si]
            si += 1
            if st < s_ms or st < busy_until:
                continue
            k = np.searchsorted(t, st)
            if k >= len(t) - 1:
                continue
            entry = ask[k] if side == 1 else bid[k]
            r = exit_fn(side, k, t, bid, ask, entry, ctx)
            if r is None:
                continue
            j, pnl = r
            trades.append((t[k], t[j], side, float(pnl)))
            busy_until = t[j]
    tr = pd.DataFrame(trades, columns=["open_ms", "close_ms", "side", "pnl_oz"])
    tr["open"] = pd.to_datetime(tr.open_ms, unit="ms")
    tr["close"] = pd.to_datetime(tr.close_ms, unit="ms")
    return tr


def pf(x):
    gl = -x[x < 0].sum()
    return round(x[x > 0].sum() / gl, 2) if gl else float("inf")


def money(tr, start, base=0.03, max_level=None, bal0=1000.0):
    """start부터 bal0로 시작했을 때 최종 잔고와 최대 낙폭 (일일 마틴: 손실 x2, 익절 복귀, 최대 단계 손실 시 그날 정지)"""
    d = tr[tr.open >= start]
    bal, peak, mdd, level, cur, stopped = bal0, bal0, 0.0, 0, None, None
    for r in d.itertuples():
        day = r.open.date()
        if day != cur:
            cur, level = day, 0
        if stopped == day:
            continue
        lot = base * 2 ** level if max_level is not None else base
        bal += r.pnl_oz * lot * 100
        peak, mdd = max(peak, bal), max(mdd, (peak - bal) / peak)
        if bal <= 0:
            return f"깡통 {r.open:%y-%m}", mdd
        if max_level is not None:
            if r.pnl_oz < 0:
                if level >= max_level:
                    stopped = day
                else:
                    level += 1
            else:
                level = 0
    return f"{bal:,.0f}", mdd


def main():
    m5 = lb.bars_from_ticks(5)
    h1 = lb.bars_from_ticks(60)
    sigs = lb.signals(m5, h1)
    up = psar(m5.high.values, m5.low.values)
    t5 = m5.time.values.astype("datetime64[ms]").astype(np.int64)
    # 각 봉에서 '그 봉 이후(포함) 처음 SAR이 하락/상승인 봉' 인덱스
    n = len(up)
    nxt_dn, nxt_up = np.full(n, -1), np.full(n, -1)
    nd = nu = -1
    for i in range(n - 1, -1, -1):
        if not up[i]:
            nd = i
        else:
            nu = i
        nxt_dn[i], nxt_up[i] = nd, nu
    ctx = (t5, nxt_dn, nxt_up)

    methods = [fixed_tp(3.0), sar_exit(), trailing(2.0, 1.5), trailing(3.0, 2.0), trailing(3.0, 3.0), trailing(5.0, 3.0)]
    start = pd.Timestamp("2025-01-01")
    rows = []
    for fn in methods:
        tr = simulate(sigs, fn, ctx)
        tr.to_csv(f"exits_{fn.name.replace(' ', '_').replace('/', '-')}.csv", index=False, encoding="utf-8-sig")
        y = tr.groupby(tr.close.dt.year).pnl_oz.apply(pf)
        mins = (tr.close_ms - tr.open_ms) / 60000
        f0, d0 = money(tr, start)
        f5, d5 = money(tr, start, max_level=5)
        rows.append({
            "청산 방식": fn.name, "거래수": len(tr), "승률%": round((tr.pnl_oz > 0).mean() * 100, 1),
            "평균익절$/oz": round(tr.pnl_oz[tr.pnl_oz > 0].mean(), 2), "평균손절$/oz": round(tr.pnl_oz[tr.pnl_oz < 0].mean(), 2),
            "보유(분,중앙)": round(mins.median()), "전체PF": pf(tr.pnl_oz),
            **{f"PF {k}": v for k, v in y.items()},
            "25~26 고정0.03": f"{f0} ({d0:.0%})", "25~26 마틴5": f"{f5} ({d5:.0%})",
        })
        print(f"완료: {fn.name}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("exits_results.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print()
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
