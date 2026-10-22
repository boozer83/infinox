"""프리미엄 뉴스: "한쪽으로 튀었다"를 판단하는 방법 비교 (실제 틱, high USD·EUR·GBP, 2022-01 ~ 2026-10)

공통: 발표 30초 전 매수(Ask)·매도(Bid) 같은 lot 진입 (돌파 진입 방식만 예외)
      판단이 5분 안에 안 나면 양쪽 동시 종료
      남은 쪽 (슬라이드 4번): 10분 뒤 이익이면 종료, 손실이면 본전 될 때 종료, 비상 손절 6$, 최대 60분
판단 방법 (진 쪽은 판단 틱의 실제 가격으로 청산):
  고정 c$        : 진입가 대비 c$ 역행 (지금 방식, c = 1 / 1.5 / 2 / 3)
  중간가 c$      : 진입 때 중간가 대비 중간가가 c$ 움직임 (스프레드 때문에 한쪽이 먼저 걸리는 것 제거)
  변동성 k배     : c = 발표 전 30분 1분봉 평균 폭 x k (k = 2 / 4 / 6), 최소 0.5$
  확인 N초       : 2$ 넘은 상태가 N초 계속돼야 인정 (휩쏘 거르기, N = 3 / 10)
  T초 뒤 방향    : 발표 T초 뒤 중간가가 m$ 이상 움직였으면 반대쪽 청산, 아니면 양쪽 종료 (T = 30 / 60, m = 1)
  돌파 진입 X$   : 미리 안 들어가고, 발표 뒤 5분 안에 중간가가 X$ 움직인 방향으로 한쪽만 진입 (X = 1 / 2 / 3)
                   0.3초 늦게 체결, 노출을 양방향과 맞추려고 lot 2배로 계산, 5분 안에 돌파 없으면 매매 안 함
앞 발표에서 60분 안에 온 발표는 건너뜀 (방법과 무관하게 같은 발표 집합)
결과: $/oz (매수+매도 합), 연도별 합계와 꼬리

사용법: python -m backtests.news_breakout       (results/news_breakout.csv)
"""
import gc

import numpy as np
import pandas as pd

import backtests.long_period as lb
import backtests.news_hk as nh

PRE, QUIET, HOLD, MAXH, ECUT = 30_000, 300_000, 600_000, 3_600_000, 6.0
FILL_MS = 300
METHODS = {**{f"고정 {c}$": ("fixed", c) for c in (1.0, 1.5, 2.0, 3.0)},
           **{f"중간가 {c}$": ("mid", c) for c in (1.0, 2.0)},
           **{f"변동성 {k}배": ("vol", k) for k in (2, 4, 6)},
           **{f"확인 {n}초 (2$)": ("confirm", n) for n in (3, 10)},
           **{f"{t}초 뒤 방향": ("dir", t) for t in (30, 60)},
           **{f"돌파 진입 {x}$": ("break", x) for x in (1.0, 2.0, 3.0)}}


def first(c):
    k = np.flatnonzero(c)
    return int(k[0]) if k.size else None


def hold_winner(tt, x, e, side, k1, ms):
    """남은 쪽(side, 진입가 e, 청산 가격 배열 x)을 k1부터 들고 감 -> $/oz"""
    pnl = (x - e) * side
    h10 = int(np.searchsorted(tt, ms + HOLD))
    stop = first(pnl[k1:] <= -ECUT)
    stop = k1 + stop if stop is not None else len(tt) - 1
    if h10 < len(tt) and pnl[h10] > 0:
        j = h10
    else:
        be = first(pnl[max(h10, k1):] >= 0)
        j = max(h10, k1) + be if be is not None else len(tt) - 1
    return pnl[min(j, stop)]


def held_for(c, tt, n_ms):
    """조건 c가 n_ms 동안 계속 참인 첫 시점 idx (그 시점에 인정)"""
    k = np.flatnonzero(np.diff(np.r_[0, c.astype(np.int8), 0]))
    for s, e in zip(k[::2], k[1::2]):
        j = int(np.searchsorted(tt, tt[s] + n_ms))
        if j < e:
            return j
    return None


def one(t, bid, ask, ms, method):
    kind, p = method
    k0 = int(np.searchsorted(t, ms - PRE))
    if k0 >= len(t) or t[k0] - (ms - PRE) > 20_000:
        return None
    kend = int(np.searchsorted(t, ms + MAXH))
    b, a, tt = bid[k0:kend], ask[k0:kend], t[k0:kend]
    if len(tt) < 10:
        return None
    eb, es = a[0], b[0]
    mid = (b + a) / 2
    m0 = mid[0]
    quiet = min(int(np.searchsorted(tt, ms + QUIET)), len(tt) - 1)

    if kind == "break":
        k = first((np.abs(mid[:quiet] - m0) >= p) & (tt[:quiet] >= ms))
        if k is None:
            return None                 # 신호 없음 (매매 안 함)
        side = 1 if mid[k] > m0 else -1
        k = min(int(np.searchsorted(tt, tt[k] + FILL_MS)), len(tt) - 1)      # 급등 틱이라 0.3초 늦게 체결
        e = a[k] if side == 1 else b[k]
        return 2 * hold_winner(tt, b if side == 1 else a, e, side, k, ms)   # 양방향과 같은 노출이 되게 lot 2배

    if kind == "dir":
        k = int(np.searchsorted(tt, ms + p * 1000))
        if k >= len(tt) or abs(mid[k] - m0) < 1.0:
            q = min(k, len(tt) - 1)
            return (b[q] - eb) + (es - a[q])
        down = mid[k] < m0
    else:
        if kind == "fixed":
            cb, cs = b <= eb - p, a >= es + p
        elif kind == "mid":
            cb, cs = mid <= m0 - p, mid >= m0 + p
        elif kind == "vol":
            i0 = int(np.searchsorted(t, ms - PRE - 1_800_000))
            seg = pd.Series(bid[i0:k0], index=pd.to_datetime(t[i0:k0], unit="ms")).resample("1min")
            c = max(0.5, float((seg.max() - seg.min()).mean()) * p) if k0 > i0 else 2.0
            cb, cs = b <= eb - c, a >= es + c
        else:   # confirm
            cb, cs = b <= eb - 2.0, a >= es + 2.0
        if kind == "confirm":
            kb, ks = held_for(cb, tt, p * 1000), held_for(cs, tt, p * 1000)
        else:
            kb, ks = first(cb), first(cs)
        kb = kb if kb is not None and kb < quiet else None
        ks = ks if ks is not None and ks < quiet else None
        if kb is None and ks is None:
            return (b[quiet] - eb) + (es - a[quiet])
        down = kb is not None and (ks is None or kb <= ks)
        k = kb if down else ks
    if down:     # 매수 청산, 매도 들고 감
        return (b[k] - eb) + hold_winner(tt, a, es, -1, k, ms)
    return (es - a[k]) + hold_winner(tt, b, eb, 1, k, ms)


def main():
    ev = nh.events()
    ev = ev[ev.importance == "high"].reset_index(drop=True)
    ev_ms = ev.t.values.astype("datetime64[ms]").astype(np.int64)
    key = ev.currency + " " + ev.event.str.replace(r"\s*\(.*\)", "", regex=True)
    kmap = dict(zip(ev_ms, key))
    rows, last = [], 0
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        f = ev_ms[(ev_ms >= int(s.timestamp() * 1000)) & (ev_ms < int(e.timestamp() * 1000))]
        if not len(f):
            continue
        t, bid, ask = lb.load_month(y, m)
        for ms in f:
            if ms - last < MAXH + PRE:  # 앞 발표 매매가 끝날 수 있는 60분 안의 발표는 건너뜀 (겹쳐 두 번 세지 않게)
                continue
            last = ms
            for name, mt in METHODS.items():
                r = one(t, bid, ask, int(ms), mt)
                if r is not None:
                    rows.append((y, name, r, kmap[ms]))
        del t, bid, ask
        gc.collect()
        print(f"  {y}-{m:02d} 완료", flush=True)
    d = pd.DataFrame(rows, columns=["y", "방법", "oz", "발표"])
    d.to_csv(lb.TICK_DIR.parent.parent / "results" / "news_breakout_events.csv", index=False, encoding="utf-8-sig")
    out = d.pivot_table(index="방법", columns="y", values="oz", aggfunc="sum").round(1)
    g = d.groupby("방법").oz
    out["5년"] = g.sum().round(1)
    out["이긴%"] = (g.apply(lambda v: (v > 0).mean()) * 100).round(1)
    out["최악"] = g.min().round(1)
    out["하위1%"] = g.quantile(0.01).round(1)
    out["상위5번 빼면"] = (g.sum() - g.apply(lambda v: v.nlargest(5).sum())).round(1)
    out = out.loc[list(METHODS)]
    out.to_csv(lb.TICK_DIR.parent.parent / "results" / "news_breakout.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print(out.to_string())


if __name__ == "__main__":
    main()
