"""PA Master Grade-A (금 1시간봉, 거래량 조건 끔) 승률 올리기 세 가지, 실제 틱 체결

신호는 tv_batch2.pa_master(use_vol=False) 그대로, 체결은 tv_pa_ticks와 같음 (다음 봉 첫 틱, 손절은 넘은 첫 틱 가격, 익절 지정가)
1. 4시간봉 추세 맞추기: 신호 시각에 마지막으로 완성된 4시간봉 기준, 같은 방향일 때만 진입
     EMA200  : 종가 > EMA200 이면 매수만 / < 이면 매도만
     EMA50/200: EMA50 > EMA200 이면 매수만 / 반대면 매도만
     구름    : 일목 구름(9/26/52, 그 봉 위치 구름) 위면 매수만 / 아래면 매도만 (trend_trader.py와 같음)
2. 나눠 익절: 절반 aR 익절, 나머지 bR 익절 (나머지 손절은 원래 자리 또는 본전)
3. 본전 손절: 가격이 tR 가면 손절을 진입가로 옮김
R = 손익 / |진입가 - 처음 손절가| (스왑 포함). 승률 = R > 0 인 거래 비율
과적합 방지: 2022~2024로 고르고 2025~2026에서 확인

사용법: python -m backtests.tv_pa_improve
"""
import numpy as np
import pandas as pd

import backtests.long_period as lb
import backtests.tv_single as ts
from backtests.tv_batch2 import bars_vol, pa_master
from backtests.tv_pa_ticks import month_ticks, next_month
from indicators import ema
import backtests.tickvol_m1 as tv


def h4_trend(m1, h1):
    """h1 봉마다 (신호 = 봉 마감 시각) 마지막 완성 4시간봉의 추세 방향 (+1/-1/0) 세 가지"""
    h4 = ts.bars(m1, 240)
    c = h4.close
    mid = lambda n: (h4.high.rolling(n).max() + h4.low.rolling(n).min()) / 2
    sa, sb = ((mid(9) + mid(26)) / 2).shift(26), mid(52).shift(26)
    up, dn = np.maximum(sa, sb), np.minimum(sa, sb)
    t = pd.DataFrame({"end": h4.time + pd.Timedelta(hours=4),
                      "EMA200": np.sign(c - ema(c, 200)),
                      "EMA50/200": np.sign(ema(c, 50) - ema(c, 200)),
                      "구름": np.where(c > up, 1, np.where(c < dn, -1, 0))})
    m = pd.merge_asof(pd.DataFrame({"end": h1.time + pd.Timedelta(hours=1)}), t, on="end", direction="backward")
    return {k: m[k].fillna(0).values.astype(int) for k in ("EMA200", "EMA50/200", "구름")}


def run(d, side, stop, legs, be=None):
    """legs = [(비중, 익절 R)], be = 본전으로 옮길 R (None이면 안 옮김). 한 번에 한 거래"""
    times = d.time.values
    rows, busy = [], np.datetime64("1970-01-01")
    last = list(lb.months())[-1]
    for i in np.flatnonzero(side[:-1]):
        s, sl0 = int(side[i]), stop[i]
        T = times[i + 1]
        if T < busy:
            continue
        y, m = pd.Timestamp(T).year, pd.Timestamp(T).month
        t, bid, ask = month_ticks(y, m)
        k = int(np.searchsorted(t, T.astype("datetime64[ms]").astype(np.int64)))
        if k >= len(t):
            y, m = next_month(y, m)
            if (y, m) > last:
                break
            t, bid, ask = month_ticks(y, m)
            k = 0
        ent, et = (ask[k] if s == 1 else bid[k]), t[k]
        risk = abs(ent - sl0)
        if (ent - sl0) * s <= 0:
            continue
        sl, open_legs, trig = sl0, [(w, ent + s * r * risk) for w, r in legs], (None if be is None else ent + s * be * risk)
        done, pnl, xt = [], 0.0, None
        while open_legs:
            px_out = bid if s == 1 else ask              # 청산 쪽 가격
            seg = px_out[k:]
            hit = lambda lv, beyond: np.flatnonzero(seg * s >= lv * s if beyond else seg * s <= lv * s)
            ev = []
            h = hit(sl, False)
            if h.size:
                ev.append((h[0], 0, None))
            for w, tp in open_legs:
                h = hit(tp, True)
                if h.size:
                    ev.append((h[0], 1, (w, tp)))
            if trig is not None:
                h = hit(trig, True)
                if h.size:
                    ev.append((h[0], 2, None))
            if not ev:
                y, m = next_month(y, m)
                if (y, m) > last:
                    break
                t, bid, ask = month_ticks(y, m)
                k = 0
                continue
            j, kind, leg = min(ev, key=lambda e: (e[0], e[1]))
            j += k
            if kind == 0:                               # 손절 (또는 본전): 남은 것 전부 그 틱 가격
                w = sum(w for w, _ in open_legs)
                pnl += w * (px_out[j] - ent) * s
                done.append((w, t[j]))
                open_legs = []
            elif kind == 1:
                pnl += leg[0] * (leg[1] - ent) * s
                done.append((leg[0], t[j]))
                open_legs.remove(leg)
            else:
                sl, trig = ent, None
            k = j
            xt = t[j]
        if open_legs:
            break
        e64 = np.datetime64(int(et), "ms")
        sw = sum(w * ts.swap_nights(e64, np.datetime64(int(x), "ms")) for w, x in done) * (ts.SWAP_LONG if s == 1 else ts.SWAP_SHORT)
        x64 = np.datetime64(int(xt), "ms")
        rows.append((e64, x64, s, (pnl + sw) / risk))
        busy = x64
    return pd.DataFrame(rows, columns=["open", "close", "side", "r"])


def stats(tr):
    if not len(tr):
        return {"거래": 0}
    r = tr.r
    eq = r.cumsum()
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    return {"거래": len(r), "승률%": round((r > 0).mean() * 100, 1), "PF": round(g / l, 2) if l else np.inf,
            "거래당R": round(r.mean(), 3), "합계R": round(r.sum(), 1), "낙폭R": round((eq.cummax() - eq).max(), 1)}


def split(tr):
    y = pd.to_datetime(tr.open).dt.year
    a, b = stats(tr[y <= 2024]), stats(tr[y >= 2025])
    out = {f"앞 {k}": v for k, v in a.items()}
    out.update({f"뒤 {k}": v for k, v in b.items()})
    out.update({f"{yy}": round(tr[y == yy].r.sum(), 1) for yy in range(2022, 2027)})
    return out


def main():
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = ts.load_m1("XAUUSD")
    h1 = bars_vol(m1, tv.load(), 60)
    side, stop, _, _ = pa_master(h1, use_vol=False)
    trend = h4_trend(m1, h1)
    exits = {"3R": ([(1, 3)], None), "2R": ([(1, 2)], None), "1R": ([(1, 1)], None),
             "반 1R + 반 3R": ([(.5, 1), (.5, 3)], None), "반 1R + 반 2R": ([(.5, 1), (.5, 2)], None),
             "반 1R + 반 3R, 나머지 본전": ([(.5, 1), (.5, 3)], 1.0),
             "3R, 1R에서 본전": ([(1, 3)], 1.0), "3R, 1.5R에서 본전": ([(1, 3)], 1.5), "3R, 2R에서 본전": ([(1, 3)], 2.0),
             "2R, 1R에서 본전": ([(1, 2)], 1.0)}
    out = []
    for name, (legs, be) in exits.items():
        tr = run(h1, side, stop, legs, be)
        out.append({"추세": "없음", "청산": name, **split(tr)})
        print(name, "끝", flush=True)
    for tname, tdir in trend.items():
        sd = np.where(side == tdir, side, 0)
        for name in ("3R", "2R", "반 1R + 반 3R"):
            legs, be = exits[name]
            tr = run(h1, sd, stop, legs, be)
            out.append({"추세": tname, "청산": name, **split(tr)})
        print(tname, "끝", flush=True)
    res = pd.DataFrame(out)
    res.to_csv("results/tv_pa_improve.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    # 이번 주 신호의 4시간봉 방향
    since = h1.time >= pd.Timestamp("2026-10-19")
    for i in np.flatnonzero(side != 0):
        if since.iat[i]:
            print(h1.time.iat[i], "신호", side[i], {k: v[i] for k, v in trend.items()})


if __name__ == "__main__":
    main()
