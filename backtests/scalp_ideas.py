"""단타 전략 3종 스크리닝 (2022-01 ~ 현재, 실제 틱 체결)

검증 방식은 strategies.py와 같음: 전략마다 설정 후보 8개 중 2022~2023년 PF가 가장 좋은 것 하나를 고르고
(학습 거래 50건 이상), 그 설정 그대로 2024 / 2025 / 2026 PF를 확인. 통과 기준: 모든 연도 PF 1.1 이상.
추가로 거래당 비용 0.2달러를 더 뺐을 때의 2024~2026 PF도 표시 (README: 비용 0.2달러 이상이면 단타 우위가 사라졌음).

시간은 모두 서버시간. 진입은 신호 봉이 끝난 뒤 다음 봉 첫 틱 (매수 Ask / 매도 Bid),
손절/익절은 처음 닿는 틱 가격으로 체결. 전략마다 포지션은 한 번에 1개.

1. 추세 눌림목  : 4시간봉 돈치안 추세(trend_trader.py와 같은 N25 진입 / N12 청산) 상태가 매수일 때만,
                  M5/M15 RSI(2)가 X 미만이면 매수 (매도는 반대). 종가가 SMA5를 넘으면(매도는 아래로) 청산,
                  손절 = k x ATR(14), 최대 보유 20봉. 4시간봉 상태는 완성된 봉만 사용
2. 아시아 되돌림: 01:00 ~ 범위 끝(03:00/05:00) 고가/저가 범위. 범위 끝 ~ 09:00 사이 5분봉이 범위 위로 꼬리를 냈다가
                  범위 안에서 마감하면 매도 (아래는 매수, 하루 1번). 손절 = 범위 끝 ± 범위폭 x b,
                  익절 = 범위 중간 또는 반대편, 10:00 시간 청산. 범위폭 1달러 미만인 날은 제외
3. 스퀴즈 돌파  : 볼린저밴드(20, 2)가 켈트너 채널(20, 1.5 ATR) 안에 6봉 이상 있다가 빠져나온 봉에서
                  종가가 SMA20 위면 매수, 아래면 매도. 손절 = k x ATR(14),
                  익절 = 위험의 1.5배 (최대 48봉) 또는 종가가 SMA20을 반대로 넘으면 청산

사용법: python -m backtests.scalp_ideas   (long_period를 먼저 한 번 실행해 틱이 저장돼 있어야 함)
"""
import itertools

import numpy as np
import pandas as pd

import backtests.long_period as lb
from backtests.strategies import (PASS_PF, TRAIN_END, YEARS, TickStore, bar_exit_specs, day_sessions, evaluate,
                                  pf, resample, to_ms)
from indicators import atr, rsi

MIN_TRAIN_TRADES = 50
EXTRA_COST = 0.2
M5_CACHE = lb.TICK_DIR.parent / "m5_bars.npz"


# ---------------------------------------------------------------- 데이터 (메모리 절약)
def m5_bars():
    """틱으로 5분봉 만들기 (Bid 기준, lb.bars_from_ticks와 같은 결과).
    한 달씩 Bid 배열만 읽어 numpy로 묶고, 결과는 data/m5_bars.npz에 저장해 다음부터 다시 쓴다.
    틱 파일이 캐시보다 새로우면 다시 만든다."""
    newest = max(p.stat().st_mtime for d in (lb.TICK_DIR, lb.TICK_Z_DIR) for p in d.glob("*.npz"))
    if M5_CACHE.exists() and M5_CACHE.stat().st_mtime >= newest:
        z = np.load(M5_CACHE)
        b = pd.DataFrame({c: z[c] for c in ("open", "high", "low", "close")})
        b["time"] = pd.to_datetime(z["k"] * 300, unit="s")
        return b
    parts = []
    for y, m in lb.months():
        t, bid, _ = lb.load_month(y, m)
        key = t // 300_000
        s = np.concatenate(([0], np.flatnonzero(np.diff(key)) + 1))
        e = np.append(s[1:], len(key)) - 1
        parts.append((key[s], bid[s], np.maximum.reduceat(bid, s), np.minimum.reduceat(bid, s), bid[e]))
        del t, bid, key
    k, o, h, lo, c = (np.concatenate(x) for x in zip(*parts))
    _, first = np.unique(k, return_index=True)       # 달 경계에서 겹친 봉은 앞의 것만
    k, o, h, lo, c = k[first], o[first], h[first], lo[first], c[first]
    np.savez(M5_CACHE, k=k, open=o, high=h, low=lo, close=c)
    b = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c})
    b["time"] = pd.to_datetime(k * 300, unit="s")
    return b


# ---------------------------------------------------------------- 체결
def execute(specs, store):
    """strategies.execute와 같고, tp_abs(익절 가격 직접 지정)를 추가로 지원"""
    out, busy = [], -1
    for s in specs:
        if s["entry"] < busy or s["exit"] <= s["entry"]:
            continue
        t, bid, ask = store.window(s["entry"], s["exit"])
        k = np.searchsorted(t, s["entry"])
        x = np.searchsorted(t, s["exit"])
        if k >= len(t) - 1 or x >= len(t):
            continue
        side = s["side"]
        e = ask[k] if side == 1 else bid[k]
        sl = s["sl_abs"] if "sl_abs" in s else e - side * s["sl_dist"]
        risk = (e - sl) * side
        if risk <= 0:
            continue
        if "tp_abs" in s:
            tp = s["tp_abs"]
            if (tp - e) * side <= 0:      # 진입가가 이미 익절가를 지나쳤으면 건너뜀
                continue
        else:
            tp = e + side * s["tp_r"] * risk if s.get("tp_r") else None
        price = bid if side == 1 else ask
        seg = price[k + 1:x]
        if side == 1:
            hit = (seg <= sl) | (seg >= tp) if tp else (seg <= sl)
        else:
            hit = (seg >= sl) | (seg <= tp) if tp else (seg >= sl)
        idx = np.flatnonzero(hit)
        j = k + 1 + idx[0] if idx.size else x
        pnl = (price[j] - e) * side
        out.append((t[k], t[j], side, pnl, pnl / risk, risk))
        busy = t[j]
    tr = pd.DataFrame(out, columns=["open_ms", "close_ms", "side", "pnl_oz", "r", "risk"])
    tr["close"] = pd.to_datetime(tr.close_ms, unit="ms")
    return tr


def next_cross(arr):
    def exit_fn(i, side):
        a = arr[side]
        p = np.searchsorted(a, i + 1)
        return a[p] if p < len(a) else None
    return exit_fn


# ---------------------------------------------------------------- 1. 추세 눌림목
def h4_state(h4, n=25):
    """4시간봉 돈치안 추세 상태 (+1 매수 / -1 매도 / 0 없음), trend_trader.py와 같은 진입/청산 규칙"""
    c = h4.close.values
    hh = h4.high.rolling(n).max().shift(1).values
    ll = h4.low.rolling(n).min().shift(1).values
    xh = h4.high.rolling(n // 2).max().shift(1).values
    xl = h4.low.rolling(n // 2).min().shift(1).values
    st, out = 0, np.zeros(len(c), dtype=int)
    for i in range(len(c)):
        if (st == 1 and c[i] < xl[i]) or (st == -1 and c[i] > xh[i]):
            st = 0
        elif st == 0:
            st = 1 if c[i] > hh[i] else -1 if c[i] < ll[i] else 0
        out[i] = st
    return out


def attach_state(bars, tf_minutes, h4, state):
    """각 봉이 끝난 시점에 이미 완성된 마지막 4시간봉의 상태"""
    left = pd.DataFrame({"done": bars.time + pd.Timedelta(minutes=tf_minutes)})
    right = pd.DataFrame({"avail": h4.time + pd.Timedelta(hours=4), "state": state})
    m = pd.merge_asof(left, right, left_on="done", right_on="avail", direction="backward")
    return m.state.fillna(0).astype(int).values


def pullback(bars, st, x, k):
    r2 = rsi(bars.close, 2).values
    c = bars.close.values
    sma5 = bars.close.rolling(5).mean().values
    longs = np.flatnonzero((st == 1) & (r2 < x))
    shorts = np.flatnonzero((st == -1) & (r2 > 100 - x))
    exit_fn = next_cross({1: np.flatnonzero(c > sma5), -1: np.flatnonzero(c < sma5)})
    entries = sorted([(i, 1) for i in longs] + [(i, -1) for i in shorts])
    return bar_exit_specs(bars, entries, exit_fn, (k * atr(bars)).values, max_hold=20)


# ---------------------------------------------------------------- 2. 아시아 되돌림
def asia_fade(m5, range_end_h, buf, tp_mode):
    t_all = to_ms(m5.time)
    pos = {tm: i for i, tm in enumerate(m5.time)}
    ranges = day_sessions(m5, (1, 0), (range_end_h, 0))
    windows = day_sessions(m5, (range_end_h, 0), (9, 0))
    specs = []
    for d, w in windows.items():
        r = ranges.get(d)
        if r is None or len(r) < 12:      # 범위 구간 5분봉이 12개(1시간) 미만인 날은 제외
            continue
        hi, lo = r.high.max(), r.low.min()
        width = hi - lo
        if width < 1.0:
            continue
        mid = (hi + lo) / 2
        exit_ms = int(to_ms(pd.Timestamp(d) + pd.Timedelta(hours=10)))
        for row in w.itertuples():
            if row.high > hi and lo < row.close < hi:
                side = -1
            elif row.low < lo and lo < row.close < hi:
                side = 1
            else:
                continue
            i = pos[row.time]
            if i + 1 >= len(m5):
                break
            sl = hi + buf * width if side == -1 else lo - buf * width
            tp = mid if tp_mode == "중간" else (lo if side == -1 else hi)
            specs.append({"entry": t_all[i + 1], "side": side, "exit": exit_ms, "sl_abs": sl, "tp_abs": tp})
            break
    return specs


# ---------------------------------------------------------------- 3. 스퀴즈 돌파
def squeeze(bars, k, exit_mode, min_sq=6):
    c = bars.close
    sma = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    a20 = atr(bars, 20)
    sq = ((sma + 2 * sd) < (sma + 1.5 * a20)) & ((sma - 2 * sd) > (sma - 1.5 * a20))
    held = sq.astype(int).rolling(min_sq).sum().shift(1) >= min_sq
    release = (held & ~sq).values
    cv, sv = c.values, sma.values
    entries = [(i, 1 if cv[i] > sv[i] else -1) for i in np.flatnonzero(release)]
    sl_dist = (k * atr(bars)).values
    if exit_mode == "SMA20":
        exit_fn = next_cross({1: np.flatnonzero(cv < sv), -1: np.flatnonzero(cv > sv)})
        return bar_exit_specs(bars, entries, exit_fn, sl_dist, max_hold=48)
    specs = bar_exit_specs(bars, entries, lambda i, s: None, sl_dist, max_hold=48)
    for s in specs:
        s["tp_r"] = 1.5
    return specs


# ---------------------------------------------------------------- 실행
def main():
    print("봉 만드는 중...", flush=True)
    m5 = m5_bars()
    print(f"5분봉 {len(m5):,}개 ({m5.time.iat[0]:%Y-%m-%d} ~ {m5.time.iat[-1]:%Y-%m-%d})", flush=True)
    m15, h4 = resample(m5, "15min"), resample(m5, "4h")
    state = h4_state(h4)
    st = {"M5": attach_state(m5, 5, h4, state), "M15": attach_state(m15, 15, h4, state)}
    bars = {"M5": m5, "M15": m15}
    store = TickStore(keep=2)

    grids = {
        "1. 추세 눌림목": [(f"{tf} RSI2<{x} 손절 {k}ATR", lambda tf=tf, x=x, k=k: pullback(bars[tf], st[tf], x, k))
                       for tf, x, k in itertools.product(["M5", "M15"], [5, 10], [1.5, 2.5])],
        "2. 아시아 되돌림": [(f"범위 01~{e:02d}시, 손절 +{b}폭, 익절 {tp}",
                          lambda e=e, b=b, tp=tp: asia_fade(m5, e, b, tp))
                         for e, b, tp in itertools.product([3, 5], [0.25, 0.5], ["중간", "반대편"])],
        "3. 스퀴즈 돌파": [(f"{tf} 손절 {k}ATR, 청산 {ex}", lambda tf=tf, k=k, ex=ex: squeeze(bars[tf], k, ex))
                       for tf, k, ex in itertools.product(["M5", "M15"], [1.5, 2.5], ["1.5R", "SMA20"])],
    }

    all_rows, picks = [], []
    for fam, grid in grids.items():
        best = None
        for name, make in grid:
            tr = execute(make(), store)
            row = {"전략": fam, "설정": name, **evaluate(tr)}
            late = tr.pnl_oz[tr.close.dt.year >= TRAIN_END]
            row[f"PF 24-26 비용+{EXTRA_COST}"] = pf(late - EXTRA_COST)
            row["평균손익$"] = round(tr.pnl_oz.mean(), 3) if len(tr) else 0
            all_rows.append(row)
            n_train = int((tr.close.dt.year < TRAIN_END).sum())
            if n_train >= MIN_TRAIN_TRADES and row["PF 22-23"] != float("inf") and \
                    (best is None or row["PF 22-23"] > best["PF 22-23"]):
                best = row
            print(f"  {fam} | {name}: 거래 {row['거래수']}, PF 22-23 {row['PF 22-23']}, PF 24-26 {row['PF 24-26']}",
                  flush=True)
        if best:
            picks.append(best)
        print(f"완료: {fam}", flush=True)

    res = pd.DataFrame(all_rows)
    res.to_csv("scalp_ideas_all.csv", index=False, encoding="utf-8-sig")
    pk = pd.DataFrame(picks)
    pk["통과"] = pk[[f"PF {y}" for y in YEARS]].min(axis=1).ge(PASS_PF).map({True: "통과", False: "-"})
    pk.to_csv("scalp_ideas_picked.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print("\n=== 전체 설정 ===")
    print(res.to_string(index=False))
    print("\n=== 2022~2023 PF로 고른 설정 -> 2024~2026 검증 ===")
    print(pk.to_string(index=False))


if __name__ == "__main__":
    main()
