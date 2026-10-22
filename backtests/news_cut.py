"""뉴스 양방향에서 '한쪽 정리' 조건 비교 (실제 틱): 많이 튄 20개 + CPI·NFP·FOMC 금리 결정 전체

공통: 발표 30초 전 매수(Ask)·매도(Bid) 같은 lot. 판단 뒤 진 쪽은 0.3초 뒤 첫 틱에 청산
      5분 안에 판단 안 나면 양쪽 정리
정리 조건 (기준 = 발표 시각 중간가)
  고정 X$        : 중간가가 X$ 움직이면 (HK = 2$)
  비율 X%        : 가격의 X% 움직이면
  변동성 k배     : 발표 전 30분 1분봉 평균 폭 x k 만큼 움직이면
  유지 N초       : 2$ (또는 0.05%) 넘은 상태가 N초 계속되면
  T초 방향       : 발표 T초 뒤 중간가가 0.5$ 넘게 움직였으면 그 방향만 남김, 아니면 양쪽 정리
남은 쪽 청산
  10분       : 발표 10분에 정리
  10분+추적  : +8$ 넘으면 최고점에서 2.5$ 밀릴 때, 아니면 10분 (캡처 BOE)
  15분       : 발표 15분에 정리
  모두 비상 손절: 발표 시각 중간가에서 반대쪽으로 2$ 넘게 되돌아오면 (판단이 틀렸던 것) 즉시 정리
결과 $/oz (매수+매도 합)

사용법: python -m backtests.news_cut      (results/news_cut.csv, news_cut_top20.csv)
"""
import gc

import numpy as np
import pandas as pd

import backtests.long_period as lb

PRE, QUIET, DELAY = 30_000, 300_000, 300
OUT = lb.TICK_DIR.parent.parent / "results"
MAJOR3 = ("USD CPI m/m", "USD 실업률", "USD 연방 금리 결정")
RULES = {**{f"고정 {x}$": ("abs", x) for x in (1, 2, 3, 5)},
         **{f"비율 {x}%": ("pct", x) for x in (0.03, 0.05, 0.1)},
         **{f"변동성 {k}배": ("vol", k) for k in (3, 5, 8)},
         **{f"유지 {n}초 (2$)": ("hold", n) for n in (3, 10)},
         "유지 5초 (0.05%)": ("hold_pct", 5),
         **{f"{s}초 방향": ("dir", s) for s in (10, 30, 60)}}
EXITS = ("10분", "10분+추적", "15분")


def first(c):
    k = np.flatnonzero(c)
    return int(k[0]) if k.size else None


def held(c, tt, n_ms):
    k = np.flatnonzero(np.diff(np.r_[0, c.astype(np.int8), 0]))
    for s, e in zip(k[::2], k[1::2]):
        j = int(np.searchsorted(tt, tt[s] + n_ms))
        if j < e:
            return j
    return None


def one(tt, b, a, ms, rule, exit_mode, vol):
    kind, p = rule
    mid = (b + a) / 2
    r0 = int(np.searchsorted(tt, ms))
    if r0 >= len(tt) - 1:
        return None
    mr = mid[r0]
    eb, es = a[0], b[0]
    quiet = min(int(np.searchsorted(tt, ms + QUIET)), len(tt) - 1)
    after = np.zeros(len(tt), bool)
    after[r0:quiet] = True
    d = mid - mr
    th = {"abs": p, "pct": mr * p / 100, "vol": vol * p, "hold": 2.0, "hold_pct": mr * 0.0005}.get(kind)
    side = None
    if kind == "dir":
        k = min(int(np.searchsorted(tt, ms + p * 1000)), len(tt) - 1)
        if abs(d[k]) >= 0.5:
            side = 1 if d[k] > 0 else -1
    else:
        up, dn = after & (d >= th), after & (d <= -th)
        if kind in ("hold", "hold_pct"):
            ku, kd = held(up, tt, p * 1000), held(dn, tt, p * 1000)
        else:
            ku, kd = first(up), first(dn)
        if ku is not None or kd is not None:
            side = 1 if kd is None or (ku is not None and ku <= kd) else -1
            k = ku if side == 1 else kd
    if side is None:
        q = quiet if kind != "dir" else k
        return (b[q] - eb) + (es - a[q]), 0
    kc = min(int(np.searchsorted(tt, tt[k] + DELAY)), len(tt) - 1)
    lose = (es - a[kc]) if side == 1 else (b[kc] - eb)
    x, e = (b, eb) if side == 1 else (a, es)
    pnl = (x - e) * side
    tend = min(int(np.searchsorted(tt, ms + (900_000 if exit_mode == "15분" else 600_000))), len(tt) - 1)
    seg = pnl[kc:tend + 1]
    stop = (d[kc:tend + 1] * side) <= -2.0                     # 발표가 반대쪽 2$ 되돌림
    if exit_mode == "10분+추적":
        peak = np.maximum.accumulate(seg)
        stop |= (peak >= 8) & (seg <= peak - 2.5)
    j = first(stop)
    if j is not None:
        jj = min(int(np.searchsorted(tt, tt[kc + j] + DELAY)), len(tt) - 1)
        win = pnl[jj]
    else:
        win = pnl[tend]
    return lose + win, side


def main():
    mv = pd.read_csv(OUT / "news_moves.csv", parse_dates=["t"])
    mv["a5"] = mv["5분 최대%"].abs()
    top = set(mv.sort_values("a5", ascending=False).head(20).t)
    ev = mv[mv.key.isin(MAJOR3) | mv.t.isin(top)].sort_values("t").reset_index(drop=True)
    ev["top20"] = ev.t.isin(top)
    ev_ms = ev.t.values.astype("datetime64[ms]").astype(np.int64)
    rows = []
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        idx = np.flatnonzero((ev_ms >= int(s.timestamp() * 1000)) & (ev_ms < int(e.timestamp() * 1000)))
        if not len(idx):
            continue
        t, bid, ask = lb.load_month(y, m)
        for i in idx:
            ms = int(ev_ms[i])
            k0, k1 = int(np.searchsorted(t, ms - PRE)), int(np.searchsorted(t, ms + 1_000_000))
            tt, b, a = t[k0:k1], bid[k0:k1], ask[k0:k1]
            vol = ev["평소 1분 폭"].iat[i] if ev["평소 1분 폭"].iat[i] == ev["평소 1분 폭"].iat[i] else 1.0
            fin = np.sign(ev["15분 순변화$"].iat[i])
            for rn, rule in RULES.items():
                for xm in EXITS:
                    r = one(tt, b, a, ms, rule, xm, vol)
                    if r is not None:
                        rows.append((i, rn, xm, r[0], r[1], r[1] == fin))
        del t, bid, ask
        gc.collect()
        print(f"  {y}-{m:02d} 완료", flush=True)
    d = pd.DataFrame(rows, columns=["i", "조건", "청산", "oz", "남긴쪽", "방향맞음"]).join(ev[["t", "key", "top20"]], on="i")
    d["y"] = d.t.dt.year
    d.to_csv(OUT / "news_cut_events.csv", index=False, encoding="utf-8-sig")
    res = []
    for (rn, xm), g in d.groupby(["조건", "청산"], sort=False):
        tp, mj = g[g.top20], g[g.key.isin(MAJOR3)]
        row = {"조건": rn, "청산": xm, "상위20 합": round(tp.oz.sum(), 1), "상위20 방향 맞음": int(tp.방향맞음.sum()),
               "상위20 최악": round(tp.oz.min(), 1), "3대 발표 횟수": len(mj)}
        for y in range(2022, 2027):
            row[str(y)] = round(mj.oz[mj.y == y].sum(), 1)
        row["3대 5년"] = round(mj.oz.sum(), 1)
        row["3대 2022~25"] = round(mj.oz[mj.y <= 2025].sum(), 1)
        row["3대 최악"] = round(mj.oz.min(), 1)
        row["3대 정리 안 함(무반응)"] = int((mj.남긴쪽 == 0).sum())
        res.append(row)
    res = pd.DataFrame(res)
    res.to_csv(OUT / "news_cut.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.sort_values("3대 5년", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
