"""HK 마틴 마스터 '프리미엄 뉴스' 규칙 재현 (실제 틱, 2022-01 ~ 2026-10)

10/22 생방송 슬라이드 + 10/21 실제 계좌 틱 대조로 확인한 규칙:
  1) 중요 지표 발표 30초 전 같은 lot으로 매수(Ask)·매도(Bid) 동시 진입
  2) 한쪽이 CUT$ 역행하면 그쪽 청산 (10/21 BOE: 2.0$ 역행에서 19초 만에 손절)
  3) 발표 후 5분 동안 어느 쪽도 CUT에 안 닿으면 양쪽 동시 종료 (10/21 09:05:00)
  4) 남은 쪽: 발표 10분 뒤 이익이면 종료, 손실이면 본전(0) 이상 될 때 종료 (최대 60분, 그 전에 CUT 손절)
     win_stop=True면 남은 쪽도 CUT$ 역행에서 손절 (슬라이드에는 없는 가정), False면 손절 없이 최대 60분
     win_cut을 주면 남은 쪽 손절폭만 따로 (비상 손절), close10=True면 10분에 손익과 무관하게 종료
     target을 주면 10분 전이라도 남은 쪽 이익이 target$에 닿으면 종료 (슬라이드 "목표수익 미도달 시 조기 종료", 값은 모름)
캘린더 시각은 지금 서버 시차(GMT+3)로 나와서 미국 서머타임이 아닌 날은 1시간 당김 (2026-10-22 고침: 전에는 겨울에 1시간 늦게 진입)
대상: MT5 캘린더(calendar_all.csv) USD/EUR/GBP, 중요도 high (moderate는 따로 비교)
같은 시각 발표가 여러 개면 한 번만. 결과는 금 1온스(1 lot의 1/100) 기준 $ 합계 (매수+매도)

사용법: python -m backtests.news_hk
"""
import gc
import os

import numpy as np
import pandas as pd

import backtests.long_period as lb

CUTS = (1.5, 2.0, 3.0)
PRE_MS, QUIET_MS, HOLD_MS, MAX_MS = 30_000, 300_000, 600_000, 3_600_000


def events():
    f = os.path.join(os.environ["APPDATA"], "MetaQuotes/Terminal/Common/Files/calendar_all.csv")
    c = pd.read_csv(f, encoding="cp949")
    c = c[c.currency.isin(["USD", "EUR", "GBP"]) & c.importance.isin(["high", "moderate"])].copy()
    c["t"] = pd.to_datetime(c.time, format="%Y.%m.%d %H:%M")
    c = c[(c.t.dt.hour > 0) | (c.t.dt.minute > 0)]       # 시각 없는 하루 종일 일정(00:00) 제외
    dst = c.t.dt.tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward").map(lambda x: bool(x.dst()))
    c["t"] = c.t - pd.to_timedelta((~dst.values).astype(int), unit="h")
    c = c[(c.t >= "2022-01-03") & (c.t < "2026-10-22")]
    c["rank"] = c.importance.map({"high": 2, "moderate": 1})
    c = c.sort_values(["t", "rank"], ascending=[True, False]).drop_duplicates("t")
    c = c[c.t.dt.weekday < 5]
    return c[["t", "currency", "importance", "event"]].reset_index(drop=True)


def first(cond):
    k = np.flatnonzero(cond)
    return k[0] if k.size else None


def one(t, bid, ask, ms, cut, target=None, win_stop=True, win_cut=None, close10=False):
    """발표 시각 ms 하나 -> (매수 $/oz, 매도 $/oz, 결과 종류, 청산 시각 ms) 또는 None"""
    k0 = int(np.searchsorted(t, ms - PRE_MS))
    if k0 >= len(t) or t[k0] - (ms - PRE_MS) > 20_000:
        return None
    eb, es = ask[k0], bid[k0]
    k_end = int(np.searchsorted(t, ms + MAX_MS))
    b, a, tt = bid[k0:k_end], ask[k0:k_end], t[k0:k_end]
    if len(tt) < 10:
        return None
    lb_ = first(b <= eb - cut)                      # 매수 손절 (Bid)
    ls_ = first(a >= es + cut)                      # 매도 손절 (Ask)
    quiet = int(np.searchsorted(tt, ms + QUIET_MS))
    if (lb_ is None or lb_ >= quiet) and (ls_ is None or ls_ >= quiet):
        q = min(quiet, len(tt) - 1)
        return b[q] - eb, es - a[q], "5분 무반응", int(tt[q])
    if lb_ is not None and (ls_ is None or lb_ <= ls_):
        loser, pb, side, e = "매수", -cut, -1, es
        x, sl = a, es + cut
    else:
        loser, pb, side, e = "매도", -cut, 1, eb
        x, sl = b, eb - cut
    k1 = lb_ if loser == "매수" else ls_
    h10 = int(np.searchsorted(tt, ms + HOLD_MS))
    pnl = (x - e) * side
    stop = first(pnl[k1:] <= -(win_cut or cut))
    stop = k1 + stop if stop is not None and win_stop else len(tt) - 1     # win_stop=False: 남은 쪽은 손절 없음 (슬라이드)
    if h10 < len(tt) and (pnl[h10] > 0 or close10):
        j = h10
    else:
        be = first(pnl[max(h10, k1):] >= 0)
        j = max(h10, k1) + be if be is not None else len(tt) - 1
    if target is not None:
        tg = first(pnl[k1:max(k1, min(h10, len(tt)))] >= target)
        if tg is not None:
            j = min(j, k1 + tg)
    j = min(j, stop)
    win = pnl[j]
    lose = es - a[k1] if loser == "매도" else b[k1] - eb       # 손절 틱 실제 가격 (발표 때 건너뛰면 -cut보다 큼)
    return (win, lose, "매도 손절 후 매수", int(tt[j])) if loser == "매도" else (lose, win, "매수 손절 후 매도", int(tt[j]))


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    ev = events()
    ev_ms = ev.t.values.astype("datetime64[ms]").astype(np.int64)
    res = {c: [None] * len(ev) for c in CUTS}
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        lo, hi = np.searchsorted(ev_ms, int(s.timestamp() * 1000)), np.searchsorted(ev_ms, int(e.timestamp() * 1000))
        if lo == hi:
            continue
        t, bid, ask = lb.load_month(y, m)
        for i in range(lo, hi):
            for c in CUTS:
                res[c][i] = one(t, bid, ask, int(ev_ms[i]), c)
        del t, bid, ask
        gc.collect()
        print(f"  {y}-{m:02d} 완료", flush=True)
    rows = []
    for c in CUTS:
        d = ev.copy()
        r = res[c]
        d["ok"] = [x is not None for x in r]
        d["oz"] = [x[0] + x[1] if x else np.nan for x in r]
        d["kind"] = [x[2] if x else "" for x in r]
        d = d[d.ok]
        d.to_csv(f"news_hk_cut{c}.csv", index=False, encoding="utf-8-sig")
        for name, g in [("전체 high", d[d.importance == "high"]), ("전체 moderate", d[d.importance == "moderate"])] + \
                [(f"high {yy}", d[(d.importance == "high") & (d.t.dt.year == yy)]) for yy in range(2022, 2027)]:
            if not len(g):
                continue
            rows.append({"손절폭$": c, "구분": name, "발표 수": len(g), "이긴 비율%": round((g.oz > 0).mean() * 100, 1),
                         "평균$/oz": round(g.oz.mean(), 2), "합계$/oz": round(g.oz.sum(), 1),
                         "5분 무반응%": round((g.kind == "5분 무반응").mean() * 100, 1)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("news_hk_summary.csv", index=False, encoding="utf-8-sig")
    d = pd.read_csv("news_hk_cut2.0.csv")
    d = d[d.importance == "high"]
    key = d.event.str.replace(r"\s*\(.*\)", "", regex=True)
    top = d.groupby([d.currency, key]).oz.agg(["count", "mean", "sum"]).query("count >= 15").sort_values("mean")
    print("\n발표 종류별 (high, 손절폭 2$, 15번 이상)")
    print(pd.concat([top.head(8), top.tail(8)]).round(2).to_string())


if __name__ == "__main__":
    main()
