"""뉴스 매매 깊게 확인 (실제 틱, high USD·EUR·GBP, 2022-01 ~ 2026-10)

진입 2가지
  돌파 X$   : 발표 전엔 안 들어가고, 발표 뒤 5분 안에 중간가가 발표 30초 전 대비 X$ 움직이면 그 방향 한쪽 (1 lot)
  양방향 HK : 발표 30초 전 매수·매도 (각 1 lot), 진입가 대비 2$ 역행한 쪽 청산, 5분 안에 판단 안 나면 둘 다 종료
  양방향 발표가 : 같은데, 판단 기준이 발표 시각 중간가 대비 2$ (10/20 09:00 세 계좌가 진입가와 무관하게 09:00:10에
                  동시 청산, 10/21 BOE 16:00:19 청산 둘 다 발표 시각 중간가 +2$ 넘은 첫 틱과 맞음)
청산 (돌파 진입 쪽 / 양방향의 남은 쪽)
  슬라이드  : 10분 뒤 이익이면 종료, 손실이면 본전 될 때 (비상 손절 6$, 최대 60분)
  익절5손절3: +5 / -3 (최대 60분)
  추적2     : 처음 -3 손절, 최고점에서 2$ 되돌리면 청산 (최대 60분)
  5분 / 15분 / 30분 : 그 시간에 종료 (비상 손절 6$)
  10분      : 발표 10분에 손익과 무관하게 종료 (비상 손절 6$)
  10분+추적 : +8$ 넘으면 최고점에서 2.5$ 밀릴 때 종료, 아니면 발표 10분에 종료 (비상 손절 6$)
              (캡처: 10/20 09:10:00 정각 +2, 10/21 BOE 최고 +9.26 뒤 16:07:04 +6.81)
체결: 시장가 신호 뒤 DELAY초 지난 첫 틱 가격 (0.3 / 1 / 3초), 체결마다 추가 미끄러짐 SLIP$ (0 / 0.3)
발표 고르기 (연도별 결과)
  전체 high / 주요 4개 (NFP 실업률·CPI m/m·FOMC 금리 결정·미국 GDP q/q, 같은 기간 결과를 보고 고른 것)
  워크포워드: 그 해 전까지 같은 방법으로 8번 이상 했고 평균이 플러스였던 발표 종류만 (2023년부터, 미래 정보 없음)
같은 묶음 안에서 앞 매매가 끝날 수 있는 60분 안의 발표는 건너뜀

사용법: python -m backtests.news_deep      (results/news_deep.csv, news_deep_events.csv)
"""
import gc
import itertools
import os

import numpy as np
import pandas as pd

import backtests.long_period as lb
import backtests.news_hk as nh

PRE, QUIET, HOLD, MAXH, ECUT = 30_000, 300_000, 600_000, 3_600_000, 6.0
MAJOR = ("USD 실업률", "USD CPI m/m", "USD 연방 금리 결정", "USD GDP q/q")
ENTRIES = [("돌파", x) for x in (1.0, 1.5, 2.0, 3.0, 4.0)] + [("양방향", 2.0), ("양방향 발표가", 2.0)]
EXITS = ("슬라이드", "익절5손절3", "추적2", "5분", "15분", "30분", "10분", "10분+추적")
ONLY = os.environ.get("NEWS_ENTRIES")      # 예: "돌파 2.0,양방향 발표가 2.0" 이면 그 진입만 (빨리 돌릴 때)
DELAYS = (300, 1000, 3000)
SLIPS = (0.0, 0.3)
OUT = lb.TICK_DIR.parent.parent / "results"


def first(c):
    k = np.flatnonzero(c)
    return int(k[0]) if k.size else None


def at(tt, k, delay):
    """k 틱에서 신호 -> delay ms 뒤 첫 틱"""
    return min(int(np.searchsorted(tt, tt[k] + delay)), len(tt) - 1)


def hold(tt, x, e, side, k, ms, mode, delay):
    """k에서 진입가 e로 들고 감 (x = 청산 가격 배열) -> $/oz"""
    pnl = (x - e) * side
    seg = pnl[k:]
    if mode == "슬라이드":
        h10 = int(np.searchsorted(tt, ms + HOLD))
        stop = first(seg <= -ECUT)
        stop = k + stop if stop is not None else len(tt) - 1
        if h10 < len(tt) and pnl[h10] > 0:
            j = h10
        else:
            be = first(pnl[max(h10, k):] >= 0)
            j = max(h10, k) + be if be is not None else len(tt) - 1
        j = min(j, stop)
        if j == stop and j < len(tt) - 1:
            j = at(tt, j, delay)
        return pnl[j]
    if mode == "익절5손절3":
        j = first((seg >= 5) | (seg <= -3))
        return pnl[at(tt, k + j, delay)] if j is not None else pnl[-1]
    if mode == "추적2":
        peak = np.maximum.accumulate(seg)
        j = first((seg <= -3) | (seg <= peak - 2) & (peak >= 1))
        return pnl[at(tt, k + j, delay)] if j is not None else pnl[-1]
    if mode == "10분+추적":
        te = min(int(np.searchsorted(tt, ms + HOLD)), len(tt) - 1)
        s2 = seg[: max(1, te - k + 1)]
        peak = np.maximum.accumulate(s2)
        j = first((s2 <= -ECUT) | ((peak >= 8) & (s2 <= peak - 2.5)))
        return pnl[at(tt, k + j, delay)] if j is not None else pnl[te]
    m = {"5분": 300_000, "15분": 900_000, "30분": 1_800_000, "10분": HOLD}[mode]
    te = min(int(np.searchsorted(tt, ms + m)), len(tt) - 1)
    j = first(seg[: max(1, te - k)] <= -ECUT)
    return pnl[at(tt, k + j, delay)] if j is not None else pnl[te]


def one(tt, b, a, ms, entry, mode, delay, slip):
    """발표 하나 -> $/oz (체결 수만큼 미끄러짐 뺌) 또는 None(매매 없음)"""
    kind, p = entry
    mid = (b + a) / 2
    m0 = mid[0]
    quiet = min(int(np.searchsorted(tt, ms + QUIET)), len(tt) - 1)
    if kind == "돌파":
        k = first((np.abs(mid[:quiet] - m0) >= p) & (tt[:quiet] >= ms))
        if k is None:
            return None
        side = 1 if mid[k] > m0 else -1
        k = at(tt, k, delay)
        e = a[k] if side == 1 else b[k]
        return hold(tt, b if side == 1 else a, e, side, k, ms, mode, delay) - 2 * slip
    eb, es = a[0], b[0]
    if kind == "양방향 발표가":
        r0 = int(np.searchsorted(tt, ms))
        mr = mid[min(r0, len(tt) - 1)]
        after = tt[:quiet] >= ms
        kb, ks = first(after & (mid[:quiet] <= mr - p)), first(after & (mid[:quiet] >= mr + p))
    else:
        kb, ks = first(b[:quiet] <= eb - p), first(a[:quiet] >= es + p)
    if kb is None and ks is None:
        return (b[quiet] - eb) + (es - a[quiet]) - 4 * slip
    down = kb is not None and (ks is None or kb <= ks)
    k = at(tt, kb if down else ks, delay)
    if down:
        return (b[k] - eb) + hold(tt, a, es, -1, k, ms, mode, delay) - 4 * slip
    return (es - a[k]) + hold(tt, b, eb, 1, k, ms, mode, delay) - 4 * slip


def main():
    ev = nh.events()
    ev = ev[ev.importance == "high"].reset_index(drop=True)
    ev["key"] = ev.currency + " " + ev.event.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    ev_ms = ev.t.values.astype("datetime64[ms]").astype(np.int64)
    ei_ok = [i for i, e in enumerate(ENTRIES) if not ONLY or f"{e[0]} {e[1]}" in ONLY.split(",")]
    combos = list(itertools.product(ei_ok, EXITS, DELAYS, SLIPS))
    res = np.full((len(ev), len(combos)), np.nan)
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        idx = np.flatnonzero((ev_ms >= int(s.timestamp() * 1000)) & (ev_ms < int(e.timestamp() * 1000)))
        if not len(idx):
            continue
        t, bid, ask = lb.load_month(y, m)
        for i in idx:
            ms = int(ev_ms[i])
            k0 = int(np.searchsorted(t, ms - PRE))
            if k0 >= len(t) or t[k0] - (ms - PRE) > 20_000:
                continue
            k1 = int(np.searchsorted(t, ms + MAXH))
            tt, b, a = t[k0:k1], bid[k0:k1], ask[k0:k1]
            if len(tt) < 10:
                continue
            for c, (ei, mode, dl, sl) in enumerate(combos):
                r = one(tt, b, a, ms, ENTRIES[ei], mode, dl, sl)
                res[i, c] = np.nan if r is None else r
        del t, bid, ask
        gc.collect()
        print(f"  {y}-{m:02d} 완료", flush=True)

    years = ev.t.dt.year.values
    keys = ev.key.values
    major = np.isin(keys, MAJOR)

    def picked(sel):
        """sel 안에서 60분 겹침 건너뛰기"""
        keep, last = np.zeros(len(ev), bool), -10 ** 15
        for i in np.flatnonzero(sel):
            if ev_ms[i] - last >= MAXH + PRE:
                keep[i], last = True, ev_ms[i]
        return keep

    rows, ev_rows = [], []
    for c, (ei, mode, dl, sl) in enumerate(combos):
        r = res[:, c]
        traded = ~np.isnan(r)
        wf = np.zeros(len(ev), bool)
        for y in range(2023, 2027):
            past = pd.DataFrame({"k": keys[(years < y) & traded], "r": r[(years < y) & traded]}).groupby("k").r.agg(["size", "mean"])
            good = past[(past["size"] >= 8) & (past["mean"] > 0)].index
            wf |= (years == y) & np.isin(keys, good)
        for name, sel in (("전체 high", np.ones(len(ev), bool)), ("주요 4개", major), ("워크포워드", wf)):
            k = picked(sel) & traded
            v = r[k]
            row = {"진입": f"{ENTRIES[ei][0]} {ENTRIES[ei][1]}$", "청산": mode, "지연초": dl / 1000, "미끄러짐": sl,
                   "고르기": name, "거래": int(k.sum())}
            for y in range(2022, 2027):
                row[str(y)] = round(np.nansum(r[k & (years == y)]), 1)
            row["합계"] = round(v.sum(), 1)
            row["2023~합계"] = round(r[k & (years >= 2023)].sum(), 1)
            row["플러스해"] = sum(row[str(y)] > 0 for y in range(2022 if name != "워크포워드" else 2023, 2027))
            row["이긴%"] = round((v > 0).mean() * 100, 1) if len(v) else np.nan
            row["최악"] = round(v.min(), 1) if len(v) else np.nan
            row["상위5번 빼면"] = round(v.sum() - np.sort(v)[-5:].sum(), 1) if len(v) > 5 else np.nan
            eq = np.r_[0, np.cumsum(v)]
            row["최대낙폭"] = round((np.maximum.accumulate(eq) - eq).max(), 1)
            rows.append(row)
        if (ei, dl, sl) == (2, 300, 0.0):
            ev_rows.append(pd.DataFrame({"청산": mode, "key": keys, "y": years, "r": r}))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"news_deep{'_part' if ONLY else ''}.csv", index=False, encoding="utf-8-sig")
    pd.concat(ev_rows).to_csv(OUT / "news_deep_events.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    base = out[(out.지연초 == 0.3) & (out.미끄러짐 == 0)]
    print(base.sort_values("2023~합계", ascending=False).head(40).to_string(index=False))


if __name__ == "__main__":
    main()
