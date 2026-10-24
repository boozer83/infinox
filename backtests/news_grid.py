"""뉴스 양방향: '한쪽 살리는 조건' x '살린 쪽 청산 조건' 전부 조합 (실제 틱, USD·EUR·GBP high + moderate, 2022-01 ~ 2026-10)

공통: 발표 30초 전 매수(Ask)·매도(Bid) 각 1 lot (돌파 진입만 예외). 판단 뒤 진 쪽은 0.3초 지난 첫 틱에 청산
      판단이 발표 5분 안에 안 나면 양쪽 정리. 남은 쪽은 발표 30분이 최대
한쪽 살리는 조건 (기준 = 발표 직전 중간가, 발표 뒤 틱만 봄. 진입가 기준만 진입 때부터)
  중간가 X$      : 중간가가 기준에서 X$ 움직인 방향을 살림
  비율 X%        : 기준가의 X% 움직인 방향
  진입가 X$      : 한쪽이 진입가 대비 X$ 역행하면 그쪽 청산 (HK 처음 방식)
  유지 X$ N초    : X$ 넘은 상태가 N초 계속되면
  T초 방향 m$    : 발표 T초 뒤 중간가가 m$ 넘게 움직였으면 그 방향, 아니면 그때 양쪽 정리
  돌파 X$        : 미리 안 들어가고 중간가 X$ 움직인 방향으로 한쪽만 진입 (0.3초 늦게 체결)
살린 쪽 청산 (p = 남은 쪽 손익 $/oz, 발표 30분이 최대)
  시간 T분 [되돌림 R$] : 발표 T분에 청산, R$이 있으면 중간가가 기준 반대쪽으로 R$ 넘어가면 즉시 (판단이 틀린 것)
  익절 TP / 손절 SL    : p >= TP 또는 p <= -SL
  추적 Z$ T분          : 최고 p에서 Z$ 밀리면, 늦어도 T분
  합산보호 A/B T분     : 진 쪽 손익 + p (뉴스 한 번 합계)가 A$ 넘으면 그 뒤 B$로 내려올 때 청산, 아니면 T분 (비상: 되돌림 4$)
  큰이익추적 A Z T분   : p가 A$ 넘으면 최고점에서 Z$ 밀릴 때, 아니면 T분 (HK 캡처 = 8 / 2.5 / 10분)
  슬라이드             : 10분에 이익이면 청산, 손실이면 본전 될 때 (비상: 되돌림 6$)
비용: 틱의 실제 Bid/Ask + 체결마다 추가 미끄러짐 0 / 0.1$ (데모 실측: 뉴스 한 번 약 0.45$ = 체결 4번 x 0.1)
같은 묶음에서 앞 매매(최대 30분)와 겹치는 발표는 건너뜀
결과: $/oz 합계 (1 lot = 100 oz면 x100 달러). 고르기 = 2022~2024 성적, 확인 = 2025·2026 (고를 때 안 본 기간)

사용법: python -m backtests.news_grid      (results/news_grid.csv, news_grid_events.npz)
"""
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.long_period as lb
import backtests.news_hk as nh

PRE, DECIDE, MAXH, DELAY = 30_000, 300_000, 1_800_000, 300
MAJOR = ("USD 실업률", "USD CPI m/m", "USD 연방 금리 결정", "USD GDP q/q")
SLIPS = (0.0, 0.1)
OUT = Path(__file__).resolve().parents[1] / "results"

SURVIVE = ([("중간가", x) for x in (1, 1.5, 2, 3, 4)] + [("비율", x) for x in (0.03, 0.05, 0.08)] +
           [("진입가", x) for x in (1.5, 2, 3)] + [("유지", x, n) for x, n in ((1, 3), (2, 3), (2, 10))] +
           [("방향", t, m) for t, m in ((10, 1), (30, 1), (60, 2))] + [("돌파", x) for x in (1, 2, 3)])
EXITS = ([("시간", t, r) for t in (5, 10, 15, 20, 30) for r in (None, 2, 4)] +
         [("익절손절", tp, sl) for tp in (3, 5, 8, 12, 20) for sl in (2, 4, 6)] +
         [("추적", z, t) for z in (2, 3, 5, 8) for t in (15, 30)] +
         [("합산보호", a, b, t) for a in (2, 4, 6) for b in (0, 1) for t in (15, 30)] +
         [("큰이익추적", a, z, t) for a in (5, 8, 12) for z in (2.5, 5) for t in (10, 15)] +
         [("슬라이드",)])


def name_s(s):
    if s[0] == "유지":
        return f"유지 {s[1]}$ {s[2]}초"
    if s[0] == "방향":
        return f"{s[1]}초 방향 {s[2]}$"
    return f"{s[0]} {s[1]}" + ("%" if s[0] == "비율" else "$")


def name_e(e):
    if e[0] == "시간":
        return f"시간 {e[1]}분" + (f" 되돌림 {e[2]}$" if e[2] else "")
    return {"익절손절": lambda: f"익절 {e[1]} 손절 {e[2]}", "추적": lambda: f"추적 {e[1]}$ {e[2]}분",
            "합산보호": lambda: f"합산보호 {e[1]}/{e[2]} {e[3]}분", "큰이익추적": lambda: f"큰이익추적 {e[1]} {e[2]} {e[3]}분",
            "슬라이드": lambda: "슬라이드"}[e[0]]()


def first(c, start=0):
    k = np.flatnonzero(c[start:])
    return start + int(k[0]) if k.size else None


def decide(tt, b, a, mid, ms, r, ref, s):
    """-> (판단 틱, 살릴 방향 +1 매수 / -1 매도) 또는 (정리 틱, 0)"""
    q = min(int(np.searchsorted(tt, ms + DECIDE)), len(tt) - 1)
    dev = mid - ref
    kind = s[0]
    if kind in ("중간가", "비율", "돌파"):
        x = s[1] if kind != "비율" else ref * s[1] / 100
        k = first(np.abs(dev[r:q + 1]) >= x)
        return (q, 0) if k is None else (r + k, 1 if dev[r + k] > 0 else -1)
    if kind == "진입가":
        x = s[1]
        lb_ = first(b[:q + 1] <= a[0] - x)          # 매수가 진 쪽
        ls_ = first(a[:q + 1] >= b[0] + x)          # 매도가 진 쪽
        if lb_ is None and ls_ is None:
            return q, 0
        if ls_ is None or (lb_ is not None and lb_ <= ls_):
            return lb_, -1
        return ls_, 1
    if kind == "유지":
        x, n = s[1], s[2] * 1000
        sg = np.where(dev >= x, 1, np.where(dev <= -x, -1, 0))[r:q + 1]
        if not sg.any():
            return q, 0
        chg = np.flatnonzero(np.r_[True, sg[1:] != sg[:-1]])
        ends = np.r_[chg[1:], len(sg)]
        for st, en in zip(chg, ends):
            if sg[st] == 0:
                continue
            k = int(np.searchsorted(tt, tt[r + st] + n))
            if k < r + en:                          # N초 뒤에도 같은 쪽에 있음
                return k, int(sg[st])
        return q, 0
    t, m = s[1], s[2]                               # 방향
    k = min(int(np.searchsorted(tt, ms + t * 1000)), len(tt) - 1)
    return (k, 1 if dev[k] > 0 else -1) if abs(dev[k]) >= m else (k, 0)


def exit_at(e, p, f, tm, ms, lose):
    """살린 쪽 청산 틱 (p, f, tm은 청산 가능한 첫 틱부터)"""
    end = len(tm) - 1

    def at(minutes):
        return min(int(np.searchsorted(tm, ms + minutes * 60_000)), end)

    def stop_f(rv, j):
        k = first(f[:j + 1] <= -rv) if rv else None
        return j if k is None else k

    kind = e[0]
    if kind == "시간":
        return stop_f(e[2], at(e[1]))
    if kind == "익절손절":
        k = first((p >= e[1]) | (p <= -e[2]))
        return end if k is None else k
    if kind == "추적":
        j = at(e[2])
        k = first(p[:j + 1] <= np.maximum.accumulate(p[:j + 1]) - e[1])
        return j if k is None else k
    if kind == "합산보호":
        j = stop_f(4, at(e[3]))
        tot = lose + p[:j + 1]
        on = first(tot >= e[1])
        if on is None:
            return j
        k = first(tot <= e[2], on)
        return j if k is None else k
    if kind == "큰이익추적":
        j10 = at(e[3])
        on = first(p >= e[1])
        if on is None or on > j10:
            return j10
        k = first(p <= np.maximum.accumulate(np.where(np.arange(len(p)) >= on, p, -np.inf)) - e[2], on)
        return end if k is None else k
    j10 = at(10)                                    # 슬라이드
    be = first(p >= 0, j10)
    j = j10 if p[j10] > 0 else (end if be is None else be)
    return stop_f(6, j)


def one_event(tt, b, a, ms):
    """발표 하나 -> (조합 수,) 손익 $/oz (미끄러짐 0), (살리는 조건 수,) 체결 횟수"""
    out = np.full(len(SURVIVE) * len(EXITS), np.nan)
    fills = np.zeros(len(SURVIVE))
    mid = (a + b) / 2
    r = int(np.searchsorted(tt, ms))
    if r == 0 or r >= len(tt) - 5:
        return out, fills
    ref = mid[r - 1]
    for si, s in enumerate(SURVIVE):
        k, d = decide(tt, b, a, mid, ms, r, ref, s)
        kc = min(int(np.searchsorted(tt, tt[k] + DELAY)), len(tt) - 1)
        if s[0] == "돌파":
            if d == 0:
                continue                            # 돌파 없음 = 매매 없음
            lose, e0 = 0.0, (a[kc] if d > 0 else b[kc])
            fills[si] = 2
        else:
            fills[si] = 4
            if d == 0:                              # 양쪽 정리
                out[si * len(EXITS):(si + 1) * len(EXITS)] = (b[kc] - a[0]) + (b[0] - a[kc])
                continue
            lose = (b[kc] - a[0]) if d < 0 else (b[0] - a[kc])
            e0 = a[0] if d > 0 else b[0]
        p = (b[kc:] - e0) if d > 0 else (e0 - a[kc:])
        f = (mid[kc:] - ref) * d
        tm = tt[kc:]
        for ei, e in enumerate(EXITS):
            j = exit_at(e, p, f, tm, ms, lose)
            out[si * len(EXITS) + ei] = lose + p[j]
    return out, fills


def month_job(args):
    y, m, items = args
    try:
        t, bid, ask = lb.load_month(y, m)
    except FileNotFoundError:
        return []
    res = []
    for i, ms in items:
        k0 = int(np.searchsorted(t, ms - PRE))
        if k0 >= len(t) or t[k0] - (ms - PRE) > 20_000:
            continue
        k1 = int(np.searchsorted(t, ms + MAXH))
        if k1 - k0 < 10:
            continue
        o, f = one_event(t[k0:k1], bid[k0:k1], ask[k0:k1], ms)
        res.append((i, o, f))
    return res


def run():
    ev = nh.events()
    ev["key"] = ev.currency + " " + ev.event.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    ev_ms = ev.t.values.astype("datetime64[ms]").astype(np.int64)
    jobs = []
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        idx = np.flatnonzero((ev_ms >= int(s.timestamp() * 1000)) & (ev_ms < int(e.timestamp() * 1000)))
        if len(idx):
            jobs.append((y, m, [(int(i), int(ev_ms[i])) for i in idx]))
    raw = np.full((len(ev), len(SURVIVE) * len(EXITS)), np.nan)
    fills = np.zeros((len(ev), len(SURVIVE)))
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 4))) as ex:
        for (y, m, _), res in zip(jobs, ex.map(month_job, jobs)):
            for i, o, f in res:
                raw[i], fills[i] = o, f
            print(f"  {y}-{m:02d} 완료", flush=True)
    np.savez_compressed(OUT / "news_grid_events.npz", raw=raw, fills=fills, ms=ev_ms)
    ev.to_csv(OUT / "news_grid_events.csv", index=False, encoding="utf-8-sig")
    return ev, ev_ms, raw, fills


def picked(sel, ev_ms):
    keep, last = np.zeros(len(sel), bool), -10 ** 15
    for i in np.flatnonzero(sel):
        if ev_ms[i] - last >= MAXH + PRE:
            keep[i], last = True, ev_ms[i]
    return keep


def summarize(ev, ev_ms, raw, fills):
    years = ev.t.dt.year.values
    groups = {"전체 (high+moderate)": np.ones(len(ev), bool), "high": (ev.importance == "high").values,
              "moderate": (ev.importance == "moderate").values, "주요 4개": np.isin(ev.key.values, MAJOR)}
    combos = list(itertools.product(range(len(SURVIVE)), range(len(EXITS))))
    rows = []
    for g, sel in groups.items():
        base = picked(sel, ev_ms)
        for slip in SLIPS:
            for c, (si, ei) in enumerate(combos):
                v = raw[:, c] - slip * fills[:, si]
                k = base & ~np.isnan(raw[:, c])
                vv, yy = v[k], years[k]
                row = {"묶음": g, "미끄러짐": slip, "살리는 조건": name_s(SURVIVE[si]), "청산": name_e(EXITS[ei]),
                       "매매 수": int(k.sum()), "승률%": round((vv > 0).mean() * 100, 1) if len(vv) else np.nan}
                for y in range(2022, 2027):
                    row[str(y)] = round(vv[yy == y].sum(), 1)
                row["2022~24"] = round(vv[yy <= 2024].sum(), 1)
                row["2025~26"] = round(vv[yy >= 2025].sum(), 1)
                row["5년"] = round(vv.sum(), 1)
                row["최악 1번"] = round(vv.min(), 1) if len(vv) else np.nan
                row["상위5 빼면"] = round(np.sort(vv)[:-5].sum(), 1) if len(vv) > 5 else np.nan
                row["플러스 해"] = sum(row[str(y)] > 0 for y in range(2022, 2027))
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "news_grid.csv", index=False, encoding="utf-8-sig")
    return out


def report(out):
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.unicode.east_asian_width", True)
    cols = ["살리는 조건", "청산", "매매 수", "승률%", "2022", "2023", "2024", "2025", "2026", "2022~24", "2025~26", "5년", "최악 1번", "상위5 빼면"]
    for g in out["묶음"].unique():
        d = out[(out["묶음"] == g) & (out["미끄러짐"] == 0.1)]
        print(f"\n=== {g}, 미끄러짐 0.1$/체결 — 2022~24로 고른 상위 10 (2025~26은 고를 때 안 본 기간)")
        print(d.sort_values("2022~24", ascending=False).head(10)[cols].to_string(index=False))
        print(f"--- {g}, 5년 모두 플러스인 조합 {int((d['플러스 해'] == 5).sum())}개 / {len(d)}개")
        print(d[d["플러스 해"] == 5].sort_values("상위5 빼면", ascending=False).head(10)[cols].to_string(index=False))


if __name__ == "__main__":
    report(summarize(*run()))
