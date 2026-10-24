"""뉴스 서프라이즈 판단 (실시간 모의 봇 news_surprise_paper.py와 다시 돌려 보기가 같이 씀, 백테스트 news_moves2 / news_surprise와 같은 계산)

z = round((실제 - 예측) / 그 지표의 2022~2024 표준편차, 2), 기여 = clip(z, ±3) x 극성, S = 기여 합
엇갈림 = +0.5 넘는 기여와 -0.5 넘는 기여가 같이 있음
같은 지표 이름이 한 발표에 두 번 있으면 둘 다 더함 (백테스트와 같음), 같은 값(event_id)이 다시 오면 덮어씀 (수정치)
청산: 발표 직전 중간가 = 기준, 발표 30분 직전 틱까지만 사용 (백테스트 news_surprise_exits와 같음)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.news_grid import exit_at

ROOT = Path(__file__).resolve().parent.parent
CURRENCIES = ("USD", "EUR", "GBP")
IMPORTANCE = ("moderate", "high")
S_MIN = 1.5       # 모의 진입 기준 (1.5 이상 전부 기록)
S_BASE = 2.0      # 기본 기준 (실주문 후보). 1.5~2는 '관찰'로 따로 표시
                  # 2026-10-24 GPT 틱 검증: 2025~26 1.5는 +105 / 낙폭 45, 2는 +89 / 낙폭 34, 1.5~2 구간 한 번 평균 +0.27 (0일 수 있음)
MAX_HOLD_MS = 30 * 60_000
EXITS = {"큰이익추적 5/2.5/15분": ("큰이익추적", 5, 2.5, 15), "합산보호 2/1 30분": ("합산보호", 2, 1, 30),
         "30분 보유": ("시간", 30, None)}
# 큰이익추적에 '아니다' 판단을 덧붙인 것 (5-21): 발표 30초 뒤 반대로 1$ 넘게 가 있으면 / 진입 뒤 마감된 1분봉 SAR이 반대면
EXTRA = ("큰이익추적+반응30초", "큰이익추적+SAR1분")
NAMES = list(EXITS) + list(EXTRA)
SAR_BEFORE_MS = 8 * 3600_000     # SAR 계산용으로 발표 전 틱을 이만큼 (짧으면 SAR이 덜 맞음)


def key_of(currency, event):
    return currency + " " + re.sub(r"\s*\(.*\)", "", str(event)).strip()


def params(cal):
    """cal: backtests.gpt_news_export.calendar() -> ({지표: 표준편차}, {지표: 극성 ±1})"""
    c = cal[cal.currency.isin(CURRENCIES) & cal.importance.isin(IMPORTANCE) & ~cal.all_day]
    c = c[(c.time >= "2022-01-03") & (c.time.dt.weekday < 5)]
    c = c.assign(key=[key_of(a, b) for a, b in zip(c.currency, c.event)], err=c.actual - c.forecast)
    sd = c[c.time.dt.year <= 2024].groupby("key").err.std()
    pol = pd.read_csv(ROOT / "results" / "news_surprise_polarity.csv", encoding="utf-8-sig", index_col=0).iloc[:, 0]
    return sd.dropna().to_dict(), pol[pol != 0].to_dict()


def selected_keys(min_trades=3):
    """2022~2024 모의 매매(news_surprise_types)에서 min_trades번 이상, 평균 +였던 대표 지표 (5-20). 기록 표시용"""
    f = ROOT / "results" / "news_surprise_trades.csv"
    if not f.exists():
        return set()
    t = pd.read_csv(f, encoding="utf-8-sig")
    g = t[t.y <= 2024].groupby("key")["큰이익추적 5/2.5/15분 1.5초"].agg(["size", "mean"])
    return set(g.index[(g["size"] >= min_trades) & (g["mean"] > 0)])


def main_key(items):
    """items: [(지표, 중요도)] 일정 순서 -> 대표 지표 (high 중 첫 번째, 없으면 첫 번째, news_moves2와 같음)"""
    for k, imp in items:
        if imp == "high":
            return k
    return items[0][0] if items else ""


def contribution(key, actual, forecast, sd, pol):
    """지표 하나의 기여 (못 쓰면 None)"""
    if actual is None or forecast is None or pd.isna(actual) or pd.isna(forecast):
        return None
    if key not in pol or not sd.get(key):
        return None
    return float(np.clip(round((actual - forecast) / sd[key], 2), -3, 3) * pol[key])


def scored_keys(group, sd, pol):
    """그 시각 일정(캘린더) 중 점수에 들어갈 지표 수 = 기다릴 지표 수. 예측치가 있고 극성이 있는 것"""
    return sum(1 for k, fc in group if k in pol and sd.get(k) and fc is not None and not pd.isna(fc))


def tier(s):
    """|S| 2 이상 = 기본, 1.5~2 = 관찰"""
    return "기본" if abs(s) >= S_BASE else "관찰"


def decide(contribs):
    """contribs: {event_id: 기여} -> (방향 +1/-1/0, S, 이유)"""
    v = np.array(list(contribs.values()), dtype=float)
    if not len(v):
        return 0, 0.0, "점수 낼 지표 없음"
    s = float(v.sum())
    if abs(s) < S_MIN:
        return 0, s, f"|S| {abs(s):.2f} < {S_MIN:g}"
    if (v > 0.5).any() and (v < -0.5).any():
        return 0, s, "지표끼리 엇갈림"
    return (1 if s > 0 else -1), s, ""


def outcome(tt, b, a, ms, side, entry_ms, entry_price=None):
    """발표 ms, entry_ms 틱에 진입 -> {청산 이름: $/oz} (틱은 발표 30분 직전까지만 씀, 미끄러짐 미포함)
    SAR1분은 넘겨준 틱 전체(발표 SAR_BEFORE_MS 전부터 주면 맞음)로 1분봉 SAR을 계산"""
    from backtests.news_surprise_stops import sar_bars, stop_sar_close
    keep = tt < ms + MAX_HOLD_MS
    bars = sar_bars(tt[keep], b[keep], 1)
    keep &= tt >= ms - 60_000
    tt, b, a = tt[keep], b[keep], a[keep]
    mid = (a + b) / 2
    r = int(np.searchsorted(tt, ms))
    ref = float(mid[r - 1]) if r else float(mid[0])
    k = int(np.searchsorted(tt, entry_ms))
    if k >= len(tt) - 2:
        return None
    e0 = entry_price if entry_price is not None else float(a[k] if side > 0 else b[k])
    p = (b[k:] - e0) if side > 0 else (e0 - a[k:])
    f = (mid[k:] - ref) * side
    tm = tt[k:]
    out = {"ref_mid": round(ref, 2), "entry_price": round(e0, 2)}
    for name, ex in EXITS.items():
        out[name] = round(float(p[exit_at(ex, p, f, tm, ms, 0.0)]), 2)
    j0 = exit_at(EXITS["큰이익추적 5/2.5/15분"], p, f, tm, ms, 0.0)
    j = int(np.searchsorted(tm, ms + 30_000))
    out["큰이익추적+반응30초"] = round(float(p[min(j0, j) if j < len(tm) and f[j] < -1 else j0]), 2)
    js = stop_sar_close(bars, tm, tm[0], side)
    out["큰이익추적+SAR1분"] = round(float(p[j0 if js is None else min(j0, js)]), 2)
    return out
