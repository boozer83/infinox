"""HK 시그널 새 기능을 우리 SAR 단타에 붙이면 나아지나 (실제 틱, 2022-01 ~ 2026-10, 연도별)

기준: 지금 봇 신호 (봉 마감 + RSI50 + 1시간 SAR + 시간/ATR/과열 필터), 손절 7.5 / 익절 3 / 반대 콩 청산 (hk_apply와 같음)
붙여 보는 것 (HK 1부 방송 슬라이드·패널):
  슈퍼쉴드 15+1h      : 15분봉·1시간봉 SAR이 신호 방향과 같을 때만 진입
  슈퍼쉴드 15+30+1h   : 패널처럼 30분봉까지 같을 때만
  만루홈런            : 거르지 않고, 15·30분·1시간이 다 같으면 lot 2배
  슈퍼쉴드 비중 1.5배 (진행자: 슈퍼쉴드 조건이면 1.5배로 들어감)
    거르기 + 1.5배    : 15분·1시간 같을 때만 진입, 30분까지 같으면 1.5배 / 15분·1시간 같으면 전부 1.5배
    안 거르고 1.5배   : 다 진입, 15분·1시간 (또는 15·30분·1시간) 같으면 1.5배
  하루 손실 정지 X     : 그날 누적 손익이 -X$/oz에 닿으면 그날 더 진입 안 함 (단타판 구조대: 손실 키우지 않고 멈춤)
  + 주요 뉴스 돌파     : NFP 실업률·CPI·FOMC 금리 결정·미국 GDP 발표 뒤 5분 안에 중간가가 2$ 움직인 방향으로 한쪽만
                         (news_breakout '돌파 진입 2$', 0.3초 늦게 체결, 단타와 같은 lot), 단타와 따로 손익만 더함
결과: $/oz (lot 1 단위), 거래당 lot 배수 반영

사용법: python -m backtests.our_upgrades 2025      (results/our_upgrades_2025.csv)
"""
import gc
import sys

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.hk_apply as ha
import backtests.long_period as lb
import backtests.news_breakout as nb
import backtests.news_hk as nh
from backtests.gold_rest_others import resample
from indicators import psar

MAJOR = "USD 실업률|USD CPI m/m|USD 연방 금리 결정|USD GDP q/q"


def tf_dir(bars, minutes, ms):
    """그 시각까지 완성된 봉의 SAR 방향 (1 / -1)"""
    d = pd.DataFrame({"end": ha.to_ms(bars.time + pd.Timedelta(minutes=minutes)),
                      "d": np.where(psar(bars.high.values, bars.low.values), 1, -1)})
    return pd.merge_asof(pd.DataFrame({"ms": ms}), d, left_on="ms", right_on="end", direction="backward").d.values


def news_trades(year):
    ev = nh.events()
    ev = ev[(ev.importance == "high") & (ev.t.dt.year == year)]
    ev = ev[(ev.currency + " " + ev.event).str.contains(MAJOR, regex=True)]
    ms_all = ha.to_ms(ev.t)
    rows, last = [], 0
    for y, m in [(y, m) for y, m in lb.months() if y == year]:
        s, e = lb.month_range(y, m)
        f = ms_all[(ms_all >= int(s.timestamp() * 1000)) & (ms_all < int(e.timestamp() * 1000))]
        if not len(f):
            continue
        t, bid, ask = lb.load_month(y, m)
        for ms in f:
            if ms - last < nb.MAXH + nb.PRE:
                continue
            last = ms
            r = nb.one(t, bid, ask, int(ms), ("break", 2.0))
            if r is not None:
                rows.append((int(ms), r / 2))           # 돌파 진입은 lot 2배로 계산돼 있어서 1배로
        del t, bid, ask
        gc.collect()
    return pd.DataFrame(rows, columns=["ms", "oz"])


def daily_stop(tr, lim):
    out, day, cum = [], None, 0.0
    for ms, oz, w in tr:
        d = pd.Timestamp(ms, unit="ms").date()
        if d != day:
            day, cum = d, 0.0
        if cum <= -lim:
            continue
        cum += oz * w
        out.append((ms, oz, w))
    return out


def stats(name, tr, year):
    if not tr:
        return {"연도": year, "방법": name, "거래": 0}
    ms = np.array([x[0] for x in tr])
    oz = np.array([x[1] * x[2] for x in tr])
    o = np.argsort(ms, kind="stable")
    eq = np.r_[0, np.cumsum(oz[o])]
    w, l = oz[oz > 0], oz[oz <= 0]
    return {"연도": year, "방법": name, "거래": len(oz), "승률%": round(len(w) / len(oz) * 100, 1),
            "PF": round(w.sum() / -l.sum(), 2) if l.sum() else np.inf, "합계$/oz": round(oz.sum(), 1),
            "최대낙폭$/oz": round((np.maximum.accumulate(eq) - eq).max(), 1)}


def main():
    year = int(sys.argv[1])
    fe.START, fe.END = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
    m5, h1 = ha.load_year(year)
    our = ha.our_signals(m5, h1)
    m15, m30 = resample(m5, 15), resample(m5, 30)
    d15, d30, d60 = tf_dir(m15, 15, our.ms.values), tf_dir(m30, 30, our.ms.values), tf_dir(h1, 60, our.ms.values)
    s = our.side.values
    up = psar(m5.high.values, m5.low.values)
    end5 = m5.time + pd.Timedelta(minutes=5)
    against = {1: ha.to_ms(end5[~up]), -1: ha.to_ms(end5[up])}
    del m5, h1, m15, m30
    gc.collect()
    agree2 = (d15 == s) & (d60 == s)
    agree3 = agree2 & (d30 == s)
    sigs = {"기준": (our, "our"), "슈퍼쉴드 15+1h": (our[agree2], "our"), "슈퍼쉴드 15+30+1h": (our[agree3], "our")}
    res = ha.run_scalp(year, sigs, against)
    a3 = dict(zip(our.ms.values, agree3))
    a2 = dict(zip(our.ms.values, agree2))
    base = [(ms, oz, 1) for ms, oz in res["기준"]]
    variants = {
        "기준": base,
        "슈퍼쉴드 15+1h": [(ms, oz, 1) for ms, oz in res["슈퍼쉴드 15+1h"]],
        "슈퍼쉴드 15+30+1h": [(ms, oz, 1) for ms, oz in res["슈퍼쉴드 15+30+1h"]],
        "만루홈런 (일치 때 2배)": [(ms, oz, 2 if a3.get(ms) else 1) for ms, oz in res["기준"]],
        "슈퍼쉴드 15+1h, 30분도 같으면 1.5배": [(ms, oz, 1.5 if a3.get(ms) else 1) for ms, oz in res["슈퍼쉴드 15+1h"]],
        "슈퍼쉴드 15+1h, 전부 1.5배": [(ms, oz, 1.5) for ms, oz in res["슈퍼쉴드 15+1h"]],
        "안 거르고 15+1h 같으면 1.5배": [(ms, oz, 1.5 if a2.get(ms) else 1) for ms, oz in res["기준"]],
        "안 거르고 15+30+1h 같으면 1.5배": [(ms, oz, 1.5 if a3.get(ms) else 1) for ms, oz in res["기준"]],
        "하루 손실 정지 5$": daily_stop(base, 5.0),
        "하루 손실 정지 10$": daily_stop(base, 10.0),
    }
    nw = news_trades(year)
    news = [(int(ms), oz, 1) for ms, oz in zip(nw.ms, nw.oz)]
    variants["주요 뉴스 돌파만"] = news
    variants["기준 + 주요 뉴스 돌파"] = base + news
    variants["만루 + 하루 정지 10$ + 주요 뉴스"] = daily_stop(variants["만루홈런 (일치 때 2배)"], 10.0) + news
    out = pd.DataFrame([stats(n, v, year) for n, v in variants.items()])
    out.to_csv(ha.OUT / f"our_upgrades_{year}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
