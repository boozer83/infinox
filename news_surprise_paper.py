"""데모 모의 매매: 뉴스 서프라이즈 방향 (주문 안 냄, docs/hk_program.md 5-17 ~ 5-19)

1) MT5 서비스 CalendarLive(mql5/CalendarLive.mq5)가 쓰는 Common\\Files\\calendar_live.csv를 0.1초마다 읽음
2) 판단은 common/surprise.py (백테스트와 같은 계산): 지표마다 기여 = clip(round(z, 2), ±3) x 극성, S = 합
3) 같은 시각 발표는 묶어서 한 번만 판단: 그 시각 일정(MT5 캘린더 + 실제로 들어온 것) 중 점수 낼 지표가 다 들어오거나,
   첫 지표를 읽고 GROUP_WAIT초가 지나면 판단 (고용지표처럼 여러 개가 같이 나올 때 첫 지표만으로 들어가지 않게)
4) |S| >= S_MIN, 엇갈림 없음, 발표 MAX_LAG초 안이면 모의 진입: 판단 ENTRY_DELAY_MS 뒤의 새 틱 가격 (매수 Ask / 매도 Bid)
   (CalendarLive가 실제치를 받은 시각보다 앞선 틱은 안 씀)
5) 발표 30분 30초 뒤 MT5에서 틱(SAR용으로 발표 8시간 전부터)을 받아 청산 5가지 계산 (발표 30분 직전 틱까지, 백테스트와 같음):
   큰이익추적 5/2.5/15분(기본), 합산보호 2/1 30분, 30분 보유, 큰이익추적+반응30초, 큰이익추적+SAR1분 (5-21). 비교용으로 '발표 1초 뒤(+0.3초) 진입했다면'도 같이
기록: logs/surprise_log/signals.csv (발표마다 판단 때 / 1분 뒤 S, 지표별 기여, 지연, 진입 여부·이유)
      selected = 대표 지표가 2022~2024 모의 매매에서 이익이던 지표인지 (5-20, 진입은 똑같이 하고 표시만)
      trades.csv (모의 매매 결과 $/oz, 미끄러짐 미포함)

사용법: python news_surprise_paper.py          (데모 계좌, Ctrl+C로 종료)
        python news_surprise_paper.py --check  (극성·표준편차 표와 CalendarLive 상태만 보고 끝)
        python news_surprise_paper.py --replay 2026-10-23 [--lag 1]  (그날을 캘린더 실제치 + 저장된 틱으로 다시 돌려 봄,
                                                모든 지표가 발표 lag초 뒤 한꺼번에 왔다고 가정, MT5 연결 안 함)
"""
import argparse
import csv
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from backtests.gpt_news_export import calendar
from common import surprise as sp
from common.news_calendar import mt5_events
from common.trader import Trader, append_csv
from config import SYMBOL

MAX_LAG = 10            # 발표 뒤 이 초 안에 판단해야 진입 (백테스트: 10초까지 플러스)
GROUP_WAIT = 0.5        # 같은 시각 지표를 기다리는 시간 (첫 지표를 읽은 뒤, 초)
ENTRY_DELAY_MS = 300    # 판단 뒤 이만큼 지난 틱으로 진입 (실제 주문 지연 가정, 백테스트 +0.3초와 같음)
POLL_SEC = 0.1
STRATEGY = "서프라이즈 모의"
ROOT = Path(__file__).resolve().parent
COMMON = Path(os.environ["APPDATA"]) / "MetaQuotes" / "Terminal" / "Common" / "Files"
LIVE_CSV = COMMON / "calendar_live.csv"
STATUS_TXT = COMMON / "calendar_live_status.txt"
LOG_DIR = ROOT / "logs" / "surprise_log"
SIGNALS_CSV = LOG_DIR / "signals.csv"
TRADES_CSV = LOG_DIR / "trades.csv"
LIVE_COLS = ["recv_server", "event_time", "delay_ms", "tick_ms", "importance", "currency", "event", "actual",
             "forecast", "previous", "revision", "impact", "event_id", "value_id"]
SIG_FIELDS = ["event_time", "main_key", "selected", "expected", "got_at_decide", "S_decide", "S_1min", "detail", "first_delay_ms", "decide_delay_ms",
              "entered", "reason"]
TRADE_FIELDS = (["event_time", "main_key", "selected", "S", "side", "detail", "first_delay_ms", "decide_delay_ms", "entry_time", "entry_delay_ms",
                 "entry_price", "entry_spread", "ref_mid"] + [f"{k} $/oz" for k in sp.NAMES]
                + [f"1초 진입 {k} $/oz" for k in sp.NAMES])
bot = None


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def to_ms(t):
    return int(t.replace(tzinfo=timezone.utc).timestamp() * 1000)


def ms_str(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%H:%M:%S.%f")[:-3]


class Feed:
    """calendar_live.csv에 새로 붙은 줄만 읽음"""

    def __init__(self):
        self.pos = LIVE_CSV.stat().st_size if LIVE_CSV.exists() else 0     # 켜기 전 기록은 건너뜀

    def rows(self):
        if not LIVE_CSV.exists() or LIVE_CSV.stat().st_size <= self.pos:
            return []
        with open(LIVE_CSV, "rb") as f:
            f.seek(self.pos)
            data = f.read()
        cut = data.rfind(b"\n") + 1               # 덜 써진 마지막 줄은 다음에
        self.pos += cut
        text = data[:cut].decode("cp949", "replace")
        return [dict(zip(LIVE_COLS, r)) for r in csv.reader(text.splitlines()) if r and r[0] != "recv_server"]


def schedule_groups():
    """MT5 캘린더 일정: {발표 시각: [(지표, 예측치, 중요도)]} (기다릴 지표 수, 대표 지표)"""
    c = mt5_events()
    if not len(c):
        return {}
    c = c[c.currency.isin(sp.CURRENCIES) & c.importance.isin(sp.IMPORTANCE)]
    out = {}
    for t, cur, ev, fc, imp in zip(c.time, c.currency, c.event, c.forecast, c.importance):
        out.setdefault(t.to_pydatetime(), []).append((sp.key_of(cur, ev), fc, imp))
    return out


def ticks(t0, t1):
    a = mt5.copy_ticks_range(SYMBOL, t0.replace(tzinfo=timezone.utc), t1.replace(tzinfo=timezone.utc), mt5.COPY_TICKS_INFO)
    if a is None or not len(a):
        return None
    return a["time_msc"].astype(np.int64), a["bid"], a["ask"]


def evaluate(tr, tk=None):
    """발표 30분 30초 뒤: 모의 진입가 그대로 청산 3가지 + 1초(+0.3초) 진입 비교"""
    rel = tr["release"]
    tk = tk or ticks(rel - timedelta(milliseconds=sp.SAR_BEFORE_MS), rel + timedelta(minutes=30))
    if tk is None:
        return None
    ms = to_ms(rel)
    res = sp.outcome(*tk, ms, tr["side"], tr["entry_ms"], tr["entry_price"])
    if res is None:
        return None
    out = {"ref_mid": res["ref_mid"], **{f"{k} $/oz": res[k] for k in sp.NAMES}}
    one = sp.outcome(*tk, ms, tr["side"], ms + 1300)
    if one:
        out.update({f"1초 진입 {k} $/oz": one[k] for k in sp.NAMES})
    return out


def replay(day, lag):
    """그날 발표를 캘린더 실제치로 다시 판단 (모든 지표가 발표 lag초 뒤 한꺼번에 왔다고 가정) -> 모의 매매 표"""
    import backtests.long_period as lb
    cal = calendar()
    sd, pol = sp.params(cal)
    selected = sp.selected_keys()
    c = cal[cal.currency.isin(sp.CURRENCIES) & cal.importance.isin(sp.IMPORTANCE) & ~cal.all_day
            & (cal.time.dt.date == pd.Timestamp(day).date())]
    t, bid, ask = lb.load_month(pd.Timestamp(day).year, pd.Timestamp(day).month)
    rows = []
    for rel, g in c.groupby("time"):
        rel = rel.to_pydatetime()
        contribs, detail = {}, []
        for cur, ev, act, fc, eid in zip(g.currency, g.event, g.actual, g.forecast, g.event_id):
            k = sp.key_of(cur, ev)
            v = sp.contribution(k, act, fc, sd, pol)
            if v is not None:
                contribs[eid] = v
                detail.append(f"{k} {act:g}/{fc:g}:{v:+.2f}")
        side, s, why = sp.decide(contribs)
        mk = sp.main_key(list(zip([sp.key_of(a, b) for a, b in zip(g.currency, g.event)], g.importance)))
        row = {"발표": f"{rel:%H:%M}", "대표 지표": mk, "선택": mk in selected, "S": round(s, 2), "지표": "; ".join(detail)[:70]}
        if not side:
            rows.append({**row, "진입": why})
            continue
        ms = to_ms(rel)
        k0, k1 = np.searchsorted(t, [ms - sp.SAR_BEFORE_MS, ms + sp.MAX_HOLD_MS])
        tk = (t[k0:k1], bid[k0:k1], ask[k0:k1])
        res = sp.outcome(*tk, ms, side, ms + int(lag * 1000) + ENTRY_DELAY_MS)
        rows.append({**row, "진입": "매수" if side > 0 else "매도", **(res or {})})
    return pd.DataFrame(rows)


def status_age():
    if not STATUS_TXT.exists():
        return None
    return time.time() - STATUS_TXT.stat().st_mtime


def main():
    global bot
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--replay", default="")
    ap.add_argument("--lag", type=float, default=1.0)
    args = ap.parse_args()
    if args.replay:
        pd.set_option("display.width", 250)
        pd.set_option("display.unicode.east_asian_width", True)
        print(replay(args.replay, args.lag).to_string(index=False))
        return
    sd, pol = sp.params(calendar())
    selected = sp.selected_keys()
    if args.check:
        age = status_age()
        print(f"표준편차 {len(sd)}개 지표, 극성 {len(pol)}개 지표 (+1 금 상승 {sum(v > 0 for v in pol.values())}, "
              f"-1 하락 {sum(v < 0 for v in pol.values())}), 선택 지표 {len(selected)}개")
        print("CalendarLive 상태:", STATUS_TXT.read_text(encoding="cp949").strip() if age is not None else "없음",
              f"({age:.0f}초 전)" if age is not None else "")
        return
    bot = Trader(STRATEGY, SYMBOL, None, 0, LOG_DIR, "surprise")      # 주문은 안 냄
    acc = bot.start()
    feed = Feed()
    sched, sched_at = schedule_groups(), time.time()
    groups, entries, pending, warned = {}, [], [], False
    bot.send(f"▶️ 시작 ({acc.login}, 주문 없음)\n발표 뒤 {MAX_LAG}초 안에 받은 실제치로 |S| >= {sp.S_MIN:g}이면 서프라이즈 방향 모의 진입 "
             f"(같은 시각 지표는 묶어서 판단, 최대 {GROUP_WAIT:g}초 기다림), 발표 30분 30초 뒤 청산 {len(sp.NAMES)}가지 결과 ({', '.join(sp.NAMES)})\n"
             f"지표 {len(pol)}개 극성 사용")
    with bot.running("⏸ 종료"):
        last_alive = 0.0
        while True:
            if time.time() - last_alive > 5:
                bot.keep_alive()
                last_alive = time.time()
                age = status_age()
                if (age is None or age > 300) and not warned:
                    bot.send("⚠️ CalendarLive 서비스 기록이 5분 넘게 안 바뀝니다. MT5 내비게이터 > 서비스에서 켜져 있는지 확인하세요")
                    warned = True
                elif age is not None and age < 120:
                    warned = False
                if time.time() - sched_at > 3600:
                    sched, sched_at = schedule_groups(), time.time()
            bot.clock.update()
            now = bot.clock.now()
            now_ms = to_ms(now)
            # 1) 새로 들어온 실제치를 발표 시각별로 모음
            for row in feed.rows():
                if row["currency"] not in sp.CURRENCIES or row["importance"] not in sp.IMPORTANCE:
                    continue
                rel = datetime.strptime(row["event_time"], "%Y.%m.%d %H:%M:%S")
                if (now - rel).total_seconds() > 120:          # 늦게 온 수정치
                    continue
                g = groups.get(rel)
                if g is None:
                    items = sched.get(rel, [])
                    g = groups[rel] = {"seen": time.monotonic(), "first_delay": int(row["delay_ms"]), "contribs": {}, "detail": {},
                                       "expected": sp.scored_keys([(k, fc) for k, fc, _ in items], sd, pol), "decided": False,
                                       "main": sp.main_key([(k, imp) for k, _, imp in items]), "arrived": []}
                k = sp.key_of(row["currency"], row["event"])
                g["arrived"].append((k, row["importance"]))
                if not g["main"]:                               # 일정에 없는 발표면 들어온 것으로 대표 지표
                    g["main"] = sp.main_key(g["arrived"])
                act, fc = num(row["actual"]), num(row["forecast"])
                v = sp.contribution(k, act, fc, sd, pol)
                if v is not None:
                    g["contribs"][row["event_id"]] = v
                    g["detail"][row["event_id"]] = f"{k} {act:g}/{fc:g}:{v:+.2f}"
            # 2) 묶음이 다 왔거나 GROUP_WAIT가 지나면 한 번만 판단
            for rel, g in groups.items():
                if g["decided"]:
                    continue
                full = g["expected"] and len(g["contribs"]) >= g["expected"]
                if not full and time.monotonic() - g["seen"] < GROUP_WAIT:
                    continue
                side, s, why = sp.decide(g["contribs"])
                lag = (now_ms - to_ms(rel)) / 1000
                if side and lag > MAX_LAG:
                    side, why = 0, f"늦게 판단 ({lag:.1f}초)"
                g.update(decided=True, S_decide=s, got=len(g["contribs"]), decide_delay=int(lag * 1000), reason=why,
                         entered=False, detail_decide="; ".join(g["detail"].values()))
                if side:
                    entries.append({"release": rel, "side": side, "S": round(s, 2), "g": g, "after_ms": now_ms + ENTRY_DELAY_MS,
                                    "min_tick_ms": to_ms(rel) + g["first_delay"]})
            # 3) 판단 0.3초 뒤의 새 틱으로 모의 진입 (실제치 받은 시각보다 앞선 틱은 안 씀)
            for e in list(entries):
                t = mt5.symbol_info_tick(SYMBOL)
                g = e["g"]
                if t is not None and t.time_msc >= max(e["after_ms"], e["min_tick_ms"]):
                    entries.remove(e)
                    price = float(t.ask if e["side"] > 0 else t.bid)
                    g["entered"] = True
                    tr = {**{k: e[k] for k in ("release", "side", "S")}, "main_key": g["main"], "selected": g["main"] in selected,
                          "entry_ms": int(t.time_msc), "entry_price": price,
                          "entry_spread": round(t.ask - t.bid, 2), "detail": g["detail_decide"],
                          "first_delay": g["first_delay"], "decide_delay": g["decide_delay"]}
                    pending.append(tr)
                    msg = (f"📝 모의 {'매수' if e['side'] > 0 else '매도'} {e['release']:%H:%M} S {e['S']:+.2f} "
                           f"({g['main']}, {'선택 지표' if g['main'] in selected else '선택 밖 지표'})\n{g['detail_decide']}\n"
                           f"실제치 {g['first_delay'] / 1000:.1f}초 뒤 받음, 진입 {(t.time_msc - to_ms(e['release'])) / 1000:.1f}초 뒤 "
                           f"{price} (스프레드 {t.ask - t.bid:.2f})")
                    bot.log.info(msg.replace("\n", " / "))
                    bot.send(msg)
                elif (now_ms - to_ms(e["release"])) / 1000 > MAX_LAG + 5:
                    entries.remove(e)
                    g["reason"] = "새 틱 없음"
            # 4) 발표 1분 뒤 신호 기록
            for rel in [r for r, g in groups.items() if g["decided"] and (now - r).total_seconds() > 60
                        and not any(e["g"] is g for e in entries)]:
                g = groups.pop(rel)
                s1 = sp.decide(g["contribs"])[1]
                append_csv(SIGNALS_CSV, SIG_FIELDS, {
                    "event_time": rel, "main_key": g["main"], "selected": g["main"] in selected, "expected": g["expected"], "got_at_decide": g["got"], "S_decide": round(g["S_decide"], 2),
                    "S_1min": round(s1, 2), "detail": "; ".join(g["detail"].values()), "first_delay_ms": g["first_delay"],
                    "decide_delay_ms": g["decide_delay"], "entered": g["entered"], "reason": g["reason"]})
            # 5) 발표 30분 30초 뒤 결과
            for tr in [x for x in pending if now > x["release"] + timedelta(minutes=30, seconds=30)]:
                pending.remove(tr)
                res = evaluate(tr)
                if res is None:
                    bot.send(f"❗ {tr['release']:%H:%M} 모의 매매 결과 계산 실패 (틱 없음)")
                    continue
                append_csv(TRADES_CSV, TRADE_FIELDS, {
                    "event_time": tr["release"], "main_key": tr["main_key"], "selected": tr["selected"], "S": tr["S"],
                    "side": tr["side"], "detail": tr["detail"],
                    "first_delay_ms": tr["first_delay"], "decide_delay_ms": tr["decide_delay"], "entry_time": ms_str(tr["entry_ms"]),
                    "entry_delay_ms": tr["entry_ms"] - to_ms(tr["release"]), "entry_price": tr["entry_price"],
                    "entry_spread": tr["entry_spread"], **res})
                lines = "\n".join(f"  {k}: {res[f'{k} $/oz']:+.2f} (1초 진입이면 {res.get(f'1초 진입 {k} $/oz', float('nan')):+.2f})"
                                  for k in sp.NAMES)
                bot.send(f"📊 {tr['release']:%H:%M} 모의 {'매수' if tr['side'] > 0 else '매도'} 결과 ({tr['main_key']}, "
                         f"{'선택 지표' if tr['selected'] else '선택 밖 지표'}, $/oz, 미끄러짐 미포함)\n{lines}")
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
