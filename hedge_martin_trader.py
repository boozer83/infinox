"""헤지 마틴 (데모 관찰용, --live면 실계좌. HK MARTIN 구조 재현, backtests/hedge_martin_ticks.py의 START_SIG + CONTINUE 모드와 같은 규칙)

- 사이클 시작: SAR 단타(demo_trader)와 같은 신호 봉(5분봉 SAR 전환 + RSI50 + 1시간봉 SAR + 시간/ATR/과열 필터)이 완성되면
  매수·매도를 같은 lot(L0)으로 동시 진입. 중요 지표(USD/EUR/GBP moderate 이상, MT5 캘린더 + ForexFactory) 앞뒤 NEWS_MIN분이면 시작 안 함
  L0 = 사이클 시작 잔고 1,000달러당 LOT_PER_1000 lot (0.01 단위 내림, 최소 0.01)
- 모든 포지션은 진입가 ± TP_USD에 서버 익절, 손절 없음 (진 쪽은 들고 감)
- 사이클 중 5분봉 SAR이 전환될 때마다 그 방향에 열린 포지션이 없으면 다음 단계 lot(L0 x 2^단계)으로 1개 진입 (최대 MAX_LEVEL)
  포지션이 다 익절로 닫혀도 사이클 목표 전이면 끝내지 않고 다음 전환에 다음 단계로 이어감
- 사이클 합계(이번 사이클 확정 손익 + 평가손익)가 시작 잔고의 +TARGET%면 전부 청산, -LIMIT%면 전부 청산
- 주말에도 보유 (백테스트와 같음). 기본 데모, --live면 실계좌 (2026-10-25 사용자 요청, 기록 logs/hedge_live_log/)
백테스트 (해마다 1만 달러, 2022~2026, 익절 5 / 목표 1% / 한도 30% / 4단계): 2022 -81%, 2023 -32%, 2024 -34%, 2025 +55%, 2026 +33%,
  최대 낙폭 85%. 캡처(9~10월 HK 계좌)와 흐름이 맞는지 보려는 관찰용
상태: logs/hedge_log/state.json (사이클 시작 시각·잔고·단계)
"""
import argparse
import logging
import time
from datetime import timedelta
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import demo_trader as dt
from common.news_calendar import near_news
from common.trader import Trader, load_json, save_json, side_of, side_name
from config import SYMBOL
from indicators import atr, psar

TP_USD = 5.0
TARGET = 0.01
LIMIT = 0.30
MAX_LEVEL = 4
LOT_PER_1000 = 0.01
NEWS_MIN = 30
NIGHT_OFF = False   # True면 SAR 단타처럼 서버 15:00~21:00(한국 21:00~03:00) 시작 안 함. 2026-10-22 끔 (밤에도 시작)
MAGIC = 20261020
POLL_SEC = 1.0
RETRY_SEC = 5       # 사이클 청산이 실패하면 이 간격으로 다시
STRATEGY = "헤지 마틴"
LOG_DIR = Path(__file__).resolve().parent / "logs" / "hedge_log"
STATE_JSON = LOG_DIR / "state.json"
SIGNAL_JSON = LOG_DIR / "signal_state.json"
log = logging.getLogger("hedge")
bot = None         # Trader, main()에서 설정


def use_live_paths():
    """실계좌 실행은 기록을 logs/hedge_live_log/에 따로 (뮤텍스 이름도 이 폴더 이름)"""
    global LOG_DIR, STATE_JSON, SIGNAL_JSON
    LOG_DIR = LOG_DIR.parent / "hedge_live_log"
    STATE_JSON, SIGNAL_JSON = LOG_DIR / "state.json", LOG_DIR / "signal_state.json"


def save_state(st):
    save_json(STATE_JSON, st, indent=1)


def realized(since):
    """since: 서버 시각 문자열. 그 뒤로 이 봇이 청산한 손익 합계 (이력 조회 실패면 None)"""
    t0 = int(pd.Timestamp(since).tz_localize("UTC").timestamp())
    deals = mt5.history_deals_get(t0, t0 + 60 * 86400)
    if deals is None:
        return None
    return sum(d.profit + d.commission + d.swap for d in deals if d.magic == MAGIC and d.entry == mt5.DEAL_ENTRY_OUT)


def lot0(bal):
    return max(0.01, np.floor(bal / 1000 * LOT_PER_1000 * 100) / 100)


def open_pos(side, lot, level):
    f = bot.open(side, round(lot, 2), f"HM L{level}", tp_dist=TP_USD)
    if f is None:
        return False
    log.info(f"진입 {'BUY' if side == 1 else 'SELL'} {lot} lot L{level} @ {f.res.price} TP {f.tp}")
    return True


def close_all(pos, why):
    """전부 시장가 청산 시도. 사이클은 정상 조회로 포지션이 0개인 걸 확인한 뒤에만 끝냄 (메인 루프가 RETRY_SEC마다 다시)"""
    for p in pos:
        dt.close_retry(p, f"HM {why}", f"사이클 청산 ({why})")


def end_cycle(st, why):
    time.sleep(1)
    total = realized(st["since"])
    if total is None:
        total = 0.0
        log.warning("사이클 끝: 이력 조회 실패로 합계 모름")
    icon = {"target": "✅", "limit": "🛑", "tp": "✅", "max": "⚠️"}.get(why, "")
    msg = {"target": f"사이클 목표 +{TARGET:.0%} 도달", "limit": f"사이클 손실 한도 -{LIMIT:.0%}",
           "tp": "포지션 모두 익절", "max": "최대 단계 다 쓰고 모두 익절"}.get(why, why)
    log.info(f"사이클 끝 ({msg}) 합계 {total:+.2f} 단계 {st['level']}")
    bot.send(f"{icon} 사이클 끝: {msg}\n합계 {total:+,.2f}$ (시작 잔고 {st['start_bal']:,.2f}의 {total / st['start_bal']:+.2%}), "
             f"최대 단계 L{st['level']}\n{bot.account_line()}")
    return {}


def start_ok(m5, h1, now):
    """SAR 단타와 같은 신호 + 필터를 통과하면 방향, 아니면 (0, 이유)"""
    side, bar, close, desc, _ = dt.compute_signal(m5, h1)
    if not side:
        return 0, None
    if now.weekday() == 4 and dt.is_after(now, dt.FRIDAY_NO_ENTRY):
        return 0, "금요일 마감 전"
    if NIGHT_OFF and dt.is_after(now, dt.NO_ENTRY_FROM) and not dt.is_after(now, dt.NO_ENTRY_UNTIL):
        return 0, "미국장 제외 시간"
    if dt.h1_flips_today(h1, now) >= dt.MAX_H1_FLIPS:
        return 0, "1시간봉 SAR 전환 많음"
    a = float(atr(m5).iat[-1])
    if a < dt.MIN_ATR:
        return 0, f"ATR {a:.2f} 작음"
    ref = dt.atr_ref()
    if ref and a > dt.MAX_ATR_MULT * ref:
        return 0, "ATR 급변동"
    if ov := dt.overextended(m5, side):
        return 0, f"과열 ({ov})"
    if near_news(pd.Timestamp(bar) + pd.Timedelta(minutes=5), NEWS_MIN):
        return 0, f"뉴스 앞뒤 {NEWS_MIN}분"
    return side, None


def main():
    global bot
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="실계좌에서 실행 (기본은 데모)")
    args = ap.parse_args()
    if args.live:
        use_live_paths()
    bot = Trader(STRATEGY, SYMBOL, MAGIC, dt.DEVIATION, LOG_DIR, "hedge", live=args.live)
    bot.bridge_id = "HEDGE"        # MT4 따라하기 이름
    acc = bot.start()
    st = load_json(STATE_JSON)
    desc = (f"SAR 단타 신호에 매수·매도 동시 진입, 각자 익절 {TP_USD:g}$ / 손절 없음, SAR 전환마다 2배 추가 (최대 {MAX_LEVEL}단계), "
            f"사이클 +{TARGET:.0%} 청산 / -{LIMIT:.0%} 손절, 뉴스 앞뒤 {NEWS_MIN}분 시작 안 함"
            + (", 미국장(한국 21~03시) 시작 안 함" if NIGHT_OFF else ", 밤에도 시작"))
    log.info(f"시작: 계좌 {acc.login} 잔고 {acc.balance}, {desc}, 이어받은 상태 {st}")
    bot.send(f"▶️ {'실계좌' if args.live else '데모'} 시작 ({acc.login})\n전략: {desc}\n시작 lot {lot0(acc.balance)} (1,000$당 {LOT_PER_1000})"
             + (f"\n진행 중 사이클 이어받음: L{st['level']}" if st else ""))
    dt.bot = bot                                       # dt.my_positions / fresh_tick / close_retry가 이 봇(매직)으로 동작
    dt.lock(LOG_DIR.name)
    sig = load_json(SIGNAL_JSON)                       # 마지막으로 처리한 봉 (재시작 때 같은 봉으로 다시 시작·추가 안 함)
    last_bar = pd.Timestamp(sig["last_bar"]) if sig.get("last_bar") else None
    last_retry, last_warn = 0.0, 0.0
    with bot.running("⏸ 종료 (열린 포지션은 서버 익절만 걸려 있음, 손절 없음)"):
        while True:
            now = bot.keep_alive()
            pos = dt.my_positions()
            if pos is None:                            # 조회 실패를 '포지션 없음'으로 보지 않음: 이 루프는 판단·주문 안 함
                if time.time() - last_warn > 60:
                    last_warn = time.time()
                    log.warning(f"포지션 조회 실패 ({mt5.last_error()}), 이번 루프는 쉼")
                time.sleep(POLL_SEC)
                continue
            if st.get("closing"):                      # 청산 중: 포지션이 정상 조회로 0개가 될 때까지 다시 청산, 그 뒤에 사이클 끝
                if pos:
                    if time.time() - last_retry > RETRY_SEC:
                        last_retry = time.time()
                        close_all(pos, st["closing"])
                    time.sleep(POLL_SEC)
                    continue
                st = end_cycle(st, st["closing"])
                save_state(st)
            elif st:
                real = realized(st["since"])
                if real is None:                       # 이력 조회 실패: 목표·한도 판단 안 함
                    time.sleep(POLL_SEC)
                    continue
                total = real + sum(p.profit + p.swap for p in pos)
                if pos and (total >= TARGET * st["start_bal"] or total <= -LIMIT * st["start_bal"]):
                    st["closing"] = "target" if total > 0 else "limit"
                    save_state(st)
                    last_retry = time.time()
                    close_all(pos, st["closing"])
                    time.sleep(POLL_SEC)
                    continue
                if not pos and (real >= TARGET * st["start_bal"] or st["level"] >= MAX_LEVEL):
                    st = end_cycle(st, "tp" if real >= TARGET * st["start_bal"] else "max")
                    save_state(st)
            m5 = bot.rates(mt5.TIMEFRAME_M5, dt.M5_BARS)
            if m5 is not None and m5.time.iat[-1] != last_bar:
                h1 = None
                if not st:                             # 1시간봉 조회 실패·부족이면 이 봉은 다음 루프에서 다시 (예외로 꺼지지 않게)
                    h1 = bot.rates(mt5.TIMEFRAME_H1, dt.H1_BARS)
                    bar_end = m5.time.iat[-1] + pd.Timedelta(minutes=5)
                    if h1 is None or (h1.time + pd.Timedelta(hours=1) <= bar_end).sum() < 50:
                        if time.time() - last_warn > 60:
                            last_warn = time.time()
                            log.warning(f"1시간봉 조회 실패 ({mt5.last_error()}), 다시 시도")
                        time.sleep(POLL_SEC)
                        continue
                last_bar = m5.time.iat[-1]
                save_json(SIGNAL_JSON, {"last_bar": str(last_bar)})
                up = psar(m5.high.values, m5.low.values)
                flip = (1 if up[-1] else -1) if up[-1] != up[-2] else 0

                def stale():
                    """신호 봉 마감 + ENTRY_LATE_SEC 넘었거나 시세가 멈췄으면 이유 (주문 직전에 부름)"""
                    late = (bot.clock.now() - (last_bar + pd.Timedelta(minutes=5))).total_seconds()
                    if late > dt.ENTRY_LATE_SEC:
                        return f"신호 봉 마감 {late:.0f}초 뒤 (재시작·멈춤)"
                    if not dt.fresh_tick():
                        return f"시세 멈춤 / 연결 끊김 (틱 {dt.TICK_FRESH_SEC}초 넘게 없음)"
                    return None

                if not st:
                    side, why = start_ok(m5, h1, now)
                    if why:
                        log.info(f"신호 있지만 시작 안 함: {why}")
                    if side and (why := stale()):
                        log.info(f"신호 있지만 시작 안 함: {why}")
                        side = 0
                    if side:
                        bal = mt5.account_info().balance
                        l0 = lot0(bal)
                        st = {"since": (now - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
                              "start_bal": bal, "level": 0, "l0": l0}
                        save_state(st)
                        open_pos(1, l0, 0)
                        open_pos(-1, l0, 0)
                        bot.send(f"🔀 사이클 시작: 매수·매도 {l0} lot 동시 진입 (SAR 단타 신호 {side_name(side)})\n"
                                 f"목표 +{TARGET * bal:,.2f}$ / 한도 -{LIMIT * bal:,.2f}$\n{bot.account_line()}")
                elif flip and st["level"] < MAX_LEVEL and not any(side_of(p) == flip for p in pos):
                    if why := stale():
                        log.info(f"SAR 전환 추가 안 함: {why}")
                        time.sleep(POLL_SEC)
                        continue
                    st["level"] += 1
                    lot = round(st["l0"] * 2 ** st["level"], 2)
                    save_state(st)
                    if open_pos(flip, lot, st["level"]):
                        bot.send(f"{'🟢 매수' if flip == 1 else '🔴 매도'} 추가 {lot} lot (L{st['level']}, SAR 전환)\n"
                                 f"{bot.account_line()}")
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
