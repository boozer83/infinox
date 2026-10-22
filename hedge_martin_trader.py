"""데모 관찰용 헤지 마틴 (HK MARTIN 구조 재현, backtests/hedge_martin_ticks.py의 START_SIG + CONTINUE 모드와 같은 규칙)

- 사이클 시작: SAR 단타(demo_trader)와 같은 신호 봉(5분봉 SAR 전환 + RSI50 + 1시간봉 SAR + 시간/ATR/과열 필터)이 완성되면
  매수·매도를 같은 lot(L0)으로 동시 진입. 중요 지표(USD/EUR/GBP moderate 이상) 앞뒤 NEWS_MIN분이면 시작 안 함
  L0 = 사이클 시작 잔고 1,000달러당 LOT_PER_1000 lot (0.01 단위 내림, 최소 0.01)
- 모든 포지션은 진입가 ± TP_USD에 서버 익절, 손절 없음 (진 쪽은 들고 감)
- 사이클 중 5분봉 SAR이 전환될 때마다 그 방향에 열린 포지션이 없으면 다음 단계 lot(L0 x 2^단계)으로 1개 진입 (최대 MAX_LEVEL)
  포지션이 다 익절로 닫혀도 사이클 목표 전이면 끝내지 않고 다음 전환에 다음 단계로 이어감
- 사이클 합계(이번 사이클 확정 손익 + 평가손익)가 시작 잔고의 +TARGET%면 전부 청산, -LIMIT%면 전부 청산
- 주말에도 보유 (백테스트와 같음). 데모 전용 (실계좌 옵션 없음)
백테스트 (해마다 1만 달러, 2022~2026, 익절 5 / 목표 1% / 한도 30% / 4단계): 2022 -81%, 2023 -32%, 2024 -34%, 2025 +55%, 2026 +33%,
  최대 낙폭 85%. 캡처(9~10월 HK 계좌)와 흐름이 맞는지 보려는 관찰용
상태: hedge_log/state.json (사이클 시작 시각·잔고·단계)
"""
import json
import logging
import time
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import demo_trader as dt
from backtests.hedge_martin_ticks import near_news
from config import SYMBOL
from indicators import atr, psar
from reports import ServerClock
from telegram_notify import Notifier

TP_USD = 5.0
TARGET = 0.01
LIMIT = 0.30
MAX_LEVEL = 4
LOT_PER_1000 = 0.01
NEWS_MIN = 30
NIGHT_OFF = False   # True면 SAR 단타처럼 서버 15:00~21:00(한국 21:00~03:00) 시작 안 함. 2026-10-22 끔 (밤에도 시작)
MAGIC = 20261020
POLL_SEC = 1.0
STRATEGY = "헤지 마틴"
LOG_DIR = dt.Path(__file__).resolve().parent / "hedge_log"
STATE_JSON = LOG_DIR / "state.json"
log = logging.getLogger("hedge")


def send(text):
    tg.send(f"[{STRATEGY}] {text}")


def load_state():
    try:
        return json.loads(STATE_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def save_state(st):
    STATE_JSON.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def my_positions():
    return [p for p in (mt5.positions_get(symbol=SYMBOL) or ()) if p.magic == MAGIC]


def realized(since):
    """since: 서버 시각 문자열. 그 뒤로 이 봇이 청산한 손익 합계"""
    t0 = int(pd.Timestamp(since).tz_localize("UTC").timestamp())
    deals = mt5.history_deals_get(t0, t0 + 60 * 86400) or ()
    return sum(d.profit + d.commission + d.swap for d in deals if d.magic == MAGIC and d.entry == mt5.DEAL_ENTRY_OUT)


def lot0(bal):
    return max(0.01, np.floor(bal / 1000 * LOT_PER_1000 * 100) / 100)


def open_pos(side, lot, level):
    info, tick = mt5.symbol_info(SYMBOL), mt5.symbol_info_tick(SYMBOL)
    otype = mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL
    price = tick.ask if side == 1 else tick.bid
    margin = mt5.order_calc_margin(otype, SYMBOL, lot, price)
    if margin is None or margin > mt5.account_info().margin_free * 0.9:
        send(f"⚠️ 증거금 부족으로 {'매수' if side == 1 else '매도'} {lot} lot 진입 안 함")
        return False
    tp = round(price + TP_USD * side, info.digits)
    res = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": round(lot, 2), "type": otype,
                          "price": price, "tp": tp, "deviation": dt.DEVIATION, "magic": MAGIC,
                          "comment": f"HM L{level}", "type_time": mt5.ORDER_TIME_GTC, "type_filling": dt.filling_mode()})
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        err = f"{res.retcode if res else mt5.last_error()} {res.comment if res else ''}"
        log.error(f"주문 실패: {err}")
        send(f"❗ 주문 실패 ({'매수' if side == 1 else '매도'} {lot} lot): {err}")
        return False
    log.info(f"진입 {'BUY' if side == 1 else 'SELL'} {lot} lot L{level} @ {res.price} TP {tp}")
    return True


def close_all(why):
    for p in my_positions():
        tick = mt5.symbol_info_tick(SYMBOL)
        buy = p.type == mt5.POSITION_TYPE_BUY
        mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": p.volume, "position": p.ticket,
                        "type": mt5.ORDER_TYPE_SELL if buy else mt5.ORDER_TYPE_BUY,
                        "price": tick.bid if buy else tick.ask, "deviation": dt.DEVIATION, "magic": MAGIC,
                        "comment": f"HM {why}", "type_time": mt5.ORDER_TIME_GTC, "type_filling": dt.filling_mode()})


def end_cycle(st, why):
    time.sleep(1)
    total = realized(st["since"])
    icon = {"target": "✅", "limit": "🛑", "tp": "✅", "max": "⚠️"}.get(why, "")
    msg = {"target": f"사이클 목표 +{TARGET:.0%} 도달", "limit": f"사이클 손실 한도 -{LIMIT:.0%}",
           "tp": "포지션 모두 익절", "max": "최대 단계 다 쓰고 모두 익절"}.get(why, why)
    log.info(f"사이클 끝 ({msg}) 합계 {total:+.2f} 단계 {st['level']}")
    send(f"{icon} 사이클 끝: {msg}\n합계 {total:+,.2f}$ (시작 잔고 {st['start_bal']:,.2f}의 {total / st['start_bal']:+.2%}), "
         f"최대 단계 L{st['level']}\n{dt.account_line()}")
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
    if near_news(pd.Series([pd.Timestamp(bar) + pd.Timedelta(minutes=5)]))[0]:
        return 0, f"뉴스 앞뒤 {NEWS_MIN}분"
    return side, None


def main():
    global tg
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(LOG_DIR / "events.log", encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)
    dt.connect(False)
    acc = dt.check_demo(False)
    tg = Notifier()
    clock = ServerClock(SYMBOL)
    st = load_state()
    desc = (f"SAR 단타 신호에 매수·매도 동시 진입, 각자 익절 {TP_USD:g}$ / 손절 없음, SAR 전환마다 2배 추가 (최대 {MAX_LEVEL}단계), "
            f"사이클 +{TARGET:.0%} 청산 / -{LIMIT:.0%} 손절, 뉴스 앞뒤 {NEWS_MIN}분 시작 안 함"
            + (", 미국장(한국 21~03시) 시작 안 함" if NIGHT_OFF else ", 밤에도 시작"))
    log.info(f"시작: 계좌 {acc.login} 잔고 {acc.balance}, {desc}, 이어받은 상태 {st}")
    send(f"▶️ 데모 시작 ({acc.login})\n전략: {desc}\n시작 lot {lot0(acc.balance)} (1,000$당 {LOT_PER_1000})"
         + (f"\n진행 중 사이클 이어받음: L{st['level']}" if st else ""))
    last_bar = None
    try:
        while True:
            dt.check_demo(False, acc.login)
            clock.update()
            now = clock.now()
            pos = my_positions()
            if st:
                real = realized(st["since"])
                total = real + sum(p.profit + p.swap for p in pos)
                if pos and total >= TARGET * st["start_bal"]:
                    close_all("target")
                    st = end_cycle(st, "target")
                elif pos and total <= -LIMIT * st["start_bal"]:
                    close_all("limit")
                    st = end_cycle(st, "limit")
                elif not pos and (real >= TARGET * st["start_bal"] or st["level"] >= MAX_LEVEL):
                    st = end_cycle(st, "tp" if real >= TARGET * st["start_bal"] else "max")
                if not st:
                    save_state(st)
            m5 = dt.rates(mt5.TIMEFRAME_M5, dt.M5_BARS)
            if m5 is not None and m5.time.iat[-1] != last_bar:
                last_bar = m5.time.iat[-1]
                up = psar(m5.high.values, m5.low.values)
                flip = (1 if up[-1] else -1) if up[-1] != up[-2] else 0
                if not st:
                    h1 = dt.rates(mt5.TIMEFRAME_H1, dt.H1_BARS)
                    side, why = start_ok(m5, h1, now)
                    if why:
                        log.info(f"신호 있지만 시작 안 함: {why}")
                    if side:
                        bal = mt5.account_info().balance
                        l0 = lot0(bal)
                        st = {"since": (now - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
                              "start_bal": bal, "level": 0, "l0": l0}
                        save_state(st)
                        open_pos(1, l0, 0)
                        open_pos(-1, l0, 0)
                        send(f"🔀 사이클 시작: 매수·매도 {l0} lot 동시 진입 (SAR 단타 신호 {'매수' if side == 1 else '매도'})\n"
                             f"목표 +{TARGET * bal:,.2f}$ / 한도 -{LIMIT * bal:,.2f}$\n{dt.account_line()}")
                elif flip and st["level"] < MAX_LEVEL and not any(
                        (1 if p.type == mt5.POSITION_TYPE_BUY else -1) == flip for p in my_positions()):
                    st["level"] += 1
                    lot = round(st["l0"] * 2 ** st["level"], 2)
                    save_state(st)
                    if open_pos(flip, lot, st["level"]):
                        send(f"{'🟢 매수' if flip == 1 else '🔴 매도'} 추가 {lot} lot (L{st['level']}, SAR 전환)\n{dt.account_line()}")
            time.sleep(POLL_SEC)
    except KeyboardInterrupt:
        send("⏸ 종료 (열린 포지션은 서버 익절만 걸려 있음, 손절 없음)")
    except BaseException as e:
        log.exception("오류로 종료")
        send(f"❗ 오류로 멈췄습니다: {e}")
        raise
    finally:
        tg.flush()
        mt5.shutdown()


if __name__ == "__main__":
    main()
