"""데모 전용 지수 장중 봇 (docs/hk_program.md 5-23 ~ 5-25, 백테스트 backtests/index_intraday.py / index_ticks.py)

--symbol NAS100 : 첫 30분 방향. 본장(뉴욕 09:30 = 서버 16:30) 첫 30분 Bid 움직임이 ATR x FIRST_MIN 넘으면
                  그 방향으로 시가 30분 + 0.3초 뒤 시장가 진입, 손절 = 진입 순간 Bid - ATR (매수) / Ask + ATR (매도), MT5 SL로 걺
--symbol GER40  : ORB 30분 매도만. Xetra 09:00(서버시간으로 바꿈) 처음 30분 Bid 저가를 마감된 1분봉 종가가 처음 깨면
                  다음 분 + 0.3초 뒤 매도 (진입 시각이 시가 + 3시간 전일 때만). 손절 = Bid가 범위 가운데에 닿으면 시장가 청산
                  (봇이 1초마다 확인, 백테스트와 같은 규칙) + 만일을 위한 MT5 SL (가운데 + 진입 스프레드 + 여유)
청산: 마감 CLOSE_BEFORE초 전 시장가 (안 닫히면 다시 시도)
ATR = 전날까지 본장 고가-저가 14일 평균 (1분봉 Bid). 휴장일·미국 조기 마감은 common/exchange_calendar.py (백테스트와 같음)
lot = 거래당 위험 RISK_PCT% / 1 lot 손실(order_calc_profit), 종목 lot 단위로 내림. 최소 lot 위험이 MAX_RISK_PCT% 넘거나
      손실 계산이 안 되면 그날 건너뜀
안전장치 (2026-10-24 GPT 검증 반영)
  - 상태 파일보다 MT5 실제 포지션·오늘 체결 이력을 먼저 봄: 이 매직의 포지션이 있거나 오늘 진입 체결이 있으면 새 진입 안 함
  - 한 종목에 한 프로세스만 (윈도우 이름 있는 뮤텍스), 상태 파일은 임시 파일에 쓰고 바꿔치기
  - 재시작 때 GER40 가운데 손절은 그날 범위를 다시 계산해서 복원 (안 되면 정리), 손절·마감 청산은 닫힐 때까지 다시 시도
  - 진입 뒤 바로 닫혀도 체결 번호로 이어받아 기록, 기록 못 한 거래는 날이 바뀌어도 따로 보관해서 다시 시도 (2026-10-24 GPT 2차 검증)
  - 주문 응답이 이상해도(부분 체결 등) 실제 포지션을 다시 조회해서 이어받음. 포지션 조회 실패면 그 루프는 새 진입 안 함
  - 진입은 한 번 읽은 호가로 lot·손절을 계산하고 손절가 방향·최소 거리 확인. 진입 예정 시각보다 LATE_SEC초 넘게 늦으면 건너뜀
  - GER40은 범위 이후 마감된 1분봉을 빠짐없이 순서대로 확인해서 '처음 돌파'를 찾음 (재시작해도 같은 봉을 찾음)
  - 터미널 연결·시세 신선도(TICK_FRESH_SEC 안의 틱) 확인, 안 되면 새 진입 안 함
기록: logs/index_<종목>_log/trades.csv (신호·예정·주문·체결 시각, 호가, 요청·체결 lot, 목표·실제 위험, 결과 코드,
      틱 기준 최대 역행, 청산 요청·체결가·미끄러짐, 부분 청산 평균가), events.log, 텔레그램

사용법: python index_trader.py --symbol NAS100        (데모 계좌, Ctrl+C로 종료)
        python index_trader.py --symbol GER40 --dry  (오늘 본장 시각·ATR만 보고 끝)
"""
import argparse
import ctypes
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np

from backtests.index_intraday import SESS, server
from common.exchange_calendar import de_closed, us_close_time, us_closed
from common.trader import Trader, append_csv, load_json, side_name
from config import SYMBOL as CLOCK_SYMBOL
from reports import ServerClock

VERSION = "2026-10-24c"
MAGICS = {"NAS100": 20261025, "GER40": 20261026}
NAMES = {"NAS100": "나스닥 첫30분", "GER40": "독일 ORB 매도"}
FIRST_MIN, FIRST_STOP = 0.2, 1.0
ORB_MIN, ENTRY_UNTIL = 30, 180
ATR_DAYS = 14
RISK_PCT = 0.25          # 목표 위험 (잔고 %)
MAX_RISK_PCT = 0.5       # 최소 lot이어도 이보다 크면 건너뜀
ENTRY_DELAY_MS = 300
LATE_SEC = 60            # 진입 예정 시각보다 이만큼 늦으면 그날 건너뜀
CLOSE_BEFORE = 2         # 마감 이 초 전에 청산 (백테스트는 마감 직전 마지막 틱)
SAFETY_SL_PTS = {"GER40": 20.0}   # GER40 MT5 SL 여유 (봇의 Bid 손절이 먼저)
TICK_FRESH_SEC = 30
DEVIATION = 500          # 0.01 단위 = 5포인트
POLL_SEC = 1.0
FIELDS = ["version", "date", "side", "signal", "atr", "signal_bar", "planned_entry", "send_time", "fill_time", "order_rtt_ms", "entry_delay_ms",
          "bid", "ask", "tick_time", "req_lot", "lot", "order", "deal", "positions", "retcode", "req_price", "fill", "slip",
          "entry_spread", "sl", "target_risk", "risk", "currency", "mae_pts", "close_req_time", "close_req_price",
          "close_spread", "close_time", "close_price", "close_slip", "close_deals", "result", "pts", "profit"]
bot = None


# ---------------------------------------------------------------- 시각·시세
def bounds(sym, d):
    """그날 본장 (서버시간 시작, 끝). 휴장일이면 None"""
    tz, o, c = SESS[sym]
    if (us_closed(d) if sym == "NAS100" else de_closed(d)):
        return None
    cl = us_close_time(d, c) if sym == "NAS100" else c
    return server(d, tz, o), server(d, tz, cl)


def utc(t):
    return t.replace(tzinfo=timezone.utc)


def m1(sym, t0, t1):
    """t0 <= 봉 시작 < t1 1분봉 (서버시간)"""
    r = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M1, utc(t0), utc(t1 - timedelta(seconds=1)))
    if r is None or not len(r):
        return None
    r = r[r["time"] < int(utc(t1).timestamp())]
    return r if len(r) else None


def atr_of(sym, d):
    """전날까지 본장 고가-저가 ATR_DAYS일 평균"""
    rng, k = [], d
    while len(rng) < ATR_DAYS and (d - k).days < 60:
        k -= timedelta(days=1)
        b = bounds(sym, k)
        if b is None:
            continue
        r = m1(sym, *b)
        if r is None or len(r) < 60:
            continue
        rng.append(float(r["high"].max() - r["low"].min()))
    return float(np.mean(rng)) if len(rng) == ATR_DAYS else None


def nas_signal(r, atr, so):
    """첫 30분 1분봉 -> (방향, 움직임) 또는 (0, 이유)"""
    if r is None or r["time"][0] > int(utc(so + timedelta(minutes=5)).timestamp()):
        return 0, "첫 30분 봉 없음"
    mv = float(r["close"][-1] - r["open"][0])
    if abs(mv) < FIRST_MIN * atr:
        return 0, f"움직임 {mv:+.1f} < ATR {atr:.0f} x {FIRST_MIN:g}"
    return (1 if mv > 0 else -1), mv


def ger_first_break(bars, lo):
    """범위 이후 마감된 1분봉들 중 종가가 lo 밑인 첫 봉의 시작 시각 (없으면 None)"""
    k = np.flatnonzero(bars["close"] < lo)
    return int(bars["time"][k[0]]) if k.size else None


def fresh(sym):
    """터미널 연결 + 그 종목 틱이 TICK_FRESH_SEC 안에 들어왔는지"""
    ti = mt5.terminal_info()
    t = mt5.symbol_info_tick(sym)
    if ti is None or not ti.connected or t is None or not t.time_msc:
        return None
    if (bot.clock.now() - datetime.fromtimestamp(t.time_msc / 1000, timezone.utc).replace(tzinfo=None)).total_seconds() > TICK_FRESH_SEC:
        return None
    return t


# ---------------------------------------------------------------- MT5 실제 상태
def my_positions():
    """이 봇 포지션 목록, 조회 실패면 None"""
    p = mt5.positions_get(symbol=bot.symbol)
    if p is None:
        return None
    return [x for x in p if x.magic == bot.magic]


def entered_today(d):
    """오늘(서버 날짜) 이 봇의 진입 체결이 있었는지 (조회 실패면 None)"""
    t0 = datetime(d.year, d.month, d.day)
    deals = mt5.history_deals_get(utc(t0), utc(t0 + timedelta(days=1)))
    if deals is None:
        return None
    return any(d.magic == bot.magic and d.symbol == bot.symbol and d.entry == mt5.DEAL_ENTRY_IN for d in deals)


def save_state(path, st):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


_mutex = None


def lock(sym):
    """한 종목에 프로세스 하나: 윈도우 이름 있는 뮤텍스 (만드는 것과 확인이 한 번에 일어나서 동시에 켜도 하나만 통과).
    프로세스가 끝나면 윈도우가 알아서 풀어 줌"""
    global _mutex
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateMutexW.restype = ctypes.c_void_p
    k.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    h = k.CreateMutexW(None, False, f"Local\\infinox_index_trader_{sym}")
    if not h:
        raise SystemExit(f"잠금 만들기 실패 ({ctypes.get_last_error()})")
    if ctypes.get_last_error() == 183:                # ERROR_ALREADY_EXISTS
        raise SystemExit("이미 실행 중 (같은 종목 봇). 중복 실행 안 함")
    _mutex = h


# ---------------------------------------------------------------- 주문
def lot_for(side, price, sl_price):
    """(lot, 실제 위험 $, 목표 위험 $) 또는 (None, 이유, 목표)"""
    info = mt5.symbol_info(bot.symbol)
    risk = mt5.account_info().balance * RISK_PCT / 100
    otype = mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL
    loss = mt5.order_calc_profit(otype, bot.symbol, 1.0, price, sl_price)
    if loss is None or loss >= 0:
        return None, "손실 계산 안 됨", risk
    loss = -loss
    step = info.volume_step or 0.01
    dec = max(0, -int(math.floor(math.log10(step) + 1e-9)))
    lot = math.floor(risk / loss / step + 1e-9) * step
    lot = round(min(max(lot, info.volume_min), info.volume_max, 1.0), dec)
    actual = loss * lot
    if actual > mt5.account_info().balance * MAX_RISK_PCT / 100:
        return None, f"최소 lot 위험 {actual:.2f}$가 {MAX_RISK_PCT:g}% 넘음", risk
    return lot, actual, risk


def enter(st, side, sl_of, signal, planned):
    """한 번 읽은 호가로 lot·손절을 정하고 시장가 진입. sl_of(tick) -> 절대 손절가"""
    t = fresh(bot.symbol)
    if t is None:
        st["done"] = "시세 멈춤 / 연결 끊김"
        bot.send(f"⚠️ 진입 건너뜀: {st['done']}")
        return
    price = t.ask if side > 0 else t.bid
    info = mt5.symbol_info(bot.symbol)
    sl = round(sl_of(t), info.digits)
    min_dist = (info.trade_stops_level or 0) * info.point
    if (side > 0 and sl >= t.bid - min_dist) or (side < 0 and sl <= t.ask + min_dist):
        st["done"] = f"손절가 {sl}가 현재가 쪽으로 너무 가까움"
        bot.send(f"⚠️ 진입 건너뜀: {st['done']}")
        return
    lot, risk, target = lot_for(side, price, sl)
    if lot is None:
        st["done"] = risk
        bot.send(f"⏭ 진입 건너뜀: {risk} (목표 위험 {target:.2f}$)")
        return
    if not bot.margin_ok(side, lot):
        st["done"] = "증거금 부족"
        return
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": bot.symbol, "volume": lot,
           "type": mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL, "price": price, "sl": sl,
           "deviation": DEVIATION, "magic": bot.magic, "comment": f"IDX {VERSION}", "type_time": mt5.ORDER_TIME_GTC,
           "type_filling": bot.filling_mode()}
    st["done"] = "주문 보냄"                       # 응답 전에 죽어도 다시 안 보냄 (다음 루프는 실제 포지션으로 복구)
    save_state(bot.state_path, st)
    sent = bot.clock.now()
    res = mt5.order_send(req)
    got = bot.clock.now()
    time.sleep(0.5)
    pos = my_positions()
    st.update(side=side, signal=signal, planned_entry=f"{planned:%H:%M:%S.%f}"[:-3], send_time=f"{sent:%H:%M:%S.%f}"[:-3],
              order_rtt_ms=int((got - sent).total_seconds() * 1000), planned_ms=int(utc(planned).timestamp() * 1000), bid=t.bid, ask=t.ask,
              tick_time=datetime.fromtimestamp(t.time_msc / 1000, timezone.utc).strftime("%H:%M:%S.%f")[:-3],
              req_lot=lot, req_price=price, entry_spread=round(t.ask - t.bid, 2), sl=sl, target_risk=round(target, 2),
              risk=round(risk, 2), currency=mt5.account_info().currency, retcode=res.retcode if res else mt5.last_error(),
              order=res.order if res else "", deal=res.deal if res else "")
    if pos is None:                                    # 조회 실패: 결과 모름 -> 메인 루프가 실제 포지션·체결로 이어받음
        bot.log.warning("주문 뒤 포지션 조회 실패, 다음 루프에서 확인")
        return
    if not pos:
        dl = mt5.history_deals_get(ticket=res.deal) if res and res.deal else None
        if dl:                                         # 진입했지만 조회 전에 이미 닫힘 (즉시 손절 등)
            adopt_deal(st, dl[0])
            st["done"] = "진입"
            bot.send(f"⚠️ 진입 체결 뒤 바로 닫힘 (포지션 {dl[0].position_id}), 결과는 청산 기록으로")
            return
        st["done"] = f"주문 실패 ({st['retcode']} {res.comment if res else ''})"
        bot.log.error(st["done"])
        bot.send(f"❗ {st['done']}")
        return
    adopt(st, pos)
    st["done"] = "진입"
    bot.log.info(f"진입 {side_name(side)} {st['lot']} lot @ {st['fill']} (요청 {price}, 손절 {sl}, 위험 {risk:.2f}) {signal}")
    bot.send(f"{'📈' if side > 0 else '📉'} {side_name(side)} {st['lot']} lot @ {st['fill']} (요청 {price}, 스프레드 {t.ask - t.bid:.2f}, "
             f"응답 {st['order_rtt_ms']}ms)\n{signal}\n손절 {sl} (위험 {risk:.2f}{st['currency']}), 마감 전 청산")


def adopt(st, pos):
    """실제 포지션으로 상태 채움 (부분 체결·재시작 복구)"""
    p = pos[0]
    vol = sum(x.volume for x in pos)
    st.update(position=p.ticket, positions=[x.ticket for x in pos], lot=round(vol, 2),
              fill=round(sum(x.price_open * x.volume for x in pos) / vol, 2), side=1 if p.type == 0 else -1,
              fill_time=datetime.fromtimestamp(p.time_msc / 1000, timezone.utc).strftime("%H:%M:%S.%f")[:-3],
              entry_ms=int(p.time_msc))
    st.setdefault("sl", p.sl)
    st["slip"] = round((st["fill"] - st.get("req_price", st["fill"])) * st["side"], 2)
    if "planned_ms" in st:
        st["entry_delay_ms"] = st["entry_ms"] - st["planned_ms"]


def adopt_deal(st, d):
    """진입 체결 하나로 상태 채움 (포지션이 이미 닫혀서 positions_get에 없을 때)"""
    st.update(position=d.position_id, positions=[d.position_id], lot=d.volume, fill=d.price,
              side=1 if d.type == mt5.DEAL_TYPE_BUY else -1, entry_ms=int(d.time_msc),
              fill_time=datetime.fromtimestamp(d.time_msc / 1000, timezone.utc).strftime("%H:%M:%S.%f")[:-3])
    st["slip"] = round((d.price - st.get("req_price", d.price)) * st["side"], 2)


def ger_range(sym, day):
    """그날 처음 30분 Bid 범위 (hi, lo) 또는 None (재시작 때 가운데 손절 복원용)"""
    b = bounds(sym, day)
    r = m1(sym, b[0], b[0] + timedelta(minutes=ORB_MIN)) if b else None
    if r is None or len(r) < 20:
        return None
    return float(r["high"].max()), float(r["low"].min())


def close_all(pos, why, st):
    """전부 시장가 청산. 실패해도 st['closing']이 남아 다음 루프에서 포지션이 없어질 때까지 다시 함"""
    st.setdefault("closing", why)
    t = mt5.symbol_info_tick(bot.symbol)
    if t is not None and "close_req_time" not in st:
        st.update(close_req_time=f"{bot.clock.now():%H:%M:%S}", close_req_price=t.bid if st.get("side", 1) > 0 else t.ask,
                  close_spread=round(t.ask - t.bid, 2), close_why=why)
    for p in pos:
        f = bot.deal(-(1 if p.type == 0 else -1), p.volume, "IDX close", position=p.ticket)
        if f.ok:
            bot.log.info(f"{why}: 티켓 {p.ticket} 청산")
        else:
            bot.log.warning(f"{why} 실패: 티켓 {p.ticket} {f.err}")
            if not st.get("close_fail_sent"):
                st["close_fail_sent"] = True
                bot.send(f"❗ {why} 실패 (티켓 {p.ticket}, {f.err}). 닫힐 때까지 1초마다 다시 시도")


def finish(st):
    """포지션이 다 닫혔으면 기록 (부분 청산은 체결량 가중 평균)"""
    deals = []
    for pid in st.get("positions") or [st["position"]]:
        got = mt5.history_deals_get(position=pid)
        if got is None:
            return False
        deals += list(got)
    out = sorted((d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT), key=lambda d: d.time_msc)
    if not out:
        return False
    vol = sum(d.volume for d in out)
    px = sum(d.price * d.volume for d in out) / vol
    profit = round(sum(d.profit + d.commission + d.swap for d in deals), 2)
    last = out[-1]
    result = {mt5.DEAL_REASON_SL: "SL"}.get(last.reason, st.get("close_why", "CLOSE"))
    side = st["side"]
    tk = mt5.copy_ticks_range(bot.symbol, datetime.fromtimestamp(st["entry_ms"] / 1000, timezone.utc),
                              datetime.fromtimestamp(last.time_msc / 1000, timezone.utc), mt5.COPY_TICKS_INFO)
    mae = float(np.max((st["fill"] - tk["bid"]) if side > 0 else (tk["ask"] - st["fill"]))) if tk is not None and len(tk) else ""
    req = st.get("close_req_price")
    append_csv(bot.log_dir / "trades.csv", FIELDS, {
        **st, "version": VERSION, "date": st["day"], "side": side_name(side), "atr": round(st["atr"], 1),
        "mae_pts": round(mae, 2) if mae != "" else "", "close_time": datetime.fromtimestamp(last.time, timezone.utc).strftime("%H:%M:%S"),
        "close_price": round(px, 2), "close_slip": round((req - px) * side, 2) if req else "", "close_deals": len(out),
        "result": result, "pts": round((px - st["fill"]) * side, 2), "profit": profit,
        "positions": "|".join(str(x) for x in st.get("positions") or [st["position"]])})
    bot.send(f"{'✅' if profit > 0 else '❌'} 청산 ({result}) @ {px:.2f}: {(px - st['fill']) * side:+.2f}pt, {profit:+.2f}{st.get('currency', '')}"
             f" (최대 역행 {mae if mae == '' else round(mae, 1)}pt)\n{bot.account_line()}")
    st["position"] = None
    return True


# ---------------------------------------------------------------- 메인
def main():
    global bot
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, choices=list(MAGICS))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    sym = args.symbol
    log_dir = Path(__file__).resolve().parent / "logs" / f"index_{sym.lower()}_log"
    bot = Trader(NAMES[sym], sym, MAGICS[sym], DEVIATION, log_dir, f"idx_{sym}")
    acc = bot.start(dry_run=args.dry)
    bot.clock = ServerClock(CLOCK_SYMBOL)          # 지수 틱은 장 밖에서 멈추거나 비어 있어서 서버 시계는 금 틱으로
    mt5.symbol_select(CLOCK_SYMBOL, True)
    for _ in range(30):
        bot.clock.update()
        time.sleep(0.1)
    now = bot.clock.now()
    if args.dry:
        b = bounds(sym, now.date())
        print(f"서버 {now:%Y-%m-%d %H:%M:%S}, 오늘 본장 {b[0]:%H:%M}~{b[1]:%H:%M}" if b else "오늘 휴장", f"ATR {atr_of(sym, now.date())}")
        mt5.shutdown()
        return
    lock(sym)
    bot.state_path = log_dir / "state.json"
    desc = (f"본장 첫 30분이 ATR x {FIRST_MIN:g} 넘게 움직이면 그 방향, 손절 ATR x {FIRST_STOP:g}" if sym == "NAS100"
            else f"본장 처음 {ORB_MIN}분 저가를 1분봉 종가가 처음 깨면 매도 (진입이 시가 + {ENTRY_UNTIL // 60}시간 전일 때), 손절 범위 가운데")
    bot.send(f"▶️ 데모 시작 ({acc.login}, {VERSION})\n전략: {desc}, 마감 {CLOSE_BEFORE}초 전 청산, 거래당 위험 {RISK_PCT:g}% "
             f"(최소 lot 위험 {MAX_RISK_PCT:g}% 넘으면 건너뜀)")
    st = load_json(bot.state_path)
    bot.log.info(f"시작 {VERSION}, 이어받은 상태 {st}")
    with bot.running("⏸ 종료"):
        while True:
            now = bot.keep_alive()
            d = now.date()
            b = bounds(sym, d)
            pos = my_positions()
            if pos is None:                          # 조회 실패: 아무것도 하지 않음
                time.sleep(POLL_SEC)
                continue
            # 1) 실제 포지션 관리: 이어받기, GER40 Bid 손절, 마감·날짜 넘김 청산
            if pos:
                p_day = datetime.fromtimestamp(pos[0].time, timezone.utc).date()
                if not st.get("position"):
                    st.setdefault("day", str(p_day))
                    st.setdefault("atr", 0.0)
                    adopt(st, pos)
                    bot.log.info(f"포지션 이어받음 {st}")
                    bot.send(f"🔁 열린 포지션 이어받음 (티켓 {st['positions']}, {st['lot']} lot @ {st['fill']})")
                if sym == "GER40" and "mid" not in st:
                    rg = ger_range(sym, p_day)
                    if rg:
                        st["hi"], st["lo"] = rg
                        st["mid"] = (rg[0] + rg[1]) / 2
                        bot.log.info(f"가운데 손절 복원 {st['mid']:.2f}")
                    else:
                        st.setdefault("closing", "범위 복원 실패, 정리")
                t = mt5.symbol_info_tick(sym)
                if b is None or p_day != d or now >= b[1] - timedelta(seconds=CLOSE_BEFORE):
                    close_all(pos, "마감 청산" if p_day == d else "지난 날 포지션 청산", st)
                elif st.get("closing"):
                    close_all(pos, st["closing"], st)
                elif sym == "GER40" and t is not None and t.bid >= st["mid"]:
                    close_all(pos, "SL", st)
                save_state(bot.state_path, st)
                time.sleep(POLL_SEC)
                continue
            if st.get("position"):
                finish(st)
                save_state(bot.state_path, st)
            for rec in list(st.get("unfinished", [])):     # 지난 날 기록 못 한 거래
                if finish(rec):
                    st["unfinished"].remove(rec)
            if st.get("day") != str(d):
                left = st.get("unfinished", []) + ([st] if st.get("position") else [])
                st = {"day": str(d), "unfinished": left} if left else {"day": str(d)}
            if b is None or st.get("done") or now < b[0] or now >= b[1]:
                save_state(bot.state_path, st)
                time.sleep(POLL_SEC if b and b[0] - now < timedelta(minutes=2) else 10)
                continue
            so, sc = b
            # 2) 오늘 이미 진입했으면 (상태 파일이 없어도) 다시 안 들어감
            ent = entered_today(d)
            if ent is None:
                time.sleep(POLL_SEC)
                continue
            if ent:
                st["done"] = "오늘 이미 진입함 (체결 이력)"
                bot.log.info(st["done"])
                continue
            if "atr" not in st:
                st["atr"] = atr_of(sym, d)
                if st["atr"] is None:
                    st["done"] = "ATR 없음"
                    bot.send("⚠️ ATR 계산할 1분봉이 부족해서 오늘은 쉽니다")
                    continue
            atr = st["atr"]
            # 3) 신호
            if sym == "NAS100":
                planned = so + timedelta(minutes=30, milliseconds=ENTRY_DELAY_MS)
                if now >= planned:
                    if (now - planned).total_seconds() > LATE_SEC:
                        st["done"] = f"늦게 켜짐 ({(now - planned).total_seconds():.0f}초)"
                    else:
                        side, info = nas_signal(m1(sym, so, so + timedelta(minutes=30)), atr, so)
                        if not side:
                            st["done"] = info
                            bot.log.info(info)
                        else:
                            enter(st, side, lambda t, s=side: (t.bid - FIRST_STOP * atr) if s > 0 else (t.ask + FIRST_STOP * atr),
                                  f"첫 30분 {info:+.1f}pt (ATR {atr:.0f})", planned)
            else:
                rng_end = so + timedelta(minutes=ORB_MIN)
                cur = now.replace(second=0, microsecond=0)
                if cur >= rng_end + timedelta(minutes=1):
                    if "lo" not in st:
                        r = m1(sym, so, rng_end)
                        if r is None or len(r) < 20:
                            st["done"] = "범위 봉 부족"
                            continue
                        st["hi"], st["lo"] = float(r["high"].max()), float(r["low"].min())
                        st["mid"] = (st["hi"] + st["lo"]) / 2
                    bars = m1(sym, rng_end, cur)                     # 범위 이후 마감된 봉 전부 (순서대로)
                    brk = ger_first_break(bars, st["lo"]) if bars is not None else None
                    if brk is not None:
                        bar = datetime.fromtimestamp(brk, timezone.utc).replace(tzinfo=None)
                        planned = bar + timedelta(minutes=1, milliseconds=ENTRY_DELAY_MS)
                        if planned >= so + timedelta(minutes=ENTRY_UNTIL):
                            st["done"] = "3시간 안에 돌파 없음"
                        elif (now - planned).total_seconds() > LATE_SEC:
                            st["done"] = f"돌파({bar:%H:%M} 봉)를 늦게 봄 ({(now - planned).total_seconds():.0f}초)"
                            bot.log.info(st["done"])
                        elif now >= planned:
                            st["signal_bar"] = f"{bar:%H:%M}"
                            spread_pad = SAFETY_SL_PTS["GER40"]
                            enter(st, -1, lambda t: st["mid"] + (t.ask - t.bid) + spread_pad,
                                  f"30분 저가 {st['lo']:.1f} 깸 ({bar:%H:%M} 봉, 범위 {st['hi'] - st['lo']:.1f}pt)", planned)
                    elif cur >= so + timedelta(minutes=ENTRY_UNTIL):
                        st["done"] = "3시간 안에 돌파 없음"
            save_state(bot.state_path, st)
            time.sleep(0.2 if st.get("done") is None else POLL_SEC)


if __name__ == "__main__":
    main()
