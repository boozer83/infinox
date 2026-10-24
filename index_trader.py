"""데모 전용 지수 장중 봇 (docs/hk_program.md 5-23, 5-24, 백테스트 backtests/index_intraday.py / index_ticks.py)

--symbol NAS100 : 첫 30분 방향. 본장(뉴욕 09:30 = 서버 16:30) 첫 30분 Bid 움직임이 ATR x FIRST_MIN 넘으면
                  그 방향으로 시장가 진입, 손절 ATR x FIRST_STOP, 마감 전 청산
--symbol GER40  : ORB 30분 매도만. Xetra 09:00(서버시간으로 바꿈) 처음 30분 Bid 저가를 마감된 1분봉 종가가 깨면 매도,
                  손절 = 범위 가운데 (Bid가 닿으면 = Ask 기준 가운데 + 스프레드로 근사), 진입은 시가 + 3시간 전까지, 마감 전 청산
ATR = 전날까지 본장 고가-저가 14일 평균 (1분봉 Bid). 휴장일·미국 조기 마감은 common/exchange_calendar.py (백테스트와 같음)
lot = 거래당 위험 RISK_PCT% (손절까지 손실 / 1 lot 손실, 0.01 단위 내림, 최소 0.01)
기록: logs/index_<종목>_log/trades.csv (요청가·체결가·스프레드·미끄러짐·보유 중 최대 역행·청산), 텔레그램

사용법: python index_trader.py --symbol NAS100        (데모 계좌, Ctrl+C로 종료)
        python index_trader.py --symbol GER40 --dry  (오늘 본장 시각·ATR만 보고 끝)
"""
import argparse
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np

from backtests.index_intraday import SESS, server
from common.exchange_calendar import de_closed, us_close_time, us_closed
from common.trader import Trader, append_csv, load_json, save_json, side_name
from config import SYMBOL as CLOCK_SYMBOL
from reports import ServerClock

MAGICS = {"NAS100": 20261025, "GER40": 20261026}
NAMES = {"NAS100": "나스닥 첫30분", "GER40": "독일 ORB 매도"}
FIRST_MIN, FIRST_STOP = 0.2, 1.0
ORB_MIN, ENTRY_UNTIL = 30, 180
ATR_DAYS = 14
RISK_PCT = 0.25
LATE_MIN = 5            # 신호 시각보다 이만큼 늦게 켜졌으면 그날은 안 들어감 (백테스트와 다른 가격으로 들어가지 않게)
CLOSE_BEFORE = 15       # 마감 이 초 전에 청산
DEVIATION = 500         # 포인트 (지수 0.01 단위 = 5포인트)
POLL_SEC = 1.0
FIELDS = ["date", "side", "signal", "atr", "lot", "entry_time", "req_price", "fill", "slip", "entry_spread", "sl",
          "mae_pts", "close_time", "close_price", "result", "pts", "profit"]
bot = None


def bounds(sym, d):
    """그날 본장 (서버시간 시작, 끝). 휴장일이면 None"""
    tz, o, c = SESS[sym]
    if (us_closed(d) if sym == "NAS100" else de_closed(d)):
        return None
    cl = us_close_time(d, c) if sym == "NAS100" else c
    return server(d, tz, o), server(d, tz, cl)


def m1(sym, t0, t1):
    """t0 <= 봉 시작 < t1 1분봉 (서버시간)"""
    r = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M1, t0.replace(tzinfo=timezone.utc), (t1 - timedelta(seconds=1)).replace(tzinfo=timezone.utc))
    if r is None or not len(r):
        return None
    r = r[r["time"] < int(t1.replace(tzinfo=timezone.utc).timestamp())]
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


def lot_for(side, price, sl_price):
    risk = mt5.account_info().balance * RISK_PCT / 100
    otype = mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL
    loss = abs(mt5.order_calc_profit(otype, bot.symbol, 1.0, price, sl_price) or 0)
    if not loss:
        return 0.01
    return max(0.01, min(1.0, math.floor(risk / loss * 100) / 100))


def enter(st, side, sl_price, signal):
    t = mt5.symbol_info_tick(bot.symbol)
    price = t.ask if side > 0 else t.bid
    lot = lot_for(side, price, sl_price)
    f = bot.open(side, lot, f"IDX {signal[:20]}", sl_dist=abs(price - sl_price))
    if f is None:
        st["done"] = "주문 실패"
        return
    st.update(ticket=f.res.order, side=side, lot=lot, req_price=f.price, fill=f.fill, sl=f.sl, signal=signal,
              entry_time=f"{bot.clock.now():%H:%M:%S}", entry_spread=round(t.ask - t.bid, 2),
              slip=round((f.fill - f.price) * side, 2), mae=0.0, done="진입")
    bot.log.info(f"진입 {side_name(side)} {lot} lot @ {f.fill} (요청 {f.price}, 손절 {f.sl}) {signal}")
    bot.send(f"{'📈' if side > 0 else '📉'} {side_name(side)} {lot} lot @ {f.fill} (요청 {f.price}, 스프레드 {t.ask - t.bid:.2f})\n"
             f"{signal}\n손절 {f.sl}, 마감 전 청산")


def finish(st):
    """포지션이 닫혔으면 기록"""
    info = bot.find_close(st["ticket"], {"IDX close": "CLOSE"})
    if info is None:
        return False
    side = st["side"]
    pts = round((info["close_price"] - st["fill"]) * side, 2)
    append_csv(bot.log_dir / "trades.csv", FIELDS, {
        "date": st["day"], "side": side_name(side), "signal": st["signal"], "atr": round(st["atr"], 1), "lot": st["lot"],
        "entry_time": st["entry_time"], "req_price": st["req_price"], "fill": st["fill"], "slip": st["slip"],
        "entry_spread": st["entry_spread"], "sl": st["sl"], "mae_pts": round(st["mae"], 2), **info, "pts": pts})
    bot.send(f"{'✅' if info['profit'] > 0 else '❌'} 청산 ({info['result']}) @ {info['close_price']}: {pts:+.2f}pt, "
             f"{info['profit']:+.2f}$ (보유 중 최대 역행 {st['mae']:.1f}pt)\n{bot.account_line()}")
    st["ticket"] = None
    return True


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
    state_json = log_dir / "state.json"
    now = bot.clock.now()
    if args.dry:
        b = bounds(sym, now.date())
        print(f"서버 {now:%Y-%m-%d %H:%M:%S}, 오늘 본장 {b[0]:%H:%M}~{b[1]:%H:%M}" if b else "오늘 휴장", f"ATR {atr_of(sym, now.date())}")
        mt5.shutdown()
        return
    desc = (f"본장 첫 30분이 ATR x {FIRST_MIN:g} 넘게 움직이면 그 방향, 손절 ATR x {FIRST_STOP:g}" if sym == "NAS100"
            else f"본장 처음 {ORB_MIN}분 저가를 1분봉 종가가 깨면 매도 (시가 + {ENTRY_UNTIL // 60}시간 전까지), 손절 범위 가운데")
    bot.send(f"▶️ 데모 시작 ({acc.login})\n전략: {desc}, 마감 전 청산, 거래당 위험 {RISK_PCT:g}%")
    st = load_json(state_json)
    with bot.running("⏸ 종료"):
        while True:
            now = bot.keep_alive()
            d = now.date()
            if st.get("day") != str(d) and not st.get("ticket"):
                st = {"day": str(d)}
            b = bounds(sym, d)
            # 남은 포지션: 기록 갱신 / 마감 전 청산
            pos = bot.positions()
            if pos:
                p = pos[0]
                t = mt5.symbol_info_tick(sym)
                adverse = (st.get("fill", p.price_open) - t.bid) if p.type == mt5.POSITION_TYPE_BUY else (t.ask - st.get("fill", p.price_open))
                st["mae"] = max(st.get("mae", 0.0), adverse)
                if b is None or now >= b[1] - timedelta(seconds=CLOSE_BEFORE) or now.date() != datetime.strptime(st["day"], "%Y-%m-%d").date():
                    bot.close(p, "IDX close", "마감 청산")
            elif st.get("ticket"):
                if finish(st):
                    save_json(state_json, st)
            if b is None or st.get("done") or now < b[0]:
                save_json(state_json, st)
                time.sleep(POLL_SEC if b and now < b[0] and (b[0] - now).total_seconds() < 120 else min(30, POLL_SEC * 10))
                continue
            so, sc = b
            if "atr" not in st:
                st["atr"] = atr_of(sym, d)
                if st["atr"] is None:
                    st["done"] = "ATR 없음"
                    bot.send("⚠️ ATR 계산할 1분봉이 부족해서 오늘은 쉽니다")
                    continue
            atr = st["atr"]
            if sym == "NAS100":
                sig_t = so + timedelta(minutes=30)
                if now >= sig_t:
                    if now > sig_t + timedelta(minutes=LATE_MIN):
                        st["done"] = "늦게 켜짐"
                    else:
                        r = m1(sym, so, sig_t)
                        if r is None or r["time"][0] > int((so + timedelta(minutes=5)).replace(tzinfo=timezone.utc).timestamp()):
                            st["done"] = "첫 30분 봉 없음"
                        else:
                            mv = float(r["close"][-1] - r["open"][0])
                            if abs(mv) < FIRST_MIN * atr:
                                st["done"] = f"움직임 {mv:+.1f} < ATR {atr:.0f} x {FIRST_MIN:g}"
                                bot.log.info(st["done"])
                            else:
                                side = 1 if mv > 0 else -1
                                t = mt5.symbol_info_tick(sym)
                                sl = (t.bid - FIRST_STOP * atr) if side > 0 else (t.ask + FIRST_STOP * atr)
                                enter(st, side, sl, f"첫 30분 {mv:+.1f}pt (ATR {atr:.0f})")
            else:
                rng_end = so + timedelta(minutes=ORB_MIN)
                if now >= so + timedelta(minutes=ENTRY_UNTIL):
                    st["done"] = "3시간 안에 돌파 없음"
                elif now >= rng_end + timedelta(minutes=1):
                    if "lo" not in st:
                        r = m1(sym, so, rng_end)
                        if r is None or len(r) < 20:
                            st["done"] = "범위 봉 부족"
                            continue
                        st["hi"], st["lo"] = float(r["high"].max()), float(r["low"].min())
                        if now > rng_end + timedelta(minutes=LATE_MIN):
                            st["late_check"] = True
                    last = m1(sym, now.replace(second=0, microsecond=0) - timedelta(minutes=1), now.replace(second=0, microsecond=0))
                    if last is not None and int(last["time"][-1]) != st.get("last_bar"):
                        st["last_bar"] = int(last["time"][-1])
                        if st.pop("late_check", False):
                            seen = m1(sym, rng_end, now.replace(second=0, microsecond=0) - timedelta(minutes=1))
                            if seen is not None and (seen["close"] < st["lo"]).any():
                                st["done"] = "늦게 켜짐 (이미 돌파)"
                                continue
                        if float(last["close"][-1]) < st["lo"]:
                            t = mt5.symbol_info_tick(sym)
                            mid = (st["hi"] + st["lo"]) / 2
                            enter(st, -1, mid + (t.ask - t.bid), f"30분 저가 {st['lo']:.1f} 깸 (범위 {st['hi'] - st['lo']:.1f}pt)")
            save_json(state_json, st)
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
