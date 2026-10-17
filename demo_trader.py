"""데모 계좌 관찰용 자동매매: XAUUSD 5분봉 SAR + RSI50 + H1 SAR + 일일 마틴

진입 규칙 (backtests/filters.py의 "RSI50+H1 SAR"와 동일):
- 5분봉 SAR(0.02, 0.2)가 전환된 봉이 완성되면 바로 시장가 진입
  매수: RSI(14) > 50 이고 마지막 완성 1시간봉 SAR이 상승
  매도: RSI(14) < 50 이고 마지막 완성 1시간봉 SAR이 하락
- 포지션은 한 번에 1개, 포지션이 있는 동안 새 신호는 무시
- SL 7.5 / TP 3 달러를 주문과 함께 서버에 설정 (스크립트가 꺼져도 SL/TP는 작동)

마틴 규칙 (backtests/daily_martin.py와 동일):
- 첫 진입 0.03 lot. 손절로 끝나면 다음 진입 수량 x2, 익절이면 0.03으로 복귀
  단계 0~5: 0.03 / 0.06 / 0.12 / 0.24 / 0.48 / 0.96
- 5단계(0.96)에서도 손절이면 그날은 신규 진입 정지
- 서버시간 날짜가 바뀌면 단계 0부터 다시 시작
- 단계는 오늘 이 스크립트가 청산한 거래내역(MT5 서버 기록)으로 계산 -> 스크립트를 껐다 켜도 이어짐

주말 보호:
- 금요일 FRIDAY_NO_ENTRY 이후 신규 진입 안 함, FRIDAY_CLOSE 이후 보유 포지션 시장가 청산
  (주말 갭은 서버 SL로도 막을 수 없어서)

안전장치:
- 데모 계좌가 아니면 시작하지 않고, 실행 중 계좌가 바뀌어도 즉시 멈춤
- 이 스크립트의 매직 넘버가 붙은 포지션만 확인/관리
- 증거금이 부족하면 진입하지 않음

기록 (demo_log/ 폴더): trades.csv (거래별 신호/체결/슬리피지/결과), events.log (실행 로그)

텔레그램 (telegram.json 설정이 있을 때, 설정법은 telegram_notify.py 참고):
- 진입/청산 때마다 거래 내용과 오늘/이번주/이번달 수익률
- 일일(월~금 23:58 서버시간), 주간(금요일), 월간(마지막 거래일) 리포트
- 시작/종료, 마틴 정지, 금요일 청산, 주문 실패

사용법:
  python demo_trader.py            # 실행 (Ctrl+C로 종료)
  python demo_trader.py --dry-run  # 주문 없이 신호만 계산해서 출력 (텔레그램 전송 안 함)
"""
import argparse
import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from config import SYMBOL, mt5_init
from indicators import psar, rsi
from reports import ReportScheduler, ServerClock, summary_line
from telegram_notify import Notifier

# 전략
SL_USD = 7.5
TP_USD = 3.0
# 마틴
BASE_LOT = 0.03
MARTIN_MULT = 2
MARTIN_MAX_LEVEL = 5
# 주말 보호 (서버시간, 금요일)
FRIDAY_NO_ENTRY = "22:30"
FRIDAY_CLOSE = "23:30"
# 주문
MAGIC = 20261017
DEVIATION = 30            # 허용 슬리피지 (포인트, 0.30달러)
POLL_SEC = 1.0
M5_BARS = 600
H1_BARS = 400

LOG_DIR = Path(__file__).resolve().parent / "demo_log"
TRADES_CSV = LOG_DIR / "trades.csv"
STATE_JSON = LOG_DIR / "state.json"
FIELDS = ["signal_bar", "side", "level", "lot", "signal_price", "order_time", "fill_price", "slippage", "spread",
          "sl", "tp", "ticket", "close_time", "close_price", "profit", "result"]

log = logging.getLogger("demo")
tg = None          # Notifier, main()에서 설정
clock = None       # ServerClock, main()에서 설정


STRATEGY = "마틴"
STRATEGY_DESC = "5분봉 SAR 전환 + RSI50 + 1시간봉 SAR, 익절 3$ / 손절 7.5$, 일일 마틴 5단계"


def send(text):
    """텔레그램 전송 (앞에 전략 이름 표시)"""
    tg.send(f"[{STRATEGY}] {text}")


def account_line():
    a = mt5.account_info()
    return f"잔고 {a.balance:,.2f} | 평가 {a.equity:,.2f} {a.currency}"


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(LOG_DIR / "events.log", encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def connect():
    if not mt5_init():
        raise SystemExit(f"MT5 연결 실패: {mt5.last_error()} (MT5를 켜고 로그인했는지, 경로 설정은 README 참고)")
    check_demo()
    if not mt5.symbol_select(SYMBOL, True):
        raise SystemExit(f"{SYMBOL} 종목을 찾을 수 없습니다")


def check_demo():
    acc = mt5.account_info()
    if acc is None:
        raise SystemExit("로그인된 계좌가 없습니다")
    if acc.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
        mt5.shutdown()
        raise SystemExit(f"데모 계좌가 아닙니다 ({acc.login}, {acc.server}). 안전을 위해 종료합니다.")
    return acc


def rates(tf, n):
    r = mt5.copy_rates_from_pos(SYMBOL, tf, 1, n)   # 1부터 = 진행 중인 봉 제외, 완성된 봉만
    if r is None or len(r) < 50:
        return None
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def compute_signal(m5, h1):
    """마지막 완성 5분봉 기준 신호: (방향 or 0, 봉 시각, 종가, 설명)"""
    up = psar(m5.high.values, m5.low.values)
    r = rsi(m5.close).values
    bar_close_time = m5.time.iat[-1] + pd.Timedelta(minutes=5)
    h1_done = h1[h1.time + pd.Timedelta(hours=1) <= bar_close_time]
    hup = psar(h1_done.high.values, h1_done.low.values)
    hdir = 1 if hup[-1] else -1
    side = 0
    if up[-1] != up[-2]:
        s = 1 if up[-1] else -1
        if (r[-1] - 50) * s > 0 and hdir == s:
            side = s
    desc = (f"SAR {'상승' if up[-1] else '하락'}{'(전환)' if up[-1] != up[-2] else ''} "
            f"RSI {r[-1]:.1f} H1 SAR {'상승' if hdir == 1 else '하락'}")
    return side, m5.time.iat[-1], float(m5.close.iat[-1]), desc


def martin_state(now):
    """오늘(서버 날짜) 이 스크립트가 청산한 거래로 (현재 단계, 오늘 정지 여부) 계산"""
    day_start = int(datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp())
    deals = mt5.history_deals_get(day_start, day_start + 2 * 86400) or ()
    closes = sorted((d for d in deals if d.magic == MAGIC and d.symbol == SYMBOL and d.entry == mt5.DEAL_ENTRY_OUT),
                    key=lambda d: d.time_msc)
    level, stopped = 0, False
    for d in closes:
        if d.profit + d.commission + d.swap < 0:
            if level >= MARTIN_MAX_LEVEL:
                stopped = True
            else:
                level += 1
        else:
            level = 0
    return level, stopped


def lot_for(level):
    info = mt5.symbol_info(SYMBOL)
    lot = BASE_LOT * MARTIN_MULT ** level
    lot = round(round(lot / info.volume_step) * info.volume_step, 2)
    return min(max(lot, info.volume_min), info.volume_max)


def my_positions():
    return [p for p in (mt5.positions_get(symbol=SYMBOL) or ()) if p.magic == MAGIC]


def filling_mode():
    # 파이썬 패키지에 SYMBOL_FILLING_* 상수가 없어서 MQL5 값 사용 (FOK = 1, IOC = 2)
    fm = mt5.symbol_info(SYMBOL).filling_mode
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def place(side, lot, level, signal_bar, signal_price):
    info, tick = mt5.symbol_info(SYMBOL), mt5.symbol_info_tick(SYMBOL)
    otype = mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL
    price = tick.ask if side == 1 else tick.bid
    margin = mt5.order_calc_margin(otype, SYMBOL, lot, price)
    free = mt5.account_info().margin_free
    if margin is None or margin > free * 0.9:
        log.warning(f"증거금 부족으로 진입 안 함: 필요 {margin}, 여유 {free:.2f} ({lot} lot)")
        send(f"⚠️ 증거금 부족으로 진입 안 함\n필요 {margin} / 여유 {free:,.2f} ({lot} lot)")
        return None
    sl = round(price - SL_USD * side, info.digits)
    tp = round(price + TP_USD * side, info.digits)
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": lot, "type": otype,
        "price": price, "sl": sl, "tp": tp, "deviation": DEVIATION, "magic": MAGIC,
        "comment": f"SAR RSI50 H1 L{level}", "type_time": mt5.ORDER_TIME_GTC, "type_filling": filling_mode(),
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        err = f"{res.retcode if res else mt5.last_error()} {res.comment if res else ''}"
        log.error(f"주문 실패: {err}")
        send(f"❗ 주문 실패 ({'매수' if side == 1 else '매도'} {lot} lot): {err}")
        return None
    fill = res.price or price
    spread = tick.ask - tick.bid
    # 슬리피지: 신호봉 종가(Bid) 대비 얼마나 불리하게 체결됐는지 (스프레드는 따로 기록하므로 제외, +가 불리)
    bid_equiv = fill - spread if side == 1 else fill
    row = {
        "signal_bar": signal_bar, "side": "BUY" if side == 1 else "SELL", "level": level, "lot": lot,
        "signal_price": signal_price, "order_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fill_price": fill, "slippage": round((bid_equiv - signal_price) * side, 3),
        "spread": round(spread, 3), "sl": sl, "tp": tp, "ticket": res.order,
    }
    log.info(f"진입 {row['side']} {lot}lot (마틴 {level}단계) @ {fill} "
             f"(신호봉 종가 {signal_price}, SL {sl}, TP {tp}, 스프레드 {row['spread']})")
    send(f"{'🟢 매수' if side == 1 else '🔴 매도'} 진입 {lot} lot (마틴 {level}단계)\n"
            f"전략: {STRATEGY_DESC}\n"
            f"가격 {fill} | SL {sl} | TP {tp}\n"
            f"최대 손실 약 {SL_USD * lot * 100:,.0f}$ / 목표 이익 약 {TP_USD * lot * 100:,.0f}$\n"
            f"{account_line()}\n{summary_line(clock.now(), MAGIC, SYMBOL)}")
    return row


def close_position(p):
    tick = mt5.symbol_info_tick(SYMBOL)
    is_buy = p.type == mt5.POSITION_TYPE_BUY
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": p.volume, "position": p.ticket,
        "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
        "price": tick.bid if is_buy else tick.ask, "deviation": DEVIATION, "magic": MAGIC,
        "comment": "friday close", "type_time": mt5.ORDER_TIME_GTC, "type_filling": filling_mode(),
    }
    res = mt5.order_send(req)
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    log.info(f"금요일 청산 {'완료' if ok else '실패'}: 티켓 {p.ticket} {res.retcode if res else mt5.last_error()}")
    if not ok:
        send(f"❗ 금요일 청산 실패: 티켓 {p.ticket} {res.retcode if res else mt5.last_error()}")


def find_close(ticket):
    deals = mt5.history_deals_get(position=ticket)
    if not deals:
        return None
    out = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
    if not out:
        return None
    d = out[-1]
    profit = sum(x.profit + x.commission + x.swap for x in deals)
    reason = {mt5.DEAL_REASON_TP: "TP", mt5.DEAL_REASON_SL: "SL"}.get(d.reason, "CLOSE")
    if d.comment == "friday close":
        reason = "FRIDAY"
    return {"close_time": datetime.fromtimestamp(d.time, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "close_price": d.price, "profit": round(profit, 2), "result": reason}


def append_row(row):
    new = not TRADES_CSV.exists()
    with open(TRADES_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def is_after(now, hhmm):
    h, m = map(int, hhmm.split(":"))
    return (now.hour, now.minute) >= (h, m)


def notify_close(row):
    profit = float(row["profit"])
    bal = mt5.account_info().balance
    pct = profit / (bal - profit) * 100 if bal - profit > 0 else 0.0
    level, stopped = martin_state(clock.now())
    icon = {"TP": "✅ 익절", "SL": "❌ 손절", "FRIDAY": "⏹ 금요일 청산"}.get(row["result"], "⏹ 청산")
    nxt = "오늘 거래 정지 (5단계 손절)" if stopped else f"다음 진입 마틴 {level}단계 ({lot_for(level)} lot)"
    log.info(f"청산 {row['result']} @ {row['close_price']} 손익 {profit} -> {nxt}")
    send(f"{icon} {row.get('side', '')} {row.get('lot', '')} lot\n"
            f"{row.get('fill_price', '')} → {row['close_price']}\n"
            f"손익 {profit:+,.2f}$ ({pct:+.2f}%)\n{nxt}\n"
            f"{account_line()}\n{summary_line(clock.now(), MAGIC, SYMBOL)}")
    if stopped:
        send("⛔ 5단계(최대 마틴)에서 손절 -> 오늘은 신규 진입을 멈춥니다. 내일 0단계부터 다시 시작합니다.")


def main():
    global tg, clock
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="주문 없이 신호만 계산")
    args = ap.parse_args()
    setup_logging()
    connect()
    acc = check_demo()
    tg = Notifier()
    if args.dry_run:
        tg.enabled = False
    clock = ServerClock(SYMBOL)
    clock.update()
    scheduler = ReportScheduler(STATE_JSON, MAGIC, SYMBOL, send)
    lots = " / ".join(f"{BASE_LOT * MARTIN_MULT ** i:g}" for i in range(MARTIN_MAX_LEVEL + 1))
    log.info(f"시작: 계좌 {acc.login} ({acc.server}, 데모) 잔고 {acc.balance} {acc.currency}, {SYMBOL} "
             f"SL {SL_USD} TP {TP_USD}, 마틴 {lots} lot{' [DRY-RUN]' if args.dry_run else ''}"
             f", 텔레그램 {'켜짐' if tg.enabled else '꺼짐'}")
    send(f"▶️ 자동매매 시작 (데모 {acc.login})\n전략: {STRATEGY_DESC}\n마틴 {lots} lot\n"
            f"{account_line()}\n{summary_line(clock.now(), MAGIC, SYMBOL)}")
    if not args.dry_run and not mt5.terminal_info().trade_allowed:
        log.warning("MT5의 'Algo Trading'이 꺼져 있습니다. 켜야 주문이 들어갑니다.")
        send("⚠️ MT5의 'Algo Trading'이 꺼져 있습니다. 켜야 주문이 들어갑니다.")

    # 이전 실행에서 열어 둔 포지션이 있으면 이어서 기록
    open_row = None
    for p in my_positions():
        open_row = {"ticket": p.ticket, "side": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                    "lot": p.volume, "fill_price": p.price_open, "sl": p.sl, "tp": p.tp}
        log.info(f"기존 포지션 이어받음: 티켓 {p.ticket} {open_row['side']} {p.volume}lot @ {p.price_open}")

    last_bar, last_close_try = None, 0.0
    try:
        while True:
            if not mt5.terminal_info():
                log.warning("터미널 연결 끊김, 재연결 시도")
                mt5.shutdown()
                time.sleep(5)
                connect()
            check_demo()
            clock.update()
            now = clock.now()

            # 진입했던 포지션이 닫혔는지 확인
            if open_row and not any(p.ticket == open_row["ticket"] for p in my_positions()):
                c = find_close(open_row["ticket"])
                if c:
                    open_row.update(c)
                    append_row(open_row)
                    notify_close(open_row)
                    open_row = None

            if not args.dry_run:
                scheduler.check(now)

            friday = now.weekday() == 4
            # 청산이 실패해도(장 마감 등) 1분에 한 번만 다시 시도
            if friday and is_after(now, FRIDAY_CLOSE) and not args.dry_run and time.time() - last_close_try > 60:
                for p in my_positions():
                    close_position(p)
                    last_close_try = time.time()

            m5 = rates(mt5.TIMEFRAME_M5, M5_BARS)
            if m5 is not None and m5.time.iat[-1] != last_bar:
                last_bar = m5.time.iat[-1]
                h1 = rates(mt5.TIMEFRAME_H1, H1_BARS)
                side, bar, close, desc = compute_signal(m5, h1)
                log.info(f"봉 {bar:%m-%d %H:%M} 종가 {close}: {desc}"
                         + (f" -> 신호 {'매수' if side == 1 else '매도'}" if side else ""))
                if side and not args.dry_run:
                    level, stopped = martin_state(now)
                    if my_positions():
                        log.info("포지션 보유 중이라 신호 무시")
                    elif stopped:
                        log.info("오늘은 5단계 손절로 거래 정지 상태라 신호 무시")
                    elif friday and is_after(now, FRIDAY_NO_ENTRY):
                        log.info("금요일 마감 전이라 신규 진입 안 함")
                    else:
                        open_row = place(side, lot_for(level), level, bar.strftime("%Y-%m-%d %H:%M"), close)
            time.sleep(POLL_SEC)
    except KeyboardInterrupt:
        log.info("사용자 종료 (열린 포지션은 서버 SL/TP로 계속 관리됨)")
        send("⏸ 자동매매 종료 (열린 포지션은 서버 SL/TP로 계속 관리됩니다)")
    except BaseException as e:
        log.exception("오류로 종료")
        send(f"❗ 자동매매가 오류로 멈췄습니다: {e}")
        raise
    finally:
        tg.flush()
        mt5.shutdown()


if __name__ == "__main__":
    main()
