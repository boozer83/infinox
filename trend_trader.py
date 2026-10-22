"""데모 계좌 관찰용 자동매매: XAUUSD 4시간봉 추세추종 (돈치안 돌파)

규칙 (backtests/strategies.py의 추세추종, backtests/trend_deep.py에서 검증한 H4 N25 k3):
- 4시간봉이 완성될 때마다 판단
- 진입: 종가가 직전 25봉 최고가 위면 매수, 최저가 아래면 매도 (바로 시장가)
- 청산: 매수는 종가가 직전 12봉 최저가 아래, 매도는 직전 12봉 최고가 위면 바로 시장가 청산
        (청산한 봉에서는 새로 진입하지 않음 - 백테스트와 동일)
- 손절: 진입가 ± 3 x ATR(14, 4시간봉)를 주문과 함께 서버에 설정. 익절 없음
- 수량: 손절에 걸리면 잔고의 RISK_PCT(2%)를 잃도록 계산 (0.01 단위 내림)
- 포지션은 한 번에 1개, 주말에도 보유 (백테스트와 동일)
- 구름 필터 (2026-10-21 추가): 돌파 방향이 4시간봉 일목 구름(9/26/52, 지금 봉 위치에 그려진 구름)과 같을 때만 진입
  (매수는 신호 봉 종가가 구름 위, 매도는 아래). 아니면 진입하지 않고 "그림자 포지션"으로 들고 있는 것처럼 기다림:
  그 사이 새 진입 안 함, 청산 신호나 손절가(진입가 ± 3 ATR)에 닿으면 그림자 끝 -> 다시 신호를 봄
  (그냥 건너뛰면 추세 중간 늦은 진입이 생겨서 나빠짐. backtests/ichimoku_others.py, 1만 달러 2% 위험 2022-01~2026-10:
   156건 21,553$ 낙폭 11.3% PF 1.76 -> 107건 21,839$ 낙폭 7.7% PF 2.34. 구름 반대 49건은 평균 0.00R)
  기록: <로그 폴더>/shadow.json (진행 중 그림자), shadow_trades.csv (끝난 그림자의 가상 결과와 R)

안전장치: 기본은 데모 계좌 전용. 실계좌는 --live 옵션을 줄 때만 실행. 실행 중 로그인 계좌가 바뀌면 즉시 멈춤
          매직 넘버 20261018 포지션만 관리, 증거금 부족 시 진입 안 함
          demo_trader.py(매직 20261017), asia_breakout_trader.py(20261019)와 같은 계좌에서 같이 돌려도
          서로의 포지션을 건드리지 않음

기록 (trend_log/ 폴더, 실계좌는 trend_live_log/): trades.csv, events.log, state.json
텔레그램: demo_trader.py와 같은 봇, 메시지 앞에 [H4 추세] 표시

사용법:
  python trend_trader.py            # 실행 (Ctrl+C로 종료)
  python trend_trader.py --dry-run  # 주문 없이 신호만 계산해서 출력 (텔레그램 전송 안 함)
  python trend_trader.py --live     # 실계좌에서 실행 (데모에서 충분히 확인한 뒤에만)
"""
import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from config import CONTRACT, SYMBOL
from demo_trader import account_kind, check_demo, connect, filling_mode, find_close, rates
from indicators import atr
from reports import ServerClock, summary_line
from telegram_notify import Notifier

# 전략
TIMEFRAME = mt5.TIMEFRAME_H4
N_ENTRY = 25
N_EXIT = N_ENTRY // 2
SL_ATR = 3.0
CLOUD_FILTER = True       # 4시간봉 구름 방향과 같을 때만 진입 (반대면 그림자 포지션)
# 수량
RISK_PCT = 0.02
MAX_LOT = 5.0
# 주문
MAGIC = 20261018
DEVIATION = 50
POLL_SEC = 2.0
BARS = 300

LOG_DIR = Path(__file__).resolve().parent / "trend_log"
TRADES_CSV = LOG_DIR / "trades.csv"
STATE_JSON = LOG_DIR / "state.json"
SHADOW_JSON = LOG_DIR / "shadow.json"
SHADOW_CSV = LOG_DIR / "shadow_trades.csv"
FIELDS = ["signal_bar", "side", "lot", "signal_price", "order_time", "fill_price", "sl", "risk_usd", "risk_pct",
          "spread", "ticket", "close_time", "close_price", "profit", "result"]

log = logging.getLogger("trend")
tg = None
clock = None


STRATEGY = "H4 추세"
STRATEGY_DESC = (f"4시간봉 {N_ENTRY}봉 돌파 추세추종, {N_EXIT}봉 반대 돌파 청산, 손절 {SL_ATR:g} ATR, 거래당 위험 {RISK_PCT:.0%}"
                 + (", 4시간봉 구름 방향과 같을 때만 진입" if CLOUD_FILTER else ""))


def send(text):
    """텔레그램 전송 (앞에 전략 이름 표시)"""
    tg.send(f"[{STRATEGY}] {text}")


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(LOG_DIR / "events.log", encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def use_live_paths():
    """실계좌 실행은 기록을 trend_live_log/에 따로 남기고, 텔레그램에 실계좌로 표시"""
    global LOG_DIR, TRADES_CSV, STATE_JSON, SHADOW_JSON, SHADOW_CSV, STRATEGY
    LOG_DIR = LOG_DIR.parent / "trend_live_log"
    TRADES_CSV, STATE_JSON, SHADOW_JSON = LOG_DIR / "trades.csv", LOG_DIR / "state.json", LOG_DIR / "shadow.json"
    SHADOW_CSV = LOG_DIR / "shadow_trades.csv"
    STRATEGY = "H4 추세·실계좌"


def account_line():
    a = mt5.account_info()
    return f"잔고 {a.balance:,.2f} | 평가 {a.equity:,.2f} {a.currency}"


def compute(df):
    """완성된 4시간봉 기준: 진입 방향, 매수/매도 청산 신호, ATR, 설명"""
    c = df.close.iat[-1]
    hh = df.high.iloc[-1 - N_ENTRY:-1].max()
    ll = df.low.iloc[-1 - N_ENTRY:-1].min()
    xh = df.high.iloc[-1 - N_EXIT:-1].max()
    xl = df.low.iloc[-1 - N_EXIT:-1].min()
    a = atr(df).iat[-1]
    entry = 1 if c > hh else -1 if c < ll else 0
    desc = (f"종가 {c:.2f} | 진입선 {ll:.2f}~{hh:.2f} | 청산선 {xl:.2f}~{xh:.2f} | ATR {a:.2f}")
    return entry, c < xl, c > xh, float(a), desc


def cloud(df):
    """마지막 완성 봉 위치에 그려진 일목 구름 (위, 아래). 선행1·2를 26봉 민 값"""
    mid = lambda n: (df.high.rolling(n).max() + df.low.rolling(n).min()) / 2
    a = ((mid(9) + mid(26)) / 2).shift(26).iat[-1]
    b = mid(52).shift(26).iat[-1]
    return float(max(a, b)), float(min(a, b))


def load_shadow():
    try:
        return json.loads(SHADOW_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def save_shadow(sh):
    if sh is None:
        SHADOW_JSON.unlink(missing_ok=True)
    else:
        SHADOW_JSON.write_text(json.dumps(sh, ensure_ascii=False), encoding="utf-8")


def end_shadow(sh, why, px):
    """그림자 포지션 끝: 가상 결과를 shadow_trades.csv에 남기고 알림. 반환 None"""
    risk = abs(sh["entry"] - sh["sl"])
    r = (px - sh["entry"]) * sh["side"] / risk if risk else 0.0
    row = {"bar": sh["bar"], "side": "BUY" if sh["side"] == 1 else "SELL", "entry": sh["entry"], "sl": sh["sl"],
           "cloud": sh.get("cloud", ""), "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "exit": px,
           "result": why, "r": round(r, 3)}
    new = not SHADOW_CSV.exists()
    with open(SHADOW_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    log.info(f"그림자 {row['side']} 끝 ({why}) {sh['entry']} → {px}, 가상 {r:+.2f}R -> 다음 신호부터 진입 가능")
    send(f"👻 그림자 {row['side']} 끝 ({why})\n{sh['entry']} → {px}, 가상 결과 {r:+.2f}R "
         f"({'건너뛴 게 손해' if r > 0 else '건너뛴 게 이득'})\n다음 신호부터 다시 진입합니다")
    save_shadow(None)
    return None


def shadow_sl_hit(sh):
    """그림자 포지션 손절가에 닿았나 (매수는 Bid, 매도는 Ask)"""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return False
    return tick.bid <= sh["sl"] if sh["side"] == 1 else tick.ask >= sh["sl"]


def my_positions():
    return [p for p in (mt5.positions_get(symbol=SYMBOL) or ()) if p.magic == MAGIC]


def lot_for(sl_dist):
    info, acc = mt5.symbol_info(SYMBOL), mt5.account_info()
    raw = acc.balance * RISK_PCT / (sl_dist * CONTRACT)
    lot = np.floor(raw / info.volume_step) * info.volume_step
    forced = lot < info.volume_min
    lot = min(max(lot, info.volume_min), MAX_LOT, info.volume_max)
    return round(lot, 2), forced


def place(side, a, signal_bar, signal_price):
    info, tick = mt5.symbol_info(SYMBOL), mt5.symbol_info_tick(SYMBOL)
    otype = mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL
    price = tick.ask if side == 1 else tick.bid
    sl_dist = SL_ATR * a
    lot, forced = lot_for(sl_dist)
    margin = mt5.order_calc_margin(otype, SYMBOL, lot, price)
    acc = mt5.account_info()
    if margin is None or margin > acc.margin_free * 0.9:
        log.warning(f"증거금 부족으로 진입 안 함: 필요 {margin}, 여유 {acc.margin_free:.2f} ({lot} lot)")
        send(f"⚠️ 증거금 부족으로 진입 안 함: 필요 {margin} / 여유 {acc.margin_free:,.2f} ({lot} lot)")
        return None
    sl = round(price - side * sl_dist, info.digits)
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": lot, "type": otype,
        "price": price, "sl": sl, "tp": 0.0, "deviation": DEVIATION, "magic": MAGIC,
        "comment": f"trend H4 N{N_ENTRY}", "type_time": mt5.ORDER_TIME_GTC, "type_filling": filling_mode(),
    }
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        err = f"{res.retcode if res else mt5.last_error()} {res.comment if res else ''}"
        log.error(f"주문 실패: {err}")
        send(f"❗ 주문 실패 ({'매수' if side == 1 else '매도'} {lot} lot): {err}")
        return None
    fill = res.price or price
    risk_usd = abs(fill - sl) * lot * CONTRACT
    row = {
        "signal_bar": signal_bar, "side": "BUY" if side == 1 else "SELL", "lot": lot, "signal_price": signal_price,
        "order_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "fill_price": fill, "sl": sl,
        "risk_usd": round(risk_usd, 2), "risk_pct": round(risk_usd / acc.balance * 100, 2),
        "spread": round(tick.ask - tick.bid, 3), "ticket": res.order,
    }
    log.info(f"진입 {row['side']} {lot}lot @ {fill}, SL {sl} (위험 {risk_usd:.2f}$ = {row['risk_pct']}%)")
    send(f"{'🟢 매수' if side == 1 else '🔴 매도'} 진입 {lot} lot\n전략: {STRATEGY_DESC}\n가격 {fill} | 손절 {sl} (익절 없음, 추세 끝나면 청산)\n"
         f"손절 시 손실 약 {risk_usd:,.0f}$ ({row['risk_pct']}%)"
         + ("\n⚠️ 최소 수량이라 목표 위험보다 큼" if forced else "")
         + f"\n{account_line()}\n{summary_line(clock.now(), MAGIC, SYMBOL)}")
    return row


def close_position(p, reason):
    tick = mt5.symbol_info_tick(SYMBOL)
    is_buy = p.type == mt5.POSITION_TYPE_BUY
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": p.volume, "position": p.ticket,
        "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
        "price": tick.bid if is_buy else tick.ask, "deviation": DEVIATION, "magic": MAGIC,
        "comment": reason, "type_time": mt5.ORDER_TIME_GTC, "type_filling": filling_mode(),
    }
    res = mt5.order_send(req)
    ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
    log.info(f"청산 주문 {'완료' if ok else '실패'} ({reason}): 티켓 {p.ticket} {res.retcode if res else mt5.last_error()}")
    if not ok:
        send(f"❗ 청산 실패 ({reason}): 티켓 {p.ticket} {res.retcode if res else mt5.last_error()}")
    return ok


def append_row(row):
    new = not TRADES_CSV.exists()
    with open(TRADES_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def handle_closed(open_row):
    """보유하던 포지션이 닫혔으면 기록/알림하고 True"""
    if any(p.ticket == open_row["ticket"] for p in my_positions()):
        return False
    c = find_close(open_row["ticket"])
    if not c:
        return False
    open_row.update(c)
    if c["result"] == "CLOSE":
        c["result"] = open_row["result"] = "추세종료"
    append_row(open_row)
    profit = float(c["profit"])
    bal = mt5.account_info().balance
    pct = profit / (bal - profit) * 100 if bal - profit > 0 else 0.0
    icon = "✅" if profit > 0 else "❌"
    log.info(f"청산 {c['result']} @ {c['close_price']} 손익 {profit}")
    send(f"{icon} {c['result']} {open_row.get('side', '')} {open_row.get('lot', '')} lot\n"
         f"{open_row.get('fill_price', '')} → {c['close_price']}\n손익 {profit:+,.2f}$ ({pct:+.2f}%)\n"
         f"{account_line()}\n{summary_line(clock.now(), MAGIC, SYMBOL)}")
    return True


def main():
    global tg, clock
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="주문 없이 신호만 계산")
    ap.add_argument("--live", action="store_true", help="실계좌에서도 실행 (기본은 데모 전용)")
    args = ap.parse_args()
    if args.live:
        use_live_paths()
    setup_logging()
    connect(args.live)
    acc = check_demo(args.live)
    kind = account_kind(acc)
    tg = Notifier()
    if args.dry_run:
        tg.enabled = False
    clock = ServerClock(SYMBOL)
    clock.update()
    log.info(f"시작: 계좌 {acc.login} ({acc.server}, {kind}) 잔고 {acc.balance} {acc.currency}, {SYMBOL} H4 "
             f"진입 {N_ENTRY}봉 / 청산 {N_EXIT}봉 / 손절 {SL_ATR} ATR / 위험 {RISK_PCT:.0%}"
             f"{' / 4시간봉 구름 필터 (반대면 그림자 포지션)' if CLOUD_FILTER else ''}"
             f"{' [DRY-RUN]' if args.dry_run else ''}, 텔레그램 {'켜짐' if tg.enabled else '꺼짐'}")
    send(f"▶️ 추세추종 시작 ({kind} {acc.login})\n전략: {STRATEGY_DESC}\n{account_line()}")
    if not args.dry_run and not mt5.terminal_info().trade_allowed:
        log.warning("MT5의 'Algo Trading'이 꺼져 있습니다. 켜야 주문이 들어갑니다.")
        send("⚠️ MT5의 'Algo Trading'이 꺼져 있습니다.")

    open_row = None
    for p in my_positions():
        open_row = {"ticket": p.ticket, "side": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                    "lot": p.volume, "fill_price": p.price_open, "sl": p.sl}
        log.info(f"기존 포지션 이어받음: 티켓 {p.ticket} {open_row['side']} {p.volume}lot @ {p.price_open}")

    shadow = load_shadow()
    if shadow:
        log.info(f"그림자 포지션 이어받음: {shadow}")
    last_bar = None
    try:
        while True:
            if not mt5.terminal_info():
                log.warning("터미널 연결 끊김, 재연결 시도")
                mt5.shutdown()
                time.sleep(5)
                connect(args.live)
            check_demo(args.live, acc.login)
            clock.update()
            now = clock.now()

            if open_row and handle_closed(open_row):
                open_row = None
            if shadow and shadow_sl_hit(shadow):
                shadow = end_shadow(shadow, "손절", shadow["sl"])

            df = rates(TIMEFRAME, BARS)
            if df is not None and len(df) > N_ENTRY + 20 and df.time.iat[-1] != last_bar:
                last_bar = df.time.iat[-1]
                entry, exit_long, exit_short, a, desc = compute(df)
                bar = df.time.iat[-1]
                log.info(f"4시간봉 {bar:%m-%d %H:%M}: {desc}"
                         + (f" -> 진입 신호 {'매수' if entry == 1 else '매도'}" if entry else ""))
                if not args.dry_run:
                    # 1) 청산 신호
                    exited = False
                    if shadow and ((shadow["side"] == 1 and exit_long) or (shadow["side"] == -1 and exit_short)):
                        tick = mt5.symbol_info_tick(SYMBOL)
                        shadow = end_shadow(shadow, "추세종료", tick.bid if shadow["side"] == 1 else tick.ask)
                        exited = True
                    for p in my_positions():
                        is_buy = p.type == mt5.POSITION_TYPE_BUY
                        if (is_buy and exit_long) or (not is_buy and exit_short):
                            if close_position(p, "trend exit"):
                                exited = True
                                time.sleep(1)
                                if open_row and handle_closed(open_row):
                                    open_row = None
                    # 2) 진입 신호 (포지션이 없고, 이번 봉에서 청산하지 않았을 때만)
                    if entry and not exited and not my_positions() and shadow:
                        log.info("그림자 포지션 중이라 신호 무시")
                    elif entry and not exited and not my_positions():
                        c = float(df.close.iat[-1])
                        top, bot = cloud(df)
                        if CLOUD_FILTER and not (c > top if entry == 1 else c < bot):
                            tick = mt5.symbol_info_tick(SYMBOL)
                            px = tick.ask if entry == 1 else tick.bid
                            where = "구름 안" if bot <= c <= top else "구름 반대편"
                            shadow = {"side": entry, "entry": px, "sl": round(px - entry * SL_ATR * a, 2),
                                      "bar": bar.strftime("%Y-%m-%d %H:%M"), "cloud": f"{where} {bot:.2f}~{top:.2f}"}
                            save_shadow(shadow)
                            log.info(f"종가 {c:.2f}가 {where} (구름 {bot:.2f}~{top:.2f}) -> 진입 안 함, 그림자 포지션 {shadow}")
                            send(f"👻 {'매수' if entry == 1 else '매도'} 신호지만 종가 {c:.2f}가 4시간봉 {where} (구름 {bot:.2f}~{top:.2f})\n"
                                 f"진입하지 않고, 추세가 끝나거나 손절가 {shadow['sl']}에 닿을 때까지 다음 신호를 쉽니다")
                        else:
                            open_row = place(entry, a, bar.strftime("%Y-%m-%d %H:%M"), c)
            time.sleep(POLL_SEC)
    except KeyboardInterrupt:
        log.info("사용자 종료 (열린 포지션은 서버 손절로 보호됨, 추세 청산은 다시 실행해야 작동)")
        send("⏸ 추세추종 종료 (열린 포지션은 서버 손절로 보호됨. 추세 종료 청산은 다시 실행해야 작동)")
    except BaseException as e:
        log.exception("오류로 종료")
        send(f"❗ 추세추종이 오류로 멈췄습니다: {e}")
        raise
    finally:
        tg.flush()
        mt5.shutdown()


if __name__ == "__main__":
    main()
