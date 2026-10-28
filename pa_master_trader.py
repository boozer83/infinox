"""데모 계좌 관찰용 자동매매: XAUUSD 1시간봉 PA Master Grade-A (트레이딩뷰 지표, pine/pa_master_grade_a.pine)

규칙 (backtests/pa_master.py의 signals를 그대로 씀, 2026-10-28 추가):
- 1시간봉이 완성될 때마다 판단
- 신호: 장악형 / 핀바 / 샛별·석별 중 Grade-A (EMA21·200 가치 구간 또는 BOS 방향). 거래량 조건은 끔
- 진입: 바로 시장가 (= 다음 봉 시가). 손절 = 패턴 끝 (매수 = 패턴 저가, 매도 = 패턴 고가)
- 익절: 신호 봉 종가 ± 3 x (종가 - 패턴 끝) (3R). 둘 다 주문과 함께 서버에 설정, 그 밖의 청산 없음
- 수량: 손절에 걸리면 잔고의 RISK_PCT(0.5%)를 잃도록 (0.01 단위 내림). 포지션은 한 번에 1개
- 지금 가격이 이미 손절가 너머면 진입 안 함
백테스트 (실제 틱 체결, 2022-01 ~ 2026-10): 358건 승률 32.4% 거래당 +0.27R 합계 +98R 최대 낙폭 15.4R, 5년 모두 플러스
  GPT·tv-indicators 브랜치도 따로 검증해서 비슷 (+89R / +92R). 약점: 1시간봉에서만 됨, 승률 낮고 10연패 있었음
  SAR 단타와 일별 손익 상관 0.01, 월별 -0.07

안전장치: 데모 계좌 전용 (실계좌 옵션 없음). 매직 넘버 20261028 포지션만 관리, 증거금 부족 시 진입 안 함
기록 (logs/pa_log/): trades.csv, events.log
텔레그램: 다른 봇과 같은 봇, 메시지 앞에 [데모 · PA 1시간]

사용법:
  python pa_master_trader.py            # 실행 (Ctrl+C로 종료)
  python pa_master_trader.py --dry-run  # 주문 없이 신호만 계산해서 출력 (텔레그램 전송 안 함)
"""
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np

from backtests.pa_master import signals
from common.trader import Trader, append_csv, side_name
from config import CONTRACT, SYMBOL

# 전략
TIMEFRAME = mt5.TIMEFRAME_H1
R_MULT = 3.0
USE_VOL = False
# 수량
RISK_PCT = 0.005
MAX_LOT = 2.0
# 주문
MAGIC = 20261028
DEVIATION = 50
POLL_SEC = 2.0
BARS = 600                # EMA200 + 피벗이 안정되게

LOG_DIR = Path(__file__).resolve().parent / "logs" / "pa_log"
TRADES_CSV = LOG_DIR / "trades.csv"
FIELDS = ["signal_bar", "pattern", "side", "lot", "signal_price", "pattern_end", "order_time", "fill_price", "sl", "tp",
          "risk_usd", "risk_pct", "spread", "ticket", "close_time", "close_price", "profit", "result"]

log = logging.getLogger("pa")
bot = None         # Trader, main()에서 설정

STRATEGY = "PA 1시간"
STRATEGY_DESC = (f"1시간봉 Grade-A 캔들 패턴 (장악형·핀바·샛별, EMA 가치 구간 또는 BOS), 손절 패턴 끝, 익절 {R_MULT:g}R, "
                 f"거래당 위험 {RISK_PCT:.1%}")


def last_signal(df):
    """마지막 완성 봉의 신호 (방향, 패턴 끝, 종가, 패턴 이름) 또는 None"""
    b = df.rename(columns={"tick_volume": "volume"})[["time", "open", "high", "low", "close", "volume"]]
    sig = [s for s in signals(b, use_vol=USE_VOL) if s[0] == len(b) - 1]
    if not sig:
        return None
    _, side, end, close, kind, _ = sig[0]
    return side, float(end), float(close), str(kind)


def lot_for(sl_dist):
    info, acc = mt5.symbol_info(SYMBOL), mt5.account_info()
    raw = acc.balance * RISK_PCT / (sl_dist * CONTRACT)
    lot = np.floor(raw / info.volume_step) * info.volume_step
    forced = lot < info.volume_min
    lot = min(max(lot, info.volume_min), MAX_LOT, info.volume_max)
    return round(lot, 2), forced


def place(side, end, close, kind, signal_bar):
    tick = mt5.symbol_info_tick(SYMBOL)
    px = tick.ask if side == 1 else tick.bid
    tp = close + R_MULT * (close - end)                            # 신호 봉 종가 ± 3R (매도는 end > close)
    sl_dist, tp_dist = (px - end) * side, (tp - px) * side
    if sl_dist <= 0 or tp_dist <= 0:
        log.info(f"지금 가격 {px}가 이미 손절가 {end} 또는 익절가 {tp:.2f} 너머라 진입 안 함")
        return None
    lot, forced = lot_for(sl_dist)
    f = bot.open(side, lot, f"PA {kind} {R_MULT:g}R", sl_dist=sl_dist, tp_dist=tp_dist)
    if f is None:
        return None
    fill, sl = f.fill, f.sl
    risk_usd = abs(fill - sl) * lot * CONTRACT
    row = {
        "signal_bar": signal_bar, "pattern": kind, "side": "BUY" if side == 1 else "SELL", "lot": lot,
        "signal_price": close, "pattern_end": end, "order_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fill_price": fill, "sl": sl, "tp": f.tp, "risk_usd": round(risk_usd, 2),
        "risk_pct": round(risk_usd / mt5.account_info().balance * 100, 2),
        "spread": round(f.tick.ask - f.tick.bid, 3), "ticket": f.res.order,
    }
    log.info(f"진입 {row['side']} {lot}lot @ {fill} ({kind}, 신호봉 종가 {close}), SL {sl} TP {f.tp} "
             f"(위험 {risk_usd:.2f}$ = {row['risk_pct']}%)")
    bot.send(f"{'🟢 매수' if side == 1 else '🔴 매도'} 진입 {lot} lot ({kind})\n전략: {STRATEGY_DESC}\n"
             f"가격 {fill} | 손절 {sl} | 익절 {f.tp}\n손절 시 손실 약 {risk_usd:,.0f}$ ({row['risk_pct']}%)"
             + ("\n⚠️ 최소 수량이라 목표 위험보다 큼" if forced else "")
             + f"\n{bot.account_line()}\n{bot.summary()}")
    return row


def handle_closed(open_row):
    """보유하던 포지션이 닫혔으면 기록/알림하고 True"""
    if any(p.ticket == open_row["ticket"] for p in bot.positions()):
        return False
    c = bot.find_close(open_row["ticket"])
    if not c:
        return False
    open_row.update(c)
    append_csv(TRADES_CSV, FIELDS, open_row)
    profit = float(c["profit"])
    bal = mt5.account_info().balance
    pct = profit / (bal - profit) * 100 if bal - profit > 0 else 0.0
    icon = "✅" if profit > 0 else "❌"
    log.info(f"청산 {c['result']} @ {c['close_price']} 손익 {profit}")
    bot.send(f"{icon} {c['result']} {open_row.get('side', '')} {open_row.get('lot', '')} lot ({open_row.get('pattern', '')})\n"
             f"{open_row.get('fill_price', '')} → {c['close_price']}\n손익 {profit:+,.2f}$ ({pct:+.2f}%)\n"
             f"{bot.account_line()}\n{bot.summary()}")
    return True


def main():
    global bot
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="주문 없이 신호만 계산")
    args = ap.parse_args()
    bot = Trader(STRATEGY, SYMBOL, MAGIC, DEVIATION, LOG_DIR, "pa")
    acc = bot.start(args.dry_run)
    log.info(f"시작: 계좌 {acc.login} ({acc.server}, {bot.kind}) 잔고 {acc.balance} {acc.currency}, {SYMBOL} H1 "
             f"PA Master Grade-A, 익절 {R_MULT:g}R, 거래량 조건 {'켬' if USE_VOL else '끔'}, 위험 {RISK_PCT:.1%}"
             f"{' [DRY-RUN]' if args.dry_run else ''}, 텔레그램 {'켜짐' if bot.tg.enabled else '꺼짐'}")
    bot.send(f"▶️ PA 1시간 시작 ({bot.kind} {acc.login})\n전략: {STRATEGY_DESC}\n{bot.account_line()}")
    if not args.dry_run:
        bot.warn_algo_off()

    open_row = bot.adopt_position(with_tp=True)
    last_bar = None
    with bot.running("⏸ PA 1시간 종료 (열린 포지션은 서버 손절·익절로 보호됨)"):
        while True:
            bot.keep_alive()
            if open_row and handle_closed(open_row):
                open_row = None
            df = bot.rates(TIMEFRAME, BARS, min_bars=300)
            if df is not None and df.time.iat[-1] != last_bar:
                last_bar = df.time.iat[-1]
                bar = df.time.iat[-1]
                s = last_signal(df)
                log.info(f"1시간봉 {bar:%m-%d %H:%M} 종가 {df.close.iat[-1]:.2f}"
                         + (f" -> 신호 {side_name(s[0])} ({s[3]}, 패턴 끝 {s[1]:.2f})" if s else ""))
                if s and not args.dry_run:
                    if bot.positions():
                        log.info("포지션 보유 중이라 신호 무시")
                    else:
                        open_row = place(*s, bar.strftime("%Y-%m-%d %H:%M"))
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
