"""데모 계좌 관찰용 자동매매: 금/은 비율 되돌림 (XAUUSD만 매매, 1시간봉, 2026-10-29 추가)

규칙 (backtests/idea_screen.py gsr, backtests/idea_deep.py):
- 1시간봉이 완성될 때마다 판단. 비율 = 금 종가 / 은(XAGUSD) 종가 (은은 금 봉 시각에 맞춰 직전 값)
- z = (비율 - 최근 Z_N봉 평균) / 최근 Z_N봉 표준편차
- 포지션 없을 때: z > Z_K 면 금 매도 (금이 은보다 비쌈), z < -Z_K 면 금 매수. 바로 시장가 (= 다음 봉 시가)
- 청산: 매수는 z > 0, 매도는 z < 0 이 된 봉에서 시장가. 손절 = 진입가 ∓ 3 x ATR(14, 1시간봉 신호 봉 값), 익절 없음
- 손절에 걸린 뒤에도 z가 아직 기준 밖이면 다음 봉에 다시 진입 (백테스트와 같음). 금요일 청산 없음 (주말 보유)
- 수량: 0.01 lot 고정 (2026년 손절폭 중간값 약 60$ -> 0.01 lot도 데모 잔고의 약 5%라 관찰용 최소 수량)
백테스트 (2022-01 ~ 2026-10, 1분봉 체결·스프레드 포함): 243건 승률 40.3% PF 해마다 1.28 / 1.56 / 1.95 / 1.59 / 2.38,
  +2,703$/oz, 청산 기준 최대 낙폭 454$/oz. N100~150 x z1.5~3 8개 설정 모두 해마다 플러스 (거래당 0.2$ 비용 더 빼도)
  GPT 독립 재계산으로 243건 일치. 약점: 수익의 69%가 상위 5건, 추세 4시간 봇과 월별 손익 상관 0.72,
  보유 중간값 47시간 (스왑 미반영), 88+60개 설정을 보고 고름 -> 데모로 관찰

안전장치: 데모 계좌 전용 (실계좌 옵션 없음), MT4 따라하기 안 보냄. 매직 넘버 20261031 포지션만 관리
기록 (logs/gsr_log/): trades.csv, events.log
텔레그램: 다른 봇과 같은 봇, 메시지 앞에 [데모 · 금은비율]

사용법:
  python gsr_trader.py            # 실행 (Ctrl+C로 종료)
  python gsr_trader.py --dry-run  # 주문 없이 신호만 계산해서 출력 (텔레그램 전송 안 함)
"""
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from common.trader import Trader, append_csv, rates, side_name, side_of
from config import CONTRACT, SYMBOL
from indicators import atr

# 전략
TIMEFRAME = mt5.TIMEFRAME_H1
SILVER = "XAGUSD"
Z_N = 100
Z_K = 2.5
SL_ATR = 3.0
LOT = 0.01
# 주문
MAGIC = 20261031
DEVIATION = 50
POLL_SEC = 2.0
BARS = 300

LOG_DIR = Path(__file__).resolve().parent / "logs" / "gsr_log"
TRADES_CSV = LOG_DIR / "trades.csv"
FIELDS = ["signal_bar", "side", "lot", "z", "ratio", "signal_price", "order_time", "fill_price", "sl", "risk_usd",
          "spread", "ticket", "close_time", "close_price", "profit", "result"]
CLOSE_REASONS = {"gsr exit": "Z0"}

log = logging.getLogger("gsr")
bot = None         # Trader, main()에서 설정

STRATEGY = "금은비율"
STRATEGY_DESC = (f"금/은 비율 되돌림 (1시간봉 {Z_N}봉 z > {Z_K:g} 금 매도 / < -{Z_K:g} 금 매수, z가 0을 넘으면 청산), "
                 f"손절 {SL_ATR:g} ATR, {LOT} lot 고정")


def zscore(gold):
    """마지막 완성 봉의 (z, 비율, ATR). 은 시세를 못 받으면 None"""
    ag = rates(SILVER, TIMEFRAME, BARS + 50, min_bars=Z_N + 10)
    if ag is None:
        return None
    s = pd.merge_asof(gold[["time"]], ag[["time", "close"]].rename(columns={"close": "ag"}), on="time")
    ratio = gold.close / s.ag
    z = (ratio - ratio.rolling(Z_N).mean()) / ratio.rolling(Z_N).std()
    if pd.isna(z.iat[-1]):
        return None
    return float(z.iat[-1]), float(ratio.iat[-1]), float(atr(gold).iat[-1])


def place(side, z, ratio, a, close, signal_bar):
    f = bot.open(side, LOT, f"GSR z{z:+.1f}", sl_dist=SL_ATR * a)
    if f is None:
        return None
    risk_usd = abs(f.fill - f.sl) * LOT * CONTRACT
    row = {"signal_bar": signal_bar, "side": "BUY" if side == 1 else "SELL", "lot": LOT, "z": round(z, 2),
           "ratio": round(ratio, 3), "signal_price": close, "order_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "fill_price": f.fill, "sl": f.sl, "risk_usd": round(risk_usd, 2), "spread": round(f.tick.ask - f.tick.bid, 3),
           "ticket": f.res.order}
    pct = risk_usd / mt5.account_info().balance * 100
    log.info(f"진입 {row['side']} {LOT}lot @ {f.fill} (z {z:+.2f}, 비율 {ratio:.2f}), SL {f.sl} (위험 {risk_usd:.2f}$ = {pct:.1f}%)")
    bot.send(f"{'🟢 매수' if side == 1 else '🔴 매도'} 진입 {LOT} lot (금/은 비율 z {z:+.2f})\n전략: {STRATEGY_DESC}\n"
             f"가격 {f.fill} | 손절 {f.sl} | 익절 없음 (z가 0을 넘으면 청산)\n손절 시 손실 약 {risk_usd:,.0f}$ ({pct:.1f}%)\n"
             f"{bot.account_line()}\n{bot.summary()}")
    return row


def handle_closed(open_row):
    """보유하던 포지션이 닫혔으면 기록/알림하고 True"""
    if any(p.ticket == open_row["ticket"] for p in bot.positions()):
        return False
    c = bot.find_close(open_row["ticket"], CLOSE_REASONS)
    if not c:
        return False
    open_row.update(c)
    append_csv(TRADES_CSV, FIELDS, open_row)
    profit = float(c["profit"])
    bal = mt5.account_info().balance
    pct = profit / (bal - profit) * 100 if bal - profit > 0 else 0.0
    why = {"Z0": "비율 되돌아옴", "SL": "손절"}.get(c["result"], c["result"])
    log.info(f"청산 {c['result']} @ {c['close_price']} 손익 {profit}")
    bot.send(f"{'✅' if profit > 0 else '❌'} {why} {open_row.get('side', '')} {open_row.get('lot', '')} lot\n"
             f"{open_row.get('fill_price', '')} → {c['close_price']}\n손익 {profit:+,.2f}$ ({pct:+.2f}%)\n"
             f"{bot.account_line()}\n{bot.summary()}")
    return True


def main():
    global bot
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="주문 없이 신호만 계산")
    args = ap.parse_args()
    bot = Trader(STRATEGY, SYMBOL, MAGIC, DEVIATION, LOG_DIR, "gsr")
    acc = bot.start(args.dry_run)
    mt5.symbol_select(SILVER, True)
    log.info(f"시작: 계좌 {acc.login} ({acc.server}, {bot.kind}) 잔고 {acc.balance} {acc.currency}, {SYMBOL}/{SILVER} H1 "
             f"z{Z_N} ±{Z_K:g}, 손절 {SL_ATR:g} ATR, {LOT} lot{' [DRY-RUN]' if args.dry_run else ''}, "
             f"텔레그램 {'켜짐' if bot.tg.enabled else '꺼짐'}")
    bot.send(f"▶️ 금은비율 시작 ({bot.kind} {acc.login})\n전략: {STRATEGY_DESC}\n{bot.account_line()}")
    if not args.dry_run:
        bot.warn_algo_off()

    open_row = bot.adopt_position(with_tp=False)
    last_bar = None
    with bot.running("⏸ 금은비율 종료 (열린 포지션은 서버 손절로 보호됨, z 청산은 다시 켜야 함)"):
        while True:
            bot.keep_alive()
            if open_row and handle_closed(open_row):
                open_row = None
            df = bot.rates(TIMEFRAME, BARS, min_bars=Z_N + 20)
            if df is not None and df.time.iat[-1] != last_bar:
                bar = df.time.iat[-1]
                res = zscore(df)
                if res is None:
                    log.warning(f"은 시세 / z 계산 실패 ({mt5.last_error()}), 다음 루프에서 다시")
                    time.sleep(POLL_SEC)
                    continue
                last_bar = bar
                z, ratio, a = res
                log.info(f"1시간봉 {bar:%m-%d %H:%M} 금 {df.close.iat[-1]:.2f} 비율 {ratio:.2f} z {z:+.2f} ATR {a:.2f}")
                if args.dry_run:
                    time.sleep(POLL_SEC)
                    continue
                pos = bot.positions()
                for p in pos:
                    if (side_of(p) == 1 and z > 0) or (side_of(p) == -1 and z < 0):
                        bot.close(p, "gsr exit", f"z {z:+.2f} 0 넘음 -> 청산")
                if pos:
                    time.sleep(1.0)
                    if open_row and handle_closed(open_row):
                        open_row = None
                    pos = bot.positions()
                if not pos and abs(z) > Z_K:
                    side = -1 if z > 0 else 1
                    log.info(f"신호 {side_name(side)} (z {z:+.2f})")
                    open_row = place(side, z, ratio, a, float(df.close.iat[-1]), bar.strftime("%Y-%m-%d %H:%M"))
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
