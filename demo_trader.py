"""데모 계좌 관찰용 자동매매: XAUUSD 5분봉 SAR + RSI50 + H1 SAR (고정 lot, 마틴은 MARTIN = True일 때만)

2026-10-19 변경: 마틴 끔, 항상 그날 시작 잔고 1,000달러당 0.05 lot.
  마틴은 첫 진입 익절에서만 돈을 벌고(TP 3 < SL 7.5라 2번째 이후 익절은 앞 손실을 못 메움) 위험만 키웠음.
  2025-01~2026-10 실제 틱, 1,000달러 시작: 마틴 0.03 시작 11,131$ / 낙폭 31% / 최악의 날 -25%
                                        -> 고정 0.05     14,279$ / 낙폭 30% / 최악의 날 -17%

진입 규칙 (backtests/filters.py의 "RSI50+H1 SAR"와 동일):
- 5분봉 SAR(0.02, 0.2)가 전환된 봉이 완성되면 바로 시장가 진입
  매수: RSI(14) > 50 이고 마지막 완성 1시간봉 SAR이 상승
  매도: RSI(14) < 50 이고 마지막 완성 1시간봉 SAR이 하락
- 포지션은 한 번에 1개, 포지션이 있는 동안 새 신호는 무시
- SL 7.5 / TP 3 달러를 주문과 함께 서버에 설정 (스크립트가 꺼져도 SL/TP는 작동)

마틴 규칙 (MARTIN = True일 때만, backtests/daily_martin.py와 동일):
- 시작 lot = 그날 시작 잔고 1,000달러당 0.03 lot (0.01 단위 내림, 최소 0.01). 하루 안에서는 같은 시작 lot 유지
- 손절로 끝나면 다음 진입 수량 x2, 익절이면 시작 lot으로 복귀
  연속 손절 중 최대 5번째 진입까지 (익절하면 1번째로 돌아가 그날 계속 거래). 잔고 1,000달러일 때 1~5번째: 0.03 / 0.06 / 0.12 / 0.24 / 0.48 (10,000달러면 0.30 ~ 4.80)
- 5번째(0.48)에서도 손절이면 그날은 신규 진입 정지
- 서버시간 날짜가 바뀌면 1번째부터 다시 시작
- 단계는 오늘 이 스크립트가 청산한 거래내역(MT5 서버 기록)으로 계산 -> 스크립트를 껐다 켜도 이어짐

주말 보호:
- 금요일 FRIDAY_NO_ENTRY 이후 신규 진입 안 함, FRIDAY_CLOSE 이후 보유 포지션 시장가 청산
미국장 제외:
- 매일 NO_ENTRY_FROM ~ NO_ENTRY_UNTIL(서버시간 15:00~21:00) 신규 진입 안 함 (backtests/martin_scaled.py)
횡보 정지:
- 오늘 완성된 1시간봉 SAR 전환이 MAX_H1_FLIPS(3)번 이상이면 그날 신규 진입 안 함 (backtests/regime_switch.py)
  (주말 갭은 서버 SL로도 막을 수 없어서)
조용한 시장 쉬기:
- 신호가 나온 5분봉(완성 봉)의 ATR(14, Wilder)이 MIN_ATR(2.5$) 미만이면 진입 안 함
  (2022~24년처럼 변동이 작을 때는 이 신호가 손익분기 승률 71.4%를 못 넘었음)
급변동 쉬기:
- 신호 봉의 5분봉 ATR이 최근 20일(5분봉 ATR_REF_BARS개) ATR 중앙값의 MAX_ATR_MULT(2.0)배를 넘으면 진입 안 함
  (손절 7.5$가 변동성에 비해 너무 좁아지는 날. 2025년으로 기준을 고르고 2026년 틱 체결로 확인:
   PF 1.20 -> 1.22, 추세추종과 같이 돌리면 낙폭 35% -> 29%, 최악의 날 -24% -> -20%)
최대 lot 제한:
- 마틴 lot은 그날 시작 lot의 MAX_LOT_MULT(5)배까지만. 1,000달러면 0.03 / 0.06 / 0.12 / 0.15 / 0.15
과열 거르기 (2026-10-21 추가, backtests/ext_filter.py. 2025로 고르고 2026으로 확인):
- 신호 봉 종가가 방금 뚫은 SAR 값에서 ATR x MAX_EXT(0.8) 넘게 갔거나,
  새 SAR 점(반대 콩 자리)에서 ATR x MAX_RISK(2.8) 넘게 떨어져 있으면 진입 안 함 (이미 많이 간 자리)
  (반대 콩 청산 기준: 2025 +66% -> +73%, 낙폭 38.9% -> 29.8% / 2026 +741% -> +967%, 24.0% -> 19.2%)
급등락 직후 쉬기 (2026-10-28 추가, backtests/sar_range_rest.py, 데모만 켬 RANGE_REST_LIVE=False):
- 최근 완성 5분봉 24개(2시간) 고가-저가가 일봉 ATR(14, 어제까지)의 0.4배 이상이면 진입 안 함
  (2022~2026.10 실제 틱: 승률 58.6% -> 58.9%, 5년 합 +651 -> +731 oz, 낙폭 166 -> 176 oz)
손절 10$ (2026-10-28, backtests/sar_tp_grid.py, 데모만 SL_USD_LIVE=7.5):
- 손절 7.5 -> 10 (익절 3 그대로). 2022~2026.10 실제 틱: 승률 58.6% -> 59.6%, 5년 합 +651 -> +719 oz, 낙폭 166 -> 166 oz
  (바뀌는 거래 9%, 8.5는 오히려 줄고 15는 같아서 데모에서 몇 주 확인)
반대 콩 청산 (2026-10-21 추가):
- 포지션 보유 중 5분봉 SAR이 반대로 전환되면(완성 봉 기준) TP/SL 전이라도 시장가 청산 (SAR_EXIT)
  (backtests/sar_exit.py, 승률 쉬기 끔 기준: 2026 +751% -> +741%, 낙폭 27.8% -> 24.0% / 2025 +70% -> +66%, 44.7% -> 38.9%)
성적 기반 쉬기 (WINRATE_PAUSE, 2026-10-21부터 꺼 둠: 반대 콩 청산으로 승률이 71.8% 근처라 켜 두면 자주 쉬게 됨):
- 최근 WINRATE_N(200)개 결과의 승률이 WINRATE_MIN(71.4%) 미만이면 실제 진입 대신 가상 진입만 함.
  가상 진입도 같은 SL/TP로 결과를 기록해서, 승률이 회복되면 자동으로 실제 진입을 다시 시작
  기록: <로그 폴더>/winrate.json (처음에는 백테스트 마지막 200건으로 채워 둠)

안전장치:
- 기본은 데모 계좌 전용. 실계좌는 --live 옵션을 줄 때만 실행. 실행 중 로그인 계좌가 바뀌면 즉시 멈춤
- 이 스크립트의 매직 넘버가 붙은 포지션만 확인/관리
- 증거금이 부족하면 진입하지 않음
- 최대 lot: 진입할 lot이 MAX_LOT을 넘으면 MAX_LOT으로 깎아서 진입 (설정 실수/버그 방지)
  (2026-10-19 변경: 전에는 그날 진입 정지 -> 잔고 약 2만$부터 거의 거래를 안 했음.
   2026-01~10, 1,500만원 계좌에서 3개 봇 같이: 정지 +180% / 깎아서 진입 +614%, 낙폭 둘 다 19.8%)
- 하루 손실 한도: 오늘 이 전략의 청산 손익이 그날 시작 잔고의 DAILY_LOSS_PCT 이상 손실이면 그날 신규 진입 정지.
  지금은 꺼 둠(None) -> 마틴이 이길 때까지 5번째 진입까지 감
- 계좌 손실 한도: 평가금이 기록된 최고 잔고보다 ACCOUNT_STOP_PCT(50%) 이상 줄면 신규 진입 중단.
  logs/demo_log/risk_state.json에 "halted"로 남아서 재시작해도 유지됨. 확인 후 그 파일을 지우면 다시 시작

기록 (logs/demo_log/ 폴더): trades.csv (거래별 신호/체결/슬리피지/결과), events.log (실행 로그)

텔레그램 (telegram.json 설정이 있을 때, 설정법은 telegram_notify.py 참고):
- 진입/청산 때마다 거래 내용과 오늘/이번주/이번달 수익률
- 일일·주간·월간 리포트는 report_bot.py가 계좌별로 한 번에 보냄 (한국시간 평일 21:00)
- 시작/종료, 마틴 정지, 금요일 청산, 주문 실패

사용법:
  python demo_trader.py            # 실행 (Ctrl+C로 종료)
  python demo_trader.py --dry-run  # 주문 없이 신호만 계산해서 출력 (텔레그램 전송 안 함)
  python demo_trader.py --live     # 실계좌에서 실행 (데모에서 충분히 확인한 뒤에만)
"""
import argparse
import csv
import ctypes
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from common.trader import Trader, append_csv, load_json, rates, save_json, side_of, side_name
from config import SYMBOL
from indicators import atr, psar, psar_levels, rsi

# 전략
SL_USD = 10.0             # 데모 (2026-10-28 7.5 -> 10, backtests/sar_tp_grid.py). main()에서 실계좌면 SL_USD_LIVE
SL_USD_LIVE = 7.5         # 실계좌는 아직 7.5
TP_USD = 3.0
# 마틴
MARTIN = False             # False면 손절해도 lot을 키우지 않고 항상 시작 lot (5번째 손절 정지도 없음)
LOT_PER_1000 = 0.05       # 그날 시작 잔고 1,000달러당 시작 lot (마틴 쓸 때는 0.03이었음)
MARTIN_MULT = 2
MARTIN_MAX_LEVEL = 4       # 0부터 셈: 첫 진입(0) + 2배 4번 = 연속 손절 시 최대 5번째 진입까지
# 주말 보호 (서버시간, 금요일)
FRIDAY_NO_ENTRY = "22:30"
# 미국장 제외 (서버시간, 매일): 이 시간에는 신규 진입 안 함. 보유 포지션은 서버 SL/TP로 계속 관리
# 서버시간은 미국 서머타임을 따름 -> 한국시간 여름 21:00~03:00, 겨울 22:00~04:00
NO_ENTRY_FROM = "15:00"
NO_ENTRY_UNTIL = "21:00"
# 횡보 정지: 오늘(서버 날짜) 완성된 1시간봉의 SAR 전환이 이 횟수 이상이면 그날 신규 진입 안 함
MAX_H1_FLIPS = 3
# 조용한 시장 쉬기: 신호 봉의 5분봉 ATR(14)이 이 값(달러) 미만이면 진입 안 함
MIN_ATR = 2.5
# 급변동 쉬기: 신호 봉의 5분봉 ATR이 최근 ATR_REF_BARS개(20일) 5분봉 ATR 중앙값의 이 배수를 넘으면 진입 안 함
MAX_ATR_MULT = 2.0
ATR_REF_BARS = 20 * 276    # 5분봉 하루 약 276개 (23시간 장)
ATR_REF_MIN = 2000         # 기록이 이보다 적으면 이 필터는 건너뜀
# 과열 거르기: (종가 - 뚫은 SAR 값) / ATR, (종가 - 새 SAR 점) / ATR 이 넘으면 진입 안 함
MAX_EXT = 0.8
MAX_RISK = 2.8
# 급등락 직후 쉬기 (2026-10-28, backtests/sar_range_rest.py): 최근 완성 5분봉 RANGE_REST_BARS개(2시간)의 고가-저가가
# 일봉 ATR(14, 어제까지)의 RANGE_REST_MULT배 이상이면 진입 안 함. 승률 58.6 -> 58.9%, 5년 합 651 -> 731 oz
RANGE_REST_MULT = 0.4
RANGE_REST_BARS = 24
RANGE_REST_LIVE = False   # 실계좌는 아직 끔 (데모·2마틴·MT4 따라하기에서 먼저 확인)
range_rest_on = True      # main()에서 실계좌면 RANGE_REST_LIVE로 바꿈
# 마틴 lot 상한: 그날 시작 lot의 몇 배까지
MAX_LOT_MULT = 5
# 성적 기반 쉬기: 최근 N개 결과(실제 + 가상) 승률이 이 값(%) 미만이면 가상 진입만
WINRATE_PAUSE = False     # 끔: 승률은 기록만
WINRATE_N = 200
WINRATE_MIN = 71.4
# 반대 콩 청산: 보유 중 5분봉 SAR이 반대로 전환되면 시장가 청산
SAR_EXIT = True
# 틱 우세 (2026-10-29 데모 실험, backtests/sar_tick_dom.py): 진입 직전 DOM_SEC초 동안 Bid가 오른 틱 / (오른 + 내린 틱)
# 기본은 끔. sar_dom_trader.py만 켬 (skip: 신호와 반대 우세면 쉬기, flip: 반대 우세면 우세 방향으로. flip 거래는 반대 콩 청산 안 함)
DOM_MODE = None
DOM_SEC = 60
DOM_TH = 0.6              # >= 0.6 매수 우세, <= 0.4 매도 우세, 그 사이는 우세 없음 (신호대로)
FLIP_TAG = "DOM flip"     # flip 진입 주문 comment 앞부분 (반대 콩 청산에서 빼는 표시)
FRIDAY_CLOSE = "23:30"
# 실계좌 안전장치
MAX_LOT = 1.0             # 한 번에 진입할 최대 lot (넘으면 이 lot으로 깎아서 진입)
DAILY_LOSS_PCT = None     # 하루 손실 한도 (그날 시작 잔고 대비 %), None = 끔. 15%는 4연속 손절 날 마틴 회복을 끊었음
ACCOUNT_STOP_PCT = 50.0   # 계좌 손실 한도 (최고 잔고 대비 평가금 감소 %). 30%는 2025-08 낙폭(34%)에 걸려 멈췄음
# 주문
MAGIC = 20261017
BRIDGE_ID = "SAR"          # MT4 따라하기 이름 (데모 실행만 진입·청산 신호를 보냄, common/signal_bridge.py)
DEVIATION = 30            # 허용 슬리피지 (포인트, 0.30달러)
POLL_SEC = 1.0
ENTRY_LATE_SEC = 60       # 신호 봉 마감보다 이만큼 늦으면 진입 안 함 (재시작·멈춤 뒤 지난 신호로 들어가지 않게)
TICK_FRESH_SEC = 30       # 진입 전 틱이 이 안에 들어왔어야 함
RETRY_SEC = 5             # 반대 콩 청산 재시도, SL/TP 확인, 계좌 손실 한도 감시 간격
M5_BARS = 600
H1_BARS = 400

LOG_DIR = Path(__file__).resolve().parent / "logs" / "demo_log"
TRADES_CSV = LOG_DIR / "trades.csv"
STATE_JSON = LOG_DIR / "state.json"
RISK_JSON = LOG_DIR / "risk_state.json"
WINRATE_JSON = LOG_DIR / "winrate.json"
FIELDS = ["signal_bar", "side", "level", "lot", "signal_price", "order_time", "fill_price", "slippage", "spread",
          "sl", "tp", "ticket", "close_time", "close_price", "profit", "result"]

CLOSE_REASONS = {"friday close": "FRIDAY", "sar exit": "SAR"}   # 청산 주문 comment -> 기록할 결과

log = logging.getLogger("demo")
bot = None         # Trader, main()에서 설정


STRATEGY = "SAR 단타"
STRATEGY_DESC = ("5분봉 SAR 전환 + RSI50 + 1시간봉 SAR, 익절 3$ / 손절 7.5$, "
                 + ("일일 마틴 최대 5번째 진입 (lot 상한 시작 lot의 5배)" if MARTIN else
                    f"고정 lot (1,000$당 {LOT_PER_1000} lot, 마틴 없음)")
                 + ", 5분봉 ATR 2.5$ 미만 쉼, ATR 20일 중앙값 2배 초과 쉼, 최근 200건 승률 71.4% 미만 쉼")


def day_start_ts(now):
    """서버 날짜 00:00의 MT5 시간값"""
    return int(datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp())


def risk_block(now, lot):
    """신규 진입을 막아야 하면 이유(문자열), 아니면 None"""
    if reason := bot.account_stop(now, ACCOUNT_STOP_PCT, RISK_JSON):
        return reason
    if DAILY_LOSS_PCT is None:
        return None
    acc = mt5.account_info()
    day_start = day_start_ts(now)
    deals = mt5.history_deals_get(day_start, day_start + 2 * 86400) or ()
    trade_deals = [d for d in deals if d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL)]
    start_bal = acc.balance - sum(d.profit + d.commission + d.swap for d in trade_deals)
    mine = sum(d.profit + d.commission + d.swap for d in trade_deals
               if d.magic == MAGIC and d.entry == mt5.DEAL_ENTRY_OUT)
    if start_bal > 0 and mine <= -DAILY_LOSS_PCT / 100 * start_bal:
        return f"하루 손실 한도 도달 (오늘 {mine:+,.2f} / 시작 잔고 {start_bal:,.2f}의 {DAILY_LOSS_PCT:g}%)"
    return None


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
    return side, m5.time.iat[-1], float(m5.close.iat[-1]), desc, (1 if up[-1] else -1)


def overextended(m5, side):
    """신호 봉이 이미 많이 간 자리면 이유, 아니면 None"""
    _, sar, trig = psar_levels(m5.high.values, m5.low.values)
    a = float(atr(m5).iat[-1])
    c = float(m5.close.iat[-1])
    ext, risk = (c - trig[-1]) * side / a, (c - sar[-1]) * side / a
    if ext > MAX_EXT:
        return f"뚫은 SAR에서 {ext:.2f} ATR (> {MAX_EXT})"
    if risk > MAX_RISK:
        return f"새 SAR 점까지 {risk:.2f} ATR (> {MAX_RISK})"
    return None


def range_rest(m5):
    """최근 2시간 범위가 일봉 ATR의 RANGE_REST_MULT배 이상이면 이유 문자열, 아니면 None (일봉 조회 실패도 None)"""
    if not range_rest_on:
        return None
    d1 = bot.rates(mt5.TIMEFRAME_D1, 20, min_bars=15)
    if d1 is None:
        return None
    pc = d1.close.shift(1)
    tr = pd.concat([d1.high - d1.low, (d1.high - pc).abs(), (d1.low - pc).abs()], axis=1).max(axis=1)
    datr = float(tr.iloc[-14:].mean())                    # 완성 일봉 14개 = 어제까지
    w = m5.iloc[-RANGE_REST_BARS:]
    rng = float(w.high.max() - w.low.min())
    if datr > 0 and rng >= RANGE_REST_MULT * datr:
        return f"최근 2시간 범위 {rng:.1f}$ ≥ 일봉 ATR {datr:.1f}$ x {RANGE_REST_MULT:g}"
    return None


def atr_ref():
    """최근 ATR_REF_BARS개 완성 5분봉의 ATR(14) 중앙값 (급변동 쉬기 기준). 기록이 부족하면 None"""
    m5 = rates(SYMBOL, mt5.TIMEFRAME_M5, ATR_REF_BARS + 100)
    if m5 is None or len(m5) < ATR_REF_MIN:
        return None
    return float(atr(m5).iloc[-ATR_REF_BARS:].median())


def h1_flips_today(h1, now):
    """오늘(서버 날짜) 완성된 1시간봉 중 SAR 방향이 바뀐 봉의 수 (backtests/regime_switch.py와 같은 계산)"""
    up = psar(h1.high.values, h1.low.values)
    flip = up[1:] != up[:-1]
    today = (h1.time.dt.date == now.date()).values[1:]
    return int((flip & today).sum())


def base_lot(now, deals=None):
    """그날 시작 잔고(현재 잔고 - 오늘 체결된 거래 손익) 기준 시작 lot"""
    if deals is None:
        day_start = day_start_ts(now)
        deals = mt5.history_deals_get(day_start, day_start + 2 * 86400) or ()
    today = sum(d.profit + d.commission + d.swap for d in deals
                if d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL))
    info = mt5.symbol_info(SYMBOL)
    raw = (mt5.account_info().balance - today) / 1000 * LOT_PER_1000
    return max(info.volume_min, round(int(raw / info.volume_step + 1e-9) * info.volume_step, 2))


def martin_state(now):
    """오늘(서버 날짜) 이 스크립트가 청산한 거래로 (현재 단계, 오늘 정지 여부, 시작 lot) 계산"""
    day_start = day_start_ts(now)
    deals = mt5.history_deals_get(day_start, day_start + 2 * 86400) or ()
    closes = sorted((d for d in deals if d.magic == MAGIC and d.symbol == SYMBOL and d.entry == mt5.DEAL_ENTRY_OUT),
                    key=lambda d: d.time_msc)
    level, stopped = 0, False
    if not MARTIN:
        return level, stopped, base_lot(now, deals)
    for d in closes:
        if d.profit + d.commission + d.swap < 0:
            if level >= MARTIN_MAX_LEVEL:
                stopped = True
            else:
                level += 1
        else:
            level = 0
    return level, stopped, base_lot(now, deals)


def level_text(level):
    return f"마틴 {level + 1}번째" if MARTIN else "고정 lot"


def lot_for(level, base):
    info = mt5.symbol_info(SYMBOL)
    lot = min(base * MARTIN_MULT ** level, base * MAX_LOT_MULT, MAX_LOT)
    lot = round(round(lot / info.volume_step) * info.volume_step, 2)
    return min(max(lot, info.volume_min), info.volume_max)


def load_wr():
    wr = load_json(WINRATE_JSON)
    wr.setdefault("results", [])
    wr.setdefault("virtual", None)
    wr.setdefault("paused", False)
    return wr


def save_wr(wr):
    save_json(WINRATE_JSON, wr, indent=1)


def winrate(wr):
    """최근 WINRATE_N개 승률(%)과 개수"""
    last = wr["results"][-WINRATE_N:]
    return (sum(last) / len(last) * 100 if last else 0.0), len(last)


def is_paused(wr):
    if not WINRATE_PAUSE:
        return False
    rate, n = winrate(wr)
    return n >= WINRATE_N and rate < WINRATE_MIN


def record_result(wr, win, how):
    """결과 한 건 추가. 쉬기 시작/끝이 바뀌면 텔레그램으로 알림"""
    wr["results"] = (wr["results"] + [bool(win)])[-WINRATE_N:]
    rate, n = winrate(wr)
    paused = is_paused(wr)
    log.info(f"승률 기록 ({how} {'익절' if win else '손절'}): 최근 {n}건 {rate:.1f}%")
    if paused != wr["paused"]:
        wr["paused"] = paused
        if paused:
            bot.send(f"⏸ 최근 {n}건 승률 {rate:.1f}% < {WINRATE_MIN}% -> 실제 진입을 쉬고 가상 진입만 기록합니다.\n"
                     f"승률이 {WINRATE_MIN}% 이상으로 돌아오면 자동으로 다시 시작합니다.")
        else:
            bot.send(f"▶️ 최근 {n}건 승률 {rate:.1f}% -> 실제 진입을 다시 시작합니다.")
    save_wr(wr)


def check_virtual(wr):
    """가상 포지션이 익절/손절 가격에 닿았는지 확인 (매수는 Bid, 매도는 Ask 기준 = 실제 SL/TP와 같은 방식)"""
    v = wr["virtual"]
    if not v:
        return
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return
    px = tick.bid if v["side"] == 1 else tick.ask
    move = (px - v["entry"]) * v["side"]
    if move >= TP_USD or move <= -SL_USD:
        wr["virtual"] = None
        log.info(f"가상 {'매수' if v['side'] == 1 else '매도'} @ {v['entry']} 종료 @ {px}")
        record_result(wr, move > 0, "가상")


def open_virtual(wr, side, bar):
    tick = mt5.symbol_info_tick(SYMBOL)
    entry = tick.ask if side == 1 else tick.bid
    wr["virtual"] = {"side": side, "entry": entry, "signal_bar": bar}
    save_wr(wr)
    rate, n = winrate(wr)
    log.info(f"승률 쉬기 중 (최근 {n}건 {rate:.1f}%) -> 가상 {'매수' if side == 1 else '매도'} @ {entry}")


def tick_dominance():
    """직전 DOM_SEC초 Bid 틱 방향 -> (우세 1/-1/0, 오른 틱 비율, 바뀐 틱 수), 조회 실패면 None"""
    t = mt5.symbol_info_tick(bot.symbol)
    if t is None or not t.time_msc:
        return None
    start = datetime.fromtimestamp((t.time_msc - DOM_SEC * 1000) / 1000, timezone.utc)
    ticks = mt5.copy_ticks_from(bot.symbol, start, 100_000, mt5.COPY_TICKS_ALL)
    if ticks is None:
        return None
    bid = ticks["bid"][ticks["time_msc"] <= t.time_msc]
    d = bid[1:] - bid[:-1]
    up, n = int((d > 0).sum()), int((d != 0).sum())
    r = up / n if n else 0.5
    return (1 if r >= DOM_TH else -1 if r <= 1 - DOM_TH else 0), r, n


def dom_side(side):
    """틱 우세를 보고 (진입 방향, comment 앞부분). 쉬면 방향 0"""
    if not DOM_MODE:
        return side, ""
    res = tick_dominance()
    if res is None:
        log.warning(f"틱 우세 조회 실패 ({mt5.last_error()}) -> 신호대로 진입")
        return side, ""
    d, r, n = res
    txt = f"직전 {DOM_SEC}초 오른 틱 {r:.0%} ({n}틱)"
    if d == 0 or d == side:
        log.info(f"틱 우세 {txt}: {'신호와 같음' if d else '우세 없음'} -> 신호대로 {side_name(side)}")
        return side, ""
    if DOM_MODE == "skip":
        log.info(f"틱 우세 {txt}: 신호({side_name(side)})와 반대라 진입 안 함")
        return 0, ""
    log.info(f"틱 우세 {txt}: 신호({side_name(side)})와 반대라 우세 방향 {side_name(-side)}로 진입 (반대 콩 청산 안 함)")
    return -side, FLIP_TAG


def sar_exitable(p):
    """반대 콩 청산 대상인지 (틱 우세 flip 거래는 SAR 반대 방향으로 들어간 거라 빼고 손절·익절로만)"""
    return not (p.comment or "").startswith(FLIP_TAG)


def my_positions():
    """이 봇 포지션 목록, 조회 실패면 None (bot.positions()는 실패도 빈 목록이라 '포지션 없음'과 구분 안 됨)"""
    p = mt5.positions_get(symbol=bot.symbol)
    return None if p is None else [x for x in p if x.magic == bot.magic]


def fresh_tick():
    """터미널이 서버에 연결돼 있고 틱이 TICK_FRESH_SEC 안에 들어왔으면 True"""
    ti, t = mt5.terminal_info(), mt5.symbol_info_tick(bot.symbol)
    if ti is None or not ti.connected or t is None or not t.time_msc:
        return False
    tick_time = datetime.fromtimestamp(t.time_msc / 1000, timezone.utc).replace(tzinfo=None)
    return abs((bot.clock.now() - tick_time).total_seconds()) <= TICK_FRESH_SEC   # 시계가 어긋나 미래 틱으로 보여도 거름


_mutex = None


def lock(name=None):
    """같은 기록 폴더(데모 / 실계좌 / 2마틴 / 헤지)로 두 개가 동시에 돌지 않게: 윈도우 이름 있는 뮤텍스 (index_trader.lock과 같음)"""
    global _mutex
    name = name or LOG_DIR.name
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateMutexW.restype = ctypes.c_void_p
    k.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    h = k.CreateMutexW(None, False, f"Local\\infinox_sar_{name}")
    if not h:
        raise SystemExit(f"잠금 만들기 실패 ({ctypes.get_last_error()})")
    if ctypes.get_last_error() == 183:                # ERROR_ALREADY_EXISTS
        raise SystemExit(f"이미 실행 중 ({name}). 중복 실행 안 함")
    _mutex = h


def row_from(p, signal_bar="", signal_price=""):
    """실제 포지션으로 기록용 row (이어받기·부분 체결·응답 실패 뒤 생긴 포지션)"""
    return {"signal_bar": signal_bar, "side": "BUY" if side_of(p) == 1 else "SELL", "level": "", "lot": p.volume,
            "signal_price": signal_price, "fill_price": p.price_open, "sl": p.sl, "tp": p.tp, "ticket": p.ticket}


_close_fail_sent = set()


def close_retry(p, comment, what):
    """청산 주문 (닫힐 때까지 부른 쪽이 RETRY_SEC마다 다시 부름). 실패 알림은 티켓마다 한 번만"""
    f = bot.deal(-side_of(p), p.volume, comment, position=p.ticket)
    code = f.res.retcode if f.res else mt5.last_error()
    log.info(f"{what} {'완료' if f.ok else '실패'}: 티켓 {p.ticket} {code}")
    if not f.ok and p.ticket not in _close_fail_sent:
        _close_fail_sent.add(p.ticket)
        bot.send(f"❗ {what} 실패: 티켓 {p.ticket} {code}. 닫힐 때까지 {RETRY_SEC}초마다 다시 시도")
    return f.ok


_protect_sent = set()


def protect(pos):
    """서버 SL/TP가 빠진 포지션에 진입가 기준 SL/TP를 다시 걸고, 이미 SL을 넘어간 가격이면 시장가 청산"""
    info, t = mt5.symbol_info(bot.symbol), mt5.symbol_info_tick(bot.symbol)
    if info is None or t is None:
        return
    for p in pos:
        if p.sl and p.tp:
            continue
        s = side_of(p)
        sl = p.sl or round(p.price_open - SL_USD * s, info.digits)
        tp = p.tp or round(p.price_open + TP_USD * s, info.digits)
        px = t.bid if s == 1 else t.ask
        if (px - sl) * s <= 0:                        # 이미 손절선 너머
            close_retry(p, "no sl", "SL 없는 포지션 손절")
            continue
        if (tp - px) * s <= 0:                        # 이미 익절선 너머
            close_retry(p, "no tp", "TP 없는 포지션 익절")
            continue
        r = mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": bot.symbol, "position": p.ticket,
                            "sl": sl, "tp": tp, "magic": bot.magic})
        ok = r is not None and r.retcode == mt5.TRADE_RETCODE_DONE
        log.warning(f"SL/TP 빠진 포지션 {p.ticket}: SL {sl} TP {tp} 다시 걸기 {'완료' if ok else '실패'} "
                    f"({r.retcode if r else mt5.last_error()})")
        if p.ticket not in _protect_sent:
            _protect_sent.add(p.ticket)
            bot.send(f"⚠️ 포지션 {p.ticket}에 서버 SL/TP가 없어서 다시 걸었어요 (SL {sl} / TP {tp}, {'완료' if ok else '실패, 다시 시도'})")


def place(side, lot, level, signal_bar, signal_price, tag=""):
    f = bot.open(side, lot, f"{tag} L{level}" if tag else f"SAR RSI50 H1 L{level}", sl_dist=SL_USD, tp_dist=TP_USD)
    time.sleep(0.5)
    pos = my_positions()
    if f is None:
        if pos:                                       # 응답은 실패(부분 체결·응답 유실)인데 실제로는 포지션이 생김
            log.warning(f"주문 응답은 실패지만 포지션이 있음 -> 이어받음 {[p.ticket for p in pos]}")
            bot.send(f"⚠️ 주문 응답은 실패였지만 포지션이 생겨서 이어받습니다 ({pos[0].volume} lot @ {pos[0].price_open})")
            return row_from(pos[0], signal_bar, signal_price)
        return None
    mine = [p for p in (pos or ()) if p.ticket == f.res.order]
    if mine and abs(mine[0].volume - lot) > 1e-9:     # 일부만 체결
        log.warning(f"요청 {lot} lot 중 {mine[0].volume} lot만 체결")
        lot = mine[0].volume
    unverified = not mine                             # 조회 실패·아직 안 보임: 메인 루프가 다음 정상 조회 때 실제 lot·체결가로 맞춤
    fill, sl, tp = f.fill, f.sl, f.tp
    spread = f.tick.ask - f.tick.bid
    # 슬리피지: 신호봉 종가(Bid) 대비 얼마나 불리하게 체결됐는지 (스프레드는 따로 기록하므로 제외, +가 불리)
    bid_equiv = fill - spread if side == 1 else fill
    row = {
        "signal_bar": signal_bar, "side": "BUY" if side == 1 else "SELL", "level": level, "lot": lot,
        "signal_price": signal_price, "order_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fill_price": fill, "slippage": round((bid_equiv - signal_price) * side, 3),
        "spread": round(spread, 3), "sl": sl, "tp": tp, "ticket": f.res.order,
    }
    if unverified:
        row["unverified"] = True
        log.warning(f"주문 뒤 포지션 {f.res.order}를 아직 확인 못 함 (조회 실패·지연), 다음 조회 때 실제 lot·체결가로 맞춤")
    log.info(f"진입 {row['side']} {lot}lot ({level_text(level)}) @ {fill} "
             f"(신호봉 종가 {signal_price}, SL {sl}, TP {tp}, 스프레드 {row['spread']})")
    bot.send(f"{'🟢 매수' if side == 1 else '🔴 매도'} 진입 {lot} lot ({level_text(level)})"
             + (" ⚠️ SAR 신호와 반대 (틱 우세 방향)" if tag else "") + "\n"
             f"전략: {STRATEGY_DESC}\n"
             f"가격 {fill} | SL {sl} | TP {tp}\n"
             f"최대 손실 약 {SL_USD * lot * 100:,.0f}$ / 목표 이익 약 {TP_USD * lot * 100:,.0f}$\n"
             f"{bot.account_line()}\n{bot.summary()}")
    return row


def recorded(ticket):
    """trades.csv에 이 티켓 기록이 이미 있는지 (기록 뒤 상태 저장 전에 꺼졌던 경우)"""
    if not TRADES_CSV.exists():
        return False
    with open(TRADES_CSV, encoding="utf-8-sig") as fh:
        return any(r.get("ticket") == str(ticket) for r in csv.DictReader(fh))


def check_closed(open_row, wr, pos):
    """진입했던 포지션이 닫혔으면 기록·알림하고 None, 아니면 그대로. pos = 방금 조회한 실제 포지션 (조회 성공한 것만)"""
    if open_row and not any(p.ticket == open_row["ticket"] for p in pos):
        c = bot.find_close(open_row["ticket"], CLOSE_REASONS)
        if c:
            if recorded(open_row["ticket"]):
                return None
            if open_row.pop("unverified", None):          # 체결을 확인 못 한 채 닫힘: 진입 체결 이력으로 lot·체결가
                ins = [d for d in (mt5.history_deals_get(position=open_row["ticket"]) or ()) if d.entry == mt5.DEAL_ENTRY_IN]
                if ins:
                    vol = sum(d.volume for d in ins)
                    open_row.update(lot=round(vol, 2), fill_price=round(sum(d.price * d.volume for d in ins) / vol, 2))
            open_row.update(c)
            append_csv(TRADES_CSV, FIELDS, open_row)
            notify_close(open_row)
            record_result(wr, float(open_row["profit"]) > 0, "실제")
            return None
    return open_row


def settle(open_row, extra, wr, pos):
    """닫힌 포지션 기록 (기록 중인 것 + 더 있는 것 전부), 응답 뒤 조회 실패로 확인 못 한 체결 대사, 새로 보인 포지션 이어받기"""
    open_row = check_closed(open_row, wr, pos)
    for t, r in list(extra.items()):
        if check_closed(r, wr, pos) is None:
            extra.pop(t)
    if open_row and open_row.get("unverified"):
        mine = [p for p in pos if p.ticket == open_row["ticket"]]
        if mine:
            p = mine[0]
            open_row.update(lot=p.volume, fill_price=p.price_open, sl=p.sl, tp=p.tp)
            open_row.pop("unverified")
            log.info(f"체결 확인: 티켓 {p.ticket} {p.volume}lot @ {p.price_open}")
    if open_row is None and pos:
        p = pos[0]
        open_row = extra.pop(p.ticket, None) or row_from(p)
        log.info(f"포지션 이어받음: 티켓 {p.ticket} {open_row['side']} {p.volume}lot @ {p.price_open}")
    for p in pos:
        if open_row and p.ticket != open_row["ticket"] and p.ticket not in extra:
            extra[p.ticket] = row_from(p)
            log.warning(f"포지션이 더 있음: 티켓 {p.ticket} {p.volume}lot @ {p.price_open} (따로 기록)")
    return open_row


def is_after(now, hhmm):
    h, m = map(int, hhmm.split(":"))
    return (now.hour, now.minute) >= (h, m)


def notify_close(row):
    profit = float(row["profit"])
    bal = mt5.account_info().balance
    pct = profit / (bal - profit) * 100 if bal - profit > 0 else 0.0
    level, stopped, base = martin_state(bot.clock.now())
    icon = {"TP": "✅ 익절", "SL": "❌ 손절", "FRIDAY": "⏹ 금요일 청산", "SAR": "↩️ 반대 콩 청산"}.get(row["result"], "⏹ 청산")
    nxt = "오늘 거래 정지 (5번째 진입 손절)" if stopped else f"다음 진입 {level_text(level)} ({lot_for(level, base)} lot)"
    log.info(f"청산 {row['result']} @ {row['close_price']} 손익 {profit} -> {nxt}")
    bot.send(f"{icon} {row.get('side', '')} {row.get('lot', '')} lot\n"
             f"{row.get('fill_price', '')} → {row['close_price']}\n"
             f"손익 {profit:+,.2f}$ ({pct:+.2f}%)\n{nxt}\n"
             f"{bot.account_line()}\n{bot.summary()}")
    if stopped:
        bot.send("⛔ 5번째 진입(최대 마틴)에서 손절 -> 오늘은 신규 진입을 멈춥니다. 내일 1번째부터 다시 시작합니다.")


def use_live_paths():
    """실계좌 실행은 기록을 logs/live_log/에 따로 남김 (텔레그램의 [실계좌] 표시는 Trader가 붙임)"""
    global LOG_DIR, TRADES_CSV, STATE_JSON, RISK_JSON, WINRATE_JSON
    LOG_DIR = LOG_DIR.parent / "live_log"
    TRADES_CSV, STATE_JSON, RISK_JSON = LOG_DIR / "trades.csv", LOG_DIR / "state.json", LOG_DIR / "risk_state.json"
    WINRATE_JSON = LOG_DIR / "winrate.json"


def main():
    global bot, range_rest_on, SL_USD
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="주문 없이 신호만 계산")
    ap.add_argument("--live", action="store_true", help="실계좌에서도 실행 (기본은 데모 전용)")
    args = ap.parse_args()
    if args.live:
        use_live_paths()
        range_rest_on = RANGE_REST_LIVE
        SL_USD = SL_USD_LIVE
    bot = Trader(STRATEGY, SYMBOL, MAGIC, DEVIATION, LOG_DIR, "demo", live=args.live)
    bot.bridge_id = BRIDGE_ID
    acc = bot.start(args.dry_run)
    kind = bot.kind
    base = base_lot(bot.clock.now())
    lots = " / ".join(f"{lot_for(i, base):g}" for i in range(MARTIN_MAX_LEVEL + 1 if MARTIN else 1))
    lots += f" lot (오늘 시작 잔고 기준, 1,000$당 {LOT_PER_1000} lot{'' if MARTIN else ', 마틴 없음'})"
    daily = "꺼짐" if DAILY_LOSS_PCT is None else f"{DAILY_LOSS_PCT:g}%"
    wr = load_wr()
    rate, n = winrate(wr)
    limits = (f"최대 lot {MAX_LOT} (넘으면 {MAX_LOT}로 진입){f' (마틴은 시작 lot의 {MAX_LOT_MULT}배까지)' if MARTIN else ''}, 하루 손실 한도 {daily}, "
              f"계좌 손실 한도 {ACCOUNT_STOP_PCT:g}%, ATR {MIN_ATR:g}$ 미만 쉼, "
              f"ATR이 20일 중앙값의 {MAX_ATR_MULT:g}배 초과면 쉼, "
              + (f"최근 {WINRATE_N}건 승률 {WINRATE_MIN:g}% 미만 쉼 (지금 {n}건 {rate:.1f}%{', 쉬는 중' if is_paused(wr) else ''})"
                 if WINRATE_PAUSE else f"승률 쉬기 꺼짐 (최근 {n}건 {rate:.1f}%)")
              + (", 5분봉 SAR 반대 전환이면 청산" if SAR_EXIT else "")
              + f", 과열 거르기 (뚫은 SAR에서 {MAX_EXT} ATR / 새 SAR 점까지 {MAX_RISK} ATR 초과면 쉼)"
              + (f", 급등락 직후 쉬기 (최근 2시간 범위 ≥ 일봉 ATR x{RANGE_REST_MULT:g})" if range_rest_on else ""))
    log.info(f"시작: 계좌 {acc.login} ({acc.server}, {kind}) 잔고 {acc.balance} {acc.currency}, {SYMBOL} "
             f"SL {SL_USD} TP {TP_USD}, {'마틴 ' if MARTIN else ''}{lots}, {limits}{' [DRY-RUN]' if args.dry_run else ''}"
             f", 텔레그램 {'켜짐' if bot.tg.enabled else '꺼짐'}")
    bot.send(f"▶️ 자동매매 시작 ({kind} {acc.login})\n전략: {STRATEGY_DESC}\n{'마틴 ' if MARTIN else 'lot '}{lots}\n"
             f"안전장치: {limits}\n{bot.account_line()}\n{bot.summary()}")
    if halted := load_json(RISK_JSON).get("halted"):
        bot.send(f"🛑 계좌 손실 한도로 중단된 상태입니다: {halted}\n"
                 f"확인 후 {LOG_DIR.name}/{RISK_JSON.name}을 지우면 다시 진입합니다.")
    if not args.dry_run:
        bot.warn_algo_off()

    # 중복 실행 막기 + 이전 실행 상태: 마지막으로 처리한 봉, 기록 중이던 포지션 (꺼져 있는 동안 닫혔으면 이력으로 기록)
    sig_json = LOG_DIR / "signal_state.json"
    if not args.dry_run:
        lock()
    sst = load_json(sig_json)
    saved_bar = pd.Timestamp(sst["last_bar"]) if sst.get("last_bar") else None
    open_row = sst.get("open_row") or None
    extra = {int(k): v for k, v in (sst.get("extra") or {}).items()}   # 기록 중인 것 말고 더 있는 포지션 (티켓 -> row)
    if open_row is None:
        open_row = bot.adopt_position()
    saved_text = [None]

    def save_sig():
        """바뀌었을 때만 저장"""
        if args.dry_run:
            return
        obj = {"last_bar": None if last_bar is None else str(last_bar), "open_row": open_row,
               "extra": {str(k): v for k, v in extra.items()}}
        text = json.dumps(obj, ensure_ascii=False, indent=1, default=str)
        if text != saved_text[0]:
            sig_json.write_text(text, encoding="utf-8")
            saved_text[0] = text

    last_bar, last_close_try, last_retry, last_guard, last_warn = saved_bar, 0.0, 0.0, 0.0, 0.0
    sar_dir = None            # 마지막 완성 5분봉(last_bar)의 SAR 방향 (반대 콩 청산 기준)
    risk_notified = None      # 안전장치 알림은 하루 한 번만
    with bot.running("⏸ 자동매매 종료 (열린 포지션은 서버 SL/TP로 계속 관리됩니다)"):
        while True:
            now = bot.keep_alive()
            pos = my_positions()
            if pos is None:                           # 조회 실패: 진입·청산 판단 안 하고 다시
                if time.time() - last_warn > 60:
                    last_warn = time.time()
                    log.warning(f"포지션 조회 실패 ({mt5.last_error()}), 이번 루프는 쉼")
                time.sleep(POLL_SEC)
                continue

            # 닫힌 포지션 기록, 기록 안 하던 포지션 이어받기
            open_row = settle(open_row, extra, wr, pos)
            save_sig()
            check_virtual(wr)

            # SL/TP 빠진 포지션 복구, 계좌 손실 한도는 신호가 없어도 감시 (한도에 닿으면 신규 진입 중단, 보유는 서버 SL/TP로)
            if time.time() - last_guard > RETRY_SEC:
                last_guard = time.time()
                if not args.dry_run:
                    protect(pos)
                bot.account_stop(now, ACCOUNT_STOP_PCT, RISK_JSON)

            friday = now.weekday() == 4
            # 청산이 실패해도(장 마감 등) 1분에 한 번만 다시 시도
            if friday and is_after(now, FRIDAY_CLOSE) and not args.dry_run and time.time() - last_close_try > 60:
                for p in pos:
                    bot.close(p, "friday close", "금요일 청산")
                    last_close_try = time.time()

            m5 = bot.rates(mt5.TIMEFRAME_M5, M5_BARS)
            # 반대 콩 청산이 실패했으면 닫힐 때까지 다시. sar_dir이 최신 완성 봉 것일 때만 (새 봉이 나왔는데 아직 계산 전이면 안 함)
            if (SAR_EXIT and sar_dir and not args.dry_run and time.time() - last_retry > RETRY_SEC
                    and m5 is not None and m5.time.iat[-1] == last_bar):
                opp = [p for p in pos if side_of(p) != sar_dir and sar_exitable(p)]
                if opp:
                    last_retry = time.time()
                    for p in opp:
                        close_retry(p, "sar exit", "반대 콩 청산")

            if m5 is not None and (m5.time.iat[-1] != last_bar or sar_dir is None):
                h1 = bot.rates(mt5.TIMEFRAME_H1, H1_BARS)
                bar_end = m5.time.iat[-1] + pd.Timedelta(minutes=5)
                if h1 is None or (h1.time + pd.Timedelta(hours=1) <= bar_end).sum() < 50:   # 1시간봉 조회 실패: 이 봉은 다음 루프에서 다시
                    if time.time() - last_warn > 60:
                        last_warn = time.time()
                        log.warning(f"1시간봉 조회 실패 ({mt5.last_error()}), 다시 시도")
                    time.sleep(POLL_SEC)
                    continue
                is_new = m5.time.iat[-1] != last_bar
                side, bar, close, desc, sar_dir = compute_signal(m5, h1)
                if is_new:
                    log.info(f"봉 {bar:%m-%d %H:%M} 종가 {close}: {desc}"
                             + (f" -> 신호 {side_name(side)}" if side else ""))
                if SAR_EXIT and not args.dry_run:
                    for p in pos:
                        if side_of(p) != sar_dir and sar_exitable(p):
                            close_retry(p, "sar exit", "반대 콩 청산")
                            last_retry = time.time()
                    pos = my_positions()
                    if pos is None:                   # 다시 조회 실패: 이 봉은 다음 루프에서 다시 판단
                        time.sleep(POLL_SEC)
                        continue
                    open_row = settle(open_row, extra, wr, pos)               # 같은 봉에서 새로 진입하기 전에 기록
                last_bar = bar
                save_sig()
                late = (now - (bar + pd.Timedelta(minutes=5))).total_seconds()
                if side and is_new and not args.dry_run:
                    level, stopped, base = martin_state(now)
                    if pos or open_row:
                        log.info("포지션 보유 중(또는 방금 청산 기록 대기)이라 신호 무시")
                    elif late > ENTRY_LATE_SEC:
                        log.info(f"신호 봉 마감 {late:.0f}초 뒤라 (재시작·멈춤) 진입 안 함")
                    elif not fresh_tick():
                        log.info(f"시세 멈춤 / 연결 끊김 (틱 {TICK_FRESH_SEC}초 넘게 없음)이라 진입 안 함")
                    elif WINRATE_PAUSE and wr["virtual"]:
                        log.info("가상 포지션 진행 중이라 신호 무시")
                    elif stopped:
                        log.info("오늘은 5번째 진입 손절로 거래 정지 상태라 신호 무시")
                    elif friday and is_after(now, FRIDAY_NO_ENTRY):
                        log.info("금요일 마감 전이라 신규 진입 안 함")
                    elif is_after(now, NO_ENTRY_FROM) and not is_after(now, NO_ENTRY_UNTIL):
                        log.info(f"미국장 제외 시간({NO_ENTRY_FROM}~{NO_ENTRY_UNTIL})이라 신규 진입 안 함")
                    elif (flips := h1_flips_today(h1, now)) >= MAX_H1_FLIPS:
                        log.info(f"오늘 1시간봉 SAR 전환 {flips}번 (횡보)이라 신규 진입 안 함")
                    elif (a := float(atr(m5).iat[-1])) < MIN_ATR:
                        log.info(f"5분봉 ATR {a:.2f}$ < {MIN_ATR}$ (조용한 시장)이라 신규 진입 안 함")
                    elif (ref := atr_ref()) and a > MAX_ATR_MULT * ref:
                        log.info(f"5분봉 ATR {a:.2f}$ > 20일 중앙값 {ref:.2f}$ x {MAX_ATR_MULT:g} (급변동)이라 신규 진입 안 함")
                    elif ov := overextended(m5, side):
                        log.info(f"과열 ({ov})이라 신규 진입 안 함")
                    elif rr := range_rest(m5):
                        log.info(f"급등락 직후 ({rr})라 신규 진입 안 함")
                    elif reason := risk_block(now, lot_for(level, base)):
                        log.info(f"안전장치: {reason} -> 신규 진입 안 함")
                        if risk_notified != now.date():
                            risk_notified = now.date()
                            bot.send(f"⛔ 안전장치: {reason}\n오늘은 신규 진입을 멈춥니다.")
                    elif is_paused(wr):
                        open_virtual(wr, side, bar.strftime("%Y-%m-%d %H:%M"))
                    elif (late := (bot.clock.now() - (bar + pd.Timedelta(minutes=5))).total_seconds()) > ENTRY_LATE_SEC or not fresh_tick():
                        log.info(f"주문 직전 다시 확인: 신호 봉 마감 {late:.0f}초 뒤 / 시세 {'정상' if fresh_tick() else '멈춤'} -> 진입 안 함")
                    elif (dom := dom_side(side))[0]:
                        open_row = place(dom[0], lot_for(level, base), level, bar.strftime("%Y-%m-%d %H:%M"), close, dom[1])
                        save_sig()
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
