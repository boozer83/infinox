"""데모 계좌 관찰용 자동매매: XAUUSD 아시아 레인지 → 런던 돌파 + 5일 모멘텀 (마틴 최대 3단계)

규칙은 모두 서버시간 기준. 하루 최대 1거래.
1. 아시아 레인지 = 오늘 01:05~08:55 5분봉의 최고가 / 최저가 (01:00 첫 봉은 롤오버 스프레드 왜곡이라 제외)
2. 09:00~17:55에 시작한 5분봉이 완성됐을 때 종가가 레인지 위 → 매수, 아래 → 매도.
   그날 첫 돌파만 봄 (첫 돌파가 아래 3번에 걸러지면 그날은 거래 없음)
3. 5일 모멘텀 필터: 전일 종가 > 6거래일 전 종가면 매수만, 작으면 매도만 (같으면 거래 없음)
4. 기준폭 U = 0.5 × 일봉 ATR(14, Wilder, 전일까지). 익절 = U, 손절 = 0.75 × U  (v2, 2026-10-21)
5. 23:00에 보유 포지션 시장가 청산 (하루를 넘기지 않음 → 스왑·주말 갭 없음)
   미국 휴일 조기 마감일(연방 공휴일·추수감사절 다음 날·12/24, 마지막 시세 19:44~21:44)은 19:30에 청산
   (전에는 23:00에 장이 닫혀 있어 다음 날 01:00에 청산됨, 2022~26 금 8건 / XAUEUR 7건)
6. 신호 강도 (v2): 돌파 봉 종가가 전일 고가 위(매수) / 전일 저가 아래(매도)면 강한 신호 가중치 1.5, 아니면 0.5

자금관리 (마틴 3단계, 날짜를 넘어 이어짐):
- lot = 사이클 시작 잔고 × RISK_PCT% × 가중치 × 2^(단계-1) / (U × 100온스), 0.01 단위 내림, 최소 0.01
  (손절폭이 아니라 U로 나눔 -> 손절되면 그 금액의 75%를 잃음. 사이클 시작 잔고는 1단계 진입 때 잔고)
- 결과가 0 이하면 다음 거래 ×2, 최대 3단계 (1·2·4배). 플러스 또는 3단계 패배 후 1단계로 돌아감
- 단계는 이 스크립트(매직 넘버)가 청산한 MT5 거래내역으로 계산 → 껐다 켜도 이어짐

v2 백테스트 (backtests/asia_v2.py, 실제 틱, 2022-01 ~ 2026-10, 1만 달러 시작):
- v1 18,645$ / 낙폭 16.7% / 최악의 달 -12.5%  ->  v2 34,104$ / 16.1% / -7.7%, 월평균 +2.29%, 다섯 해 모두 v1보다 좋음
- 승률 51.6% (강한 57.1% / 약한 47.5%, 강한 신호가 해마다 더 높음), 최장 6연패
- 최소 lot 0.01 때문에 잔고가 작으면 가중치 효과가 없음 (ATR 78$ 기준 강한 신호 0.02 lot은 잔고 약 5,200$부터)

백테스트 (실제 틱 체결, 진입 시 실제 스프레드 평균 0.20$, 2022-01 ~ 2026-10, 580건):
- 고정 lot 거래당 R: 2022~24 +0.060 / 2025~26H1 +0.198 / 숨겨 둔 검증구간 2026-07~10 +0.208
  연도별 R 2022 +0.05, 2023 +0.09, 2024 +0.05, 2025 +0.14, 2026 +0.28 (다섯 해 모두 플러스), 승률 약 57%
- 마틴 3단계, 6,000$ 시작: 13,667$ / 최대낙폭 17% (2024년 -1%)
  1,000$ 시작: 4,646$ / 낙폭 20% (단, 최소 lot 0.01 때문에 거래당 실제 리스크가 1%가 아니라 3~6%)
- 약점: 2022~24 우위가 얇음(t 1.4), 하루 1건이라 연 110~140건뿐, 우위 상당 부분이 고변동 2025~26년에서 나옴
- 잔고가 약 6,000$ 미만이면 0.01 lot이 1%보다 큰 리스크가 됨 → 시작할 때 실제 리스크를 로그/텔레그램에 표시

안전장치 (demo_trader.py와 같은 방식):
- 기본은 데모 계좌 전용. 실계좌는 --live 옵션을 줄 때만. 실행 중 로그인 계좌가 바뀌면 즉시 멈춤
- 매직 넘버 MAGIC 포지션만 관리 (demo_trader.py와 같은 계좌에서 나란히 돌려도 됨)
- 증거금 부족, lot이 MAX_LOT 초과면 진입 안 함
- 계좌 손실 한도: 평가금이 최고 잔고보다 ACCOUNT_STOP_PCT% 이상 줄면 신규 진입 중단 (<로그폴더>/risk_state.json)

기록: logs/asia_log/ (실계좌는 logs/asia_live_log/) — trades.csv, events.log, state.json

사용법:
  python asia_breakout_trader.py            # 데모 실행 (Ctrl+C로 종료)
  python asia_breakout_trader.py --dry-run  # 주문 없이 신호만 계산해서 출력
  python asia_breakout_trader.py --live     # 실계좌 (데모에서 충분히 확인한 뒤에만)
  python asia_breakout_trader.py --symbol XAUEUR          # 유로 표시 금 (매직 20261024, 기록 logs/asia_xaueur_log/)
  python asia_breakout_trader.py --symbol XAUEUR --live   # 실계좌 (종목명 XAUEUR#, 기록 logs/asia_xaueur_live_log/)
"""
import argparse
import functools
import logging
import math
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, USLaborDay, USMartinLutherKingJr,
                                    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday)

from common.trader import Trader, append_csv, load_json, save_json
from config import CONTRACT, SYMBOL, live_symbol

# 전략 (서버시간)
RANGE_FROM = "01:05"       # 아시아 레인지 시작 봉
RANGE_UNTIL = "09:00"      # 이 시각 전에 시작한 봉까지 레인지
ENTRY_UNTIL = "18:00"      # 이 시각 전에 시작한 봉의 돌파만 진입 (마지막 17:55 봉)
EXIT_AT = "23:00"          # 보유 포지션 시장가 청산
EARLY_EXIT_AT = "19:30"    # 미국 휴일 조기 마감일 청산 (가장 이른 마감 19:44)
MIN_RANGE_BARS = 50        # 레인지 봉이 이보다 적으면 (휴장일 등) 그날 거래 없음
ATR_MULT = 0.5             # 기준폭 U = 일봉 ATR(14) × 0.5
TP_RATIO = 1.0             # 익절 = U × 1.0
SL_RATIO = 0.75            # 손절 = U × 0.75
W_STRONG, W_WEAK = 1.5, 0.5    # 신호 강도 가중치 (전일 고가/저가 돌파면 강함)
MOM_DAYS = 5               # 모멘텀: 전일 종가 vs (MOM_DAYS+1)거래일 전 종가
# 자금관리
RISK_PCT = 0.6             # 1단계 lot = 사이클 시작 잔고의 0.6% / 손절폭 (XAUEUR 아시아와 같이 돌려서 1% -> 0.6%, 2026-10-24)
MARTIN_MULT = 2
MARTIN_STEPS = 3           # 1·2·4배
# 안전장치
MAX_LOT = 1.0
ACCOUNT_STOP_PCT = 50.0
# 주문
MAGIC = 20261019
SYMBOL_MAGIC = {"XAUEUR": 20261024}   # --symbol 로 다른 종목을 돌릴 때 매직 넘버
SYMBOL_RISK = {"XAUEUR": 0.6}         # 그 종목의 RISK_PCT (금 0.6% + XAUEUR 0.6%: 금 1% 혼자보다 수익 1.6배, 낙폭 16.4 -> 14.1%)
DEVIATION = 50             # 허용 슬리피지 (포인트, 0.50달러). 손절폭이 30$ 이상이라 여유 있게
POLL_SEC = 1.0

LOG_DIR = Path(__file__).resolve().parent / "logs" / "asia_log"
TRADES_CSV = LOG_DIR / "trades.csv"
STATE_JSON = LOG_DIR / "state.json"
BOT_JSON = LOG_DIR / "bot_state.json"        # 오늘 처리 여부, 사이클 1단계 lot
RISK_JSON = LOG_DIR / "risk_state.json"
FIELDS = ["date", "signal_bar", "side", "level", "lot", "range_hi", "range_lo", "daily_atr", "sl_usd",
          "signal_price", "order_time", "fill_price", "spread", "sl", "tp", "ticket",
          "close_time", "close_price", "profit", "result"]

STRATEGY = "아시아 3마틴"
STRATEGY_DESC = (f"아시아 레인지({RANGE_FROM}~{RANGE_UNTIL}) 돌파 {RANGE_UNTIL}~{ENTRY_UNTIL} + {MOM_DAYS}일 모멘텀, "
                 f"U=일봉ATR×{ATR_MULT}, 익절 {TP_RATIO:g}U / 손절 {SL_RATIO:g}U, 전일 고저 돌파면 가중치 {W_STRONG:g} "
                 f"아니면 {W_WEAK:g}, {EXIT_AT} 청산, 마틴 {MARTIN_STEPS}단계 (v2)")

CLOSE_REASONS = {"time exit": "TIME"}   # 청산 주문 comment -> 기록할 결과

log = logging.getLogger("asia")
bot = None         # Trader, main()에서 설정


# ---------------------------------------------------------------- 시간
def hm(s):
    h, m = map(int, s.split(":"))
    return h * 60 + m


def minutes(ts):
    return ts.hour * 60 + ts.minute


class EarlyCloseCalendar(AbstractHolidayCalendar):
    """금이 일찍 닫힌 미국 휴일 (2022~26 데이터와 일치. 콜럼버스·재향군인의 날은 정상 마감)"""
    rules = [USMartinLutherKingJr, USPresidentsDay, USMemorialDay,
             Holiday("Juneteenth", month=6, day=19, observance=nearest_workday),
             Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
             USLaborDay, USThanksgivingDay]


@functools.lru_cache(maxsize=32)
def early_close_day(d):
    hol = set(EarlyCloseCalendar().holidays(date(d.year, 1, 1), date(d.year, 12, 31)).date)
    thanks = [x for x in hol if x.month == 11]
    return d in hol or (bool(thanks) and d == thanks[0] + timedelta(days=1)) or (d.month == 12 and d.day == 24)


def exit_at(d):
    return EARLY_EXIT_AT if early_close_day(d) else EXIT_AT


def use_live_paths():
    """실계좌 실행은 기록을 logs/asia_live_log/에 따로 남김 (텔레그램의 [실계좌] 표시는 Trader가 붙임)"""
    global LOG_DIR, TRADES_CSV, STATE_JSON, BOT_JSON, RISK_JSON
    LOG_DIR = LOG_DIR.parent / "asia_live_log"
    TRADES_CSV, STATE_JSON = LOG_DIR / "trades.csv", LOG_DIR / "state.json"
    BOT_JSON, RISK_JSON = LOG_DIR / "bot_state.json", LOG_DIR / "risk_state.json"


def use_symbol(sym, live=False):
    """금 말고 다른 종목으로 (--symbol): 매직·위험·기록 폴더·이름을 따로 씀. 실계좌 종목명은 뒤에 # (XAUEUR#)
    XAUEUR 2022~2026 틱 백테스트: 전반 +75% / 후반 +126%, 낙폭 11~13%, 금 아시아와 같은 날 같은 방향 95%"""
    global SYMBOL, MAGIC, RISK_PCT, LOG_DIR, TRADES_CSV, STATE_JSON, BOT_JSON, RISK_JSON, STRATEGY
    base = sym.rstrip("#")
    SYMBOL = live_symbol(base) if live else base
    MAGIC = SYMBOL_MAGIC.get(base, MAGIC)
    RISK_PCT = SYMBOL_RISK.get(base, RISK_PCT)
    LOG_DIR = Path(__file__).resolve().parent / "logs" / f"asia_{base.lower()}{'_live' if live else ''}_log"
    TRADES_CSV, STATE_JSON = LOG_DIR / "trades.csv", LOG_DIR / "state.json"
    BOT_JSON, RISK_JSON = LOG_DIR / "bot_state.json", LOG_DIR / "risk_state.json"
    STRATEGY = f"아시아 3마틴·{base}"


# ---------------------------------------------------------------- 지표
def daily_features(today):
    """오늘 시작 시점에 알 수 있는 값: (일봉 ATR14 전일까지, 5일 모멘텀 방향 +1/-1/0, 전일 고가, 전일 저가)"""
    d1 = bot.rates(mt5.TIMEFRAME_D1, 400, min_bars=1)
    if d1 is None or len(d1) < 30:
        return None, 0, None, None
    d1 = d1[d1.time.dt.date < today].reset_index(drop=True)   # 완성된 전일까지
    pc = d1.close.shift()
    tr = np.maximum(d1.high - d1.low, np.maximum((d1.high - pc).abs(), (d1.low - pc).abs()))
    tr.iat[0] = d1.high.iat[0] - d1.low.iat[0]
    atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iat[-1])
    mom = int(np.sign(d1.close.iat[-1] - d1.close.iat[-1 - MOM_DAYS]))
    return atr, mom, float(d1.high.iat[-1]), float(d1.low.iat[-1])


def asian_range(m5, today):
    t = m5[(m5.time.dt.date == today)]
    mins = t.time.dt.hour * 60 + t.time.dt.minute
    r = t[(mins >= hm(RANGE_FROM)) & (mins < hm(RANGE_UNTIL))]
    if len(r) < MIN_RANGE_BARS:
        return None
    return float(r.high.max()), float(r.low.min()), len(r)


def first_breakout(m5, today, hi, lo):
    """오늘 첫 돌파 봉: (봉 시각, 방향, 종가) 또는 None"""
    t = m5[(m5.time.dt.date == today)]
    mins = t.time.dt.hour * 60 + t.time.dt.minute
    w = t[(mins >= hm(RANGE_UNTIL)) & (mins < hm(ENTRY_UNTIL))]
    for row in w.itertuples():
        if row.close > hi:
            return row.time, 1, float(row.close)
        if row.close < lo:
            return row.time, -1, float(row.close)
    return None


# ---------------------------------------------------------------- 자금관리
def martin_level(days=120):
    """청산 기록으로 다음 진입 단계(0부터) 계산"""
    level = 0
    now = int(time.time())
    for d in bot.closes_since(now - days * 86400, now + 2 * 86400):
        if d.profit + d.commission + d.swap > 0:
            level = 0
        else:
            level = level + 1 if level + 1 < MARTIN_STEPS else 0
    return level


def round_lot(x):
    info = mt5.symbol_info(SYMBOL)
    lot = math.floor(x / info.volume_step + 1e-9) * info.volume_step
    return round(min(max(lot, info.volume_min), info.volume_max), 2)


def plan_lot(level, u, w, save=True):
    """(lot, 사이클 시작 잔고, 손절 시 잔고 대비 %). 사이클 시작 잔고는 1단계 진입 때 bot_state.json에 보관"""
    st = load_json(BOT_JSON)
    bal = mt5.account_info().balance
    cyc = bal if level == 0 or "cycle_bal" not in st else st["cycle_bal"]
    if save and cyc != st.get("cycle_bal"):
        st["cycle_bal"] = cyc
        save_json(BOT_JSON, st)
    lot = round_lot(cyc * RISK_PCT / 100 * w * MARTIN_MULT ** level / (u * CONTRACT))
    risk = lot * u * SL_RATIO * CONTRACT / bal * 100 if bal > 0 else 0
    return lot, cyc, risk


def risk_block(now, lot):
    if reason := bot.account_stop(now, ACCOUNT_STOP_PCT, RISK_JSON):
        return reason
    if lot > MAX_LOT:
        return f"진입 lot {lot}이 최대 lot {MAX_LOT}을 넘음"
    return None


# ---------------------------------------------------------------- 주문
def place(side, lot, level, u, w, info_row):
    sl_usd, tp_usd = u * SL_RATIO, u * TP_RATIO
    f = bot.open(side, lot, f"asia brk L{level}", sl_dist=sl_usd, tp_dist=tp_usd)
    if f is None:
        return None
    fill, sl, tp = f.fill, f.sl, f.tp
    row = dict(info_row, side="BUY" if side == 1 else "SELL", level=level + 1, lot=lot, sl_usd=round(sl_usd, 2),
               order_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), fill_price=fill,
               spread=round(f.tick.ask - f.tick.bid, 3), sl=sl, tp=tp, ticket=f.res.order)
    strength = "강한 신호" if w == W_STRONG else "약한 신호"
    log.info(f"진입 {row['side']} {lot}lot (마틴 {level + 1}단계, {strength} x{w:g}) @ {fill}, SL {sl}, TP {tp} "
             f"(손절 {sl_usd:.2f}$ / 익절 {tp_usd:.2f}$)")
    bot.send(f"{'🟢 매수' if side == 1 else '🔴 매도'} 진입 {lot} lot (마틴 {level + 1}/{MARTIN_STEPS}단계, {strength} x{w:g})\n"
             f"레인지 {info_row['range_lo']} ~ {info_row['range_hi']} 돌파, 일봉 ATR {info_row['daily_atr']}\n"
             f"가격 {fill} | SL {sl} | TP {tp}\n"
             f"최대 손실 약 {sl_usd * lot * CONTRACT:,.0f}$ / 목표 이익 약 {tp_usd * lot * CONTRACT:,.0f}$\n"
             f"{bot.account_line()}\n{bot.summary()}")
    return row


def notify_close(row):
    profit = float(row["profit"])
    icon = {"TP": "✅ 익절", "SL": "❌ 손절", "TIME": "⏹ 시간 청산"}.get(row["result"], "⏹ 청산")
    nxt = martin_level()
    log.info(f"청산 {row['result']} @ {row['close_price']} 손익 {profit} -> 다음 진입 {nxt + 1}단계")
    bot.send(f"{icon} {row.get('side', '')} {row.get('lot', '')} lot\n"
             f"{row.get('fill_price', '')} → {row['close_price']}\n손익 {profit:+,.2f}$\n"
             f"다음 진입: 마틴 {nxt + 1}/{MARTIN_STEPS}단계\n{bot.account_line()}\n{bot.summary()}")


# ---------------------------------------------------------------- 메인
def main():
    global bot, CONTRACT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="주문 없이 신호만 계산")
    ap.add_argument("--live", action="store_true", help="실계좌에서도 실행 (기본은 데모 전용)")
    ap.add_argument("--symbol", help="금 말고 다른 종목 (예: XAUEUR)")
    args = ap.parse_args()
    if args.live:
        use_live_paths()
    if args.symbol:
        use_symbol(args.symbol, args.live)
    bot = Trader(STRATEGY, SYMBOL, MAGIC, DEVIATION, LOG_DIR, "asia", live=args.live)
    acc = bot.start(args.dry_run)
    if args.symbol:
        # 계좌 통화(달러)로 1 lot이 가격 1 움직일 때 손익 (XAUEUR은 유로 손익을 달러로 바꿔서)
        info = mt5.symbol_info(SYMBOL)
        CONTRACT = info.trade_tick_value / info.trade_tick_size
    kind = bot.kind

    now = bot.clock.now()
    datr, mom, _, _ = daily_features(now.date())
    lvl = martin_level()
    msg = (f"오늘 일봉 ATR {datr:.2f}$ → U {datr * ATR_MULT:.2f}$ (손절 {datr * ATR_MULT * SL_RATIO:.2f}$ / "
           f"익절 {datr * ATR_MULT * TP_RATIO:.2f}$), 5일 모멘텀 {'+' if mom > 0 else '-' if mom < 0 else '0'}") \
        if datr else "일봉 데이터 부족"
    if datr:
        for name, w in (("강한 신호", W_STRONG), ("약한 신호", W_WEAK)):
            lot, _, risk = plan_lot(lvl, datr * ATR_MULT, w, save=False)
            aim = RISK_PCT * w * SL_RATIO * MARTIN_MULT ** lvl
            msg += (f"\n{lvl + 1}단계 {name} {lot} lot (손절 시 잔고의 {risk:.2f}%, 목표 {aim:.2f}%"
                    f"{' — 최소 lot 때문에 큼' if risk > aim * 1.3 else ''})")
    if early_close_day(now.date()):
        msg += f"\n오늘은 미국 휴일 조기 마감일 -> {EARLY_EXIT_AT}에 청산"
    log.info(f"시작: 계좌 {acc.login} ({acc.server}, {kind}) 잔고 {acc.balance}, 다음 진입 마틴 {lvl + 1}단계. {msg}"
             f"{' [DRY-RUN]' if args.dry_run else ''}")
    bot.send(f"▶️ 자동매매 시작 ({kind} {acc.login})\n전략: {STRATEGY_DESC}\n다음 진입 마틴 {lvl + 1}단계\n{msg}\n"
             f"{bot.account_line()}")
    if not args.dry_run:
        bot.warn_algo_off()

    open_row = bot.adopt_position()

    last_bar, last_close_try, day_info = None, 0.0, {}
    with bot.running("⏸ 자동매매 종료 (열린 포지션은 서버 SL/TP로 관리됨. 23:00 시간 청산은 스크립트가 켜져 있어야 합니다)"):
        while True:
            now = bot.keep_alive()
            today = now.date()

            if open_row and not any(p.ticket == open_row["ticket"] for p in bot.positions()):
                c = bot.find_close(open_row["ticket"], CLOSE_REASONS)
                if c:
                    open_row.update(c)
                    append_csv(TRADES_CSV, FIELDS, open_row)
                    notify_close(open_row)
                    open_row = None

            # 23:00 시간 청산, 조기 마감일은 19:30 (실패하면 1분 뒤 재시도)
            at = exit_at(today)
            if minutes(now) >= hm(at) and not args.dry_run and time.time() - last_close_try > 60:
                for p in bot.positions():
                    bot.close(p, "time exit", f"{at} 청산")
                    last_close_try = time.time()

            m5 = bot.rates(mt5.TIMEFRAME_M5, 400, min_bars=1)
            if m5 is None or m5.time.iat[-1] == last_bar:
                time.sleep(POLL_SEC)
                continue
            last_bar = m5.time.iat[-1]
            bar_min = minutes(last_bar)
            if last_bar.date() != today or not (hm(RANGE_UNTIL) <= bar_min < hm(ENTRY_UNTIL)):
                time.sleep(POLL_SEC)
                continue

            st = load_json(BOT_JSON)
            if st.get("done_date") == str(today):
                time.sleep(POLL_SEC)
                continue
            if day_info.get("date") != today:
                datr, mom, ph, pl = daily_features(today)
                day_info = {"date": today, "atr": datr, "mom": mom, "ph": ph, "pl": pl}
            rng = asian_range(m5, today)
            if rng is None or not day_info["atr"]:
                log.info("레인지 봉 부족 또는 일봉 ATR 없음 -> 오늘 거래 없음")
                st["done_date"] = str(today)
                save_json(BOT_JSON, st)
                continue
            hi, lo, nbars = rng
            brk = first_breakout(m5, today, hi, lo)
            if brk is None:
                time.sleep(POLL_SEC)
                continue
            bar, side, close = brk
            st["done_date"] = str(today)          # 첫 돌파가 나오면 결과와 상관없이 오늘은 끝
            save_json(BOT_JSON, st)
            u = day_info["atr"] * ATR_MULT
            strong = close > day_info["ph"] if side == 1 else close < day_info["pl"]
            w = W_STRONG if strong else W_WEAK
            desc = (f"봉 {bar:%m-%d %H:%M} 종가 {close} {'상향' if side == 1 else '하향'} 돌파 "
                    f"(레인지 {lo}~{hi}, {nbars}봉), 5일 모멘텀 {day_info['mom']:+d}, 일봉 ATR {day_info['atr']:.2f}, "
                    f"전일 고/저 {day_info['ph']}/{day_info['pl']} -> {'강한' if strong else '약한'} 신호 x{w:g}")
            if bar != last_bar:
                log.info(f"{desc} -> 첫 돌파가 이미 지난 봉이라 (재시작 등) 오늘은 진입 안 함")
            elif side != day_info["mom"]:
                log.info(f"{desc} -> 모멘텀과 반대라 오늘은 거래 없음")
            elif args.dry_run:
                lvl = martin_level()
                lot, _, risk = plan_lot(lvl, u, w, save=False)
                log.info(f"{desc} -> [DRY-RUN] {'매수' if side == 1 else '매도'} {lot} lot (마틴 {lvl + 1}단계, 리스크 {risk:.1f}%)")
            elif bot.positions():
                log.info(f"{desc} -> 포지션 보유 중이라 진입 안 함")
            else:
                lvl = martin_level()
                lot, _, risk = plan_lot(lvl, u, w)
                if reason := risk_block(now, lot):
                    log.info(f"{desc} -> 안전장치: {reason}")
                    bot.send(f"⛔ 안전장치: {reason}\n오늘은 진입하지 않습니다.")
                else:
                    log.info(f"{desc} -> 진입 (손절 시 잔고의 {risk:.2f}%)")
                    info_row = {"date": str(today), "signal_bar": bar.strftime("%Y-%m-%d %H:%M"), "range_hi": hi,
                                "range_lo": lo, "daily_atr": round(day_info["atr"], 2), "signal_price": close}
                    open_row = place(side, lot, lvl, u, w, info_row)
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
