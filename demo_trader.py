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
성적 기반 쉬기:
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
  demo_log/risk_state.json에 "halted"로 남아서 재시작해도 유지됨. 확인 후 그 파일을 지우면 다시 시작

기록 (demo_log/ 폴더): trades.csv (거래별 신호/체결/슬리피지/결과), events.log (실행 로그)

텔레그램 (telegram.json 설정이 있을 때, 설정법은 telegram_notify.py 참고):
- 진입/청산 때마다 거래 내용과 오늘/이번주/이번달 수익률
- 일일(월~금 23:58 서버시간), 주간(금요일), 월간(마지막 거래일) 리포트
- 시작/종료, 마틴 정지, 금요일 청산, 주문 실패

사용법:
  python demo_trader.py            # 실행 (Ctrl+C로 종료)
  python demo_trader.py --dry-run  # 주문 없이 신호만 계산해서 출력 (텔레그램 전송 안 함)
  python demo_trader.py --live     # 실계좌에서 실행 (데모에서 충분히 확인한 뒤에만)
"""
import argparse
import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from config import SYMBOL, mt5_init
from indicators import atr, psar, rsi
from reports import ReportScheduler, ServerClock, summary_line
from telegram_notify import Notifier

# 전략
SL_USD = 7.5
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
# 마틴 lot 상한: 그날 시작 lot의 몇 배까지
MAX_LOT_MULT = 5
# 성적 기반 쉬기: 최근 N개 결과(실제 + 가상) 승률이 이 값(%) 미만이면 가상 진입만
WINRATE_N = 200
WINRATE_MIN = 71.4
FRIDAY_CLOSE = "23:30"
# 실계좌 안전장치
MAX_LOT = 1.0             # 한 번에 진입할 최대 lot (넘으면 이 lot으로 깎아서 진입)
DAILY_LOSS_PCT = None     # 하루 손실 한도 (그날 시작 잔고 대비 %), None = 끔. 15%는 4연속 손절 날 마틴 회복을 끊었음
ACCOUNT_STOP_PCT = 50.0   # 계좌 손실 한도 (최고 잔고 대비 평가금 감소 %). 30%는 2025-08 낙폭(34%)에 걸려 멈췄음
# 주문
MAGIC = 20261017
DEVIATION = 30            # 허용 슬리피지 (포인트, 0.30달러)
POLL_SEC = 1.0
M5_BARS = 600
H1_BARS = 400

LOG_DIR = Path(__file__).resolve().parent / "demo_log"
TRADES_CSV = LOG_DIR / "trades.csv"
STATE_JSON = LOG_DIR / "state.json"
RISK_JSON = LOG_DIR / "risk_state.json"
WINRATE_JSON = LOG_DIR / "winrate.json"
FIELDS = ["signal_bar", "side", "level", "lot", "signal_price", "order_time", "fill_price", "slippage", "spread",
          "sl", "tp", "ticket", "close_time", "close_price", "profit", "result"]

log = logging.getLogger("demo")
tg = None          # Notifier, main()에서 설정
clock = None       # ServerClock, main()에서 설정


STRATEGY = "SAR 단타"
STRATEGY_DESC = ("5분봉 SAR 전환 + RSI50 + 1시간봉 SAR, 익절 3$ / 손절 7.5$, "
                 + ("일일 마틴 최대 5번째 진입 (lot 상한 시작 lot의 5배)" if MARTIN else
                    f"고정 lot (1,000$당 {LOT_PER_1000} lot, 마틴 없음)")
                 + ", 5분봉 ATR 2.5$ 미만 쉼, ATR 20일 중앙값 2배 초과 쉼, 최근 200건 승률 71.4% 미만 쉼")


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


def connect(allow_live=False):
    if not mt5_init():
        raise SystemExit(f"MT5 연결 실패: {mt5.last_error()} (MT5를 켜고 로그인했는지, 경로 설정은 README 참고)")
    check_demo(allow_live)
    if not mt5.symbol_select(SYMBOL, True):
        raise SystemExit(f"{SYMBOL} 종목을 찾을 수 없습니다")


def check_demo(allow_live=False, login=None):
    """데모 계좌인지 확인 (allow_live면 실계좌도 허용). login을 주면 그 계좌에서 바뀌었는지도 확인"""
    acc = mt5.account_info()
    if acc is None:
        raise SystemExit("로그인된 계좌가 없습니다")
    if login is not None and acc.login != login:
        mt5.shutdown()
        raise SystemExit(f"로그인 계좌가 바뀌었습니다 ({login} -> {acc.login}). 안전을 위해 종료합니다.")
    if acc.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO and not allow_live:
        mt5.shutdown()
        raise SystemExit(f"데모 계좌가 아닙니다 ({acc.login}, {acc.server}). 실계좌는 --live 옵션으로만 실행합니다.")
    if acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO and allow_live:
        mt5.shutdown()
        raise SystemExit(f"--live인데 데모 계좌에 연결됐습니다 ({acc.login}). MT5_TERMINAL_PATH / MT5_PORTABLE / "
                         f"MT5_SYMBOL 환경변수로 실계좌 터미널을 지정하세요.")
    return acc


def account_kind(acc):
    return "데모" if acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else "실계좌"


def load_risk():
    try:
        return json.loads(RISK_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def risk_block(now, lot):
    """신규 진입을 막아야 하면 이유(문자열), 아니면 None"""
    acc = mt5.account_info()
    risk = load_risk()
    if risk.get("halted"):
        return f"계좌 손실 한도로 중단된 상태 ({risk['halted']}). 확인 후 {RISK_JSON.name}을 지우면 다시 시작"
    peak = max(risk.get("peak_balance", 0.0), acc.balance)
    if peak != risk.get("peak_balance"):
        risk["peak_balance"] = peak
        RISK_JSON.write_text(json.dumps(risk, ensure_ascii=False, indent=2), encoding="utf-8")
    if acc.equity <= peak * (1 - ACCOUNT_STOP_PCT / 100):
        risk["halted"] = f"{now:%Y-%m-%d %H:%M} 평가금 {acc.equity:,.2f} / 최고 잔고 {peak:,.2f}"
        RISK_JSON.write_text(json.dumps(risk, ensure_ascii=False, indent=2), encoding="utf-8")
        send(f"🛑 계좌 손실 한도({ACCOUNT_STOP_PCT:g}%) 도달 -> 신규 진입을 중단합니다.\n{risk['halted']}\n"
             f"확인 후 {LOG_DIR.name}/{RISK_JSON.name}을 지우고 다시 실행하세요.")
        return f"계좌 손실 한도 도달 ({risk['halted']})"
    day_start = int(datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp())
    deals = mt5.history_deals_get(day_start, day_start + 2 * 86400) or ()
    trade_deals = [d for d in deals if d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL)]
    start_bal = acc.balance - sum(d.profit + d.commission + d.swap for d in trade_deals)
    mine = sum(d.profit + d.commission + d.swap for d in trade_deals
               if d.magic == MAGIC and d.entry == mt5.DEAL_ENTRY_OUT)
    if DAILY_LOSS_PCT is not None and start_bal > 0 and mine <= -DAILY_LOSS_PCT / 100 * start_bal:
        return f"하루 손실 한도 도달 (오늘 {mine:+,.2f} / 시작 잔고 {start_bal:,.2f}의 {DAILY_LOSS_PCT:g}%)"
    return None


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


def atr_ref():
    """최근 ATR_REF_BARS개 완성 5분봉의 ATR(14) 중앙값 (급변동 쉬기 기준). 기록이 부족하면 None"""
    m5 = rates(mt5.TIMEFRAME_M5, ATR_REF_BARS + 100)
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
        day_start = int(datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp())
        deals = mt5.history_deals_get(day_start, day_start + 2 * 86400) or ()
    today = sum(d.profit + d.commission + d.swap for d in deals
                if d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL))
    info = mt5.symbol_info(SYMBOL)
    raw = (mt5.account_info().balance - today) / 1000 * LOT_PER_1000
    return max(info.volume_min, round(int(raw / info.volume_step + 1e-9) * info.volume_step, 2))


def martin_state(now):
    """오늘(서버 날짜) 이 스크립트가 청산한 거래로 (현재 단계, 오늘 정지 여부, 시작 lot) 계산"""
    day_start = int(datetime(now.year, now.month, now.day, tzinfo=timezone.utc).timestamp())
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
    try:
        wr = json.loads(WINRATE_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        wr = {}
    wr.setdefault("results", [])
    wr.setdefault("virtual", None)
    wr.setdefault("paused", False)
    return wr


def save_wr(wr):
    WINRATE_JSON.write_text(json.dumps(wr, ensure_ascii=False, indent=1), encoding="utf-8")


def winrate(wr):
    """최근 WINRATE_N개 승률(%)과 개수"""
    last = wr["results"][-WINRATE_N:]
    return (sum(last) / len(last) * 100 if last else 0.0), len(last)


def is_paused(wr):
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
            send(f"⏸ 최근 {n}건 승률 {rate:.1f}% < {WINRATE_MIN}% -> 실제 진입을 쉬고 가상 진입만 기록합니다.\n"
                 f"승률이 {WINRATE_MIN}% 이상으로 돌아오면 자동으로 다시 시작합니다.")
        else:
            send(f"▶️ 최근 {n}건 승률 {rate:.1f}% -> 실제 진입을 다시 시작합니다.")
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
    log.info(f"진입 {row['side']} {lot}lot ({level_text(level)}) @ {fill} "
             f"(신호봉 종가 {signal_price}, SL {sl}, TP {tp}, 스프레드 {row['spread']})")
    send(f"{'🟢 매수' if side == 1 else '🔴 매도'} 진입 {lot} lot ({level_text(level)})\n"
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
    level, stopped, base = martin_state(clock.now())
    icon = {"TP": "✅ 익절", "SL": "❌ 손절", "FRIDAY": "⏹ 금요일 청산"}.get(row["result"], "⏹ 청산")
    nxt = "오늘 거래 정지 (5번째 진입 손절)" if stopped else f"다음 진입 {level_text(level)} ({lot_for(level, base)} lot)"
    log.info(f"청산 {row['result']} @ {row['close_price']} 손익 {profit} -> {nxt}")
    send(f"{icon} {row.get('side', '')} {row.get('lot', '')} lot\n"
            f"{row.get('fill_price', '')} → {row['close_price']}\n"
            f"손익 {profit:+,.2f}$ ({pct:+.2f}%)\n{nxt}\n"
            f"{account_line()}\n{summary_line(clock.now(), MAGIC, SYMBOL)}")
    if stopped:
        send("⛔ 5번째 진입(최대 마틴)에서 손절 -> 오늘은 신규 진입을 멈춥니다. 내일 1번째부터 다시 시작합니다.")


def use_live_paths():
    """실계좌 실행은 기록을 live_log/에 따로 남기고, 텔레그램에 실계좌로 표시"""
    global LOG_DIR, TRADES_CSV, STATE_JSON, RISK_JSON, WINRATE_JSON, STRATEGY
    LOG_DIR = LOG_DIR.parent / "live_log"
    TRADES_CSV, STATE_JSON, RISK_JSON = LOG_DIR / "trades.csv", LOG_DIR / "state.json", LOG_DIR / "risk_state.json"
    WINRATE_JSON = LOG_DIR / "winrate.json"
    STRATEGY = "SAR 단타·실계좌"


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
    scheduler = ReportScheduler(STATE_JSON, MAGIC, SYMBOL, send)
    base = base_lot(clock.now())
    lots = " / ".join(f"{lot_for(i, base):g}" for i in range(MARTIN_MAX_LEVEL + 1 if MARTIN else 1))
    lots += f" lot (오늘 시작 잔고 기준, 1,000$당 {LOT_PER_1000} lot{'' if MARTIN else ', 마틴 없음'})"
    daily = "꺼짐" if DAILY_LOSS_PCT is None else f"{DAILY_LOSS_PCT:g}%"
    wr = load_wr()
    rate, n = winrate(wr)
    limits = (f"최대 lot {MAX_LOT} (넘으면 {MAX_LOT}로 진입){f' (마틴은 시작 lot의 {MAX_LOT_MULT}배까지)' if MARTIN else ''}, 하루 손실 한도 {daily}, "
              f"계좌 손실 한도 {ACCOUNT_STOP_PCT:g}%, ATR {MIN_ATR:g}$ 미만 쉼, "
              f"ATR이 20일 중앙값의 {MAX_ATR_MULT:g}배 초과면 쉼, "
              f"최근 {WINRATE_N}건 승률 {WINRATE_MIN:g}% 미만 쉼 (지금 {n}건 {rate:.1f}%{', 쉬는 중' if is_paused(wr) else ''})")
    log.info(f"시작: 계좌 {acc.login} ({acc.server}, {kind}) 잔고 {acc.balance} {acc.currency}, {SYMBOL} "
             f"SL {SL_USD} TP {TP_USD}, {'마틴 ' if MARTIN else ''}{lots}, {limits}{' [DRY-RUN]' if args.dry_run else ''}"
             f", 텔레그램 {'켜짐' if tg.enabled else '꺼짐'}")
    send(f"▶️ 자동매매 시작 ({kind} {acc.login})\n전략: {STRATEGY_DESC}\n{'마틴 ' if MARTIN else 'lot '}{lots}\n안전장치: {limits}\n"
            f"{account_line()}\n{summary_line(clock.now(), MAGIC, SYMBOL)}")
    if load_risk().get("halted"):
        send(f"🛑 계좌 손실 한도로 중단된 상태입니다: {load_risk()['halted']}\n"
             f"확인 후 {LOG_DIR.name}/{RISK_JSON.name}을 지우면 다시 진입합니다.")
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
    risk_notified = None      # 안전장치 알림은 하루 한 번만
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

            # 진입했던 포지션이 닫혔는지 확인
            if open_row and not any(p.ticket == open_row["ticket"] for p in my_positions()):
                c = find_close(open_row["ticket"])
                if c:
                    open_row.update(c)
                    append_row(open_row)
                    notify_close(open_row)
                    record_result(wr, float(open_row["profit"]) > 0, "실제")
                    open_row = None
            check_virtual(wr)

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
                    level, stopped, base = martin_state(now)
                    if my_positions():
                        log.info("포지션 보유 중이라 신호 무시")
                    elif wr["virtual"]:
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
                    elif reason := risk_block(now, lot_for(level, base)):
                        log.info(f"안전장치: {reason} -> 신규 진입 안 함")
                        if risk_notified != now.date():
                            risk_notified = now.date()
                            send(f"⛔ 안전장치: {reason}\n오늘은 신규 진입을 멈춥니다.")
                    elif is_paused(wr):
                        open_virtual(wr, side, bar.strftime("%Y-%m-%d %H:%M"))
                    else:
                        open_row = place(side, lot_for(level, base), level, bar.strftime("%Y-%m-%d %H:%M"), close)
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
