"""SAR 단타 + 2마틴 (진 만큼만 되찾는 lot) + 2연속 손절이면 그날 쉬기 + 매일 초기화

backtests/sar_martin.py "2마틴 + 2연속 손절 쉬기 + 매일 초기화"와 같은 규칙
- 진입·청산은 demo_trader.py(SAR 단타)와 똑같음 (같은 신호, 손절 7.5 / 익절 3, 반대 콩 청산, 쉬기 필터 전부)
- 지면 다음 lot = max(시작 lot, 이번 묶음 누적 손실$ / (익절 3$ x 100)) -> 익절 한 번이면 진 금액 회수 (더 벌진 않음)
  묶음 합계가 0 이상이 되면 시작 lot으로. 2마틴(3번째 진입)까지
- 오늘 2연속 손절이면 그날은 신규 진입 안 함
- 서버 날짜가 바뀌면 묶음 초기화 (어제 손실은 확정, 시작 lot부터)
백테스트 (2022~24 ATR 금값 비례, 1,000$ 시작 복리): 2022 -12%, 2023 -17%, 2024 +35%, 2025 +19%, 2026 +1,531%,
  최대 낙폭 43%, 최악의 날 -15%, 최대 lot 시작 lot의 2.9배 (마틴 없음: 낙폭 53%, 최악의 날 -13%)
demo_trader.py의 함수를 그대로 쓰고 magic·기록 폴더·마틴 계산만 바꿈. 기록: logs/sar_martin_log/ (실계좌는 logs/sar_martin_live_log/)

사용법: python sar_martin_trader.py          (데모 계좌)
        python sar_martin_trader.py --live   (실계좌. SAR 단타 실계좌 봇과 같은 신호라 같은 시각에 따로 진입함)
"""
import math

import MetaTrader5 as mt5

import demo_trader as dt
from config import SYMBOL

MAX_STEP = 2            # 마틴 단계 (0 = 시작 lot, 2 = 2마틴)
STOP_STREAK = 2         # 오늘 이만큼 연속 손절이면 그날 쉼
TP_OZ = dt.TP_USD
CHAIN = 0.0             # 이번 묶음 누적 손익 $ (martin_state가 계산)

dt.MAGIC = 20261022
dt.BRIDGE_ID = "SARM"
dt.MARTIN = True
dt.MARTIN_MAX_LEVEL = MAX_STEP
dt.LOG_DIR = dt.LOG_DIR.parent / "sar_martin_log"
dt.TRADES_CSV, dt.STATE_JSON = dt.LOG_DIR / "trades.csv", dt.LOG_DIR / "state.json"
dt.RISK_JSON, dt.WINRATE_JSON = dt.LOG_DIR / "risk_state.json", dt.LOG_DIR / "winrate.json"
dt.STRATEGY = "단타 2마틴"
dt.STRATEGY_DESC = (f"SAR 단타와 같은 진입·청산, 지면 다음 lot은 진 금액을 익절 {TP_OZ:g}$로 딱 되찾는 크기 ({MAX_STEP}마틴까지), "
                    f"오늘 {STOP_STREAK}연속 손절이면 그날 쉼, 매일 초기화")


def use_live_paths():
    """실계좌 실행은 기록을 logs/sar_martin_live_log/에 (SAR 단타 실계좌의 live_log와 따로, 뮤텍스 이름도 이 폴더 이름)"""
    dt.LOG_DIR = dt.LOG_DIR.parent / "sar_martin_live_log"
    dt.TRADES_CSV, dt.STATE_JSON = dt.LOG_DIR / "trades.csv", dt.LOG_DIR / "state.json"
    dt.RISK_JSON, dt.WINRATE_JSON = dt.LOG_DIR / "risk_state.json", dt.LOG_DIR / "winrate.json"


def martin_state(now):
    """오늘(서버 날짜) 이 봇이 청산한 거래로 (단계, 오늘 쉼 여부, 시작 lot). CHAIN도 갱신"""
    global CHAIN
    day_start = dt.day_start_ts(now)
    deals = mt5.history_deals_get(day_start, day_start + 2 * 86400) or ()
    closes = sorted((d for d in deals if d.magic == dt.MAGIC and d.symbol == SYMBOL and d.entry == mt5.DEAL_ENTRY_OUT),
                    key=lambda d: d.time_msc)
    chain, level, streak = 0.0, 0, 0
    for d in closes:
        p = d.profit + d.commission + d.swap
        streak = streak + 1 if p < 0 else 0
        chain += p
        if chain >= 0 or level >= MAX_STEP:
            chain, level = 0.0, 0
        else:
            level += 1
    CHAIN = chain
    return level, streak >= STOP_STREAK, dt.base_lot(now, deals)


def lot_for(level, base):
    info = mt5.symbol_info(SYMBOL)
    lot = base
    if level > 0:
        lot = max(base, math.ceil(-CHAIN / (TP_OZ * 100) / info.volume_step) * info.volume_step)
    lot = min(lot, base * dt.MAX_LOT_MULT, dt.MAX_LOT)
    lot = round(round(lot / info.volume_step) * info.volume_step, 2)
    return min(max(lot, info.volume_min), info.volume_max)


def level_text(level):
    return "시작 lot" if level == 0 else f"{level}마틴 (묶음 {CHAIN:+,.2f}$ 회수)"


def notify_close(row):
    profit = float(row["profit"])
    bal = mt5.account_info().balance
    pct = profit / (bal - profit) * 100 if bal - profit > 0 else 0.0
    level, stopped, base = martin_state(dt.bot.clock.now())
    icon = {"TP": "✅ 익절", "SL": "❌ 손절", "FRIDAY": "⏹ 금요일 청산", "SAR": "↩️ 반대 콩 청산"}.get(row["result"], "⏹ 청산")
    nxt = f"오늘 {STOP_STREAK}연속 손절 -> 오늘은 쉼 (내일 시작 lot부터)" if stopped else \
        f"다음 진입 {level_text(level)} ({lot_for(level, base)} lot)"
    dt.log.info(f"청산 {row['result']} @ {row['close_price']} 손익 {profit} -> {nxt}")
    dt.bot.send(f"{icon} {row.get('side', '')} {row.get('lot', '')} lot\n"
                f"{row.get('fill_price', '')} → {row['close_price']}\n"
                f"손익 {profit:+,.2f}$ ({pct:+.2f}%)\n{nxt}\n"
                f"{dt.bot.account_line()}\n{dt.bot.summary()}")


dt.martin_state, dt.lot_for, dt.level_text, dt.notify_close = martin_state, lot_for, level_text, notify_close
dt.use_live_paths = use_live_paths

if __name__ == "__main__":
    dt.main()
