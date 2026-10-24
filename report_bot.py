"""계좌별 텔레그램 리포트: 계좌 하나당 메시지 하나에 전략별(SAR 단타 / H4 추세 / 아시아 3마틴) 손익

보내는 시각: 한국시간 월~금 21:00
- 매일: 일일 리포트 = 지난 리포트(전 평일 21:00) 이후
- 금요일: 주간 리포트 추가 = 지난 금요일 21:00 이후
- 그 달 마지막 평일: 월간 리포트 추가 = 지난달 마지막 평일 21:00 이후
기간이 리포트 시각끼리 이어져서 빠지는 거래가 없음 (금요일 밤~토요일 새벽 거래는 다음 월요일 일일·다음 주 주간에 들어감)
손익 계산은 reports.stats와 같음 (실현손익, 수익률 = 손익 / (기간 시작 잔고 + 기간 중 입금)). MT5 거래내역은 서버시간이라 한국시간을 바꿔서 조회

기록: report_log/ (실계좌는 report_live_log/) — state.json(마지막으로 보낸 날짜), events.log

사용법:
  python report_bot.py           # 데모 계좌 리포트 (Ctrl+C로 종료)
  python report_bot.py --live    # 실계좌 리포트 (실계좌 봇과 같은 환경변수로 실행)
  python report_bot.py --now     # 지금 바로 오늘 리포트를 한 번 보내고 끝 (확인용, 보낸 기록은 안 남김)
"""
import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5

import asia_breakout_trader
import demo_trader
import trend_trader
from common.trader import Trader, account_kind, load_json
from config import SYMBOL
from reports import balance_at, deposits, last_weekday_of_month, stats

KST = timezone(timedelta(hours=9))
REPORT_HOUR = 21
POLL_SEC = 20
SFX = "#" if SYMBOL.endswith("#") else ""   # 실계좌 종목명은 뒤에 #
STRATS = (("SAR 단타", demo_trader.MAGIC, SYMBOL), ("H4 추세", trend_trader.MAGIC, SYMBOL),
          ("아시아 3마틴", asia_breakout_trader.MAGIC, SYMBOL),
          ("아시아 3마틴·XAUEUR", asia_breakout_trader.SYMBOL_MAGIC["XAUEUR"], "XAUEUR" + SFX))
WEEKDAYS = "월화수목금토일"

LOG_DIR = Path(__file__).resolve().parent / "report_log"


def prev_weekday(d):
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def at_report(d):
    return datetime(d.year, d.month, d.day, REPORT_HOUR, tzinfo=KST)


def server_hours(clock):
    """서버시간 - UTC (시간 단위, 서머타임 3 / 겨울 2)"""
    return round((clock.now() - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds() / 3600)


def section(title, start, end, off):
    """start ~ end(한국시간) 전략별 손익 블록"""
    s = (start - timedelta(hours=9 - off)).replace(tzinfo=None)
    e = (end - timedelta(hours=9 - off)).replace(tzinfo=None)
    b0, b1, dep = balance_at(s), balance_at(e), deposits(s, e)
    lines = [f"{title} ({start:%m/%d %H:%M} ~ {end:%m/%d %H:%M})"]
    total = 0.0
    for name, magic, sym in STRATS:
        st = stats(s, e, magic, sym)
        total += st["pnl"]
        n, w = st["trades"], st["wins"]
        trades = f"{n}건 (익절 {w} / 손절 {n - w}, 승률 {w / n * 100:.0f}%)" if n else "거래 없음"
        lines.append(f"• {name}: {st['pct']:+.2f}% ({st['pnl']:+,.2f}$) · {trades}")
    lines.append(f"합계 {total / (b0 + dep) * 100 if b0 + dep > 0 else 0:+.2f}% ({total:+,.2f}$)")
    lines.append(f"잔고 {b0:,.2f} → {b1:,.2f}" + (f" (입금 +{dep:,.2f}$ 포함, 수익률은 입금액까지 기준)" if dep else ""))
    return "\n".join(lines)


def report_text(today, now, off):
    acc = mt5.account_info()
    parts = [f"📊 {account_kind(acc)} {acc.login} 리포트 {today:%m/%d}({WEEKDAYS[today.weekday()]})",
             section("일일", at_report(prev_weekday(today)), now, off)]
    if today.weekday() == 4:
        parts.append(section("📅 주간", at_report(today - timedelta(days=7)), now, off))
    if today == last_weekday_of_month(today):
        prev_month_end = last_weekday_of_month(today.replace(day=1) - timedelta(days=1))
        parts.append(section(f"🗓 월간 {today:%Y-%m}", at_report(prev_month_end), now, off))
    floating = acc.equity - acc.balance
    if abs(floating) >= 0.01:
        parts.append(f"열린 포지션 평가손익 {floating:+,.2f}$ (위 손익에는 미포함)")
    return "\n\n".join(parts)


def main():
    global LOG_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="실계좌 리포트")
    ap.add_argument("--now", action="store_true", help="지금 바로 한 번 보내고 끝")
    args = ap.parse_args()
    if args.live:
        LOG_DIR = LOG_DIR.parent / "report_live_log"
    state_path = LOG_DIR / "state.json"
    bot = Trader("리포트", SYMBOL, None, 0, LOG_DIR, "report", live=args.live)   # 주문은 안 냄
    acc = bot.start()
    bot.name = f"리포트·{bot.kind}"
    log, clock = bot.log, bot.clock

    if args.now:
        now = datetime.now(KST).replace(second=0, microsecond=0)
        bot.send(report_text(now.date(), now, server_hours(clock)))
        bot.tg.flush()
        mt5.shutdown()
        return

    state = load_json(state_path)
    if not state:   # 첫 실행: 오늘 21:00이 이미 지났으면 오늘 것은 보낸 것으로 침
        now = datetime.now(KST)
        first = now.date() if now.hour >= REPORT_HOUR else now.date() - timedelta(days=1)
        state = {"last": first.isoformat()}
        state_path.write_text(json.dumps(state), encoding="utf-8")
    log.info(f"시작: 계좌 {acc.login} ({bot.kind}), 평일 {REPORT_HOUR}:00 (한국시간) 리포트, 마지막 전송 {state['last']}")
    with bot.running("⏸ 리포트 봇 종료"):
        while True:
            bot.keep_alive()
            now = datetime.now(KST).replace(second=0, microsecond=0)
            today = now.date()
            if today.weekday() < 5 and now.hour >= REPORT_HOUR and state["last"] < today.isoformat():
                text = report_text(today, now, server_hours(clock))
                bot.send(text)
                state["last"] = today.isoformat()
                state_path.write_text(json.dumps(state), encoding="utf-8")
                log.info("리포트 전송\n" + text)
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
