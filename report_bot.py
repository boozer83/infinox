"""계좌별 텔레그램 리포트: 계좌 하나당 메시지 하나에 전략별(매직 번호별) 손익. 그 기간에 거래가 있는 전략만,
모르는 매직(수동 거래 포함)은 '기타'로 묶어서 합계가 잔고 변화(입금 제외)와 맞음

보내는 시각: 한국시간 월~금 21:00
- 매일: 일일 리포트 = 지난 리포트(전 평일 21:00) 이후
- 금요일: 주간 리포트 추가 = 지난 금요일 21:00 이후
- 그 달 마지막 평일: 월간 리포트 추가 = 지난달 마지막 평일 21:00 이후
기간이 리포트 시각끼리 이어져서 빠지는 거래가 없음 (금요일 밤~토요일 새벽 거래는 다음 월요일 일일·다음 주 주간에 들어감)
손익 계산은 reports.stats와 같은 방식 (실현손익, 수익률 = 손익 / (기간 시작 잔고 + 기간 중 입금)). MT5 거래내역은 서버시간이라 한국시간을 바꿔서 조회

기록: logs/report_log/ (실계좌는 logs/report_live_log/) — state.json(마지막으로 보낸 날짜), events.log

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
import gsr_trader
import hedge_martin_trader
import index_trader
import news_surprise_paper
import news_trader
import pa_master_trader
import trend_trader
from common.trader import Trader, account_kind, load_json
from config import SYMBOL
from reports import balance_at, deposits, epoch, last_weekday_of_month

KST = timezone(timedelta(hours=9))
REPORT_HOUR = 21
POLL_SEC = 20
# 매직 -> 이름 (이 순서로 표시). DOM·SAR 마틴은 demo_trader 매직을 바꿔 쓰는 봇이라 번호를 직접 적음 (sar_dom_trader.py, sar_martin_trader.py)
STRATS = {demo_trader.MAGIC: "SAR 단타", 20261022: "SAR 마틴", 20261029: "SAR DOM skip", 20261030: "SAR DOM flip",
          trend_trader.MAGIC: "H4 추세", trend_trader.SYMBOL_MAGIC["XETUSD"]: "H4 추세·이더",
          asia_breakout_trader.MAGIC: "아시아 3마틴", asia_breakout_trader.SYMBOL_MAGIC["XAUEUR"]: "아시아 3마틴·XAUEUR",
          hedge_martin_trader.MAGIC: "헤지마틴", index_trader.MAGICS["NAS100"]: "나스닥 첫30분",
          index_trader.MAGICS["GER40"]: "독일 ORB 매도", news_surprise_paper.MAGIC: "서프라이즈", news_trader.MAGIC: "뉴스",
          pa_master_trader.MAGIC: "PA Master", gsr_trader.MAGIC: "금은비 GSR"}
WEEKDAYS = "월화수목금토일"

LOG_DIR = Path(__file__).resolve().parent / "logs" / "report_log"


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


def by_strategy(s, e):
    """기간(서버시간) 안에 청산된 포지션의 손익을 매직별로: {이름: [포지션 손익 ...]} (진입 수수료·스왑 포함)"""
    deals = mt5.history_deals_get(epoch(s - timedelta(days=60)), epoch(e)) or ()   # 진입이 기간 전인 포지션의 진입 수수료까지
    t0, t1 = epoch(s), epoch(e)
    pos_pnl, closed = {}, {}
    for d in deals:
        if d.type not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL):
            continue
        pos_pnl[d.position_id] = pos_pnl.get(d.position_id, 0.0) + d.profit + d.commission + d.swap + getattr(d, "fee", 0.0)
        if d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY) and t0 <= d.time < t1:
            closed[d.position_id] = d.magic
    out = {}
    for pid, magic in closed.items():
        out.setdefault(STRATS.get(magic, "기타"), []).append(pos_pnl[pid])
    order = list(STRATS.values()) + ["기타"]
    return dict(sorted(out.items(), key=lambda kv: order.index(kv[0])))


def section(title, start, end, off):
    """start ~ end(한국시간) 전략별 손익 블록"""
    s = (start - timedelta(hours=9 - off)).replace(tzinfo=None)
    e = (end - timedelta(hours=9 - off)).replace(tzinfo=None)
    b0, b1, dep = balance_at(s), balance_at(e), deposits(s, e)
    base = b0 + dep
    lines = [f"{title} ({start:%m/%d %H:%M} ~ {end:%m/%d %H:%M})"]
    total = 0.0
    groups = by_strategy(s, e)
    for name, pnls in groups.items():
        pnl, n, w = sum(pnls), len(pnls), sum(p > 0 for p in pnls)
        total += pnl
        pct = pnl / base * 100 if base > 0 else 0.0
        lines.append(f"• {name}: {pct:+.2f}% ({pnl:+,.2f}$) · {n}건 (이익 {w} / 손실 {n - w}, 승률 {w / n * 100:.0f}%)")
    if not groups:
        lines.append("• 거래 없음")
    lines.append(f"합계 {total / base * 100 if base > 0 else 0:+.2f}% ({total:+,.2f}$)")
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
