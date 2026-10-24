"""데모 전용 뉴스 양방향 봇 (미국 CPI·고용·FOMC 금리 결정만, docs/hk_program.md 5-12 ~ 5-15)

규칙 (백테스트 backtests/news_cut.py '비율 0.05% / 15분'과 같음)
1) 발표 PRE_SEC초 전 매수·매도 같은 lot 시장가 진입 (lot = 잔고 1,000달러당 LOT_PER_1000)
2) 발표 시각 중간가 기준으로 중간가가 TH_PCT% 움직이면 반대쪽 청산 (위로 가면 매도 청산, 아래로 가면 매수 청산)
3) 발표 뒤 QUIET_SEC초 안에 판단이 안 나면 양쪽 정리
4) 남은 쪽: 발표 HOLD_SEC초 뒤 정리. 그 전에 발표 시각 중간가 반대쪽으로 REVERT$ 되돌아오면 바로 정리
발표 시각: 서버 시간 = 뉴욕 시간 + 7시간 (서머타임 따라 같이 바뀜) -> CPI·고용 15:30, FOMC 21:00
  날짜는 MT5 캘린더 내보내기(Common\\Files\\calendar_all.csv, mql5/ExportCalendar.mq5를 Currency 빈칸으로)에서 읽음
목적: 발표 때 실제 체결가·스프레드·미끄러짐 측정. 매 발표 결과를 news_log/trades.csv와 텔레그램으로

사용법: python news_trader.py           (데모 계좌에서만)
        python news_trader.py --dry     (앞으로 할 발표 일정만 보고 끝)
        python news_trader.py --test 90 (90초 뒤를 가짜 발표 시각으로 한 번 해 보고 끝, 데모 주문이 실제로 나감)
"""
import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from common.trader import Trader, append_csv, side_of
from config import SYMBOL

LOT_PER_1000 = 0.05
PRE_SEC = 30
TH_PCT = 0.05
QUIET_SEC = 300
HOLD_SEC = 900
REVERT = 2.0
MAX_SPREAD = 2.0          # 진입 직전 스프레드가 이보다 넓으면 이번 발표는 건너뜀
EVENTS = {"CPI m/m": "08:30", "실업률": "08:30", "연방 금리 결정": "14:00"}   # 캘린더 이름 -> 뉴욕 발표 시각
MAGIC = 20261021
STRATEGY = "뉴스 양방향"
CALENDAR = Path(os.environ["APPDATA"]) / "MetaQuotes" / "Terminal" / "Common" / "Files" / "calendar_all.csv"
LOG_DIR = Path(__file__).resolve().parent / "news_log"
TRADES_CSV = LOG_DIR / "trades.csv"
STATE_JSON = LOG_DIR / "state.json"
FIELDS = ["event", "release", "lot", "entry_time", "buy_fill", "sell_fill", "entry_spread", "release_mid", "threshold",
          "decide_time", "kept", "decide_mid", "cut_fill", "cut_slip", "cut_pnl_oz", "close_time", "close_reason",
          "close_fill", "close_slip", "keep_pnl_oz", "total_oz", "total_usd"]
NY = ZoneInfo("America/New_York")
DEVIATION = 300           # 발표 때는 가격이 튀어서 넉넉하게 (포인트)
log = logging.getLogger("news")
bot = None         # Trader, main()에서 설정


def schedule(now):
    """앞으로 2주 안 발표 [(서버 시각, 이름)]. 캘린더 파일이 없거나 오래되면 경고 문구도 같이"""
    if not CALENDAR.exists():
        return [], f"캘린더 파일 없음 ({CALENDAR})"
    c = pd.read_csv(CALENDAR, encoding="cp949")
    c = c[(c.currency == "USD") & (c.importance == "high") & c.event.isin(list(EVENTS))]
    out = {}
    for day, name in zip(pd.to_datetime(c.time, format="%Y.%m.%d %H:%M").dt.date, c.event):
        hh, mm = map(int, EVENTS[name].split(":"))
        ny = datetime(day.year, day.month, day.day, hh, mm, tzinfo=NY)
        server = (ny + timedelta(hours=7)).replace(tzinfo=None)   # 서버 = 뉴욕 + 7시간
        if now - timedelta(minutes=20) < server < now + timedelta(days=14):
            out.setdefault(server, name)                         # CPI·고용이 같은 날이면 한 번만
    age = (time.time() - CALENDAR.stat().st_mtime) / 86400
    warn = f"캘린더 파일이 {age:.0f}일 지났습니다. MT5에서 ExportCalendar를 다시 실행하세요" if age > 7 else ""
    return sorted(out.items()), warn


def mid():
    t = mt5.symbol_info_tick(SYMBOL)
    return (t.bid + t.ask) / 2, t


def lot_now():
    return max(0.01, np.floor(mt5.account_info().balance / 1000 * LOT_PER_1000 * 100) / 100)


def deal(side, lot, position=None, comment=""):
    """시장가 주문 -> (체결가, 주문 직전 틱 가격) 또는 (None, 이유)"""
    f = bot.deal(side, lot, comment, position=position)
    if not f.ok:
        log.error(f"주문 실패 {comment}: {f.err}")
        return None, f.err
    return f.res.price, f.price


def close_side(st, side, why):
    """side 포지션 청산 -> (체결가, 주문 직전 가격)"""
    for p in bot.positions():
        if side_of(p) == side:
            return deal(-side, p.volume, p.ticket, f"NEWS {why}")
    return None, "포지션 없음"


def save(st):
    STATE_JSON.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def run_event(release, name, clock):
    """발표 하나 처리 (발표 PRE_SEC초 전부터 HOLD_SEC초 뒤까지 이 함수 안에서 돔)"""
    st = {"event": name, "release": release.strftime("%Y-%m-%d %H:%M:%S")}
    while clock.now() < release - timedelta(seconds=PRE_SEC):
        clock.update()
        time.sleep(0.05)
    m0, t = mid()
    if t.ask - t.bid > MAX_SPREAD:
        log.info(f"{name}: 스프레드 {t.ask - t.bid:.2f} 넓어서 건너뜀")
        bot.send(f"⏭ {name} 건너뜀: 진입 직전 스프레드 {t.ask - t.bid:.2f}$")
        return
    lot = lot_now()
    margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, SYMBOL, lot, t.ask) or 0
    if margin * 2 > mt5.account_info().margin_free * 0.9:
        bot.send(f"⚠️ {name}: 증거금 부족으로 건너뜀 ({lot} lot)")
        return
    st.update(lot=lot, entry_time=clock.now().strftime("%H:%M:%S.%f")[:-3])
    bf, _ = deal(1, lot, comment="NEWS buy")
    sf, _ = deal(-1, lot, comment="NEWS sell")
    if bf is None or sf is None:
        bot.send(f"❗ {name}: 진입 실패 (매수 {bf}, 매도 {sf}). 열린 것 정리")
        for p in bot.positions():
            deal(-side_of(p), p.volume, p.ticket, "NEWS fail")
        return
    st.update(buy_fill=bf, sell_fill=sf, entry_spread=round(bf - sf, 2))
    save(st)
    log.info(f"{name} 진입 {lot} lot 매수 {bf} 매도 {sf} (스프레드 {bf - sf:.2f})")
    bot.send(f"🔀 {name} 발표 {PRE_SEC}초 전 진입: 매수 {bf} / 매도 {sf} ({lot} lot씩, 진입 스프레드 {bf - sf:.2f}$)")
    while clock.now() < release:
        clock.update()
        time.sleep(0.02)
    mr, _ = mid()
    th = mr * TH_PCT / 100
    st.update(release_mid=round(mr, 3), threshold=round(th, 2))
    kept = 0
    while clock.now() < release + timedelta(seconds=QUIET_SEC):
        clock.update()
        m, t = mid()
        if abs(m - mr) >= th:
            kept = 1 if m > mr else -1
            break
        time.sleep(0.05)
    if not kept:
        bc, _ = close_side(st, 1, "quiet")
        sc, _ = close_side(st, -1, "quiet")
        tot = ((bc or 0) - bf) + (sf - (sc or 0))
        st.update(kept=0, close_time=clock.now().strftime("%H:%M:%S.%f")[:-3], close_reason="5분 무반응",
                  total_oz=round(tot, 2), total_usd=round(tot * lot * 100, 2))
        append_csv(TRADES_CSV, FIELDS, st)
        bot.send(f"😐 {name}: 5분 동안 {th:.2f}$ 안 움직여 양쪽 정리. 합계 {tot:+.2f}$/oz ({tot * lot * 100:+.2f}$)\n"
                 f"{bot.account_line()}")
        return
    cf, cpx = close_side(st, -kept, "cut")
    cut_pnl = (sf - cf) if kept == 1 else (cf - bf)
    st.update(decide_time=clock.now().strftime("%H:%M:%S.%f")[:-3], kept=kept, decide_mid=round(m, 3), cut_fill=cf,
              cut_slip=round((cf - cpx) * (1 if kept == 1 else -1), 2), cut_pnl_oz=round(cut_pnl, 2))
    save(st)
    bot.send(f"{'⬆️' if kept == 1 else '⬇️'} {name}: 발표가 {mr:.2f}에서 {m - mr:+.2f}$ -> {'매도' if kept == 1 else '매수'} 청산 {cf} "
         f"({cut_pnl:+.2f}$/oz, 주문 직전 대비 미끄러짐 {st['cut_slip']:+.2f}). {'매수' if kept == 1 else '매도'} 들고 감")
    entry = bf if kept == 1 else sf
    reason = f"발표 {HOLD_SEC // 60}분"
    while clock.now() < release + timedelta(seconds=HOLD_SEC):
        clock.update()
        m, t = mid()
        if (m - mr) * kept <= -REVERT:
            reason = f"발표가 반대쪽 {REVERT:g}$ 되돌림"
            break
        time.sleep(0.2)
    xf, xpx = close_side(st, kept, "close")
    keep_pnl = ((xf or 0) - entry) * kept
    tot = cut_pnl + keep_pnl
    st.update(close_time=clock.now().strftime("%H:%M:%S.%f")[:-3], close_reason=reason, close_fill=xf,
              close_slip=round((xpx - xf) * kept, 2) if xf else "", keep_pnl_oz=round(keep_pnl, 2),   # +면 불리
              total_oz=round(tot, 2), total_usd=round(tot * lot * 100, 2))
    append_csv(TRADES_CSV, FIELDS, st)
    log.info(f"{name} 끝: {st}")
    bot.send(f"{'✅' if tot > 0 else '🔻'} {name} 끝 ({reason}): 남긴 쪽 {keep_pnl:+.2f}$/oz, 합계 {tot:+.2f}$/oz "
             f"({tot * lot * 100:+,.2f}$)\n{bot.account_line()}")


def main():
    global bot
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--test", type=int, default=0)
    args = ap.parse_args()
    bot = Trader(STRATEGY, SYMBOL, MAGIC, DEVIATION, LOG_DIR, "news")
    acc = bot.start()                      # 데모 계좌가 아니면 여기서 종료
    clock = bot.clock
    for _ in range(50):
        clock.update()
        time.sleep(0.1)
    sched, warn = schedule(clock.now())
    lines = "\n".join(f"  {t:%m-%d %a %H:%M} {n}" for t, n in sched) or "  (없음)"
    print(f"서버 시각 {clock.now():%Y-%m-%d %H:%M:%S}\n앞으로 2주 발표:\n{lines}\n{warn}")
    if args.dry:
        mt5.shutdown()
        return
    left = bot.positions()
    if left:
        for p in left:
            deal(-side_of(p), p.volume, p.ticket, "NEWS restart")
        bot.send(f"⚠️ 시작할 때 남아 있던 뉴스 포지션 {len(left)}개 정리 (재시작 중간에 끊긴 발표)")
    desc = (f"CPI·고용·FOMC 금리 결정 발표 {PRE_SEC}초 전 양방향, 발표가 {TH_PCT}% 움직이면 반대쪽 청산, "
            f"{QUIET_SEC // 60}분 무반응이면 정리, 남은 쪽 {HOLD_SEC // 60}분 또는 반대 {REVERT:g}$ 되돌림에 정리")
    bot.send(f"▶️ 데모 시작 ({acc.login})\n전략: {desc}\nlot {lot_now()} (1,000$당 {LOT_PER_1000})\n다음 발표:\n{lines}"
             + (f"\n⚠️ {warn}" if warn else ""))
    if args.test:
        run_event(clock.now().replace(microsecond=0) + timedelta(seconds=args.test), "테스트", clock)
        bot.tg.flush()
        mt5.shutdown()
        return
    last_check, warned = 0.0, False
    with bot.running("⏸ 종료"):
        while True:
            now = bot.keep_alive()
            if time.time() - last_check > 600:
                sched, warn = schedule(now)
                last_check = time.time()
                if warn and not warned:
                    bot.send(f"⚠️ {warn}")
                    warned = True
            nxt = next(((t, n) for t, n in sched if t - timedelta(seconds=PRE_SEC) > now - timedelta(seconds=5)), None)
            if nxt and nxt[0] - now < timedelta(minutes=2):
                bot.send(f"⏰ {nxt[1]} 발표 {nxt[0]:%H:%M} (서버), {PRE_SEC}초 전에 진입합니다")
                run_event(nxt[0], nxt[1], clock)
                sched = [x for x in sched if x[0] != nxt[0]]
                if STATE_JSON.exists():
                    STATE_JSON.unlink()
            time.sleep(1.0)


if __name__ == "__main__":
    main()
