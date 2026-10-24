"""수익률 계산과 일/주/월 리포트 (MT5 서버 거래내역 기준)

- 손익: 이 전략(매직 넘버) 거래의 실현손익 합계 (수수료/스왑 포함, 열린 포지션 평가손익 제외)
- 수익률: 손익 / (기간 시작 시점 잔고 + 기간 중 입금액)  (입금이 있어도 % 가 부풀지 않게. 출금은 빼지 않음)
  기간 시작 잔고 = 현재 잔고 - 그 이후 계좌의 모든 잔고 변동 (입출금 포함)
- 기간은 서버시간 기준: 일 = 00:00, 주 = 월요일 00:00, 월 = 1일 00:00

리포트 시점 (서버시간): 월~금 REPORT_TIME에 일일, 금요일엔 주간, 그 달 마지막 거래일엔 월간을 이어서 보냄.
보낸 기록은 logs/demo_log/state.json에 저장 -> 껐다 켜도 중복 전송 안 함 (꺼져 있던 동안 놓친 건 가장 최근 것만 보냄).
"""
import json
import time
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5

REPORT_TIME = (23, 58)      # 금 시장 마감(23:57) 직후


def epoch(dt):
    """서버시간 datetime -> MT5 시간값 (MT5는 서버시간을 UTC처럼 표기)"""
    return int(datetime(dt.year, dt.month, dt.day, getattr(dt, "hour", 0), getattr(dt, "minute", 0),
                        tzinfo=timezone.utc).timestamp())


class ServerClock:
    """서버 시각 추정. 새 틱이 들어오는 동안 '서버 - PC' 시계 차이를 재 두고, 주말처럼 틱이 멈춰도 계속 흘러가게 함"""

    def __init__(self, symbol, offset=None):
        self.symbol, self.offset, self.last_msc = symbol, offset, None

    def update(self):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return
        if self.last_msc is not None and tick.time_msc != self.last_msc:
            self.offset = tick.time_msc / 1000 - time.time()   # 방금 새 틱 -> 시계 차이 측정
        self.last_msc = tick.time_msc
        self._fallback = tick.time

    def now(self):
        t = time.time() + self.offset if self.offset is not None else getattr(self, "_fallback", time.time())
        return datetime.fromtimestamp(t, timezone.utc).replace(tzinfo=None)


def period_start(now, kind):
    d = now.date()
    if kind == "week":
        d -= timedelta(days=d.weekday())
    elif kind == "month":
        d = d.replace(day=1)
    return datetime(d.year, d.month, d.day)


def balance_at(start):
    """start 시점 잔고 = 현재 잔고 - start 이후 모든 잔고 변동"""
    deals = mt5.history_deals_get(epoch(start), int(time.time()) + 7 * 86400) or ()
    change = sum(d.profit + d.commission + d.swap + getattr(d, "fee", 0.0) for d in deals)
    return mt5.account_info().balance - change


def deposits(start, end):
    """start ~ end 입금 합계 (출금 제외)"""
    deals = mt5.history_deals_get(epoch(start), epoch(end)) or ()
    return sum(d.profit for d in deals if d.type == mt5.DEAL_TYPE_BALANCE and d.profit > 0)


def stats(start, end, magic, symbol):
    deals = mt5.history_deals_get(epoch(start), epoch(end)) or ()
    mine = [d for d in deals if d.magic == magic and d.symbol == symbol]
    closes = [d for d in mine if d.entry == mt5.DEAL_ENTRY_OUT]
    # 포지션별 손익 (진입 수수료 포함)
    by_pos = {}
    for d in mine:
        by_pos[d.position_id] = by_pos.get(d.position_id, 0.0) + d.profit + d.commission + d.swap
    closed_pnl = [by_pos[d.position_id] for d in closes]
    pnl = sum(closed_pnl)
    levels = [int(d.comment.rsplit("L", 1)[1]) for d in mine
              if d.entry == mt5.DEAL_ENTRY_IN and d.comment.rsplit("L", 1)[-1].isdigit()]
    b0 = balance_at(start)
    dep = deposits(start, end)
    return {
        "trades": len(closes), "wins": sum(p > 0 for p in closed_pnl), "pnl": pnl,
        "pct": pnl / (b0 + dep) * 100 if b0 + dep > 0 else 0.0, "start_balance": b0, "deposit": dep,
        "end_balance": balance_at(end),
        "max_level": max(levels) if levels else None,
    }


def fmt_pct(s):
    return f"{s['pct']:+.2f}% ({s['pnl']:+,.2f}$)"


def summary_line(now, magic, symbol):
    """오늘 / 이번주 / 이번달 수익률 한 줄"""
    end = now + timedelta(days=1)
    parts = []
    for kind, name in (("day", "오늘"), ("week", "이번주"), ("month", "이번달")):
        parts.append(f"{name} {fmt_pct(stats(period_start(now, kind), end, magic, symbol))}")
    return "\n".join(parts)


def report_text(kind, start, end, magic, symbol):
    s = stats(start, end, magic, symbol)
    title = {"day": f"📊 일일 리포트 {start:%Y-%m-%d}",
             "week": f"📅 주간 리포트 {start:%m/%d} ~ {(end - timedelta(days=1)):%m/%d}",
             "month": f"🗓 월간 리포트 {start:%Y-%m}"}[kind]
    wr = f"{s['wins'] / s['trades'] * 100:.0f}%" if s["trades"] else "-"
    lines = [
        title,
        f"수익률 {fmt_pct(s)}",
        f"거래 {s['trades']}건 (익절 {s['wins']} / 손절 {s['trades'] - s['wins']}, 승률 {wr})",
        f"최고 마틴 단계 {s['max_level'] if s['max_level'] is not None else '-'}",
        f"잔고 {s['start_balance']:,.2f} → {s['end_balance']:,.2f}",
    ]
    return "\n".join(lines)


def last_weekday_of_month(d):
    nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    x = nxt - timedelta(days=1)
    while x.weekday() >= 5:
        x -= timedelta(days=1)
    return x


class ReportScheduler:
    def __init__(self, state_file, magic, symbol, send):
        self.file, self.magic, self.symbol, self.send = state_file, magic, symbol, send
        self.state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}

    def _save(self):
        self.file.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _targets(now):
        """지금 시점에서 이미 마감된 가장 최근 (거래일, 금요일, 월말 거래일)"""
        d = now.date() if (now.hour, now.minute) >= REPORT_TIME else now.date() - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        fri = d - timedelta(days=(d.weekday() - 4) % 7)
        lm = last_weekday_of_month(d)
        month_end = lm if d >= lm else last_weekday_of_month(d.replace(day=1) - timedelta(days=1))
        return d, fri, month_end

    def check(self, now):
        d, fri, month_end = self._targets(now)
        if not self.state:      # 첫 실행: 과거 리포트는 보내지 않고 기준만 잡음
            self.state = {"day": d.isoformat(), "week": fri.isoformat(), "month": month_end.isoformat()}
            self._save()
            return
        for kind, target in (("day", d), ("week", fri), ("month", month_end)):
            if target.isoformat() > self.state.get(kind, ""):
                if kind == "day":
                    start = datetime(d.year, d.month, d.day)
                elif kind == "week":
                    m = target - timedelta(days=4)
                    start = datetime(m.year, m.month, m.day)
                else:
                    start = datetime(target.year, target.month, 1)
                end = datetime(target.year, target.month, target.day) + timedelta(days=1)
                self.send(report_text(kind, start, end, self.magic, self.symbol))
                self.state[kind] = target.isoformat()
                self._save()
