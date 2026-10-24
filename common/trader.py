"""봇 공통: MT5 연결·계좌 확인, 시장가 주문·청산, 거래 기록, 텔레그램, 로그

각 봇은 main()에서 Trader 하나를 만들어 씀. 전략 설정(SYMBOL, MAGIC, LOG_DIR …)은 봇 파일의 모듈 변수에 두고,
--live / --symbol 처리가 끝난 뒤 그 값으로 Trader를 만든다.

  bot = Trader("SAR 단타", SYMBOL, MAGIC, DEVIATION, LOG_DIR, "demo", live=args.live)
  bot.start()                  # 로그 설정 + MT5 연결 + 계좌 확인 + 텔레그램 + 서버 시계
  with bot.running("종료 메시지"):
      while True:
          bot.keep_alive()     # 연결 끊김이면 재연결, 계좌가 바뀌면 종료
          ...
"""
import csv
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

from config import mt5_init
from reports import ServerClock, summary_line
from telegram_notify import Notifier


# ---------------------------------------------------------------- 파일
def load_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {} if default is None else default


def save_json(path, obj, indent=2):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def append_csv(path, fields, row):
    """거래 한 줄 추가 (처음이면 머리줄도). fields에 없는 키는 버리고, 빠진 키는 빈칸"""
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


# ---------------------------------------------------------------- 시세
def rates(symbol, tf, n, min_bars=50):
    """완성된 봉 n개 (진행 중인 봉 제외). 봉 시각 = 서버시간. min_bars보다 적으면 None"""
    r = mt5.copy_rates_from_pos(symbol, tf, 1, n)
    if r is None or len(r) < min_bars:
        return None
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def account_kind(acc):
    return "데모" if acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else "실계좌"


def side_of(p):
    """포지션 방향 +1 매수 / -1 매도"""
    return 1 if p.type == mt5.POSITION_TYPE_BUY else -1


def side_name(side):
    return "매수" if side == 1 else "매도"


@dataclass
class Fill:
    """시장가 주문 결과. ok가 아니면 err에 이유"""
    ok: bool
    err: str
    res: object
    price: float        # 주문 직전 틱 가격 (매수 Ask / 매도 Bid)
    tick: object
    sl: float
    tp: float

    @property
    def fill(self):
        return (self.res.price if self.res else 0) or self.price


# ---------------------------------------------------------------- 봇
class Trader:
    def __init__(self, name, symbol, magic, deviation, log_dir, logger, live=False):
        self.name, self.symbol, self.magic, self.deviation = name, symbol, magic, deviation
        self.log_dir, self.live = log_dir, live
        self.log = logging.getLogger(logger)
        self.tg = None
        self.clock = None
        self.acc = None

    # ---- 시작 / 연결
    def setup_logging(self):
        self.log_dir.mkdir(exist_ok=True)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for h in (logging.FileHandler(self.log_dir / "events.log", encoding="utf-8"), logging.StreamHandler()):
            h.setFormatter(fmt)
            self.log.addHandler(h)
        self.log.setLevel(logging.INFO)

    def connect(self):
        if not mt5_init():
            raise SystemExit(f"MT5 연결 실패: {mt5.last_error()} (MT5를 켜고 로그인했는지, 경로 설정은 README 참고)")
        self.check_account()
        if not mt5.symbol_select(self.symbol, True):
            raise SystemExit(f"{self.symbol} 종목을 찾을 수 없습니다")

    def check_account(self, login=None):
        """데모 계좌인지 확인 (live면 실계좌만). login을 주면 그 계좌에서 바뀌었는지도 확인"""
        acc = mt5.account_info()
        if acc is None:
            raise SystemExit("로그인된 계좌가 없습니다")
        if login is not None and acc.login != login:
            mt5.shutdown()
            raise SystemExit(f"로그인 계좌가 바뀌었습니다 ({login} -> {acc.login}). 안전을 위해 종료합니다.")
        demo = acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
        if not demo and not self.live:
            mt5.shutdown()
            raise SystemExit(f"데모 계좌가 아닙니다 ({acc.login}, {acc.server}). 실계좌는 --live 옵션으로만 실행합니다.")
        if demo and self.live:
            mt5.shutdown()
            raise SystemExit(f"--live인데 데모 계좌에 연결됐습니다 ({acc.login}). MT5_TERMINAL_PATH / MT5_PORTABLE / "
                             f"MT5_SYMBOL 환경변수로 실계좌 터미널을 지정하세요.")
        return acc

    def start(self, dry_run=False):
        """로그 + 연결 + 계좌 확인 + 텔레그램 + 서버 시계. 계좌 정보를 돌려줌"""
        self.setup_logging()
        self.connect()
        self.acc = self.check_account()
        self.tg = Notifier()
        if dry_run:
            self.tg.enabled = False
        self.clock = ServerClock(self.symbol)
        self.clock.update()
        return self.acc

    @property
    def kind(self):
        return account_kind(self.acc)

    def keep_alive(self):
        """루프마다: 터미널 연결이 끊겼으면 재연결, 로그인 계좌가 바뀌었으면 종료, 서버 시계 갱신. 서버 시각을 돌려줌"""
        if not mt5.terminal_info():
            self.log.warning("터미널 연결 끊김, 재연결 시도")
            mt5.shutdown()
            time.sleep(5)
            self.connect()
        self.check_account(self.acc.login)
        self.clock.update()
        return self.clock.now()

    def warn_algo_off(self):
        if not mt5.terminal_info().trade_allowed:
            self.log.warning("MT5의 'Algo Trading'이 꺼져 있습니다. 켜야 주문이 들어갑니다.")
            self.send("⚠️ MT5의 'Algo Trading'이 꺼져 있습니다. 켜야 주문이 들어갑니다.")

    @contextmanager
    def running(self, stop_msg="⏸ 종료"):
        """메인 루프 감싸기: Ctrl+C면 stop_msg, 오류면 알리고 다시 던짐. 끝나면 텔레그램 비우고 MT5 끊기"""
        try:
            yield
        except KeyboardInterrupt:
            self.log.info(f"사용자 종료: {stop_msg}")
            self.send(stop_msg)
        except BaseException as e:
            self.log.exception("오류로 종료")
            self.send(f"❗ 오류로 멈췄습니다: {e}")
            raise
        finally:
            if self.tg:
                self.tg.flush()
            mt5.shutdown()

    # ---- 알림
    def send(self, text):
        """텔레그램 전송 (앞에 계좌 종류와 전략 이름 표시: [데모 · SAR 단타] / [실계좌 · SAR 단타])"""
        kind = self.kind if self.acc else ("실계좌" if self.live else "데모")
        self.tg.send(f"[{kind} · {self.name}] {text}")

    def account_line(self):
        a = mt5.account_info()
        return f"잔고 {a.balance:,.2f} | 평가 {a.equity:,.2f} {a.currency}"

    def summary(self):
        """오늘 / 이번주 / 이번달 이 전략 수익률"""
        return summary_line(self.clock.now(), self.magic, self.symbol)

    # ---- 시세 / 포지션
    def rates(self, tf, n, min_bars=50):
        return rates(self.symbol, tf, n, min_bars)

    def positions(self):
        """이 봇(매직 넘버) 포지션만"""
        return [p for p in (mt5.positions_get(symbol=self.symbol) or ()) if p.magic == self.magic]

    def closes_since(self, t0, t1):
        """t0~t1(MT5 시간값) 이 봇이 청산한 거래, 시간순"""
        deals = mt5.history_deals_get(t0, t1) or ()
        return sorted((d for d in deals if d.magic == self.magic and d.symbol == self.symbol
                       and d.entry == mt5.DEAL_ENTRY_OUT), key=lambda d: d.time_msc)

    # ---- 주문
    def filling_mode(self):
        # 파이썬 패키지에 SYMBOL_FILLING_* 상수가 없어서 MQL5 값 사용 (FOK = 1, IOC = 2)
        fm = mt5.symbol_info(self.symbol).filling_mode
        if fm & 1:
            return mt5.ORDER_FILLING_FOK
        if fm & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def deal(self, side, lot, comment, position=None, sl_dist=None, tp_dist=None, deviation=None):
        """시장가 주문 하나 (position을 주면 그 포지션 청산). sl/tp는 주문 가격에서의 거리($)"""
        info, tick = mt5.symbol_info(self.symbol), mt5.symbol_info_tick(self.symbol)
        price = tick.ask if side == 1 else tick.bid
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL, "price": price,
            "deviation": self.deviation if deviation is None else deviation, "magic": self.magic,
            "comment": comment, "type_time": mt5.ORDER_TIME_GTC, "type_filling": self.filling_mode(),
        }
        if position:
            req["position"] = position
        if sl_dist is not None:
            req["sl"] = round(price - sl_dist * side, info.digits)
        if tp_dist is not None:
            req["tp"] = round(price + tp_dist * side, info.digits)
        res = mt5.order_send(req)
        ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
        err = "" if ok else f"{res.retcode if res else mt5.last_error()} {res.comment if res else ''}"
        return Fill(ok, err, res, price, tick, req.get("sl", 0.0), req.get("tp", 0.0))

    def margin_ok(self, side, lot, n=1):
        """lot짜리 n개 진입할 증거금이 여유 증거금의 90% 안이면 True, 아니면 알리고 False"""
        tick = mt5.symbol_info_tick(self.symbol)
        otype = mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL
        margin = mt5.order_calc_margin(otype, self.symbol, lot, tick.ask if side == 1 else tick.bid)
        free = mt5.account_info().margin_free
        if margin is None or margin * n > free * 0.9:
            self.log.warning(f"증거금 부족으로 진입 안 함: 필요 {margin}, 여유 {free:.2f} ({lot} lot)")
            self.send(f"⚠️ 증거금 부족으로 진입 안 함\n필요 {margin} / 여유 {free:,.2f} ({lot} lot)")
            return False
        return True

    def open(self, side, lot, comment, sl_dist=None, tp_dist=None):
        """증거금 확인 후 시장가 진입. 실패하면 알리고 None"""
        if not self.margin_ok(side, lot):
            return None
        f = self.deal(side, lot, comment, sl_dist=sl_dist, tp_dist=tp_dist)
        if not f.ok:
            self.log.error(f"주문 실패: {f.err}")
            self.send(f"❗ 주문 실패 ({side_name(side)} {lot} lot): {f.err}")
            return None
        return f

    def close(self, p, comment, what):
        """포지션 시장가 청산. 실패하면 알림. 성공 여부"""
        f = self.deal(-side_of(p), p.volume, comment, position=p.ticket)
        code = f.res.retcode if f.res else mt5.last_error()
        self.log.info(f"{what} {'완료' if f.ok else '실패'}: 티켓 {p.ticket} {code}")
        if not f.ok:
            self.send(f"❗ {what} 실패: 티켓 {p.ticket} {code}")
        return f.ok

    def find_close(self, ticket, comment_reasons=None):
        """닫힌 포지션의 청산 정보 (아직 안 닫혔으면 None). 손익은 진입·청산 수수료/스왑 포함
        결과: TP / SL / CLOSE, 또는 청산 주문 comment가 comment_reasons에 있으면 그 값"""
        deals = mt5.history_deals_get(position=ticket) or ()
        out = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
        if not out:
            return None
        d = out[-1]
        profit = sum(x.profit + x.commission + x.swap for x in deals)
        reason = {mt5.DEAL_REASON_TP: "TP", mt5.DEAL_REASON_SL: "SL"}.get(d.reason, "CLOSE")
        reason = (comment_reasons or {}).get(d.comment, reason)
        return {"close_time": datetime.fromtimestamp(d.time, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "close_price": d.price, "profit": round(profit, 2), "result": reason}

    def adopt_position(self, with_tp=True):
        """이전 실행에서 열어 둔 포지션이 있으면 기록용 row로 이어받음 (없으면 None)"""
        row = None
        for p in self.positions():
            row = {"ticket": p.ticket, "side": "BUY" if side_of(p) == 1 else "SELL",
                   "lot": p.volume, "fill_price": p.price_open, "sl": p.sl}
            if with_tp:
                row["tp"] = p.tp
            self.log.info(f"기존 포지션 이어받음: 티켓 {p.ticket} {row['side']} {p.volume}lot @ {p.price_open}")
        return row

    # ---- 안전장치
    def account_stop(self, now, stop_pct, risk_json):
        """계좌 손실 한도: 평가금이 기록된 최고 잔고보다 stop_pct% 넘게 줄면 halted로 남기고 이유, 아니면 None
        risk_json에 남은 halted는 재시작해도 유지 (확인 후 그 파일을 지우면 다시 시작)"""
        acc = mt5.account_info()
        risk = load_json(risk_json)
        if risk.get("halted"):
            return f"계좌 손실 한도로 중단된 상태 ({risk['halted']}). 확인 후 {risk_json.name}을 지우면 다시 시작"
        peak = max(risk.get("peak_balance", 0.0), acc.balance)
        if peak != risk.get("peak_balance"):
            risk["peak_balance"] = peak
            save_json(risk_json, risk)
        if acc.equity <= peak * (1 - stop_pct / 100):
            risk["halted"] = f"{now:%Y-%m-%d %H:%M} 평가금 {acc.equity:,.2f} / 최고 잔고 {peak:,.2f}"
            save_json(risk_json, risk)
            self.send(f"🛑 계좌 손실 한도({stop_pct:g}%) 도달 -> 신규 진입을 중단합니다.\n{risk['halted']}\n"
                      f"확인 후 {risk_json.parent.name}/{risk_json.name}을 지우고 다시 실행하세요.")
            return f"계좌 손실 한도 도달 ({risk['halted']})"
        return None
