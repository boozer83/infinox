"""MT5 연결 테스트 (읽기 전용 - 주문하지 않음)

실행 전: MT5 터미널을 켜고 계좌로 로그인해 두세요.
사용법: python check_connection.py
"""
import MetaTrader5 as mt5

from config import SYMBOL, mt5_init

if not mt5_init():
    print("연결 실패:", mt5.last_error())
    raise SystemExit(1)

try:
    term = mt5.terminal_info()
    acc = mt5.account_info()
    if acc is None:
        print("터미널은 연결됐지만 로그인된 계좌가 없습니다:", mt5.last_error())
        raise SystemExit(1)

    trade_mode = {0: "데모", 1: "콘테스트", 2: "실계좌"}.get(acc.trade_mode, acc.trade_mode)
    print("=== 계좌 정보 ===")
    print(f"터미널      : {term.path}")
    print(f"계좌 번호   : {acc.login}")
    print(f"서버        : {acc.server}")
    print(f"계좌 종류   : {trade_mode}")
    print(f"통화        : {acc.currency}")
    print(f"잔고        : {acc.balance:,.2f}")
    print(f"평가금액    : {acc.equity:,.2f}")
    print(f"레버리지    : 1:{acc.leverage}")
    print(f"알고리즘 거래 허용(터미널): {term.trade_allowed}")

    positions = mt5.positions_get() or ()
    print(f"\n보유 포지션 : {len(positions)}개")
    for p in positions:
        side = "매수" if p.type == mt5.POSITION_TYPE_BUY else "매도"
        print(f"  {p.symbol} {side} {p.volume}lot  손익 {p.profit:,.2f}")

    if mt5.symbol_select(SYMBOL, True):
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick:
            print(f"\n{SYMBOL} 현재가  매수(Ask) {tick.ask}  매도(Bid) {tick.bid}")
    else:
        print(f"\n{SYMBOL} 종목을 찾을 수 없습니다 (브로커 종목명이 다를 수 있음)")
finally:
    mt5.shutdown()
