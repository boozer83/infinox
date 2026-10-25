"""공통 설정

MT5 터미널 경로는 PC마다 다를 수 있어서 기본은 자동 감지.
자동 감지가 안 되거나 MT5가 여러 개 설치돼 있으면 환경변수로 지정:
  PowerShell:  $env:MT5_TERMINAL_PATH = "C:\\Program Files\\Infinox MetaTrader 5 Terminal\\terminal64.exe"
"""
import os

import MetaTrader5 as mt5

TERMINAL_PATH = os.environ.get("MT5_TERMINAL_PATH") or None
PORTABLE = os.environ.get("MT5_PORTABLE") == "1"
SYMBOL = os.environ.get("MT5_SYMBOL") or "XAUUSD"
CONTRACT = 100          # 금 1 lot = 100온스
# 실계좌(88004888) 종목명. 새 종목을 실계좌에서 돌릴 때는 여기만 추가 (데모는 이름 그대로)
LIVE_SYMBOLS = {"XAUUSD": "XAUUSD#", "XAUEUR": "XAUEUR#", "XETUSD": "XETUSD.a", "NAS100": "NAS100.a", "GER40": "GER40.a"}


def live_symbol(base):
    """데모 종목명 -> 실계좌 종목명 (표에 없으면 그대로)"""
    return LIVE_SYMBOLS.get(base, base)


def mt5_init():
    """MT5 터미널 연결. 성공하면 True"""
    return mt5.initialize(path=TERMINAL_PATH, portable=PORTABLE) if TERMINAL_PATH else mt5.initialize()
