"""MT4 따라하기 공통 통로 (2026-10-26 사용자 요청)

데모 파이썬 봇의 진입·청산을 MT5 공용 폴더 Common\\Files\\infinox_bridge.csv에 한 줄씩 붙임.
MT4 EA(mql4/mod_follow.mqh)가 0.1초마다 읽어서, 같은 전략 이름의 따라하기 전략이 자기 계좌에서 진입·청산.
판단은 파이썬 한 곳에서만 하고 MT4는 따라만 함 (MT4·MT5 서버 봉 가격 차이로 신호가 갈리지 않게).
봇은 직접 부르지 않고 Trader.open(데모 체결 확인 뒤) / Trader.deal(청산 주문 직전)이 bot.bridge_id 이름으로 보냄 (common/trader.py).

한 줄: 시각(UTC ms),전략,동작(open/close),방향,손절 거리,익절 거리,손절 절대 가격,1000$당 lot,데모 티켓,E  (ascii, CRLF)
- 시각 = 데모 체결(청산 주문) 시각. MT4는 open을 이 시각 뒤 5초 안에 봤을 때만 따라 들어감
- 마지막 E = 줄 끝 표시 (MT4가 쓰는 중인 줄을 반만 읽고 처리하지 않게)
- open: 방향 1/-1. 손절·익절 거리는 MT4가 자기 체결가 기준으로 검 (0이면 없음). 손절 절대 가격(지수)이 0보다 크면 거리 대신 씀
  1000$당 lot = 데모 lot / 데모 잔고 x 1000 -> MT4가 자기 잔고로 lot 계산 (전략의 위험 비율·마틴 단계가 그대로 따라감)
- close: 방향 = 닫을 포지션 방향. MT4는 같은 데모 티켓을 주문 설명에 가진 짝 포지션만 닫음
  데모 티켓마다 한 번만 씀 (청산 재시도마다 쌓이지 않게)
- 쓰기에 실패한 줄은 밀린 목록에 두고 다음 send와 봇 루프(Trader.keep_alive -> flush)에서 다시 씀.
  open은 4초가 지나면 MT4가 어차피 안 받으니 버리고 경고 (청산은 끝까지 다시)
- 서버 SL/TP로 닫히는 건 보내지 않음 (MT4는 자기 SL/TP로 닫힘)
- 실계좌(--live)·dry-run 실행은 보내지 않음 (같은 신호가 두 번 나가지 않게 데모 한 곳만): Trader.bridge가 거름
"""
import logging
import os
import time
from pathlib import Path

BRIDGE_FILE = Path(os.environ.get("APPDATA", "")) / "MetaQuotes" / "Terminal" / "Common" / "Files" / "infinox_bridge.csv"
OPEN_RETRY_SEC = 4
log = logging.getLogger("bridge")
_closed_refs = set()          # 청산 신호를 이미 쓴(또는 밀린 목록에 넣은) (전략, 데모 티켓)
_pending = []                 # 쓰기 실패한 줄: (줄, 동작, 시각 ms)


_maybe_partial = False         # 직전 쓰기가 도중에 실패함 (반쯤 쓴 줄이 파일 끝에 남았을 수 있음)


def _write(line):
    global _maybe_partial
    for i in range(3):
        try:
            BRIDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(BRIDGE_FILE, "a", encoding="ascii", newline="") as f:
                # 반쯤 쓴 줄이 남았을 수 있으면 줄을 먼저 끊음: 조각은 E가 없는 따로 된 줄이 돼서 MT4가 건너뜀
                f.write(("\r\n" if _maybe_partial else "") + line)
            _maybe_partial = False
            return True
        except OSError as e:
            _maybe_partial = True
            if i == 2:
                log.warning(f"MT4 통로 쓰기 실패 ({e}): {line.strip()}")
            else:
                time.sleep(0.2)
    return False


def flush():
    """쓰기 실패해 밀린 줄 다시 쓰기 (send와 Trader.keep_alive가 부름)"""
    while _pending:
        line, action, ts = _pending[0]
        if action == "open" and time.time() * 1000 - ts > OPEN_RETRY_SEC * 1000:
            _pending.pop(0)
            log.warning(f"MT4 통로: 진입 신호를 {OPEN_RETRY_SEC}초 안에 못 써서 버림 (MT4는 이 진입을 안 따라감) {line.strip()}")
            continue
        if not _write(line):
            return
        _pending.pop(0)
        log.info(f"MT4 통로: 밀린 줄 다시 씀 {line.strip()}")


def send(strategy, action, side=0, sl=0.0, tp=0.0, sl_abs=0.0, lot1000=0.0, ref=0, ts_ms=None):
    """한 줄 붙이기. 성공하면 True, 실패는 밀린 목록에 두고 봇은 계속"""
    flush()
    key = (strategy, int(ref))
    if action == "close" and ref:
        if key in _closed_refs:
            return False
        _closed_refs.add(key)
    ts = int(ts_ms if ts_ms is not None else time.time() * 1000)
    line = f"{ts},{strategy},{action},{int(side)},{sl:.5f},{tp:.5f},{sl_abs:.5f},{lot1000:.6f},{int(ref)},E\r\n"
    if not _pending and _write(line):
        log.info(f"MT4 통로: {strategy} {action} {side} (티켓 {ref})")
        return True
    _pending.append((line, action, ts))               # 앞에 밀린 줄이 있으면 순서를 지키려고 뒤에 붙임
    flush()
    return False
