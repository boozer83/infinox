"""다른 종목 틱 받기 (주말용: 받는 동안 MT5 터미널이 바빠져서 평일에는 그 터미널의 봇이 늦어짐)

- 2022-01부터 이번 달까지 한 달씩, 하루 단위로 나눠 받음. 이미 받은 달은 건너뜀 (이번 달은 다시 받음)
- 저장: data/ticks_z/<종목>/YYYY-MM.npz (long_period.save_z와 같은 압축, 가격 단위는 종목 소수 자릿수)
  원본은 저장 안 함 (압축본이 그대로 복원되는지 저장할 때 확인)
- 끝나면 텔레그램으로 종목별 기간/틱 수/용량 알림

사용법: python -m backtests.download_symbol BTCUSD XAGUSD
        평일에 받을 땐 --slow (하루치마다 2초 쉼)
        --one: 새로 한 달 받으면 끝 (download_monthly.sh가 한 달씩 새 프로세스로 반복), --quiet: 텔레그램 알림 끔
읽기: from backtests.download_symbol import load_month; t, bid, ask = load_month("XAGUSD", 2025, 3)
"""
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np

from backtests.long_period import month_range, months
from telegram_notify import Notifier

ROOT = Path(__file__).resolve().parent.parent / "data" / "ticks_z"
PAUSE = 0.0           # 하루치 받을 때마다 쉬는 초 (--slow: 평일에 봇이 덜 밀리게)
ONE = False           # --one: 새로 한 달 받으면 끝 (한 달마다 프로세스를 새로 띄워 메모리를 비우기 위해)


def pack(t, bid, ask, digits):
    sc = 10 ** digits
    bi, ai = np.rint(bid * sc).astype(np.int64), np.rint(ask * sc).astype(np.int64)
    z = {"t0": t[:1].astype(np.int64), "b0": bi[:1], "scale": np.array([sc]),
         "dt": np.diff(t, prepend=t[0]).astype(np.int32), "db": np.diff(bi, prepend=bi[0]).astype(np.int32),
         "sp": (ai - bi).astype(np.int32)}
    return z


def unpack(z):
    t = z["t0"][0] + np.cumsum(z["dt"], dtype=np.int64)
    bi = z["b0"][0] + np.cumsum(z["db"], dtype=np.int64)
    sc = z["scale"][0]
    return t, bi / sc, (bi + z["sp"]) / sc


def past_deadline():
    """월요일 06:00(이 PC 시간, 한국) 이후면 True: 장이 열리기 전에 터미널을 봇에게 돌려줌"""
    now = datetime.now()
    return now.weekday() == 0 and now.hour >= 6


def load_month(symbol, y, m):
    with np.load(ROOT / symbol / f"{y}-{m:02d}.npz") as z:
        return unpack(z)


def download(symbol):
    info = mt5.symbol_info(symbol)
    if info is None or not mt5.symbol_select(symbol, True):
        return f"{symbol}: 종목 없음"
    d_out = ROOT / symbol
    d_out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    got, total = [], 0
    for y, m in months():
        if past_deadline():
            print(f"  {symbol}: 월요일 장 시작 전이라 중단 (다음 주말에 이어서 받음)", flush=True)
            break
        f = d_out / f"{y}-{m:02d}.npz"
        if f.exists() and (y, m) != (now.year, now.month):
            got.append((y, m))
            continue
        s, e = month_range(y, m)
        parts, d = [], s
        while d < e:
            de = min(d + timedelta(days=1), e)
            t = mt5.copy_ticks_range(symbol, d, de, mt5.COPY_TICKS_INFO)
            if t is not None and len(t):
                # 하루치를 바로 필요한 세 칸만 따로 복사 (MT5 틱 구조체 전체를 한 달 모으면 메모리가 모자람)
                k = (t["time_msc"] < int(de.timestamp() * 1000)) & (t["bid"] > 0) & (t["ask"] > 0)
                parts.append((t["time_msc"][k].astype(np.int64), t["bid"][k].copy(), t["ask"][k].copy()))
            del t
            d = de
            time.sleep(PAUSE)
        if not parts:
            print(f"  {symbol} {y}-{m:02d}: 틱 없음", flush=True)
            continue
        tm = np.concatenate([x[0] for x in parts])
        bd = np.concatenate([x[1] for x in parts])
        ak = np.concatenate([x[2] for x in parts])
        del parts
        z = pack(tm, bd, ak, info.digits)
        t2, b2, a2 = unpack(z)
        ok = (np.array_equal(t2, tm) and np.allclose(b2, bd, atol=0.5 / z["scale"][0])
              and np.allclose(a2, ak, atol=0.5 / z["scale"][0]))
        del t2, b2, a2
        if not ok:
            raise RuntimeError(f"{symbol} {y}-{m:02d} 압축 복원 불일치")
        np.savez_compressed(f, **z)
        n = len(tm)
        del tm, bd, ak, z
        total += n
        got.append((y, m))
        print(f"  {symbol} {y}-{m:02d}: 틱 {n:,}개", flush=True)
        if ONE:
            return f"{symbol} {y}-{m:02d}: 한 달 받음 (--one)"
    if not got:
        return f"{symbol}: 받은 틱 없음"
    mb = sum(p.stat().st_size for p in d_out.glob("*.npz")) / 1e6
    (y0, m0), (y1, m1) = got[0], got[-1]
    return f"{symbol}: {y0}-{m0:02d} ~ {y1}-{m1:02d}, 이번에 받은 틱 {total:,}개, 용량 {mb:,.0f}MB"


def main():
    global PAUSE, ONE
    args = [a for a in sys.argv[1:] if a not in ("--slow", "--one", "--quiet")]
    if "--slow" in sys.argv:
        PAUSE = 2.0
    ONE = "--one" in sys.argv
    symbols = args or ["BTCUSD", "XAGUSD"]
    tg = Notifier()
    if not mt5.initialize():
        tg.send(f"[틱 다운로드] ❗ MT5 연결 실패: {mt5.last_error()}")
        tg.flush()
        raise SystemExit(1)
    quiet = "--quiet" in sys.argv          # 한 달씩 반복 실행할 때 텔레그램 알림 끔
    if not quiet:
        tg.send(f"[틱 다운로드] 시작: {', '.join(symbols)}")
    lines = []
    for s in symbols:
        try:
            lines.append(download(s))
        except Exception as ex:   # 한 종목이 실패해도 다음 종목은 받음
            lines.append(f"{s}: 오류 {ex}")
        print(lines[-1], flush=True)
    mt5.shutdown()
    if not quiet:
        tg.send("[틱 다운로드] 끝\n" + "\n".join(lines))
    tg.flush()


if __name__ == "__main__":
    main()
