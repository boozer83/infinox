"""다른 종목(은, 비트코인 등) 1분봉을 2022-01부터 받아서 연도별 zip CSV로 저장 (다른 PC에서 받아 옮기기용)

받는 방법 (종목마다 자동으로 고름):
  1. MT5 1분봉 (빠름). MT5 "도구 -> 옵션 -> 차트 -> 차트 최대 봉 수"가 무제한이어야 오래된 봉까지 받아짐
  2. 1분봉이 비어 있는 달은 틱을 일주일씩 받아 바로 1분봉으로 만듦 (느리지만 메모리를 적게 씀)

결과: data/bars/<종목>_M1_<연도>.csv.zip  (열: time(서버 시간), open, high, low, close, spread(포인트), source)
      가격은 Bid 기준 (MT5 봉과 같음), spread는 그 1분 동안의 평균 스프레드 (포인트)

사용법: python -m backtests.download_bars XAGUSD BTCUSD
        (종목 이름은 MT5 "종목" 창에 보이는 그대로. 실계좌는 XAGUSD# 처럼 다를 수 있음)
"""
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from config import mt5_init

FIRST = datetime(2022, 1, 1, tzinfo=timezone.utc)
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "bars"


def month_starts():
    m, now = FIRST, datetime.now(timezone.utc)
    while m < now:
        yield m
        m = (m.replace(day=28) + timedelta(days=4)).replace(day=1)


def bars_from_rates(symbol, s, e):
    r = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, s, e)
    if r is None or len(r) == 0:
        return None
    df = pd.DataFrame(r)[["time", "open", "high", "low", "close", "spread"]]
    df = df[(df.time >= int(s.timestamp())) & (df.time < int(e.timestamp()))]   # 범위 밖 봉이 섞여 오는 경우 제외
    df["source"] = "rates"
    return df


def bars_from_ticks(symbol, s, e):
    point = mt5.symbol_info(symbol).point
    parts, d = [], s
    while d < e:
        de = min(d + timedelta(days=7), e)
        t = mt5.copy_ticks_range(symbol, d, de, mt5.COPY_TICKS_INFO)
        if t is not None and len(t):
            df = pd.DataFrame({"k": t["time_msc"] // 60_000, "bid": t["bid"], "sp": (t["ask"] - t["bid"]) / point})
            df = df[(df.bid > 0) & (df.k < int(de.timestamp()) // 60)]
            g = df.groupby("k")
            parts.append(pd.DataFrame({"time": g.size().index * 60, "open": g.bid.first().values,
                                       "high": g.bid.max().values, "low": g.bid.min().values,
                                       "close": g.bid.last().values, "spread": g.sp.mean().round(1).values}))
        d = de
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df["source"] = "ticks"
    return df


def download(symbol):
    if not mt5.symbol_select(symbol, True):
        print(f"{symbol}: 종목을 찾을 수 없음 ({mt5.last_error()})")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    years = {}
    for s in month_starts():
        e = min((s.replace(day=28) + timedelta(days=4)).replace(day=1), datetime.now(timezone.utc))
        df = bars_from_rates(symbol, s, e)
        how = "1분봉"
        if df is None or len(df) < 1000:
            df, how = bars_from_ticks(symbol, s, e), "틱"
        n = 0 if df is None else len(df)
        print(f"  {symbol} {s:%Y-%m}: {n:,}개 ({how})", flush=True)
        if n:
            years.setdefault(s.year, []).append(df)
    for y, parts in years.items():
        df = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time")
        df["time"] = pd.to_datetime(df.time, unit="s").dt.strftime("%Y-%m-%d %H:%M")
        name = f"{symbol.replace('#', '')}_M1_{y}"
        with zipfile.ZipFile(OUT_DIR / f"{name}.csv.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{name}.csv", df.to_csv(index=False))
        print(f"저장: {OUT_DIR / (name + '.csv.zip')} ({len(df):,}개, {(OUT_DIR / (name + '.csv.zip')).stat().st_size / 1e6:.1f}MB)")


def main():
    symbols = sys.argv[1:] or ["XAGUSD", "BTCUSD"]
    if not mt5_init():
        raise SystemExit(f"MT5 연결 실패: {mt5.last_error()}")
    try:
        print(f"MT5 차트 최대 봉 수: {mt5.terminal_info().maxbars:,} (작으면 오래된 1분봉이 안 받아져서 틱으로 받음)")
        for s in symbols:
            download(s)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
