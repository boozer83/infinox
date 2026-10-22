"""장기 백테스트: 2022-01 ~ 현재, RSI50 + H1 SAR (실제 틱 체결, 마틴 없음)

- 틱을 한 달씩 받아 data/ticks/YYYY-MM.npz로 저장 (다음 실행부터는 저장본 사용)
  깃에는 압축본 data/ticks_z/YYYY-MM.npz를 올림 (약 1/10 크기). data/ticks에 없으면 압축본을 풀어서 씀
- 5분봉/1시간봉은 MT5 봉 기록이 짧아서 틱 Bid로 직접 만듦 (MT5 봉도 Bid 기준)
- 신호: 5분봉 SAR(0.02, 0.2) 전환 + RSI(14) 50 방향 + 마지막 완성 1시간봉 SAR 방향 일치
- 체결: 다음 5분봉 첫 틱 (매수 Ask / 매도 Bid), SL/TP 처음 닿는 틱 가격으로 청산
- SL/TP 두 방식:
    고정    : SL 7.5 / TP 3 달러
    가격비례: 금값 4,150달러 기준 SL 7.5 / TP 3을 진입 시점 금값에 비례해 조정
- 수량 두 방식: 0.02 lot 고정 / 복리 (월초 잔고 1,000달러당 0.02 lot, 최소 0.01)

사용법: python -m backtests.long_period
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from backtests.m5_ticks import first_hit
from config import CONTRACT, SYMBOL, mt5_init
from indicators import psar, rsi

FIRST_MONTH = (2022, 1)
TICK_DIR = Path(__file__).resolve().parent.parent / "data" / "ticks"
TICK_Z_DIR = TICK_DIR.parent / "ticks_z"
SL, TP, REF_PRICE = 7.5, 3.0, 4150.0
START_BALANCE = 1000.0
FIXED_LOT = 0.02
LOT_PER_1000 = 0.02


def months():
    y, m = FIRST_MONTH
    now = datetime.now(timezone.utc)
    while (y, m) <= (now.year, now.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def month_range(y, m):
    s = datetime(y, m, 1, tzinfo=timezone.utc)
    e = datetime(y + 1, 1, 1, tzinfo=timezone.utc) if m == 12 else datetime(y, m + 1, 1, tzinfo=timezone.utc)
    return s, min(e, datetime.now(timezone.utc) + timedelta(hours=12))   # 틱 시각은 서버 시간(UTC보다 앞섬)이라 여유를 둠


def download_all():
    TICK_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    for y, m in months():
        f = TICK_DIR / f"{y}-{m:02d}.npz"
        current = (y, m) == (now.year, now.month)
        if f.exists() and not current:
            continue
        s, e = month_range(y, m)
        parts, d = [], s
        while d < e:
            de = min(d + timedelta(days=7), e)
            t = mt5.copy_ticks_range(SYMBOL, d, de, mt5.COPY_TICKS_INFO)
            if t is None:
                raise RuntimeError(f"틱 데이터 없음 ({d:%Y-%m-%d}): {mt5.last_error()}")
            parts.append(t[t["time_msc"] < int(de.timestamp() * 1000)][["time_msc", "bid", "ask"]])
            d = de
        a = np.concatenate(parts)
        np.savez(f, t=a["time_msc"], bid=a["bid"], ask=a["ask"])
        save_z(y, m, a["time_msc"], a["bid"], a["ask"])
        print(f"  {y}-{m:02d}: 틱 {len(a):,}개 저장", flush=True)


def save_z(y, m, t, bid, ask):
    """압축 저장: 시간/Bid는 앞 틱과의 차이(ms / 0.01달러), Ask는 스프레드(0.01달러). 그대로 복원 안 되면 에러"""
    TICK_Z_DIR.mkdir(parents=True, exist_ok=True)
    bi, ai = np.round(bid * 100).astype(np.int64), np.round(ask * 100).astype(np.int64)
    dt, db = np.diff(t, prepend=t[0]), np.diff(bi, prepend=bi[0])
    sp = ai - bi
    for x in (dt, db, sp):
        if len(x) and np.abs(x).max() >= 2 ** 31:
            raise ValueError(f"{y}-{m:02d}: 차이가 너무 커서 int32로 저장 불가")
    z = dict(t0=t[:1], b0=bi[:1], dt=dt.astype(np.int32), db=db.astype(np.int32), sp=sp.astype(np.int32))
    if not all(np.array_equal(u, v) for u, v in zip(unpack_z(z), (t, bid, ask))):
        raise ValueError(f"{y}-{m:02d}: 압축 후 원본과 다름 (가격이 0.01 단위가 아님?)")
    np.savez_compressed(TICK_Z_DIR / f"{y}-{m:02d}.npz", **z)


def unpack_z(z):
    t = z["t0"][0] + np.cumsum(z["dt"], dtype=np.int64) if len(z["dt"]) else np.zeros(0, np.int64)
    bi = z["b0"][0] + np.cumsum(z["db"], dtype=np.int64) if len(z["db"]) else np.zeros(0, np.int64)
    return t, bi / 100, (bi + z["sp"]) / 100


def load_month(y, m):
    f = TICK_DIR / f"{y}-{m:02d}.npz"
    if f.exists():
        z = np.load(f)
        return z["t"], z["bid"], z["ask"]
    return unpack_z(np.load(TICK_Z_DIR / f"{y}-{m:02d}.npz"))


def bars_from_ticks(minutes):
    out = []
    for y, m in months():
        t, bid, _ = load_month(y, m)
        key = t // (minutes * 60_000)
        df = pd.DataFrame({"k": key, "p": bid})
        g = df.groupby("k").p
        b = pd.DataFrame({"open": g.first(), "high": g.max(), "low": g.min(), "close": g.last()})
        out.append(b)
    b = pd.concat(out)
    b = b[~b.index.duplicated()]
    b["time"] = pd.to_datetime(b.index * minutes * 60, unit="s")
    return b.reset_index(drop=True)


def check_bars(m5):
    """틱으로 만든 5분봉과 MT5 5분봉 비교 (최근 2주)"""
    end = datetime.now(timezone.utc)
    r = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M5, end - timedelta(days=14), end)
    ref = pd.DataFrame(r)
    ref["time"] = pd.to_datetime(ref["time"], unit="s")
    j = ref.merge(m5, on="time", suffixes=("_mt5", "_tick"))
    diff = (j.close_mt5 - j.close_tick).abs()
    print(f"봉 검증: {len(j):,}개 비교, 종가 차이 최대 {diff.max():.3f} / 평균 {diff.mean():.4f}")


def signals(m5, h1):
    up = psar(m5.high.values, m5.low.values)
    r = rsi(m5.close).values
    h = pd.DataFrame({"close_time": h1.time + pd.Timedelta(hours=1),
                      "hdir": np.where(psar(h1.high.values, h1.low.values), 1, -1)})
    mm = pd.merge_asof(pd.DataFrame({"close_time": m5.time + pd.Timedelta(minutes=5)}), h,
                       on="close_time", direction="backward")
    hdir = mm.hdir.values
    t = m5.time.values.astype("datetime64[ms]").astype(np.int64)
    out = []
    for i in np.nonzero(up[1:-1] != up[:-2])[0] + 1:
        s = 1 if up[i] else -1
        if (r[i] - 50) * s > 0 and hdir[i] == s:
            out.append((t[i + 1], s))
    return out


def simulate(sigs, scaled):
    """달마다 (이번 달 + 다음 달) 틱을 불러 거래를 시뮬레이션. 손익은 달러/온스 단위."""
    ms = list(months())
    trades, busy_until, si = [], -1, 0
    for idx, (y, m) in enumerate(ms):
        s, e = month_range(y, m)
        s_ms, e_ms = int(s.timestamp() * 1000), int(e.timestamp() * 1000)
        cur = [load_month(y, m)] + ([load_month(*ms[idx + 1])] if idx + 1 < len(ms) else [])
        t = np.concatenate([c[0] for c in cur])
        bid = np.concatenate([c[1] for c in cur])
        ask = np.concatenate([c[2] for c in cur])
        while si < len(sigs) and sigs[si][0] < e_ms:
            st, side = sigs[si]
            si += 1
            if st < s_ms or st < busy_until:
                continue
            k = np.searchsorted(t, st)
            if k >= len(t) - 1:
                continue
            e_price = ask[k] if side == 1 else bid[k]
            f = e_price / REF_PRICE if scaled else 1.0
            sl, tp = SL * f, TP * f
            if side == 1:
                j = first_hit(bid, k + 1, e_price - sl, e_price + tp)
                if j < 0:
                    continue
                pnl = bid[j] - e_price
            else:
                j = first_hit(ask, k + 1, e_price - tp, e_price + sl)
                if j < 0:
                    continue
                pnl = e_price - ask[j]
            trades.append((t[k], t[j], side, e_price, float(pnl)))
            busy_until = t[j]
    tr = pd.DataFrame(trades, columns=["open_ms", "close_ms", "side", "entry", "pnl_oz"])
    tr["close"] = pd.to_datetime(tr.close_ms, unit="ms")
    tr["month"] = tr.close.dt.to_period("M")
    return tr


def monthly(tr):
    rows = []
    bal_fixed = bal_comp = START_BALANCE
    for mo, g in tr.groupby("month"):
        lot_c = max(0.01, np.floor(bal_comp / 1000 * LOT_PER_1000 * 100) / 100)
        p_fixed = g.pnl_oz.sum() * FIXED_LOT * CONTRACT
        p_comp = g.pnl_oz.sum() * lot_c * CONTRACT
        gl = -g.pnl_oz[g.pnl_oz < 0].sum()
        rows.append({
            "월": str(mo), "거래수": len(g), "승률%": round((g.pnl_oz > 0).mean() * 100, 1),
            "PF": round(g.pnl_oz[g.pnl_oz > 0].sum() / gl, 2) if gl else np.inf,
            "고정 손익$": round(p_fixed, 2), "고정 수익률%": round(p_fixed / bal_fixed * 100, 1),
            "복리 lot": lot_c, "복리 수익률%": round(p_comp / bal_comp * 100, 1),
            "복리 잔고": round(bal_comp + p_comp, 2),
        })
        bal_fixed += p_fixed
        bal_comp += p_comp
    return pd.DataFrame(rows)


def main():
    if not mt5_init():
        raise SystemExit(f"MT5 연결 실패: {mt5.last_error()}")
    try:
        print("틱 다운로드...", flush=True)
        download_all()
        m5 = bars_from_ticks(5)
        h1 = bars_from_ticks(60)
        check_bars(m5)
    finally:
        mt5.shutdown()
    sigs = signals(m5, h1)
    print(f"5분봉 {len(m5):,}개 ({m5.time.iat[0]:%Y-%m-%d} ~ {m5.time.iat[-1]:%Y-%m-%d}), 신호 {len(sigs):,}개\n")

    for name, scaled in (("고정 SL7.5/TP3", False), ("가격비례 SL/TP", True)):
        tr = simulate(sigs, scaled)
        mt = monthly(tr)
        tag = "fixed" if not scaled else "scaled"
        mt.to_csv(f"long_monthly_{tag}.csv", index=False, encoding="utf-8-sig")
        tr.drop(columns=["month"]).to_csv(f"long_trades_{tag}.csv", index=False, encoding="utf-8-sig")
        gl = -tr.pnl_oz[tr.pnl_oz < 0].sum()
        eq = START_BALANCE + (tr.pnl_oz * FIXED_LOT * CONTRACT).cumsum()
        peak = np.maximum.accumulate(np.r_[START_BALANCE, eq.values])[1:]
        print(f"===== {name} =====")
        print(f"거래 {len(tr):,}건, 승률 {(tr.pnl_oz > 0).mean():.1%}, PF {tr.pnl_oz[tr.pnl_oz > 0].sum() / gl:.3f}")
        print(f"0.02lot 고정: 최종 {eq.iat[-1]:,.0f}달러, 최대낙폭 {((peak - eq) / peak).max():.1%}, "
              f"손실월 {(mt['고정 손익$'] < 0).sum()}/{len(mt)}")
        yearly = tr.groupby(tr.close.dt.year).pnl_oz.agg(
            거래수="count", PF=lambda x: round(x[x > 0].sum() / -x[x < 0].sum(), 3))
        print("연도별\n" + yearly.to_string())
        print(mt.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
