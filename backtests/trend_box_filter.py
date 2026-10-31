"""H4 추세: 박스권(횡보) 진입을 거르는 필터 + 스왑 (2026-10-31 사용자 요청: 박스권에 추세 들어가면 덜 먹는다)

기준 = 지금 봇: N25 돌파 진입, 12봉 청산, 손절 3 ATR, 4시간봉 구름 같은 쪽만 (아니면 그림자), 2% 위험, 1만 달러, 실제 틱 체결.
필터는 신호 봉까지 완성된 봉으로만 계산. 걸린 신호도 그림자처럼 자리를 차지 (거래 순서는 필터 없는 순서 그대로, 구름 필터와 같은 방식)
  ADX(14) 4시간봉 / 일봉 < 기준이면 쉼
  Choppiness(14) > 기준이면 쉼  (100 x log10(ATR1 합 / (최고-최저)) / log10(n))
  돌파 직전 25봉 폭 / ATR(14) < 기준이면 쉼 (좁은 박스)
  효율비 ER(25) = |종가 변화| / |봉마다 변화| 합 < 기준이면 쉼
  GPT 제안: ER(20), ATR / 직전 50봉 ATR 중앙값, 10봉 평균 봉 겹침률, 돌파 여유 (종가 - 25봉 극값) / ATR
스왑: 지금 계좌 값 (포인트, 매수 -78.9 / 매도 +25.57 x 0.01$ = 1온스당 하룻밤 -0.789$ / +0.256$, 수요일 밤 3배, 주말 없음)

사용법: python -m backtests.trend_box_filter
"""
import gc

import numpy as np
import pandas as pd

from backtests.gold_rest_others import load_m1, resample
from backtests.ichimoku_others import attach, keep_mask, trend_entries
from backtests.strategies import TickStore, bar_exit_specs, execute, pf
from backtests.trend_portfolio import rollovers
from indicators import adx, atr

SWAP_LONG, SWAP_SHORT, TRIPLE = -0.789, 0.2557, 3    # $/oz 하룻밤, MQL 요일 3 = 수요일
BAL0 = 10_000.0


def chop(h, n=14):
    tr = pd.concat([h.high - h.low, (h.high - h.close.shift()).abs(), (h.low - h.close.shift()).abs()], axis=1).max(axis=1)
    rng = h.high.rolling(n).max() - h.low.rolling(n).min()
    return 100 * np.log10(tr.rolling(n).sum() / rng) / np.log10(n)


def er(c, n=25):
    return (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()


def sized(tr, risk_pct=0.02):
    """잔고 % 위험으로 1만 달러 복리: 최종$, 최대낙폭%"""
    bal, peak, mdd = BAL0, BAL0, 0.0
    for r in tr.itertuples():
        lot = max(0.01, np.floor(bal * risk_pct / (r.risk * 100) * 100 + 1e-9) / 100)
        bal += r.pnl_oz * lot * 100
        peak, mdd = max(peak, bal), max(mdd, (peak - bal) / peak)
    return bal, mdd


def row(name, tr, col):
    t = tr.assign(pnl_oz=tr[col])
    bal, mdd = sized(t)
    y = t.close.dt.year
    r = {"방식": name, "거래": len(t), "승률%": round((t.pnl_oz > 0).mean() * 100, 1), "PF": pf(t.pnl_oz),
         "R합": round((t.pnl_oz / t.risk).sum(), 1), "1만$->": round(bal), "최대낙폭%": round(mdd * 100, 1)}
    for yr in range(2022, 2027):
        r[str(yr)] = round((t.pnl_oz[y == yr] / t.risk[y == yr]).sum(), 1)
    return r


def main():
    m1 = load_m1("XAUUSD")
    h4, d1 = resample(m1, 240), resample(m1, 1440)
    del m1
    gc.collect()
    ent, exit_fn = trend_entries(h4)
    sl = (3.0 * atr(h4)).values
    specs = bar_exit_specs(h4, ent, exit_fn, sl)
    tr = execute(specs, TickStore())
    spec_t = np.array([s["entry"] for s in specs])
    t_ms = h4.time.values.astype("datetime64[ms]").astype(np.int64)
    k = np.searchsorted(spec_t, tr.open_ms.values, side="right") - 1
    sig = np.searchsorted(t_ms, spec_t[k]) - 1
    side = tr.side.values
    t_sig = (h4.time + pd.Timedelta(hours=4)).values[sig]
    cloud = np.asarray(keep_mask(attach(t_sig, h4, 240, h4.close.values[sig]), side, "구름 위/아래만"))

    tr["open"] = pd.to_datetime(tr.open_ms, unit="ms")
    tr["close"] = pd.to_datetime(tr.close_ms, unit="ms")
    nights = np.array([rollovers(o, c, TRIPLE) for o, c in zip(tr.open, tr.close)])
    tr["swap"] = nights * np.where(side == 1, SWAP_LONG, SWAP_SHORT)
    tr["pnl_sw"] = tr.pnl_oz + tr.swap

    # 신호 봉 시점 지표 (4시간봉은 신호 봉 그대로, 일봉은 신호 시각 전에 완성된 봉)
    a4 = atr(h4)
    feat = pd.DataFrame({
        "adx4": adx(h4).values[sig] if not isinstance(adx(h4), tuple) else adx(h4)[0].values[sig],
        "chop": chop(h4).values[sig],
        "width": ((h4.high.rolling(25).max() - h4.low.rolling(25).min()).shift(1) / a4).values[sig],
        "er": er(h4.close).values[sig],
    "er20": er(h4.close, 20).values[sig],
    "atr_rel": (a4 / a4.shift(1).rolling(50).median()).values[sig],
    "overlap": ((np.minimum(h4.high, h4.high.shift()) - np.maximum(h4.low, h4.low.shift())).clip(lower=0)
                / (h4.high - h4.low).replace(0, np.nan)).rolling(10).mean().values[sig],
    "margin": (np.where(side == 1, h4.close.values[sig] - h4.high.rolling(25).max().shift(1).values[sig],
                        h4.low.rolling(25).min().shift(1).values[sig] - h4.close.values[sig]) / a4.values[sig]),
    })
    ad = adx(d1)
    ad = ad[0] if isinstance(ad, tuple) else ad
    dd = pd.DataFrame({"end": d1.time + pd.Timedelta(days=1), "adxd": ad.values})
    feat["adxd"] = pd.merge_asof(pd.DataFrame({"t": t_sig}), dd, left_on="t", right_on="end").adxd.values

    cases = [("기준 (지금 봇)", np.ones(len(tr), bool))]
    for th in (15, 18, 20, 22, 25):
        cases.append((f"4시간 ADX < {th} 쉼", feat.adx4.values >= th))
    for th in (15, 20, 25):
        cases.append((f"일봉 ADX < {th} 쉼", feat.adxd.values >= th))
    for th in (50, 55, 61.8):
        cases.append((f"Chop > {th} 쉼", feat.chop.values <= th))
    for th in (3, 4, 5, 6):
        cases.append((f"25봉 폭 < {th} ATR 쉼", feat.width.values >= th))
    for th in (0.15, 0.2, 0.3):
        cases.append((f"ER(25) < {th} 쉼", feat.er.values >= th))

    # GPT 제안 (2026-10-31): ER(20), ATR 상대수준, 봉 겹침률, 돌파 여유
    for th in (0.15, 0.2, 0.25):
        cases.append((f"ER(20) < {th} 쉼", feat.er20.values >= th))
    for th in (0.9, 1.0, 1.1):
        cases.append((f"ATR/50봉 중앙 < {th} 쉼", feat.atr_rel.values >= th))
    for th in (0.6, 0.7, 0.8):
        cases.append((f"10봉 겹침률 > {th} 쉼", feat.overlap.values <= th))
    for th in (0.05, 0.1, 0.15):
        cases.append((f"돌파 여유 < {th} ATR 쉼", feat.margin >= th))

    rows = []
    for name, m in cases:
        t = tr[cloud & m].reset_index(drop=True)
        rows.append(row(name, t, "pnl_oz"))
        rows.append(row(name + " +스왑", t, "pnl_sw"))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    print(out.to_string(index=False))
    base = tr[cloud]
    print(f"\n스왑: 거래당 평균 {base.swap.mean():+.2f}$/oz, 합계 {base.swap.sum():+.1f}$/oz (매수 {base.swap[base.side == 1].sum():+.1f} / 매도 {base.swap[base.side == -1].sum():+.1f}), "
          f"평균 보유 {((base.close - base.open).dt.total_seconds() / 86400).mean():.1f}일")


if __name__ == "__main__":
    main()
