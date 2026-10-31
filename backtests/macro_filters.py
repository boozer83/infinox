"""거시 필터 2개를 지금 봇 거래에 (2026-10-31, 다른 AI 후보 1·2번, xauusd_new_strategy_candidates.md)

데이터 (data/macro/, git 제외):
  DFII10.csv     : 미국 10년 TIPS 실질금리 (FRED, 일별, 미국 날짜 종가)
  cot_gold.csv   : CFTC 레거시 COT, 금 COMEX (publicreporting.cftc.gov 6dca-aqww), 화요일 기준, 금요일 발표

1. 실질금리 레짐: z = (5일 변화) / 최근 252일 동안 5일 변화의 표준편차 (평균 빼고). 전날까지 값만 씀 (미국 날짜 d 값 -> 서버 d+1부터)
   z <= -k: 금 매수 쪽 (매도 진입 안 함), z >= +k: 금 매도 쪽 (매수 진입 안 함). k = 0 / 0.5 / 1.0
2. COT 극단: 상업 순포지션, 비상업(투기) 순포지션 각각 156주 z. 발표 다음 주 월요일(서버)부터 씀
   상업 z >= +k 이고 투기 z <= -k: 금 매수 쪽 (매도 진입 안 함), 반대면 매수 진입 안 함. k = 1.0 / 1.5 / 2.0
   + 단독 스윙: 신호가 켜진 주 월요일 첫 틱 진입, 두 z가 ±1 안으로 돌아온 다음 월요일 청산, 손절 3 x 일봉 ATR (실제 틱)

대상 거래: SAR 단타 results/sar_regime_trades.csv (2,441건, oz), 아시아 3마틴 v2 (asia_v2, 걸러진 날은 신호가 없던 것처럼 마틴 순서 다시),
          H4 추세 + 구름 (trend_box_filter와 같은 거래, 2% 위험 1만$)
통과 = 승률과 수익이 전체 / 2022~23 / 2024~26 모두 지금보다 나음 (기존 전략 수정 규칙)

사용법: python -m backtests.macro_filters
"""
import gc
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.asia_v2 as av
from backtests.gold_rest_others import load_m1, resample
from backtests.ichimoku_others import attach, keep_mask, trend_entries
from backtests.strategies import TickStore, bar_exit_specs, execute, pf
from backtests.trend_box_filter import sized
from indicators import atr

ROOT = Path(__file__).resolve().parent.parent
MACRO = ROOT / "data" / "macro"
RES = ROOT / "results"


def real_yield_z():
    d = pd.read_csv(MACRO / "DFII10.csv", na_values=".").dropna()
    d["date"] = pd.to_datetime(d.observation_date)
    s = d.set_index("date").DFII10.astype(float)
    ch = s.diff(5)
    z = (ch - ch.rolling(252).mean()) / ch.rolling(252).std()
    z.index = z.index + pd.Timedelta(days=1)            # 미국 날짜 d 종가 -> 다음 날부터 씀
    return z


def cot_z():
    d = pd.read_csv(MACRO / "cot_gold.csv")
    d = d[d.cftc_contract_market_code.astype(str).str.lstrip("0") == "88691"].copy()
    d["date"] = pd.to_datetime(d.report_date_as_yyyy_mm_dd).dt.normalize()
    d = d.sort_values("date").drop_duplicates("date").set_index("date")
    comm = d.comm_positions_long_all - d.comm_positions_short_all
    spec = d.noncomm_positions_long_all - d.noncomm_positions_short_all
    z = pd.DataFrame({"comm": (comm - comm.rolling(156).mean()) / comm.rolling(156).std(),
                      "spec": (spec - spec.rolling(156).mean()) / spec.rolling(156).std()})
    z.index = z.index + pd.Timedelta(days=6)            # 화요일 기준 -> 금요일 발표 -> 다음 월요일부터
    return z


def bias_ry(z, k):
    """+1 = 금 매수 쪽 (매도 막음), -1 = 매도 쪽 (매수 막음), 0 = 중립"""
    return pd.Series(np.where(z <= -k, 1, np.where(z >= k, -1, 0)), index=z.index) if k > 0 else -np.sign(z)


def bias_cot(z, k):
    return pd.Series(np.where((z.comm >= k) & (z.spec <= -k), 1, np.where((z.comm <= -k) & (z.spec >= k), -1, 0)), index=z.index)


def at(bias, times):
    """거래 시각마다 그 전에 알려진 마지막 바이어스"""
    b = bias.dropna().sort_index()
    i = np.searchsorted(b.index.values, pd.to_datetime(times).values, side="right") - 1
    out = np.where(i >= 0, b.values[np.maximum(i, 0)], 0)
    return np.nan_to_num(out.astype(float))


def keep_of(bias_vals, side):
    return ~((bias_vals == 1) & (side == -1) | (bias_vals == -1) & (side == 1))


def judge(name, base, new):
    """base / new: dict 기간 -> (승률, 수익). 전체·앞·뒤 모두 둘 다 나아야 통과"""
    ok = all(new[p][0] > base[p][0] and new[p][1] > base[p][1] for p in base)
    return {"필터": name, **{f"{p} 승률": new[p][0] for p in base}, **{f"{p} 수익": new[p][1] for p in base}, "통과": ok}


def periods(t):
    y = pd.to_datetime(t).dt.year if hasattr(t, "dt") else pd.DatetimeIndex(t).year
    return {"전체": np.ones(len(y), bool), "22-23": np.asarray(y < 2024), "24-26": np.asarray(y >= 2024)}


def sar_stats(d, keep):
    out = {}
    for p, m in periods(d.open).items():
        x = d.oz[m & keep]
        out[p] = (round((x > 0).mean() * 100, 1), round(x.sum(), 1))
    return out


def h4_trades():
    m1 = load_m1("XAUUSD")
    h4 = resample(m1, 240)
    del m1
    gc.collect()
    ent, exit_fn = trend_entries(h4)
    specs = bar_exit_specs(h4, ent, exit_fn, (3.0 * atr(h4)).values)
    tr = execute(specs, TickStore())
    spec_t = np.array([s["entry"] for s in specs])
    t_ms = h4.time.values.astype("datetime64[ms]").astype(np.int64)
    k = np.searchsorted(spec_t, tr.open_ms.values, side="right") - 1
    sig = np.searchsorted(t_ms, spec_t[k]) - 1
    t_sig = (h4.time + pd.Timedelta(hours=4)).values[sig]
    cloud = np.asarray(keep_mask(attach(t_sig, h4, 240, h4.close.values[sig]), tr.side.values, "구름 위/아래만"))
    tr["open"] = pd.to_datetime(tr.open_ms, unit="ms")
    tr["close"] = pd.to_datetime(tr.close_ms, unit="ms")
    return tr[cloud].reset_index(drop=True), h4


def h4_stats(tr, keep):
    out = {}
    for p, m in periods(tr.close).items():
        t = tr[m & keep]
        bal, _ = sized(t)
        out[p] = (round((t.pnl_oz > 0).mean() * 100, 1), round(bal - 10_000))
    return out


def asia_stats(sig, oc, keep):
    out = {}
    s = sig[keep]
    for p, m in periods(s.day).items():
        tr, _ = av.simulate(s[m], oc, True, 0.75, True)
        out[p] = (round((tr.pnl > 0).mean() * 100, 1), round(tr.bal.iat[-1] - av.BAL0) if len(tr) else 0)
    return out


def cot_swing(h4, z, k):
    """COT 극단 단독 스윙: 바이어스가 켜진 주 월요일 서버 01:05 진입, 꺼지거나 바뀌면 그 주 월요일 청산, 손절 3 x 일봉 ATR"""
    d1 = h4.set_index("time").resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    a = atr(d1).shift(1)
    b = bias_cot(z, k)
    b = b[b.index >= "2021-12-27"]
    specs, cur = [], None
    for t, v in b.items():
        mon = pd.Timestamp(t) + pd.Timedelta(hours=1, minutes=5)
        ms = int(mon.value // 1_000_000)
        if cur and v != cur["side"]:
            cur["exit"] = ms
            specs.append(cur)
            cur = None
        if cur is None and v != 0:
            i = a.index.searchsorted(pd.Timestamp(t))
            if i < len(a) and a.iat[i] > 0:
                cur = {"entry": ms, "side": int(v), "sl_dist": 3 * a.iat[i]}
    tr = execute(specs, TickStore()) if specs else pd.DataFrame()
    return tr


def main():
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    ry, cz = real_yield_z(), cot_z()
    filters = [(f"실질금리 k{k}", bias_ry(ry, k)) for k in (0, 0.5, 1.0)] + \
              [(f"COT k{k}", bias_cot(cz, k)) for k in (1.0, 1.5, 2.0)]

    rows = []
    sar = pd.read_csv(RES / "sar_regime_trades.csv", parse_dates=["open"])
    base = sar_stats(sar, np.ones(len(sar), bool))
    rows.append({"봇": "SAR 단타", **judge("지금", base, base)})
    for name, b in filters:
        keep = keep_of(at(b, sar.open), sar.side.values)
        rows.append({"봇": "SAR 단타", **judge(name, base, sar_stats(sar, keep)), "거른 비율%": round((~keep).mean() * 100, 1)})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    tr, h4 = h4_trades()
    base = h4_stats(tr, np.ones(len(tr), bool))
    rows.append({"봇": "H4 추세", **judge("지금", base, base)})
    for name, b in filters:
        keep = keep_of(at(b, tr.open), tr.side.values)
        rows.append({"봇": "H4 추세", **judge(name, base, h4_stats(tr, keep)), "거른 비율%": round((~keep).mean() * 100, 1)})
    for k in (1.0, 1.5, 2.0):
        sw = cot_swing(h4, cz, k)
        if len(sw):
            y = sw.close.dt.year
            print(f"COT 단독 스윙 k{k}: {len(sw)}건 승률 {(sw.pnl_oz > 0).mean() * 100:.0f}% 합 {sw.pnl_oz.sum():+.0f}$/oz "
                  f"PF " + " / ".join(f"{yr}:{pf(sw.pnl_oz[y == yr])}" for yr in range(2022, 2027)), flush=True)
        else:
            print(f"COT 단독 스윙 k{k}: 신호 없음", flush=True)
    del h4
    gc.collect()
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    av.BAL0 = 10_000.0
    sig = av.day_signals()
    sig = sig[sig.side == sig.mom].reset_index(drop=True)
    oc = av.outcomes(sig)
    base = asia_stats(sig, oc, np.ones(len(sig), bool))
    rows.append({"봇": "아시아", **judge("지금", base, base)})
    for name, b in filters:
        keep = keep_of(at(b, sig.t), sig.side.values)
        rows.append({"봇": "아시아", **judge(name, base, asia_stats(sig, oc, keep)), "거른 비율%": round((~keep).mean() * 100, 1)})

    out = pd.DataFrame(rows)
    out.to_csv(RES / "macro_filters.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    for name, b in filters:
        print(f"{name}: 2022-01 이후 매수 쪽 {(b[b.index >= '2022'] == 1).mean() * 100:.0f}% / 매도 쪽 {(b[b.index >= '2022'] == -1).mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
