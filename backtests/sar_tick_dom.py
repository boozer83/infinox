"""SAR 단타: 진입 직전 매수·매도 우세를 보고 (2026-10-29 사용자 아이디어, 실제 틱)

우세 = 신호 시각 직전 W초 동안 Bid가 오른 틱 수 / (오른 틱 + 내린 틱)
       >= 0.55 매수 우세, <= 0.45 매도 우세, 그 사이는 우세 없음 (금은 실제 거래량이 없어서 틱 방향으로)
  skip_W : 우세가 SAR 신호 방향과 반대면 쉬기
  flip_W : 우세가 반대면 우세 방향으로 진입 (반대 콩 청산은 의미 없어서 이 거래만 손절 7.5 / 익절 3, 금요일 마감)
  W = 30 / 60 / 120 / 300 / 900 / 3600초. 끝에 _0.6 처럼 붙이면 우세 기준을 0.6 / 0.4로
나머지는 지금 봇 그대로 (backtests.sar_martin.trades). 기준은 results/sar_us_now_YYYY.csv
결과 results/sar_dom_<방식>_YYYY.csv

사용법: python -m backtests.sar_tick_dom 2025 skip_300 flip_300 ...
        python -m backtests.sar_tick_dom sum
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtests.fast_entry as fe
import backtests.long_period as lb
import backtests.sar_martin as sm
import backtests.winrate_ideas as wi

RES = Path(__file__).resolve().parent.parent / "results"
WINDOWS = (30, 60, 120, 300, 900, 3600)
HI, LO = 0.55, 0.45
MODE = "skip_300"
DOM = {}                  # 신호 ms -> {W: 우세 (1, -1, 0)}
_orig_sig = fe.bar_close_signals
_orig_trade = sm.TRADE
_flip = {"on": False}
EMPTY = np.array([], dtype=np.int64)


def dominance(sig_ms):
    """신호마다 창별 우세 (한 달씩 틱을 읽음)"""
    out = {w: np.full(len(sig_ms), 0.5) for w in WINDOWS}
    months = [(y, m) for y, m in lb.months()]
    for idx, (y, m) in enumerate(months):
        s, e = lb.month_range(y, m)
        sel = np.flatnonzero((sig_ms >= s.timestamp() * 1000) & (sig_ms < e.timestamp() * 1000))
        if not len(sel):
            continue
        t, bid, _ = lb.load_month(y, m)
        if idx > 0:                                   # 월초 신호용으로 전달 끝부분
            pt, pb, _ = lb.load_month(*months[idx - 1])
            keep = pt >= pt[-1] - 2 * 3600_000
            t, bid = np.concatenate([pt[keep], t]), np.concatenate([pb[keep], bid])
        d = np.sign(np.diff(bid))
        up, dn = np.concatenate([[0], np.cumsum(d > 0)]), np.concatenate([[0], np.cumsum(d < 0)])
        k1 = np.searchsorted(t, sig_ms[sel])              # 신호 시각 전 마지막 틱까지
        for w in WINDOWS:
            k0 = np.searchsorted(t, sig_ms[sel] - w * 1000)
            k1c = np.maximum(k1 - 1, 0)
            u, n = up[k1c] - up[k0], (up[k1c] - up[k0]) + (dn[k1c] - dn[k0])
            r = np.where(n > 0, u / np.maximum(n, 1), 0.5)
            out[w][sel] = r
        print(f"  우세 {y}-{m:02d}", flush=True)
    return out


def signals(m5, *a, **k):
    sig = _orig_sig(m5, *a, **k)
    ms = sig.t.values.astype("datetime64[ms]").astype(np.int64)
    miss = [x for x in ms if x not in DOM]
    if miss:
        dm = dominance(np.array(miss))
        for i, x in enumerate(miss):
            DOM[x] = {w: dm[w][i] for w in WINDOWS}
    kind, w, *th = MODE.split("_")
    hi = float(th[0]) if th else HI
    r = np.array([DOM[x][int(w)] for x in ms])
    d = np.where(r >= hi, 1, np.where(r <= 1 - hi, -1, 0))
    against = (d != 0) & (d != sig.side.values)
    sig["flip"] = False
    if kind == "skip":
        sig = sig[~against].reset_index(drop=True)
    else:
        sig.loc[against, "side"] = -sig.loc[against, "side"]
        sig.loc[against, "flip"] = True
    print(f"  반대 우세 {against.sum()}건 / 신호 {len(against)}건 ({kind})", flush=True)
    return sig


class Acc(wi.Acc):
    def on_signal(self, s, t, bid, ask, x):
        _flip["on"] = bool(getattr(s, "flip", False))
        return super().on_signal(s, t, bid, ask, x)


def trade(t, bid, ask, k, side, fc, against, o, sig_px):
    if _flip["on"]:
        against = {side: EMPTY, -side: EMPTY}
    return _orig_trade(t, bid, ask, k, side, fc, against, o, sig_px)


def run(year, modes):
    global MODE
    fe.bar_close_signals = signals
    sm.Acc, sm.TRADE = Acc, trade
    for m in modes:
        MODE = m
        sm.MAE.clear()
        scaled = year < 2025
        sm.trades(year, scaled)
        os.replace(f"sar_martin_{'scaled' if scaled else 'trades'}_{year}.csv", RES / f"sar_dom_{m}_{year}.csv")


def summary():
    rows = []
    names = {"지금 봇": "sar_us_now_{y}.csv"}
    for f in sorted(RES.glob("sar_dom_*_2025.csv"), key=lambda f: (f.stem.split("_")[2], int(f.stem.split("_")[3]), f.stem)):
        kind, w, *th = f.stem.split("_")[2:-1]
        names[f"{'쉬기' if kind == 'skip' else '우세 방향'} {w}초" + (f" 기준 {th[0]}" if th else "")] =             f.name.replace("2025", "{y}")
    for name, pat in names.items():
        parts = [pd.read_csv(RES / pat.format(y=y)).assign(y=y) for y in range(2022, 2027) if (RES / pat.format(y=y)).exists()]
        tr = pd.concat(parts)
        row = {"규칙": name, "건수": len(tr), "승률%": round((tr.oz > 0).mean() * 100, 1), "oz합": round(tr.oz.sum(), 1)}
        row.update({f"승률{y % 100}": round((tr.oz[tr.y == y] > 0).mean() * 100, 1) for y in range(2022, 2027)})
        row.update({f"oz{y % 100}": round(tr.oz[tr.y == y].sum(), 1) for y in range(2022, 2027)})
        eq = tr.oz.cumsum()
        row["낙폭oz"] = round((eq.cummax() - eq).max(), 1)
        rows.append(row)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    pd.set_option("display.unicode.east_asian_width", True)
    print(res.to_string(index=False))
    res.to_csv(RES / "sar_tick_dom.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    if sys.argv[1] == "sum":
        summary()
    else:
        run(int(sys.argv[1]), sys.argv[2:] or [MODE])
