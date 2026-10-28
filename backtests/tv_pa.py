"""PA Master Grade-A (tv_batch2.pa_master) 더 파보기 - 금 (2022-01 ~ 2026-10)

1. 설정 흔들기: 원본 기본값에서 한 번에 하나씩만 바꿈 (1시간봉, 익절 1R / 2R / 3R)
   피벗 길이 3/5/7/10, 가치 구간 폭 0.3/0.5/0.8 ATR, 거래량 배수 1.0/1.1/1.3·끔, 몸통 최소 0.4/0.6/0.8 ATR,
   EMA 21/200 -> 20/100, 34/200, 50/200
2. 봉 길이: 30분 / 1시간 / 2시간 / 3시간 / 4시간 (기본 설정)
3. 나눠 보기 (1시간봉, 2R): 패턴 종류, 가치 구간 대 BOS만, 매수 대 매도, 진입 시간대(서버 시간)
체결·비용: tv_batch2.rr_trades (다음 봉 시가 진입, 손절 = 패턴 끝, 한 봉에 둘 다 닿으면 손절, 실제 스프레드·스왑)

사용법: python -m backtests.tv_pa
"""
import numpy as np
import pandas as pd

import backtests.tickvol_m1 as tv
import backtests.tv_single as ts
from backtests.tv_batch2 import bars_vol, pa_master, rr_trades

GRID = [("원본 기본값", {})] + \
    [(f"피벗 {x}", {"piv": x}) for x in (3, 7, 10)] + \
    [(f"가치 구간 {x} ATR", {"touch": x}) for x in (0.3, 0.8)] + \
    [(f"거래량 x{x}", {"volm": x}) for x in (1.0, 1.3)] + [("거래량 조건 끔", {"use_vol": False})] + \
    [(f"몸통 최소 {x} ATR", {"body_min": x}) for x in (0.4, 0.8)] + \
    [(f"EMA {a}/{b}", {"ef_len": a, "em_len": b}) for a, b in ((20, 100), (34, 200), (50, 200))]
YC = [str(y) for y in ts.YEARS]


def row(name, tr):
    if len(tr) < 5:
        return {"구분": name, "거래": len(tr)}
    r = ts.summary(tr)
    return {"구분": name, **{k: r[k] for k in ["거래", "승률%", "PF", "합계%", "최대낙폭%", "매수PF", "매도PF"] + YC}}


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = ts.load_m1("XAUUSD")
    vol = tv.load()
    h1 = bars_vol(m1, vol, 60)

    print("===== 1. 설정 흔들기 (1시간봉) =====")
    out = []
    for name, kw in GRID:
        side, stop, _, _ = pa_master(h1, **kw)
        r = {"설정": name}
        for rr in (1.0, 2.0, 3.0):
            tr = rr_trades(h1, side, stop, rr)
            r[f"PF {rr:.0f}R"] = round(ts.pf(tr.pct), 2)
            if rr == 2.0:
                s = ts.summary(tr)
                r.update({"거래": s["거래"], "매수PF 2R": s["매수PF"], "매도PF 2R": s["매도PF"],
                          **{f"{y} 2R": s[y] for y in YC}})
        out.append(r)
    g = pd.DataFrame(out)
    print(g.to_string(index=False))

    print("\n===== 2. 봉 길이 (기본 설정) =====")
    out = []
    for tf, mins in (("M30", 30), ("H1", 60), ("H2", 120), ("H3", 180), ("H4", 240)):
        d = h1 if mins == 60 else bars_vol(m1, vol, mins)
        side, stop, _, _ = pa_master(d)
        for rr in (1.0, 2.0, 3.0):
            out.append(row(f"{tf} {rr:.0f}R", rr_trades(d, side, stop, rr)))
    print(pd.DataFrame(out).to_string(index=False))

    print("\n===== 3. 나눠 보기 (1시간봉, 2R) =====")
    side, stop, pat, ctx = pa_master(h1, parts=True)
    tr = rr_trades(h1, side, stop, 2.0)
    k = np.searchsorted(h1.time.values, tr.open.values) - 1        # 신호 봉
    tr["패턴"], tr["구간"] = pat[k], ctx[k]
    tr["시간"] = pd.to_datetime(tr.open).dt.hour
    tr["시간대"] = pd.cut(tr.시간, [-1, 7, 14, 20, 23], labels=["01-07 아시아", "08-14 유럽", "15-20 미국", "21-23 미국 후반"])
    out = [row("전체", tr)]
    for col in ("패턴", "구간", "시간대"):
        for v, t in tr.groupby(col, observed=True):
            out.append(row(f"{col}: {v}", t))
    for s, nm in ((1, "매수"), (-1, "매도")):
        out.append(row(f"방향: {nm}", tr[tr.side == s]))
    print(pd.DataFrame(out).to_string(index=False))
    tr.to_csv("results/tv_pa_trades_h1_2r.csv", index=False, encoding="utf-8-sig")
    g.to_csv("results/tv_pa_grid.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
