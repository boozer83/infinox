"""뉴스 발표 뒤 금값이 얼마나 튀었나 (실제 틱, USD·EUR·GBP high/moderate, 2022-01 ~ 2026-10)

발표마다 발표 시각 중간가 기준으로 1분 / 5분 / 15분 안 최대 움직임(위·아래 중 큰 쪽, $와 %),
15분 뒤 순변화, 발표 직후 10초 최대 스프레드, 발표 전 30분 1분봉 평균 폭(평소 변동) 대비 배수
결과: results/news_moves.csv (발표마다), news_moves_types.csv (발표 종류별)

사용법: python -m backtests.news_moves
"""
import gc

import numpy as np
import pandas as pd

import backtests.long_period as lb
import backtests.news_hk as nh

OUT = lb.TICK_DIR.parent.parent / "results"
WIN = {"1분": 60_000, "5분": 300_000, "15분": 900_000}


def main():
    ev = nh.events()
    ev["key"] = ev.currency + " " + ev.event.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    ev_ms = ev.t.values.astype("datetime64[ms]").astype(np.int64)
    rows = []
    for y, m in lb.months():
        s, e = lb.month_range(y, m)
        idx = np.flatnonzero((ev_ms >= int(s.timestamp() * 1000)) & (ev_ms < int(e.timestamp() * 1000)))
        if not len(idx):
            continue
        t, bid, ask = lb.load_month(y, m)
        for i in idx:
            ms = int(ev_ms[i])
            k0 = int(np.searchsorted(t, ms))
            if k0 >= len(t) or t[k0] - ms > 20_000:
                continue
            k1 = int(np.searchsorted(t, ms + WIN["15분"]))
            if k1 - k0 < 10:
                continue
            mid = (bid[k0:k1] + ask[k0:k1]) / 2
            m0 = mid[0]
            r = {"i": i, "가격": round(m0, 2)}
            for n, w in WIN.items():
                kk = int(np.searchsorted(t[k0:k1], ms + w))
                seg = mid[: max(1, kk)] - m0
                up, dn = seg.max(), seg.min()
                r[f"{n} 최대$"] = round(up if up >= -dn else dn, 2)
            r["15분 순변화$"] = round(mid[-1] - m0, 2)
            k10 = int(np.searchsorted(t, ms + 10_000))
            r["발표 10초 최대 스프레드"] = round((ask[k0:k10] - bid[k0:k10]).max(), 2) if k10 > k0 else np.nan
            p0 = int(np.searchsorted(t, ms - 1_800_000))
            if k0 - p0 > 10:
                pre = pd.Series(bid[p0:k0], index=pd.to_datetime(t[p0:k0], unit="ms")).resample("1min")
                r["평소 1분 폭"] = round(float((pre.max() - pre.min()).mean()), 2)
            rows.append(r)
        del t, bid, ask
        gc.collect()
        print(f"  {y}-{m:02d} 완료", flush=True)
    d = pd.DataFrame(rows).set_index("i").join(ev[["t", "key", "importance"]])
    for n in WIN:
        d[f"{n} 최대%"] = (d[f"{n} 최대$"] / d.가격 * 100).round(3)
    d["5분/평소"] = (d["5분 최대$"].abs() / d["평소 1분 폭"]).round(1)
    d = d.sort_values("t")
    d.to_csv(OUT / "news_moves.csv", index=False, encoding="utf-8-sig")
    a5 = d["5분 최대%"].abs()
    g = d.assign(a5=a5, a15=d["15분 최대%"].abs(), big=a5 >= 0.1, huge=a5 >= 0.2).groupby(["key", "importance"])
    ty = pd.DataFrame({"횟수": g.size(), "5분 최대% 중간값": g.a5.median().round(3), "5분 최대% 평균": g.a5.mean().round(3),
                       "15분 최대% 중간값": g.a15.median().round(3), "0.1% 넘은 비율": (g.big.mean() * 100).round(0),
                       "0.2% 넘은 비율": (g.huge.mean() * 100).round(0),
                       "발표 스프레드 중간값": g["발표 10초 최대 스프레드"].median().round(2),
                       "5분/평소 중간값": g["5분/평소"].median().round(1)}).reset_index()
    ty = ty[ty.횟수 >= 10].sort_values("5분 최대% 중간값", ascending=False)
    ty.to_csv(OUT / "news_moves_types.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print(ty.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
