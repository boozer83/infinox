"""사용자가 받은 트레이딩뷰 전략 2개를 금에 (2022-01 ~ 2026-10, 기본 설정 그대로)

1) Supertrend Mult TP (keio1989)
   일봉 SuperTrend(10, 2) 와 차트 봉 SuperTrend(10, 4) 가 둘 다 위 -> 매수, 둘 다 아래 -> 매도, 하나라도 바뀌면 청산
   일봉 값은 그날 일봉이 끝난 뒤부터 씀 (request.security, lookahead 없음)
   나눠 익절: 매수 +3/5/7/10/15/20/25/30/40%, 매도 -1/3/5/7/10/15/20/25/30% 에서 처음 수량의 10%씩. 손절 없음 (기본 꺼짐)
2) 볼린저밴드돌파_20240228
   BB(SMA 15, 표준편차 x 2), 폭은 최소 가운데선의 3% (코인용 기본값). 종가가 위 밴드 상향 돌파 -> 매수 (매도 중이면 갈아탐),
   아래 밴드 하향 돌파 -> 매도. 매수 청산 = 아래 밴드 하향 돌파, 매도 청산 = 위 밴드 상향 돌파
   손절: 진입가 기준 max(-7%, -2 x ATR14) (매수), 고정 (트레일링 OFF). 익절은 수량 0%라 꺼짐. 거래량·ADX 등 필터 기본 꺼짐
   비교용: 최소 폭 0% (그냥 볼린저 2 표준편차)
체결·비용: tv_single과 같음 (봉 마감 신호 -> 다음 봉 시가, 실제 스프레드, 미끄러짐 0.02, 스왑). 손절·익절은 봉 고가·저가로 확인
           (한 봉에 손절과 익절이 같이 닿으면 손절 먼저)

사용법: python -m backtests.tv_bbst
"""
import numpy as np
import pandas as pd

import backtests.tv_single as ts
from indicators import atr

LONG_TP = [3, 5, 7, 10, 15, 20, 25, 30, 40]
SHORT_TP = [1, 3, 5, 7, 10, 15, 20, 25, 30]
TP_PART = 0.10


def daily_trend_on(d, d1):
    """차트 봉마다 '이미 끝난 마지막 일봉'의 SuperTrend(10, 2) 방향"""
    tr = pd.DataFrame({"end": d1.time + pd.Timedelta(days=1), "st": ts.supertrend_dir(d1, 10, 2.0)})
    end = d.time + (d.time.iat[1] - d.time.iat[0])
    end = pd.DataFrame({"end": end.values})
    m = pd.merge_asof(end, tr, on="end", direction="backward")
    return m.st.fillna(0).values


def run(d, want, stop_fn=None, tps=None):
    """want[i]: 봉 i 마감에 원하는 포지션 -> 봉 i+1 시가 체결
    stop_fn(i, side, entry) -> 손절가 (진입 봉 직전 값 기준), tps: 나눠 익절 % 목록 (매수용, 매도용)"""
    o, h, l, sp, t = (d[k].values for k in ("open", "high", "low", "sp", "time"))
    rows = []
    cur, ent, et, left, sl, hit = 0, 0.0, None, 1.0, np.nan, 0
    pnl = 0.0

    def close_all(i, px):
        nonlocal cur, pnl
        p = pnl + (px - ent) * cur * left
        p += ts.swap_nights(et, t[i]) * (ts.SWAP_LONG if cur == 1 else ts.SWAP_SHORT)
        rows.append((et, t[i], cur, ent, p / ent * 100))
        cur = 0

    for i in range(1, len(d)):
        w = want[i - 1]
        if w != cur:                                  # 봉 시가에 청산/진입
            if cur:
                close_all(i, o[i] - ts.SLIP if cur == 1 else o[i] + sp[i] + ts.SLIP)
            if w:
                cur = int(w)
                ent = o[i] + sp[i] + ts.SLIP if cur == 1 else o[i] - ts.SLIP
                et, left, pnl, hit = t[i], 1.0, 0.0, 0
                sl = stop_fn(i - 1, cur, ent) if stop_fn else np.nan
        if not cur:
            continue
        bid_lo, bid_hi = l[i], h[i]
        ask_lo, ask_hi = l[i] + sp[i], h[i] + sp[i]
        if sl == sl and ((cur == 1 and bid_lo <= sl) or (cur == -1 and ask_hi >= sl)):
            op = o[i] if cur == 1 else o[i] + sp[i]
            close_all(i, (min(sl, op) - ts.SLIP) if cur == 1 else (max(sl, op) + ts.SLIP))
            continue
        if tps:
            lv = tps[0] if cur == 1 else tps[1]
            while hit < len(lv) and left > 1e-9:
                px = ent * (1 + cur * lv[hit] / 100)
                if (cur == 1 and bid_hi >= px) or (cur == -1 and ask_lo <= px):
                    pnl += (px - ent) * cur * TP_PART
                    left -= TP_PART
                    hit += 1
                else:
                    break
    return pd.DataFrame(rows, columns=["open", "close", "side", "entry", "pct"])


def st_mult(d, d1):
    a = daily_trend_on(d, d1)
    b = ts.supertrend_dir(d, 10, 4.0)
    return np.where((a == 1) & (b == 1), 1, np.where((a == -1) & (b == -1), -1, 0)).astype(float)


def bb_break(d, min_w):
    c = d.close
    mid = c.rolling(15).mean()
    dev = np.maximum(2 * c.rolling(15).std(ddof=1).values, mid.values * min_w / 100)   # ta.stdev는 모표준편차지만 차이 작음
    up, lo = mid.values + dev, mid.values - dev
    cu, cd = ts.cross_up(c.values, up), ts.cross_dn(c.values, lo)
    return ts.hold(cu, cd)                             # 위 돌파 매수 / 아래 이탈 매도 (서로 청산 겸 갈아타기)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = ts.load_m1("XAUUSD")
    d1 = ts.bars(m1, 1440)
    out = []
    for tf, mins in {"M15": 15, "H1": 60, "H4": 240}.items():
        d = ts.bars(m1, mins)
        a14 = atr(d).values
        c = d.close.values
        stop = lambda i, s, e: max(e * 0.93, e - 2 * a14[i]) if s == 1 else min(e * 1.07, e + 2 * a14[i])
        cases = {
            "ST Mult TP (원본)": run(d, st_mult(d, d1), tps=(LONG_TP, SHORT_TP)),
            "ST Mult TP (익절 없이)": run(d, st_mult(d, d1)),
            "볼밴돌파 최소폭3% (원본)": run(d, bb_break(d, 3.0), stop),
            "볼밴돌파 최소폭 0%": run(d, bb_break(d, 0.0), stop),
            "볼밴돌파 최소폭 0%, 손절 없이": run(d, bb_break(d, 0.0)),
        }
        for name, tr in cases.items():
            if len(tr) < 5:
                out.append({"봉": tf, "전략": name, "거래": len(tr)})
                continue
            out.append({"봉": tf, "전략": name, **ts.summary(tr)})
        print(f"{tf} 끝", flush=True)
    res = pd.DataFrame(out)
    res.to_csv("results/tv_bbst.csv", index=False, encoding="utf-8-sig")
    print(res.drop(columns=["평균%"], errors="ignore").to_string(index=False))


if __name__ == "__main__":
    main()
