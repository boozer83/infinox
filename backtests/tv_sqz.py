"""Squeeze Momentum [LazyBear] 전략판 (사용자가 받은 Pine strategy) 을 금에 (2022-01 ~ 2026-10)

원본 규칙 그대로:
  BB(20, 표준편차 x 1.5 - 원본 LazyBear도 BB에 multKC를 씀), KC(SMA20 +- SMA(TR,20) x 1.5)
  스퀴즈 풀림 = BB가 KC 밖 (lowerBB < lowerKC and upperBB > upperKC)
  val = linreg(close - avg(avg(20봉 최고, 20봉 최저), SMA20), 20, 0)
  진입: 스퀴즈 풀림이 막 시작된 봉 (0 -> 1) 이고 val > 0 -> 매수 (원본은 매수만)
  청산: val 부호가 바뀐 봉
  원본은 봉 종가 체결(process_orders_on_close). 여기서는 다음 봉 시가 + 실제 스프레드 (tv_single과 같은 비용)
추가: 매도도 하는 대칭판 (val < 0 이면 매도, 부호 바뀌면 청산)
비교: 같은 거래 수·같은 보유 봉 수로 무작위 시각에 매수 (200번) -> 금 상승장 덕분인지 확인

사용법: python -m backtests.tv_sqz
"""
import numpy as np
import pandas as pd

import backtests.tv_single as ts

TFS = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}
N, MULT_KC = 20, 1.5
RANDOM_RUNS = 200


def squeeze(d):
    c = d.close
    basis = c.rolling(N).mean()
    dev = MULT_KC * c.rolling(N).std(ddof=0)
    tr = pd.concat([d.high - d.low, (d.high - c.shift()).abs(), (d.low - c.shift()).abs()], axis=1).max(axis=1)
    rng = tr.rolling(N).mean()
    sqz_off = ((basis - dev) < (basis - rng * MULT_KC)) & ((basis + dev) > (basis + rng * MULT_KC))
    val = ts.squeeze_mom(d, N).values
    return sqz_off.values, val


def signals(d, both):
    off, val = squeeze(d)
    start = off & ~np.r_[False, off[:-1]]
    pos_v = np.nan_to_num(val) > 0
    flip = np.r_[False, pos_v[1:] != pos_v[:-1]]
    long_on = start & pos_v
    short_on = start & ~pos_v & (np.nan_to_num(val) < 0) if both else np.zeros(len(d), bool)
    # 진입 봉에서는 청산 확인 안 함 (원본: entry 다음 close 순서지만 같은 봉 부호 전환이면 바로 청산됨 -> 그대로 반영)
    want = np.zeros(len(d))
    p = 0
    for i in range(len(d)):
        if long_on[i]:
            p = 1
        elif short_on[i]:
            p = -1
        if flip[i]:
            p = 0
        want[i] = p
    return want


def random_baseline(d, tr, rng):
    """같은 보유 봉 수의 매수를 무작위 시각에 -> PF 분포"""
    t = d.time.values
    o, sp = d.open.values, d.sp.values
    k0 = np.searchsorted(t, tr.open.values)
    k1 = np.searchsorted(t, tr.close.values)
    hold = k1 - k0
    pfs = []
    for _ in range(RANDOM_RUNS):
        s = rng.integers(0, len(d) - hold.max() - 1, len(hold))
        e = s + hold
        ent = o[s] + sp[s] + ts.SLIP
        ex = o[e] - ts.SLIP
        days = (t[e] - t[s]).astype("timedelta64[D]").astype(int)
        pct = (ex - ent + ts.SWAP_LONG * days * 5 / 7) / ent * 100     # 스왑은 대략 (평일 밤 비율)
        pfs.append(ts.pf(pd.Series(pct)))
    return np.percentile(pfs, [50, 95])


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = ts.load_m1("XAUUSD")
    rng = np.random.default_rng(1)
    out = []
    for tf, mins in TFS.items():
        d = ts.bars(m1, mins)
        for both in (False, True):
            tr = ts.trades(d, signals(d, both))
            if len(tr) < 10:
                continue
            r = ts.summary(tr)
            row = {"봉": tf, "방식": "매수+매도" if both else "매수만 (원본)", **r}
            if not both:
                med, p95 = random_baseline(d, tr[tr.side == 1], rng)
                row["무작위매수 PF 중간"] = round(med, 2)
                row["무작위 상위5%"] = round(p95, 2)
            out.append(row)
        print(f"{tf} 끝", flush=True)
    res = pd.DataFrame(out)
    res.to_csv("results/tv_sqz.csv", index=False, encoding="utf-8-sig")
    print(res.drop(columns=["평균%"]).to_string(index=False))


if __name__ == "__main__":
    main()
