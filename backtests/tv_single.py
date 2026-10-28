"""트레이딩뷰 지표를 하나씩만 써서 금을 거래하면 (지표 1개 = 전략 1개, 다른 필터·손절·익절 없음)

데이터: 금 1분봉 zip (2022-01 ~ 2026-10) -> 15분 / 1시간 / 4시간봉
체결: 봉 마감에 신호 -> 다음 봉 시가에 체결. 매수는 Ask(= 시가 + 그 시각 스프레드), 매도는 Bid
비용: 실제 1분봉 스프레드 + 체결마다 0.02$ 미끄러짐, 스왑(매수 -0.78 / 매도 +0.26 $/oz/밤, 수요일 3배, 지금 MT5 값)
설정: 전부 트레이딩뷰 기본값 (고르지 않음). 손익은 거래마다 진입가 대비 % (금값이 1,800 -> 4,000으로 변해서)

추세형: 지표가 가리키는 방향으로 항상 보유 (뒤집히면 반대로 갈아탐)
역추세형: 과매도에서 돌아오면 매수 / 과매수에서 내려오면 매도, 가운데(50 등)에 오면 청산

금이 이 기간 2배 넘게 올라서 매수만 하는 지표는 좋아 보일 수 있음 -> 매수·매도 PF를 따로, 연도별 PF를 같이 봄

사용법: python -m backtests.tv_single [지표이름 일부]   (없으면 전부)
"""
import sys

import numpy as np
import pandas as pd

from backtests.gold_rest_others import load_m1
from indicators import atr, ema, rsi

TFS = {"M15": 15, "H1": 60, "H4": 240}
SLIP = 0.02
SWAP_LONG, SWAP_SHORT = -0.7833, 0.2613          # $/oz/밤
YEARS = range(2022, 2027)


def bars(m1, minutes):
    g = m1.set_index("time").resample(f"{minutes}min", label="left", closed="left")
    b = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(),
                      "close": g.close.last(), "sp": g.sp.first()})
    return b.dropna().reset_index()


def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def sma(s, n):
    return s.rolling(n).mean()


def wma(s, n):
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: (x * w).sum() / w.sum(), raw=True)


def hold(long_on, short_on, exit_long=None, exit_short=None):
    """신호 배열 -> 봉마다 원하는 포지션(+1/-1/0). 신호가 없으면 직전 포지션 유지"""
    n = len(long_on)
    pos = np.zeros(n)
    p = 0
    for i in range(n):
        if long_on[i]:
            p = 1
        elif short_on[i]:
            p = -1
        elif p == 1 and exit_long is not None and exit_long[i]:
            p = 0
        elif p == -1 and exit_short is not None and exit_short[i]:
            p = 0
        pos[i] = p
    return pos


def side(x):
    """x > 0 이면 매수, < 0 이면 매도 (항상 보유)"""
    return np.sign(np.nan_to_num(np.asarray(x, float)))


def cross_up(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.r_[False, (a[1:] > b[1:]) & (a[:-1] <= b[:-1])]


def cross_dn(a, b):
    return cross_up(-np.asarray(a, float), -np.asarray(b, float))


def osc_revert(x, lo, hi, mid):
    """lo 아래에서 위로 올라오면 매수, hi 위에서 내려오면 매도, mid 닿으면 청산"""
    x = np.asarray(x, float)
    return hold(cross_up(x, np.full_like(x, lo)), cross_dn(x, np.full_like(x, hi)), x >= mid, x <= mid)


# ---------- 지표 (트레이딩뷰 기본값) ----------

def supertrend_dir(d, n=10, mult=3.0):
    a = atr(d, n).values
    src = ((d.high + d.low) / 2).values
    c = d.close.values
    up, dn = src - mult * a, src + mult * a
    tr = np.ones(len(c))
    for i in range(1, len(c)):
        if c[i - 1] > up[i - 1]:
            up[i] = max(up[i], up[i - 1])
        if c[i - 1] < dn[i - 1]:
            dn[i] = min(dn[i], dn[i - 1])
        if tr[i - 1] == -1 and c[i] > dn[i - 1]:
            tr[i] = 1
        elif tr[i - 1] == 1 and c[i] < up[i - 1]:
            tr[i] = -1
        else:
            tr[i] = tr[i - 1]
    return tr


def psar_dir(d, step=0.02, mx=0.2):
    from indicators import psar
    return np.where(psar(d.high.values, d.low.values, step, mx), 1.0, -1.0)


def ut_bot_dir(d, key=1.0, n=10):
    """UT Bot Alerts: 종가 기준 ATR 트레일링 선, 종가가 선 위면 매수"""
    a = atr(d, n).values * key
    c = d.close.values
    stop = np.zeros(len(c))
    for i in range(1, len(c)):
        prev = stop[i - 1]
        if c[i] > prev and c[i - 1] > prev:
            stop[i] = max(prev, c[i] - a[i])
        elif c[i] < prev and c[i - 1] < prev:
            stop[i] = min(prev, c[i] + a[i])
        else:
            stop[i] = c[i] - a[i] if c[i] > prev else c[i] + a[i]
    return np.where(c > stop, 1.0, -1.0)


def chandelier_dir(d, n=22, mult=3.0):
    """Chandelier Exit (everget, 종가 사용): 최고 종가 - 3 ATR / 최저 종가 + 3 ATR"""
    a = atr(d, n).values * mult
    c = d.close.values
    hh, ll = d.close.rolling(n).max().values, d.close.rolling(n).min().values
    tr = np.ones(len(c))
    ls_p, ss_p = hh[0] - a[0], ll[0] + a[0]
    for i in range(1, len(c)):
        ls, ss = hh[i] - a[i], ll[i] + a[i]
        if c[i - 1] > ls_p:
            ls = max(ls, ls_p) if ls_p == ls_p else ls
        if c[i - 1] < ss_p:
            ss = min(ss, ss_p) if ss_p == ss_p else ss
        tr[i] = 1 if c[i] > ss_p else (-1 if c[i] < ls_p else tr[i - 1])
        ls_p, ss_p = ls, ss
    return tr


def dmi(d, n=14):
    h, l, c = d.high, d.low, d.close
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0))
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0))
    a = atr(d, n).reset_index(drop=True)
    return 100 * rma(pdm, n) / a, 100 * rma(ndm, n) / a


def aroon(d, n=14):
    up = d.high.rolling(n + 1).apply(lambda x: x.argmax(), raw=True) / n * 100
    dn = d.low.rolling(n + 1).apply(lambda x: x.argmin(), raw=True) / n * 100
    return up, dn


def heikin_dir(d):
    hc = (d.open + d.high + d.low + d.close) / 4
    ho = np.zeros(len(d))
    o, c = d.open.values, d.close.values
    ho[0] = (o[0] + c[0]) / 2
    hcv = hc.values
    for i in range(1, len(d)):
        ho[i] = (ho[i - 1] + hcv[i - 1]) / 2
    return np.sign(hcv - ho)


def vortex(d, n=14):
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    vp = (d.high - d.low.shift()).abs().rolling(n).sum() / tr.rolling(n).sum()
    vm = (d.low - d.high.shift()).abs().rolling(n).sum() / tr.rolling(n).sum()
    return vp - vm


def squeeze_mom(d, n=20):
    """LazyBear Squeeze Momentum의 막대 값 (선형회귀)"""
    c = d.close
    mid = ((d.high.rolling(n).max() + d.low.rolling(n).min()) / 2 + sma(c, n)) / 2
    x = c - mid
    t = np.arange(n)
    tm = t.mean()
    den = ((t - tm) ** 2).sum()
    return x.rolling(n).apply(lambda y: y.mean() + ((t - tm) * (y - y.mean())).sum() / den * (n - 1 - tm), raw=True)


def wavetrend(d, n1=10, n2=21):
    ap = (d.high + d.low + d.close) / 3
    esa = ema(ap, n1)
    dd = ema((ap - esa).abs(), n1)
    ci = (ap - esa) / (0.015 * dd)
    wt1 = ema(ci, n2)
    return wt1, sma(wt1, 4)


def cci(d, n=20):
    tp = (d.high + d.low + d.close) / 3
    m = sma(tp, n)
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - m) / (0.015 * md)


def stoch_k(d, k=14, smooth=3):
    ll, hh = d.low.rolling(k).min(), d.high.rolling(k).max()
    return sma(100 * (d.close - ll) / (hh - ll), smooth)


def hull(s, n=9):
    return wma(2 * wma(s, n // 2) - wma(s, n), int(np.sqrt(n)))


def ichimoku_tk(d):
    tk = (d.high.rolling(9).max() + d.low.rolling(9).min()) / 2
    kj = (d.high.rolling(26).max() + d.low.rolling(26).min()) / 2
    return tk - kj


def breakout(d, upper, lower):
    c = d.close.values
    return hold(c > np.asarray(upper, float), c < np.asarray(lower, float))


def band_revert(d, upper, lower, mid):
    """밴드 밖으로 나갔다가 안으로 돌아오면 반대로, 가운데선에서 청산"""
    c = d.close.values
    u, l, m = (np.asarray(x, float) for x in (upper, lower, mid))
    return hold(cross_up(c, l), cross_dn(c, u), c >= m, c <= m)


def macd(c, f=12, s=26, sg=9):
    m = ema(c, f) - ema(c, s)
    return m, ema(m, sg)


def bb(c, n=20, k=2.0):
    m = sma(c, n)
    sd = c.rolling(n).std(ddof=0)
    return m + k * sd, m - k * sd, m


def keltner(d, n=20, k=2.0, an=10):
    m = ema(d.close, n)
    a = atr(d, an)
    return m + k * a, m - k * a, m


INDICATORS = {
    # 추세형 (항상 보유)
    "EMA 9/21 교차": ("추세", lambda d: side(ema(d.close, 9) - ema(d.close, 21))),
    "종가 vs EMA50": ("추세", lambda d: side(d.close - ema(d.close, 50))),
    "종가 vs SMA200": ("추세", lambda d: side(d.close - sma(d.close, 200))),
    "MACD 12/26/9 교차": ("추세", lambda d: side(np.subtract(*macd(d.close)))),
    "MACD 0선": ("추세", lambda d: side(macd(d.close)[0])),
    "SuperTrend 10/3": ("추세", supertrend_dir),
    "Parabolic SAR": ("추세", psar_dir),
    "UT Bot 1/10": ("추세", ut_bot_dir),
    "Chandelier 22/3": ("추세", chandelier_dir),
    "Donchian 20 돌파": ("추세", lambda d: breakout(d, d.high.rolling(20).max().shift(), d.low.rolling(20).min().shift())),
    "Bollinger 20/2 돌파": ("추세", lambda d: breakout(d, *bb(d.close)[:2])),
    "Keltner 20/2 돌파": ("추세", lambda d: breakout(d, *keltner(d)[:2])),
    "DMI 14 (+DI vs -DI)": ("추세", lambda d: side(np.subtract(*dmi(d)))),
    "Aroon 14": ("추세", lambda d: side(np.subtract(*aroon(d)))),
    "Hull MA 9 기울기": ("추세", lambda d: side(hull(d.close).diff())),
    "Heikin Ashi 색": ("추세", heikin_dir),
    "Vortex 14": ("추세", lambda d: side(vortex(d))),
    "TRIX 18": ("추세", lambda d: side(ema(ema(ema(np.log(d.close), 18), 18), 18).diff())),
    "Awesome Osc": ("추세", lambda d: side(sma((d.high + d.low) / 2, 5) - sma((d.high + d.low) / 2, 34))),
    "Momentum 10": ("추세", lambda d: side(d.close.diff(10))),
    "Squeeze Mom (LazyBear)": ("추세", lambda d: side(squeeze_mom(d))),
    "WaveTrend 교차": ("추세", lambda d: side(np.subtract(*wavetrend(d)))),
    "Ichimoku 전환/기준": ("추세", lambda d: side(ichimoku_tk(d))),
    "RSI 14 50선": ("추세", lambda d: side(rsi(d.close) - 50)),
    "CCI 20 0선": ("추세", lambda d: side(cci(d))),
    # 역추세형 (과매수·과매도에서 반대로, 가운데에서 청산)
    "RSI 14 30/70 역": ("역추세", lambda d: osc_revert(rsi(d.close), 30, 70, 50)),
    "RSI 2 10/90 역": ("역추세", lambda d: osc_revert(rsi(d.close, 2), 10, 90, 50)),
    "Stoch 14 20/80 역": ("역추세", lambda d: osc_revert(stoch_k(d), 20, 80, 50)),
    "Williams %R 14 역": ("역추세", lambda d: osc_revert(stoch_k(d, 14, 1), 20, 80, 50)),
    "CCI 20 ±100 역": ("역추세", lambda d: osc_revert(cci(d), -100, 100, 0)),
    "WaveTrend ±53 역": ("역추세", lambda d: osc_revert(wavetrend(d)[0], -53, 53, 0)),
    "Bollinger 20/2 역": ("역추세", lambda d: band_revert(d, *bb(d.close))),
    "Keltner 20/2 역": ("역추세", lambda d: band_revert(d, *keltner(d))),
}


# ---------- 체결 ----------

def swap_nights(t0, t1):
    """t0~t1 사이 넘긴 자정 수 (월~금 밤, 수요일 3배)"""
    d0 = t0.astype("datetime64[D]")
    d1 = t1.astype("datetime64[D]")
    days = np.arange(d0, d1)
    wd = (days.astype(np.int64) + 3) % 7          # 0=월
    return int(np.where(wd == 2, 3, np.where(wd < 5, 1, 0)).sum())


def trades(d, want):
    """봉 i 마감의 원하는 포지션 -> 봉 i+1 시가 체결. 거래 목록"""
    want = np.nan_to_num(np.asarray(want, float))
    o, sp, t = d.open.values, d.sp.values, d.time.values
    rows = []
    cur, ent, et = 0, 0.0, None
    for i in range(len(want) - 1):
        w = want[i]
        if w == cur:
            continue
        k = i + 1
        if cur != 0:                  # 청산: 매수 청산은 Bid, 매도 청산은 Ask
            px = o[k] - SLIP if cur == 1 else o[k] + sp[k] + SLIP
            pnl = (px - ent) * cur
            n = swap_nights(et, t[k])
            pnl += n * (SWAP_LONG if cur == 1 else SWAP_SHORT)
            rows.append((et, t[k], cur, ent, pnl / ent * 100))
        if w != 0:
            ent = o[k] + sp[k] + SLIP if w == 1 else o[k] - SLIP
            et = t[k]
        cur = int(w)
    return pd.DataFrame(rows, columns=["open", "close", "side", "entry", "pct"])


def pf(x):
    w, l = x[x > 0].sum(), -x[x < 0].sum()
    return w / l if l > 0 else np.nan


def summary(tr):
    p = tr.pct
    eq = p.cumsum()
    r = {"거래": len(tr), "승률%": round((p > 0).mean() * 100, 1), "PF": round(pf(p), 2),
         "합계%": round(p.sum(), 1), "평균%": round(p.mean(), 3), "최대낙폭%": round((eq.cummax() - eq).max(), 1),
         "매수PF": round(pf(p[tr.side == 1]), 2), "매도PF": round(pf(p[tr.side == -1]), 2)}
    yr = pd.to_datetime(tr.close).dt.year
    for y in YEARS:
        r[str(y)] = round(pf(p[yr == y]), 2)
    return r


def main():
    pick = sys.argv[1] if len(sys.argv) > 1 else ""
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = load_m1("XAUUSD")
    rows = []
    for tf, mins in TFS.items():
        d = bars(m1, mins)
        bh = (d.close.iat[-1] / d.close.iat[0] - 1) * 100
        print(f"{tf}: 봉 {len(d):,}개, 그 기간 금 매수 보유 +{bh:.0f}%", flush=True)
        for name, (kind, fn) in INDICATORS.items():
            if pick and pick not in name:
                continue
            tr = trades(d, fn(d))
            if len(tr) < 10:
                continue
            rows.append({"지표": name, "종류": kind, "봉": tf, **summary(tr)})
    res = pd.DataFrame(rows)
    res.to_csv("results/tv_single.csv", index=False, encoding="utf-8-sig")
    yc = [str(y) for y in YEARS]
    res["이긴 해"] = (res[yc] > 1).sum(axis=1)
    print(res.sort_values(["봉", "PF"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
