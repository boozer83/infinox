"""지표 계산 (MT5 기본 지표와 같은 방식)"""
import numpy as np
import pandas as pd


def psar(h, l, step=0.02, mx=0.2):
    """파라볼릭 SAR. 각 봉에서 상승 추세(True) / 하락 추세(False) 배열을 반환"""
    return psar_levels(h, l, step, mx)[0]


def psar_levels(h, l, step=0.02, mx=0.2):
    """파라볼릭 SAR. (추세 배열, 각 봉 끝의 SAR 점, 그 봉에서 뚫어야 했던 SAR 값)"""
    n = len(h)
    up = np.zeros(n, bool)
    sar = np.zeros(n)
    trig = np.full(n, np.nan)
    up[0], sar[0], ep, af = True, l[0], h[0], step
    for i in range(1, n):
        s = sar[i-1] + af * (ep - sar[i-1])
        if up[i-1]:
            s = min(s, l[i-1], l[i-2] if i > 1 else l[i-1])
            trig[i] = s
            if l[i] < s:
                up[i], s, ep, af = False, ep, l[i], step
            else:
                up[i] = True
                if h[i] > ep:
                    ep, af = h[i], min(af + step, mx)
        else:
            s = max(s, h[i-1], h[i-2] if i > 1 else h[i-1])
            trig[i] = s
            if h[i] > s:
                up[i], s, ep, af = True, ep, h[i], step
            else:
                up[i] = False
                if l[i] < ep:
                    ep, af = l[i], min(af + step, mx)
        sar[i] = s
    return up, sar, trig


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l)


def atr(df, n=14):
    """ATR (Wilder 방식)"""
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def adx(df, n=14):
    h, l, c = df.high, df.low, df.close
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr
    ndi = 100 * pd.Series(ndm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    return dx.ewm(alpha=1/n, adjust=False).mean()
