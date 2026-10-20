"""길게 가는 추세에서만 들어가서 같은 방향으로 계속 더 쌓기 (0.01 lot씩, 1분봉, 2022-01 ~ 현재)

진입 신호: SAR 단타와 같음 (5분봉 SAR 전환 + RSI 50 + H1 SAR 같은 방향, ATR 가격의 0.0625%(4,000$에서 2.5$) 미만 / 20일 중앙값 2배 초과면 쉼,
          H1 전환 3번이면 쉼, 금요일 22:30 이후 쉼). 미국장(15~21시) 쉬기는 있음/없음 둘 다
추세 필터 ("길게 먹을 수 있는 곳"):
  없음   : 위 신호 그대로
  H4 SAR : 4시간봉 SAR도 같은 방향
  ADX    : 1시간봉 ADX(14) > 25 이고 DI 방향이 같음
  돌파   : 5분봉 종가가 직전 24봉(2시간) 고점 위 / 저점 아래
쌓기: 포지션이 있어도 같은 방향 새 신호면 하나 더 (최대 CAP개). 반대 신호는 무시
청산: 묶음 전체를 추세 끝 신호에서 한꺼번에 (H1 SAR 반전 / 15분봉 SAR 반전, 완성 봉 기준 다음 1분봉 시가)
      각 포지션 손절 SL_USD (1분봉 고저로 확인), 금요일 23:30 전부 청산
가격: 1분봉 Bid + 스프레드 (매수 진입·매도 청산은 Ask)

사용법: python -m backtests.trend_pyramid
"""
import numpy as np
import pandas as pd

from backtests.add_positions import ATR_REF_BARS, ATR_REF_MIN
from backtests.gold_rest_others import (FRIDAY_CLOSE, FRIDAY_NO_ENTRY, MAX_H1_FLIPS, MIN_ATR, NO_ENTRY_FROM,
                                        NO_ENTRY_UNTIL, h1_flips_by_time, load_m1, resample)
from indicators import atr, psar, rsi

CAP, SL_USD, LOT = 5, 20.0, 0.01
MAX_ATR_MULT = 2.0


def adx(df, n=14):
    up, dn = df.high.diff(), -df.low.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0))
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0))
    a = atr(df, n).values
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean().values / a
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean().values / a
    dx = 100 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-9)
    return pd.Series(dx).ewm(alpha=1 / n, adjust=False).mean().values, np.where(pdi > mdi, 1, -1)


def on_m1(m1_time, bars, minutes, values):
    """완성된 봉 값을 1분봉 시각에 붙임 (봉 끝 <= 시각)"""
    end = bars.time + pd.Timedelta(minutes=minutes)
    df = pd.DataFrame({"end": end, **values})
    return pd.merge_asof(pd.DataFrame({"t": m1_time}), df, left_on="t", right_on="end", direction="backward")


def build():
    m1 = load_m1("XAUUSD")
    m5, m15, h1, h4 = (resample(m1, n) for n in (5, 15, 60, 240))
    t = m1.time
    up5 = psar(m5.high.values, m5.low.values)
    flip = np.zeros(len(m5), bool)
    flip[1:] = up5[1:] != up5[:-1]
    r5 = rsi(m5.close).values
    a5 = atr(m5).values
    ref = pd.Series(a5).rolling(ATR_REF_BARS, min_periods=ATR_REF_MIN).median().values
    side5 = np.where(up5, 1, -1)
    hi24 = m5.high.shift(1).rolling(24).max().values
    lo24 = m5.low.shift(1).rolling(24).min().values
    brk = np.where(side5 == 1, m5.close.values > hi24, m5.close.values < lo24)
    sig5 = pd.DataFrame({"flip": flip, "side": side5, "rsi_ok": (r5 - 50) * side5 > 0, "atr": a5, "ref": ref,
                         "brk": brk})
    x = on_m1(t, m5, 5, {k: sig5[k].values for k in sig5})
    # 5분봉이 끝난 바로 그 1분봉에서만 신호
    x["new"] = x.end.values == t.values
    h1dir = on_m1(t, h1, 60, {"d": np.where(psar(h1.high.values, h1.low.values), 1, -1)}).d.values
    h4dir = on_m1(t, h4, 240, {"d": np.where(psar(h4.high.values, h4.low.values), 1, -1)}).d.values
    m15dir = on_m1(t, m15, 15, {"d": np.where(psar(m15.high.values, m15.low.values), 1, -1)}).d.values
    ax, adir = adx(h1)
    ad = on_m1(t, h1, 60, {"adx": ax, "adir": adir})
    fl = h1_flips_by_time(h1)[["end", "flips", "day"]]
    f = pd.merge_asof(pd.DataFrame({"t": t}), fl, left_on="t", right_on="end", direction="backward")
    flips = np.where(f.day.values == t.dt.date.values, f.flips.fillna(0).values, 0)
    mins = (t.dt.hour * 60 + t.dt.minute).values
    wd = t.dt.weekday.values
    # ATR 하한 2.5$는 금값 4,000$ 기준이라 2022~2023(1,800~2,000$)엔 거의 다 막힘 -> 가격 비율로 (2.5 / 4000)
    base = (x.new & x.flip & x.rsi_ok & (x.side == h1dir) & (x.atr >= MIN_ATR / 4000 * m1.close.values)
            & ~((x.ref == x.ref) & (x.atr > MAX_ATR_MULT * x.ref)) & (flips < MAX_H1_FLIPS)
            & ~((wd == 4) & (mins >= FRIDAY_NO_ENTRY))).values
    us = (mins >= NO_ENTRY_FROM) & (mins < NO_ENTRY_UNTIL)
    filt = {"없음": np.ones(len(t), bool), "H4 SAR": x.side.values == h4dir,
            "ADX": (ad.adx.values > 25) & (ad.adir.values == x.side.values), "돌파": x.brk.fillna(False).values.astype(bool)}
    side = np.nan_to_num(x.side.values).astype(int)
    px = {"o": m1.open.values, "h": m1.high.values, "l": m1.low.values, "sp": m1.sp.values}
    fri = (wd == 4) & (mins >= FRIDAY_CLOSE)
    return t, px, side, base, us, filt, {"H1 SAR": h1dir, "M15 SAR": m15dir}, fri


def simulate(t, px, side, entry, exitdir, fri):
    o, h, l, sp = (px[k].tolist() for k in ("o", "h", "l", "sp"))
    ex, fr, sd = exitdir.tolist(), fri.tolist(), side.tolist()
    ent = np.flatnonzero(entry).tolist()
    tms = t.values.astype("datetime64[m]").astype(np.int64).tolist()
    out, pos, gs = [], [], 0          # pos: [진입가, 손절가, 진입 분]
    n, k, i = len(o), 0, 0
    while k < len(ent) or pos:
        if not pos:
            if k >= len(ent):
                break
            i = ent[k]
        # i분 처리: 먼저 청산 확인 (시가 기준 추세 끝 / 금요일), 그다음 손절, 그다음 진입
        if pos and (ex[i] != gs or fr[i] or (i > 0 and tms[i] - tms[i - 1] > 600)):
            xp = o[i] if gs == 1 else o[i] + sp[i]
            out += [(tms[p[2]], tms[i], gs, (xp - p[0]) * gs, "추세끝") for p in pos]
            pos = []
        if k < len(ent) and ent[k] == i:
            k += 1
            if sd[i] == ex[i] and (not pos or (sd[i] == gs and len(pos) < CAP)):
                gs = sd[i]
                ep = o[i] + sp[i] if gs == 1 else o[i]
                pos.append([ep, ep - gs * SL_USD, i])
        if pos:
            keep = []
            for p in pos:
                if (gs == 1 and l[i] <= p[1]) or (gs == -1 and h[i] + sp[i] >= p[1]):
                    out.append((tms[p[2]], tms[i], gs, -SL_USD, "손절"))
                else:
                    keep.append(p)
            pos = keep
            i += 1
            if i >= n:
                break
    return pd.DataFrame(out, columns=["open", "close", "side", "oz", "why"])


def summarize(tr, label):
    tr = tr.sort_values("close")
    pnl = tr.oz * LOT * 100
    eq = pnl.cumsum()
    dd = (eq.cummax().clip(lower=0) - eq).max()
    yr = pd.to_datetime(tr.close, unit="m").dt.year
    gl = -pnl[pnl < 0].sum()
    res = {"방식": label, "거래": len(tr), "승률%": round((pnl > 0).mean() * 100, 1),
           "PF": round(pnl[pnl > 0].sum() / gl, 2) if gl else float("inf"), "손익$": round(pnl.sum()),
           "최대낙폭$": round(dd), "평균이익$": round(pnl[pnl > 0].mean(), 1), "평균손실$": round(pnl[pnl < 0].mean(), 1),
           "보유(시간,중간)": round(((tr.close - tr.open) / 60).median(), 1)}
    for y, v in pnl.groupby(yr).sum().sort_index().items():
        res[str(y)] = round(v)
    return res


def main():
    pd.set_option("display.width", 320)
    pd.set_option("display.unicode.east_asian_width", True)
    t, px, side, base, us, filt, exits, fri = build()
    print("준비 완료", flush=True)
    rows = []
    for us_rest in (True, False):
        for fn, fm in filt.items():
            for xn, xd in exits.items():
                entry = base & fm & (~us if us_rest else True)
                label = f"{fn} / 청산 {xn}{' / 미국장 쉼' if us_rest else ''}"
                rows.append(summarize(simulate(t, px, side, entry, xd, fri), label))
                print(f"  {label} 완료", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("trend_pyramid_result.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
