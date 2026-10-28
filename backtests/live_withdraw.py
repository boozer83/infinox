"""실계좌 봇 전체를 한 계좌로 합쳐서: 계속 두기 vs 1만$ 넘으면 빼서 연 3% 적금 (2026-10-27 사용자 요청)

실계좌에서 도는 봇 (scripts/bots.ps1 live): SAR 단타, H4 추세 (금 + XETUSD), 아시아 돌파 (금 + XAUEUR 각 0.6%), NAS100 / GER40
각 봇 백테스트의 거래별 수익률(그 거래 직전 잔고 대비 %)을 청산 시각 순서로 한 계좌 잔고에 곱함 (lot이 잔고 비례라서 같은 근사)
  SAR 단타   : results/sar_us_now_YYYY.csv (지금 봇 그대로, backtests.sar_us_hours), 1,000$당 0.05 lot -> 1$/oz = 잔고 0.5%
  H4 추세    : backtests.all_symbols.trend_row (구름 필터 + 그림자), 거래당 2% 위험
  아시아     : backtests.asia_xaueur outcomes (results/asia_xaueur/outcomes.pkl), 금 + XAUEUR 마틴 따로, 각 0.6%
  지수       : results/index_ticks.csv, 0.25% 위험. NAS는 손절 = ATR. GER은 손절폭(범위 가운데까지)이 CSV에 없어서
               그 해 손절 거래 |pts| 중앙값으로 근사. 청산 시각은 NAS 서버 23:00 / GER 18:30으로 둠
근사: 동시 보유를 순서대로 곱함, 최소 lot·지수 최소 lot 위험 건너뛰기 무시, 수수료·스왑 없음
빼기 규칙은 매달 말 잔고로 판단: 계속 두기 / 1만$ 넘는 만큼 전부 / 절반. 뺀 돈은 연 3% (월복리) 적금

사용법: python -m backtests.live_withdraw [시작 잔고=3333]   (거래 목록 results/live_withdraw_trades.csv 저장 후 다시 쓰려면 --cached, 최대 lot 반영은 --caps, 빼는 기준은 --cap=20000)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TRADES = RES / "live_withdraw_trades.csv"
CAP = 10_000.0
SAVE_RATE = 0.03
SAR_MAX_LOT = 1.0
CAPS = False


def sar():
    parts = []
    for y in range(2022, 2027):
        d = pd.read_csv(RES / f"sar_us_now_{y}.csv", parse_dates=["open"])
        parts.append(pd.DataFrame({"t": d.open + pd.to_timedelta(d.mins, unit="min"), "ret": d.oz * 0.05 / 1000 * 100}))
    return pd.concat(parts).assign(bot="SAR 단타")


def trend(sym):
    import backtests.all_symbols as al
    got = {}
    orig = al.yearly

    def grab(times, bal, bal0):
        got["t"], got["bal"] = pd.Series(times).reset_index(drop=True), np.asarray(bal)
        return orig(times, bal, bal0)

    al.yearly = grab
    try:
        al.trend_row(sym, al.m1_bars(sym))
    finally:
        al.yearly = orig
    bal = got["bal"]
    return pd.DataFrame({"t": got["t"], "ret": bal / np.r_[10_000.0, bal[:-1]] - 1, "bot": f"추세 {sym}"})


def asia():
    import backtests.asia_xaueur as ax
    d = ax.get_outcomes(False)
    b = ax.martin({"g": ax.pick(d, "XAUUSD"), "e": ax.pick(d, "XAUEUR")}, bal0=10_000.0, risk=0.006)
    return pd.DataFrame({"t": b.t, "ret": b.bal.values / np.r_[10_000.0, b.bal.values[:-1]] - 1, "bot": "아시아 금+XAUEUR"})


def index():
    d = pd.read_csv(RES / "index_ticks.csv", parse_dates=["날짜"])
    out = []
    for (sym, y), g in d.groupby(["종목", "y"]):
        if sym == "NAS100":
            risk, hour = g.atr, 23
        else:
            risk, hour = g.pts[g["손절"]].abs().median(), 18.5
        out.append(pd.DataFrame({"t": g["날짜"] + pd.Timedelta(hours=hour), "ret": 0.0025 * g.pts / risk, "bot": sym}))
    return pd.concat(out)


def trades(cached):
    if cached and TRADES.exists():
        return pd.read_csv(TRADES, parse_dates=["t"])
    parts = [sar(), asia(), index()]
    for sym in ("XAUUSD", "XETUSD"):
        parts.append(trend(sym))
        print("  추세", sym, "끝", flush=True)
    tr = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)
    tr = tr[tr.t < "2026-10-17"]
    tr.to_csv(TRADES, index=False, encoding="utf-8-sig")
    return tr


def run(tr, bal0, rule):
    """rule: 0 계속 두기 / 1 넘는 만큼 전부 / 0.5 절반. 월말마다 판단 -> (월별 표)
    CAPS면 봇의 최대 lot: SAR 단타 1.0 lot (demo_trader.MAX_LOT, 잔고 2만$부터 lot 고정) / 추세 5.0 lot은 이 범위에선 안 걸림"""
    bal, save, peak_total, mdd_usd, rows = bal0, 0.0, bal0, 0.0, []
    month = tr.t.dt.to_period("M")
    for m, g in tr.groupby(month, sort=True):
        for r, sar_ in zip(g.ret.values, (g.bot == "SAR 단타").values):
            if CAPS and sar_:
                r = r / 0.005 * min(bal / 1000 * 0.05, SAR_MAX_LOT) * 100 / bal     # $/oz x lot x 100 / 잔고
            bal *= 1 + r
            total = bal + save
            mdd_usd = max(mdd_usd, peak_total - total)
            peak_total = max(peak_total, total)
        save *= 1 + SAVE_RATE / 12
        out = max(0.0, bal - CAP) * rule
        bal -= out
        save += out
        rows.append({"월": str(m), "계좌": bal, "적금": save, "합계": bal + save})
    return pd.DataFrame(rows), mdd_usd


def summary(tr, bal0, start):
    t = tr[tr.t >= start]
    res = []
    for name, rule in (("계속 두기", 0.0), (f"{CAP / 10000:g}만$ 넘으면 전부 빼기", 1.0), (f"{CAP / 10000:g}만$ 넘으면 절반 빼기", 0.5)):
        m, mdd_usd = run(t, bal0, rule)
        tot = m.set_index("월").합계
        yr = {f"{y}말": round(tot[[k for k in tot.index if k.startswith(str(y))]].iat[-1]) for y in range(int(start[:4]), 2027)}
        worst = (m.합계 / m.합계.cummax() - 1).min()
        res.append({"방법": name, **yr, "계좌$": round(m.계좌.iat[-1]), "적금$": round(m.적금.iat[-1]),
                    "최종 합계$": round(m.합계.iat[-1]), "최대 손실$": round(mdd_usd), "월말 기준 최대 낙폭%": round(worst * 100, 1)})
    return pd.DataFrame(res)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bal0 = float(args[0]) if args else 3333.0
    tr = trades("--cached" in sys.argv)
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    y = tr.assign(y=tr.t.dt.year).groupby(["bot", "y"]).ret.agg(["size", lambda r: (np.prod(1 + r) - 1) * 100]).round(1)
    print(y.unstack().to_string())
    global CAPS, CAP
    CAPS = "--caps" in sys.argv
    CAP = next((float(a[6:]) for a in sys.argv if a.startswith("--cap=")), CAP)
    print("최대 lot 반영" if CAPS else "최대 lot 무시")
    out = []
    for start in ("2022-01-01", "2024-01-01", "2025-01-01"):
        s = summary(tr, bal0, start)
        print(f"\n=== {start[:4]}-01부터, 시작 {bal0:,.0f}$")
        print(s.to_string(index=False))
        out.append(s.assign(시작=start[:7]))
    pd.concat(out).to_csv(RES / f"live_withdraw{'_caps' if CAPS else ''}_{CAP:.0f}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
