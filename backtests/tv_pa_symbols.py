"""PA Master Grade-A 를 다른 종목에도 (2022-01 ~ 2026-10, 1시간봉, 원본 기본 설정)

종목: 금, XAUEUR, 은, 나스닥, 독일, 일본, 비트코인, 이더리움, BNB, 솔라나
봉·스프레드: tv_symbols와 같음 (틱으로 만든 1분봉, 종목별 시간대 스프레드 중앙값, 미끄러짐 = 스프레드 10%)
거래량: 종목별 1분 틱 수 (tickvol_m1). 비교로 거래량 조건을 끈 판도
스왑: 금만 (다른 종목은 0이라 조금 좋게 나옴, 몇 시간 보유라 영향은 작음)

사용법: python -m backtests.tv_pa_symbols
"""
import pandas as pd

import backtests.tickvol_m1 as tv
import backtests.tv_single as ts
from backtests.tv_batch2 import bars_vol, pa_master, rr_trades
from backtests.tv_symbols import GOLD_SWAP, load_sym, spread_table

SYMS = ["XAUUSD", "XAUEUR", "XAGUSD", "NAS100", "GER40", "JPN225", "BTCUSD", "XETUSD", "BNBUSD", "SOLUSD"]


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    sp_tab = spread_table()
    out = []
    for s in SYMS:
        m1 = load_sym(s, sp_tab)
        ts.SLIP = 0.02 if s == "XAUUSD" else 0.1 * m1.sp.median()
        ts.SWAP_LONG, ts.SWAP_SHORT = GOLD_SWAP if s == "XAUUSD" else (0.0, 0.0)
        d = bars_vol(m1, tv.load(s), 60)
        for use_vol in (True, False):
            side, stop, _, _ = pa_master(d, use_vol=use_vol)
            r = {"종목": s, "거래량 조건": "켬" if use_vol else "끔"}
            for rr in (1.0, 2.0, 3.0):
                tr = rr_trades(d, side, stop, rr)
                r[f"PF {rr:.0f}R"] = round(ts.pf(tr.pct), 2) if len(tr) else None
                if rr == 2.0:
                    sm = ts.summary(tr)
                    r.update({"거래": sm["거래"], "매수PF": sm["매수PF"], "매도PF": sm["매도PF"],
                              **{y: sm[y] for y in ["2022", "2023", "2024", "2025", "2026"]}})
            out.append(r)
        print(f"{s} 끝", flush=True)
    res = pd.DataFrame(out)
    res.to_csv("results/tv_pa_symbols.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
