"""금 마틴 전략이 쉬는 동안 은/비트코인을 같은 규칙으로 거래하면 수익이 나는가 (2022-01 ~ 현재, 1분봉 체결)

1. 금(XAUUSD): demo_trader.py 규칙을 그대로 시뮬레이션해서 시점마다 금 상태를 구함
   - 신호: 5분봉 SAR(0.02, 0.2) 전환 + RSI(14) 50 방향 + 마지막 완성 1시간봉 SAR 방향, SL 7.5 / TP 3 달러
   - 쉬기 (demo_trader.py와 같은 순서): 마틴 정지(5번째 진입 손절) / 금요일 22:30 이후 / 미국장 15~21시 /
     오늘 1시간봉 SAR 전환 3번 이상 / 5분봉 ATR 2.5달러 미만 / 최근 200건(실제+가상) 승률 71.4% 미만(가상 진입)
   - 금요일 23:30 보유 포지션 청산
2. 은(XAGUSD), 비트코인(BTCUSD): 같은 신호, 한 번에 1포지션. SL/TP는 금 기준(4,150달러에서 7.5 / 3)을 두 방식으로 옮김
     가격비례 : 진입가 x (7.5 / 4150), 진입가 x (3 / 4150)  (금과 같은 %)
     변동성비례: 가격비례 x (그 종목 5분봉 ATR% 중앙값 / 금 5분봉 ATR% 중앙값)  (금과 같은 '흔들림 대비' 거리)
   각 거래를 진입 시점의 금 상태로 분류:
     금 휴장 / 금 보유 중 / 금 대기(쉬기 아님, 신호 기다림) / 쉬기 이유별
3. 실제 운용 흉내: 금이 '쉬기 필터'로 쉬는 동안에만 은/비트코인 진입 (금 휴장, 금 보유, 금 대기 때는 진입 안 함)
4. 3에 일일 마틴 (demo_trader.py와 같은 단계: x2, 시작 lot의 5배까지, 5번째 진입 손절이면 그날 정지, 날짜 바뀌면 처음부터)
   시작 lot은 손절 1번에 그날 시작 잔고의 2.25% 손실 (금 1,000달러당 0.03 lot x 7.5달러와 같은 위험), 1,000달러 시작.
   lot 반올림/최소 lot은 무시. 비교용으로 금 자체에도 같은 마틴을 적용한 결과를 함께 표시

체결: 신호 봉이 끝난 다음 1분봉 시가 (매수 Ask = Bid + 스프레드, 매도 Bid). 매수 청산은 Bid, 매도 청산은 Ask.
      한 1분봉 안에서 SL과 TP가 둘 다 닿으면 SL이 먼저 (보수적). 갭으로 SL을 넘으면 그 봉 시가에 청산.
      스프레드는 1분봉 기록값(MT5 봉의 스프레드, 실제 평균보다 작을 수 있음). 수수료/스왑 없음.
      금/은은 금요일 23:30에 청산, 비트코인은 주말에도 보유.

손익은 '금 손절 1번 = -1R' 단위(진입가 대비 % / 금 SL %)로 표시해서 종목끼리 비교할 수 있게 함.
사용법: python -m backtests.gold_rest_others   (data/bars/<종목>_M1_<연도>.csv.zip 필요: backtests.download_bars)
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from indicators import atr, psar, rsi

BARS = Path(__file__).resolve().parent.parent / "data" / "bars"
POINT = {"XAUUSD": 0.01, "XAGUSD": 0.001, "BTCUSD": 0.01}
WEEKEND_CLOSE = {"XAUUSD": True, "XAGUSD": True, "BTCUSD": False}
SL_USD, TP_USD, REF_PRICE = 7.5, 3.0, 4150.0
SL_PCT, TP_PCT = SL_USD / REF_PRICE, TP_USD / REF_PRICE
MARTIN_MAX_LEVEL = 4
MARTIN_MULT, MAX_LOT_MULT = 2, 5
RISK_PCT = 7.5 * 0.03 * 100 / 1000 * 100   # 금 시작 lot(1,000달러당 0.03)으로 손절 1번 = 잔고의 2.25%
MARTIN_STARTS = (pd.Timestamp("2022-01-01"), pd.Timestamp("2025-01-01"))
NO_ENTRY_FROM, NO_ENTRY_UNTIL = 15 * 60, 21 * 60       # 분 (서버시간)
FRIDAY_NO_ENTRY, FRIDAY_CLOSE = 22 * 60 + 30, 23 * 60 + 30
MAX_H1_FLIPS = 3
MIN_ATR = 2.5
WINRATE_N, WINRATE_MIN = 200, 71.4
REST = ["마틴 정지", "금요일 마감", "미국장", "횡보", "ATR 낮음", "승률 쉬기"]
CATS = ["금 휴장", "금 보유 중", "금 대기"] + REST


def load_m1(symbol):
    parts = []
    for f in sorted(BARS.glob(f"{symbol}_M1_*.csv.zip")):
        with zipfile.ZipFile(f) as z:
            parts.append(pd.read_csv(z.open(z.namelist()[0])))
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df.time)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df["sp"] = df.spread * POINT[symbol]
    return df


def resample(m1, minutes):
    g = m1.set_index("time").resample(f"{minutes}min", label="left", closed="left")
    b = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last()})
    return b.dropna().reset_index()


def signal_table(m1):
    """신호 봉마다 (진입 시각 = 봉 끝, 방향, 그 봉 ATR). 1시간봉은 신호 봉 끝 이전에 완성된 것만 사용"""
    m5, h1 = resample(m1, 5), resample(m1, 60)
    up = psar(m5.high.values, m5.low.values)
    r = rsi(m5.close).values
    a = atr(m5).values
    end = m5.time + pd.Timedelta(minutes=5)
    h = pd.DataFrame({"end": h1.time + pd.Timedelta(hours=1),
                      "hdir": np.where(psar(h1.high.values, h1.low.values), 1, -1)})
    hdir = pd.merge_asof(pd.DataFrame({"end": end}), h, on="end", direction="backward").hdir.values
    rows = []
    for i in np.nonzero(up[1:] != up[:-1])[0] + 1:
        s = 1 if up[i] else -1
        if (r[i] - 50) * s > 0 and hdir[i] == s:
            rows.append((end.iat[i], s, a[i]))
    return pd.DataFrame(rows, columns=["t", "side", "atr"]), m5, h1


def h1_flips_by_time(h1):
    """시각 -> 그날(서버 날짜) 그 시각까지 완성된 1시간봉 SAR 전환 수 (demo_trader.h1_flips_today와 같은 계산)"""
    up = psar(h1.high.values, h1.low.values)
    flip = np.r_[False, up[1:] != up[:-1]]
    d = h1.time.dt.date
    return pd.DataFrame({"end": h1.time + pd.Timedelta(hours=1), "flips": pd.Series(flip).groupby(d.values).cumsum().values,
                         "day": d.values})


class Market:
    """1분봉 배열과 거래 체결"""

    def __init__(self, symbol, m1):
        self.symbol = symbol
        self.t = m1.time.values
        self.o, self.h, self.l = m1.open.values, m1.high.values, m1.low.values
        self.sp = m1.sp.values
        tt = m1.time
        mins = tt.dt.hour * 60 + tt.dt.minute
        # 금요일 23:30 이후 첫 봉 = 강제 청산 봉
        self.fri_close = ((tt.dt.weekday == 4) & (mins >= FRIDAY_CLOSE)).values if WEEKEND_CLOSE[symbol] else None

    def trade(self, t_entry, side, sl_pct, tp_pct):
        """(진입 idx, 청산 idx, 손익 %) 또는 None. 진입은 t_entry 이후 첫 1분봉 시가"""
        k = int(np.searchsorted(self.t, t_entry))
        if k >= len(self.t) or self.t[k] - t_entry > np.timedelta64(10, "m"):
            return None   # 장이 닫혀 있어서 바로 체결할 수 없음
        entry = self.o[k] + (self.sp[k] if side == 1 else 0.0)
        d_sl, d_tp = entry * sl_pct, entry * tp_pct
        n, w = len(self.t), 512
        j0 = k
        while j0 < n:
            j1 = min(n, j0 + w)
            sl_px, tp_px = (entry - d_sl, entry + d_tp) if side == 1 else (entry + d_sl, entry - d_tp)
            if side == 1:
                hit_sl = self.l[j0:j1] <= sl_px
                hit_tp = self.h[j0:j1] >= tp_px
            else:
                hit_sl = self.h[j0:j1] + self.sp[j0:j1] >= sl_px
                hit_tp = self.l[j0:j1] + self.sp[j0:j1] <= tp_px
            fri = self.fri_close[j0:j1] if self.fri_close is not None else np.zeros(j1 - j0, bool)
            any_ = hit_sl | hit_tp | fri
            if any_.any():
                i = int(np.argmax(any_))
                j = j0 + i
                if fri[i] and not (hit_sl[i] or hit_tp[i]):
                    px = self.o[j] + (0.0 if side == 1 else self.sp[j])
                elif hit_sl[i]:
                    op = self.o[j] + (0.0 if side == 1 else self.sp[j])
                    px = min(sl_px, op) if side == 1 else max(sl_px, op)   # 갭이면 시가
                else:
                    px = tp_px
                return k, j, (px - entry) * side / entry * 100
            j0, w = j1, w * 2
        return None


def run_gold():
    m1 = load_m1("XAUUSD")
    sig, m5, h1 = signal_table(m1)
    mk = Market("XAUUSD", m1)
    flips = h1_flips_by_time(h1)
    fl = pd.merge_asof(sig[["t"]], flips[["end", "flips", "day"]], left_on="t", right_on="end", direction="backward")
    sig["flips"] = np.where(fl.day.values == sig.t.dt.date.values, fl.flips.fillna(0).values, 0)

    trades, results = [], []
    busy_until = np.datetime64("1970-01-01")
    day, level, stopped, stop_at = None, 0, False, {}
    pause_changes = [(np.datetime64("1970-01-01"), False)]
    for t, side, a, nfl in sig[["t", "side", "atr", "flips"]].itertuples(index=False):
        t64 = np.datetime64(t)
        if t64 < busy_until:
            continue
        if t.date() != day:
            day, level, stopped = t.date(), 0, False
        mins = t.hour * 60 + t.minute
        if stopped or (t.weekday() == 4 and mins >= FRIDAY_NO_ENTRY) or NO_ENTRY_FROM <= mins < NO_ENTRY_UNTIL \
                or nfl >= MAX_H1_FLIPS or a < MIN_ATR:
            continue
        last = results[-WINRATE_N:]
        paused = len(last) >= WINRATE_N and np.mean(last) * 100 < WINRATE_MIN
        px = mk.o[min(np.searchsorted(mk.t, t64), len(mk.t) - 1)]
        res = mk.trade(t64, side, SL_USD / px, TP_USD / px)
        if res is None:
            continue
        k, j, pct = res
        win = pct > 0
        results.append(win)
        busy_until = mk.t[j]
        trades.append((mk.t[k], mk.t[j], side, pct, pct / (SL_USD / px * 100), paused))
        last = results[-WINRATE_N:]
        now_paused = len(last) >= WINRATE_N and np.mean(last) * 100 < WINRATE_MIN
        if now_paused != pause_changes[-1][1]:
            pause_changes.append((mk.t[j], now_paused))
        if not paused:      # 마틴 단계는 실제 거래만 (demo_trader.martin_state)
            if win:
                level = 0
            elif level >= MARTIN_MAX_LEVEL:
                stopped = True
                stop_at[day] = mk.t[j]
            else:
                level += 1
    tr = pd.DataFrame(trades, columns=["open", "close", "side", "pct", "r", "virtual"])
    return tr, m5, flips, pause_changes, stop_at


def gold_state(ts, gold_tr, gold_m5, flips, pause_changes, stop_at):
    """시각 배열 ts에서의 금 상태 (CATS 중 하나). demo_trader.py 신호 처리 순서와 같은 우선순위"""
    ts = pd.Series(pd.to_datetime(ts)).reset_index(drop=True)
    out = np.full(len(ts), "금 대기", dtype=object)
    t64 = ts.values
    mins = (ts.dt.hour * 60 + ts.dt.minute).values

    # 금 5분봉 ATR (그 시각에 끝난 5분봉) / 휴장 = 그 시각에 끝나는 금 5분봉이 없음
    g = pd.DataFrame({"end": gold_m5.time + pd.Timedelta(minutes=5), "atr": atr(gold_m5).values})
    m = pd.merge_asof(pd.DataFrame({"t": ts}), g, left_on="t", right_on="end", direction="backward")
    closed = (m.end.values != t64)
    fl = pd.merge_asof(pd.DataFrame({"t": ts}), flips, left_on="t", right_on="end", direction="backward")
    nfl = np.where(fl.day.values == ts.dt.date.values, fl.flips.fillna(0).values, 0)

    # 보유 구간 (실제 / 가상)
    def inside(df):
        o, c = df.open.values, df.close.values
        i = np.searchsorted(o, t64, side="right") - 1
        return (i >= 0) & (t64 < c[np.clip(i, 0, None)])
    real = inside(gold_tr[~gold_tr.virtual].reset_index(drop=True))
    virt = inside(gold_tr[gold_tr.virtual].reset_index(drop=True))
    pc_t = np.array([p[0] for p in pause_changes])
    pc_v = np.array([p[1] for p in pause_changes])
    paused = pc_v[np.searchsorted(pc_t, t64, side="right") - 1]
    stop_t = pd.to_datetime(ts.dt.date.map(stop_at)).fillna(pd.Timestamp("2100-01-01")).values
    stopped = t64 >= stop_t

    conds = [
        (closed, "금 휴장"), (real, "금 보유 중"), (virt, "승률 쉬기"), (stopped, "마틴 정지"),
        ((ts.dt.weekday.values == 4) & (mins >= FRIDAY_NO_ENTRY), "금요일 마감"),
        ((mins >= NO_ENTRY_FROM) & (mins < NO_ENTRY_UNTIL), "미국장"),
        (nfl >= MAX_H1_FLIPS, "횡보"), (m.atr.values < MIN_ATR, "ATR 낮음"), (paused, "승률 쉬기"),
    ]
    done = np.zeros(len(ts), bool)
    for c, name in conds:
        sel = c & ~done
        out[sel] = name
        done |= c
    return out


def simulate(mk, sig, sl_pct, tp_pct, allow=None):
    """한 번에 1포지션. allow가 있으면 그 신호만 진입"""
    rows, busy = [], np.datetime64("1970-01-01")
    for i, (t, side) in enumerate(sig[["t", "side"]].itertuples(index=False)):
        t64 = np.datetime64(t)
        if t64 < busy or (allow is not None and not allow[i]):
            continue
        res = mk.trade(t64, side, sl_pct, tp_pct)
        if res is None:
            continue
        k, j, pct = res
        busy = mk.t[j]
        rows.append((i, mk.t[k], mk.t[j], side, pct))
    return pd.DataFrame(rows, columns=["sig", "open", "close", "side", "pct"])


def simulate_martin(mk, sig, sl_pct, tp_pct, allow, start, start_bal=1000.0):
    """일일 마틴 (demo_trader.py 규칙): 손절마다 2배(시작 lot의 MAX_LOT_MULT배까지), 5번째 진입 손절이면 그날 정지,
    날짜(서버)가 바뀌면 처음부터. 시작 lot = 손절 1번에 그날 시작 잔고의 RISK_PCT% 손실 (금 0.03 lot / 1,000달러와 같은 위험).
    잔고 0 이하이면 깡통으로 중단. -> (거래표, 최대 낙폭 %, 깡통 시각)"""
    rows, busy = [], np.datetime64("1970-01-01")
    bal, peak, mdd, bust = start_bal, start_bal, 0.0, None
    day, level, stopped, day_bal = None, 0, False, bal
    for i, (t, side) in enumerate(sig[["t", "side"]].itertuples(index=False)):
        t64 = np.datetime64(t)
        if t < start or t64 < busy or not allow[i]:
            continue
        if t.date() != day:
            day, level, stopped, day_bal = t.date(), 0, False, bal
        if stopped:
            continue
        res = mk.trade(t64, side, sl_pct, tp_pct)
        if res is None:
            continue
        k, j, pct = res
        busy = mk.t[j]
        mult = min(MARTIN_MULT ** level, MAX_LOT_MULT)
        pnl = pct / (sl_pct * 100) * RISK_PCT / 100 * day_bal * mult
        bal += pnl
        rows.append((mk.t[k], mk.t[j], level, pct, pnl, bal))
        peak = max(peak, bal)
        mdd = max(mdd, (peak - bal) / peak * 100)
        if bal <= 0:
            bust = mk.t[j]
            break
        if pct > 0:
            level = 0
        elif level >= MARTIN_MAX_LEVEL:
            stopped = True
        else:
            level += 1
    return pd.DataFrame(rows, columns=["open", "close", "level", "pct", "pnl", "bal"]), mdd, bust


def gold_martin(real, start, start_bal=1000.0):
    """금 실제 거래(run_gold, 마틴 정지는 이미 반영)에 같은 마틴 lot을 적용한 잔고 (비교 기준)"""
    rows, bal, peak, mdd, bust = [], start_bal, start_bal, 0.0, None
    day, level, day_bal = None, 0, bal
    for o, c, r in real[["open", "close", "r"]].itertuples(index=False):
        if o < np.datetime64(start):
            continue
        if pd.Timestamp(o).date() != day:
            day, level, day_bal = pd.Timestamp(o).date(), 0, bal
        pnl = r * RISK_PCT / 100 * day_bal * min(MARTIN_MULT ** level, MAX_LOT_MULT)
        bal += pnl
        rows.append((o, c, level, r, pnl, bal))
        peak, mdd = max(peak, bal), max(mdd, (peak - bal) / peak * 100)
        if bal <= 0:
            bust = c
            break
        level = 0 if r > 0 else min(level + 1, MARTIN_MAX_LEVEL)
    return pd.DataFrame(rows, columns=["open", "close", "level", "pct", "pnl", "bal"]), mdd, bust


def martin_report(tr, mdd, bust, start_bal=1000.0):
    """연도별 수익률과 최종 잔고 한 줄"""
    if tr.empty:
        return "거래 없음"
    yr = pd.to_datetime(tr.close).dt.year
    parts, prev = [], start_bal
    for y, g in tr.groupby(yr):
        end = g.bal.iat[-1]
        parts.append(f"{y} {(end / prev - 1) * 100:+.0f}%")
        prev = end
    stops = int(((tr.level == MARTIN_MAX_LEVEL) & (tr.pct < 0)).sum())
    return (f"{' / '.join(parts)} | 최종 {tr.bal.iat[-1]:,.0f}$ | 최대 낙폭 {mdd:.0f}% | 마틴 정지 {stops}일"
            + (f" | 깡통 {pd.Timestamp(bust):%Y-%m-%d}" if bust is not None else ""))


def stats(pct):
    """pct: 손익 % -> 거래수, 승률, PF, 합계 R (금 손절 1번 = SL_PCT*100 %)"""
    if len(pct) == 0:
        return {"거래": 0, "승률%": np.nan, "PF": np.nan, "합계R": 0.0}
    gl = -pct[pct < 0].sum()
    return {"거래": len(pct), "승률%": round((pct > 0).mean() * 100, 1),
            "PF": round(pct[pct > 0].sum() / gl, 2) if gl else np.inf,
            "합계R": round(pct.sum() / (SL_PCT * 100), 1)}


def table(groups):
    return pd.DataFrame({k: stats(v) for k, v in groups}).T


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print("금 시뮬레이션 (demo_trader.py 쉬기 규칙)...", flush=True)
    gold_tr, gold_m5, flips, pause_changes, stop_at = run_gold()
    real = gold_tr[~gold_tr.virtual]
    gold_atrp = float(np.nanmedian(atr(gold_m5) / gold_m5.close))
    print(f"금 실제 거래 {len(real):,}건 / 가상 {gold_tr.virtual.sum():,}건, 마틴 정지일 {len(stop_at)}일")
    print(table([(str(y), g.pct.values) for y, g in real.groupby(pd.to_datetime(real.open).dt.year)]
                + [("전체", real.pct.values)]).to_string())
    for st in MARTIN_STARTS:
        print(f"[금 + 일일 마틴, {st:%Y-%m}부터 1,000$] " + martin_report(*gold_martin(real, st)))
    print()

    for sym in ("XAGUSD", "BTCUSD"):
        m1 = load_m1(sym)
        sig, m5, _ = signal_table(m1)
        mk = Market(sym, m1)
        sig["gold"] = gold_state(sig.t, gold_tr, gold_m5, flips, pause_changes, stop_at)
        k_vol = float(np.nanmedian(atr(m5) / m5.close)) / gold_atrp
        sp_r = float(np.median(m1.sp / m1.close)) * 100 / (TP_PCT * 100)
        print(f"==================== {sym} ====================")
        print(f"신호 {len(sig):,}개, 5분봉 ATR% 중앙값 = 금의 {k_vol:.2f}배, 스프레드 중앙값 = 가격비례 TP의 {sp_r * 100:.0f}%")
        for name, f in (("가격비례", 1.0), ("변동성비례", k_vol)):
            sl, tp = SL_PCT * f, TP_PCT * f
            print(f"\n--- SL/TP {name}: SL {sl * 100:.3f}% / TP {tp * 100:.3f}% ---")
            tr = simulate(mk, sig, sl, tp)
            tr["gold"] = sig.gold.values[tr.sig.values]
            print("[항상 진입] 진입 시점 금 상태별")
            print(table([(c, tr.pct[tr.gold == c].values) for c in CATS]
                        + [("금 쉬기 합계", tr.pct[tr.gold.isin(REST)].values), ("전체", tr.pct.values)]).to_string())
            only = simulate(mk, sig, sl, tp, allow=sig.gold.isin(REST).values)
            yr = pd.to_datetime(only.open).dt.year
            print("[금이 쉬기 필터로 쉴 때만 진입] 연도별")
            print(table([(str(y), g.pct.values) for y, g in only.groupby(yr)] + [("전체", only.pct.values)]).to_string())
            allow = sig.gold.isin(REST).values
            for st in MARTIN_STARTS:
                print(f"[금 쉴 때만 + 일일 마틴, {st:%Y-%m}부터 1,000$] "
                      + martin_report(*simulate_martin(mk, sig, sl, tp, allow, st)))
        print()


if __name__ == "__main__":
    main()
