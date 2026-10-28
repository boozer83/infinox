"""사용자가 준 트레이딩뷰 지표 4개를 금에 (2022-01 ~ 2026-10, 기본 설정, 원본 Pine 규칙대로 옮김)

1) Adaptive Donchian Breakout Channel (SchizoQuant): 지표의 LONG/SHORT 신호로 항상 보유 (갈아타기)
2) Institutional Price Action Master v2.1: 장악형·핀바·샛별 패턴 + EMA21/200 + BOS. 거래량 = 1분 틱 수 합
   a. Grade-A 신호만 (EMA 가치 구간 또는 BOS 방향 패턴) / b. 화면에 표시되는 모든 패턴 신호
   진입 다음 봉 시가, 손절 = 지표가 그리는 손절점 (패턴 저가/고가), 익절 1R / 2R / 3R. 한 번에 한 거래
3) Delta Pressure Clock (AlgoNorth): 1분봉 틱 규칙 델타 (종가가 앞 1분 종가보다 높으면 + 틱 수, 낮으면 -)
   창 15분 / 1시간 / 4시간, 원본처럼 합 -> EMA(4 x sqrt(봉 수)) -> 90% 분위로 -1~+1. |값| < 0.05 는 중립
   a. 4시간 창 방향으로 보유 (중립이면 쉼) / b. 세 창이 모두 같은 방향일 때만 보유
4) Liquidity Sweep Map (GBB): 유동성 풀(스윙 고저 5/3, 같은 고저, 전일·전주 고저, 아시아·런던 고저, 뉴욕 시간 기준)을
   꼬리로 0.02 ATR 이상 찌르고(1.5 ATR 넘으면 돌파) 종가가 안으로 돌아오면 '스윕'. 원본 확인 체인:
   스윕 직전 5봉 안의 FVG를 10봉 안에 종가로 뚫으면 확인 -> 반대 방향 진입 (다음 봉 시가),
   손절 = 스윕 꼬리 끝 + 0.1 ATR, 목표 = 반대쪽 가장 가까운 풀, 100봉 지나면 청산 (원본 알림 체인 그대로)
   (서버 시간 = 뉴욕 + 7시간으로 계산. 풀 무게 감쇠·터치 가중·최대 150개는 원본대로, 같은 고저 병합 포함)
비용·체결: tv_single과 같음. 손절·익절은 봉 고가·저가, 한 봉에 둘 다 닿으면 손절

사용법: python -m backtests.tv_batch2
"""
import math

import numpy as np
import pandas as pd

import backtests.tickvol_m1 as tv
import backtests.tv_single as ts
from indicators import atr, ema

TFS = {"M5": 5, "M15": 15, "H1": 60, "H4": 240}


def bars_vol(m1, vol, mins):
    d = ts.bars(m1, mins)
    v = vol.resample(f"{mins}min", label="left", closed="left").sum()
    d["vol"] = v.reindex(d.time).fillna(0).values
    return d


def rr_trades(d, side, stop, rr):
    """side[i] (+1/-1/0) 봉 i 마감 신호 -> i+1 시가 진입, 손절 stop[i], 익절 rr x 위험. 한 번에 한 거래"""
    o, h, l, sp, t = (d[k].values for k in ("open", "high", "low", "sp", "time"))
    rows, i, n = [], 0, len(d)
    while i < n - 1:
        s = side[i]
        if not s:
            i += 1
            continue
        k = i + 1
        ent = o[k] + sp[k] + ts.SLIP if s == 1 else o[k] - ts.SLIP
        sl = stop[i]
        if (ent - sl) * s <= 0:
            i += 1
            continue
        tp = ent + s * rr * abs(ent - sl)
        j = k
        while j < n:
            if s == 1:
                if l[j] <= sl:
                    px = min(sl, o[j]) - ts.SLIP
                    break
                if h[j] >= tp:
                    px = max(tp, o[j])
                    break
            else:
                if h[j] + sp[j] >= sl:
                    px = max(sl, o[j] + sp[j]) + ts.SLIP
                    break
                if l[j] + sp[j] <= tp:
                    px = min(tp, o[j] + sp[j])
                    break
            j += 1
        if j >= n:
            break
        pnl = (px - ent) * s + ts.swap_nights(t[k], t[j]) * (ts.SWAP_LONG if s == 1 else ts.SWAP_SHORT)
        rows.append((t[k], t[j], s, ent, pnl / ent * 100, pnl / abs(ent - sl)))
        i = j
    return pd.DataFrame(rows, columns=["open", "close", "side", "entry", "pct", "r"])


# ---------- 1. Adaptive Donchian ----------

def adaptive_donchian(d, bl=24, fl=8, sl=21, acc_len=6, depth=0.7, buf=0.02):
    h1, l1 = d.high.shift(), d.low.shift()
    up, lo = h1.rolling(bl).max(), l1.rolling(bl).min()
    width = np.maximum(up - lo, 0.01)
    fh, flo, shh, slo = h1.rolling(fl).max(), l1.rolling(fl).min(), h1.rolling(sl).max(), l1.rolling(sl).min()
    near_s, deep_s = np.maximum(flo, slo), np.minimum(flo, slo)
    near_r, deep_r = np.minimum(fh, shh), np.maximum(fh, shh)
    loc = ((d.close - lo) / width).clip(0, 1)
    acc = ema(loc, acc_len)
    bq = ((acc - 0.35) / 0.55).clip(0, 1)
    sq = ((0.65 - acc) / 0.55).clip(0, 1)
    ldef = (near_s * (1 - depth * bq) + deep_s * depth * bq).values
    sdef = (near_r * (1 - depth * sq) + deep_r * depth * sq).values
    bd = (width * buf).values
    c, upv, lov = d.close.values, up.values, lo.values
    pos, q = np.zeros(len(d)), 0
    for i in range(len(d)):
        if np.isnan(sdef[i]) or np.isnan(ldef[i]) or np.isnan(upv[i]):
            continue
        if q == 0:
            q = 1 if c[i] > upv[i] else (-1 if c[i] < lov[i] else 0)
        elif q == -1 and c[i] > sdef[i] + bd[i]:
            q = 1
        elif q == 1 and c[i] < ldef[i] - bd[i]:
            q = -1
        pos[i] = q
    return pos


# ---------- 2. Price Action Master ----------

def pivots(x, L, R, high):
    """Pine ta.pivothigh/low(L, R): 봉 i-R 이 왼쪽 L봉보다 크고(작고) 오른쪽 R봉 이상(이하). 확인되는 봉 i에 값"""
    x = np.asarray(x)
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(L + R, n):
        c = x[i - R]
        left = x[i - R - L:i - R]
        right = x[i - R + 1:i + 1]
        if high and (c > left).all() and (c >= right).all():
            out[i] = c
        if not high and (c < left).all() and (c <= right).all():
            out[i] = c
    return out


def pa_master(d, piv=5, touch=0.5, volm=1.1, use_vol=True, body_min=0.6, ef_len=21, em_len=200, parts=False):
    """기본값 = 원본. parts=True면 (매수/매도 신호, 패턴 종류, 가치 구간/BOS) 도 같이 돌려줌"""
    o, h, l, c, v = (d[k] for k in ("open", "high", "low", "close", "vol"))
    a = atr(d, 14)
    vsma = v.rolling(20).mean()
    ef, em = ema(c, ef_len), ema(c, em_len)
    low3, high3 = l.rolling(3).min(), h.rolling(3).max()
    low4, high4 = l.rolling(4).min(), h.rolling(4).max()
    has_v = (v > 0) & use_vol
    vol_base = np.where(has_v, v > vsma * volm, True)
    vol_aplus = np.where(has_v, v > vsma * 1.2, True)
    rng = h - l
    body, body_p = (c - o).abs(), (c - o).abs().shift()
    bcs = np.where(rng > 0, (c - l) / rng, 0)
    scs = np.where(rng > 0, (h - c) / rng, 0)
    c1, o1 = c.shift(), o.shift()
    bull_eng = (c1 < o1) & (c > o) & (o <= c1) & (c > o1) & (body > body_p) & (body >= a * body_min) & (body <= a * 2.8) & (bcs >= 0.65)
    bear_eng = (c1 > o1) & (c < o) & (o >= c1) & (c < o1) & (body > body_p) & (body >= a * body_min) & (body <= a * 2.8) & (scs >= 0.65)
    prev_down = (c1 < c.shift(5)) & (l <= low3)
    prev_up = (c1 > c.shift(5)) & (h >= high3)
    eng_vol = vol_base & np.where(has_v, v > v.shift(), True)
    bull_eng_s = bull_eng & prev_down & eng_vol
    bear_eng_s = bear_eng & prev_up & eng_vol
    btop, bbot = np.maximum(o, c), np.minimum(o, c)
    uw, lw = h - btop, bbot - l
    lwr = np.where(rng > 0, lw / rng, 0)
    uwr = np.where(rng > 0, uw / rng, 0)
    bpos = np.where(rng > 0, (h - bbot) / rng, 1)
    spos = np.where(rng > 0, (btop - l) / rng, 1)
    pin_ok = (rng >= a * 0.8) & (rng <= a * 3)
    bull_pin_s = pin_ok & (lwr >= 0.6) & (uwr <= 0.15) & (bpos <= 0.35) & prev_down & vol_base
    bear_pin_s = pin_ok & (uwr >= 0.6) & (lwr <= 0.15) & (spos <= 0.35) & prev_up & vol_base
    c2, o2 = c.shift(2), o.shift(2)
    b1, b2 = (c2 - o2).abs(), (c1 - o1).abs()
    b2t, b2b = np.maximum(o1, c1), np.minimum(o1, c1)
    star_low = np.minimum(l.shift(2), np.minimum(l.shift(), l))
    star_high = np.maximum(h.shift(2), np.maximum(h.shift(), h))
    v1 = b1 >= a.shift(2) * body_min
    vs = (b2 <= a.shift() * 0.45) & (b2 < b1 * 0.6)
    morning = (c2 < o2) & v1 & vs & (b2b <= c2) & (b2t <= c2 + b1 * 0.5) & (c > o) & (c >= c2 + b1 * 0.5) & (c > b2t) & (bcs >= 0.6)
    evening = (c2 > o2) & v1 & vs & (b2t >= c2) & (b2b >= c2 - b1 * 0.5) & (c < o) & (c <= c2 - b1 * 0.5) & (c < b2b) & (scs >= 0.6)
    s_down = (c2 < c.shift(7)) & (star_low <= low4)
    s_up = (c2 > c.shift(7)) & (star_high >= high4)
    bull_star_s = morning & s_down & vol_base
    bear_star_s = evening & s_up & vol_base
    any_bull = (bull_eng_s | bull_pin_s | bull_star_s).fillna(False).values
    any_bear = (bear_eng_s | bear_pin_s | bear_star_s).fillna(False).values
    pat_low = np.where(bull_star_s, star_low, np.minimum(l, l.shift()))
    pat_high = np.where(bear_star_s, star_high, np.maximum(h, h.shift()))

    # 구조 (BOS)
    ph, pl = pivots(h.values, piv, piv, True), pivots(l.values, piv, piv, False)
    cv = c.values
    n = len(d)
    bos_up, bos_dn = np.zeros(n, bool), np.zeros(n, bool)
    last_ph = last_pl = np.nan
    trd, lbb, lbd = 0, 0, 0
    prev_ph0 = prev_pl0 = 0.0
    for i in range(n):
        ph0, pl0 = (0.0 if np.isnan(last_ph) else last_ph), (0.0 if np.isnan(last_pl) else last_pl)
        xo = i > 0 and cv[i] > ph0 and cv[i - 1] <= prev_ph0
        xu = i > 0 and cv[i] < pl0 and cv[i - 1] >= prev_pl0
        prev_ph0, prev_pl0 = ph0, pl0
        if not np.isnan(ph[i]):
            if not np.isnan(last_ph):
                trd = 1 if ph[i] > last_ph else -1
            last_ph = ph[i]
        if not np.isnan(pl[i]):
            last_pl = pl[i]
        bu = trd == 1 and not np.isnan(last_ph) and xo
        bdn = trd == -1 and not np.isnan(last_pl) and xu
        if bu and i - lbb >= 1 and (lbd != 1 or cv[i] > last_ph * 1.001):
            bos_up[i], lbb, lbd = True, i, 1
        if bdn and i - lbb >= 1 and (lbd != -1 or cv[i] < last_pl * 0.999):
            bos_dn[i], lbb, lbd = True, i, -1
        if trd == -1 and not np.isnan(last_ph) and xo:
            trd = 1
        elif trd == 1 and not np.isnan(last_pl) and xu:
            trd = -1

    av, efv, emv = a.values, ef.values, em.values
    bull_macro, bear_macro = cv > emv, cv < emv
    in_bull_vz = ((pat_low <= efv + av * touch) & (cv >= efv - av * 0.25)) | ((pat_low <= emv + av * touch) & (cv >= emv - av * 0.25))
    in_bear_vz = ((pat_high >= efv - av * touch) & (cv <= efv + av * 0.25)) | ((pat_high >= emv - av * touch) & (cv <= emv + av * 0.25))
    grade_bull = any_bull & ((bull_macro & in_bull_vz) | bos_up)
    grade_bear = any_bear & ((bear_macro & in_bear_vz) | bos_dn)
    side_a = np.where(grade_bull, 1, np.where(grade_bear, -1, 0))
    side_all = np.where(any_bull & ~any_bear, 1, np.where(any_bear & ~any_bull, -1, 0))
    stop = np.where(side_all == 1, pat_low, pat_high)
    stop_a = np.where(side_a == 1, pat_low, pat_high)
    if parts:
        pat = np.where(side_a == 1, np.where(bull_star_s, "샛별", np.where(bull_eng_s, "장악형", "핀바")),
                       np.where(bear_star_s, "샛별", np.where(bear_eng_s, "장악형", "핀바")))
        vz = np.where(side_a == 1, bull_macro & in_bull_vz, bear_macro & in_bear_vz)
        return side_a, stop_a, pat, np.where(vz, "가치 구간", "BOS만")
    return side_a, stop_a, side_all, stop


# ---------- 3. Delta Pressure Clock ----------

def delta_tone(bar_delta, ratio):
    raw = bar_delta.rolling(ratio).sum()
    sm = round(4 * math.sqrt(ratio))
    eff = min(4000, max(300, 6 * (ratio + 3 * sm)))
    s = ema(raw, sm)
    scale = s.abs().rolling(eff).quantile(0.9)
    return (s / scale).clip(-1, 1).where(scale > 0)


def delta_clock(m1, vol, mins):
    c = m1.set_index("time").close
    sgn = np.sign(c.diff()).replace(0, np.nan).ffill().fillna(0)
    dv = sgn * vol.reindex(c.index).fillna(0)
    bd = dv.resample(f"{mins}min", label="left", closed="left").sum()
    u = [delta_tone(bd, max(1, round(w / mins))) for w in (15, 60, 240)]
    if len({max(1, round(w / mins)) for w in (15, 60, 240)}) < 3:
        return None
    tone = [np.where(x.abs() < 0.05, 0, np.sign(x)) for x in (y.fillna(0) for y in u)]
    slow = pd.Series(tone[2], index=bd.index)
    agree = pd.Series(np.where((tone[0] == tone[1]) & (tone[1] == tone[2]), tone[2], 0), index=bd.index)
    return slow, agree


# ---------- 4. Liquidity Sweep Map ----------

def lsm_trades(d):
    o, h, l, c, sp, t = (d[k].values for k in ("open", "high", "low", "close", "sp", "time"))
    a = atr(d, 14).values
    n = len(d)
    ny = d.time - pd.Timedelta(hours=7)
    tday = (ny + pd.Timedelta(hours=6)).dt.floor("D").values.astype("datetime64[D]").astype(np.int64)
    week = (tday + 3) // 7
    mins = (ny.dt.hour * 60 + ny.dt.minute).values
    in_a = mins >= 20 * 60                      # 20:00-24:00
    in_l = (mins >= 120) & (mins < 300)         # 02:00-05:00
    phv, plv = pivots(h, 5, 3, True), pivots(l, 5, 3, False)
    half, minw = 200.0, 0.15
    base = {"SW": 1.0, "EQ": 1.6, "SS": 1.4, "PD": 1.8, "PW": 2.0}
    pools = []                                  # dict: level side src created pivot touches members sess

    def expiry(p):
        b = base[p["src"]] * min(1 + 0.25 * (p["touches"] - 1), 2.0)
        if b <= minw:
            return 0.0
        return 1e12 if p["src"] not in ("SW", "EQ") else half * math.log(b / minw) / math.log(2)

    def weight(p, i):
        b = base[p["src"]] * min(1 + 0.25 * (p["touches"] - 1), 2.0)
        return b * (0.5 ** ((i - p["created"]) / half) if p["src"] in ("SW", "EQ") else 1.0)

    def new_pool(lvl, side, src, i, pb, sess=-1):
        p = dict(level=lvl, side=side, src=src, created=i, pivot=pb, touches=1, members=1, sess=sess)
        p["exp"] = expiry(p)
        pools.append(p)

    trk = {k: [np.nan, np.nan, False] for k in ("D", "W", "A", "L")}
    chains, rows = [], []
    first = 15
    for i in range(first, n):
        ai = a[i]
        thr, maxpen, tol = max(0.02, 0.02 * ai), 1.5 * ai, 0.10 * ai
        new_day, new_week = tday[i] != tday[i - 1], week[i] != week[i - 1]

        def commit(key, src, sess):
            hi, lo, valid = trk[key]
            if valid:
                for p in pools:
                    if p["src"] == src and p["sess"] == sess:
                        p["dead"] = True
                pools[:] = [p for p in pools if not p.get("dead")]
                new_pool(hi, 1, src, i, i, sess)
                new_pool(lo, -1, src, i, i, sess)

        if new_day:
            commit("D", "PD", -1)
        if new_week:
            commit("W", "PW", -1)
        if in_a[i - 1] and not in_a[i]:
            commit("A", "SS", 0)
        if in_l[i - 1] and not in_l[i]:
            commit("L", "SS", 1)
        for key, isnew in (("D", new_day), ("W", new_week)):
            tr = trk[key]
            if isnew or i == first:
                trk[key] = [h[i], l[i], i != first]
            else:
                tr[0], tr[1] = max(tr[0], h[i]), min(tr[1], l[i])
        for key, f, fp in (("A", in_a[i], in_a[i - 1]), ("L", in_l[i], in_l[i - 1])):
            tr = trk[key]
            if f:
                if i == first or not fp:
                    trk[key] = [h[i], l[i], i != first]
                else:
                    tr[0], tr[1] = max(tr[0], h[i]), min(tr[1], l[i])
            else:
                tr[2] = False

        # 스윕
        cand = {1: [], -1: []}
        keep = []
        for p in pools:
            pen = h[i] - p["level"] if p["side"] == 1 else p["level"] - l[i]
            if pen < thr:
                touch = (h[i] >= p["level"] - tol) if p["side"] == 1 else (l[i] <= p["level"] + tol)
                if touch:
                    p["touches"] += 1
                    p["exp"] = expiry(p)
                keep.append(p)
                continue
            back = c[i] < p["level"] if p["side"] == 1 else c[i] > p["level"]
            cls = "B" if pen > maxpen else ("R" if back else "B")
            cand[p["side"]].append((p, cls, h[i] if p["side"] == 1 else l[i]))
        pools[:] = keep
        new_chains = []
        for side in (1, -1):
            lst = cand[side]
            if not lst:
                continue
            ws = [weight(p, i) for p, _, _ in lst]
            k = int(np.argmax(ws))
            p, cls, ext = lst[k]
            if cls == "R":
                lvl = np.nan                      # 스윕 직전 다리의 FVG
                for j in range(0, 6):
                    if i - j < 2:
                        break
                    gap = (l[i - j] - h[i - j - 2]) if side == 1 else (l[i - j - 2] - h[i - j])
                    if gap > 0:
                        lvl = h[i - j - 2] if side == 1 else l[i - j - 2]
                        break
                if lvl == lvl:
                    new_chains.append(dict(side=side, bar=i, ext=ext, buf=0.1 * ai, inv=lvl, state="S"))
        # 체인
        chains.extend(new_chains)
        alive = []
        for ch in chains:
            s = ch["side"]
            beyond = h[i] >= ch["ext"] + ch["buf"] if s == 1 else l[i] <= ch["ext"] - ch["buf"]
            if i > ch["bar"] and beyond:
                continue
            if (c[i] < ch["inv"]) if s == 1 else (c[i] > ch["inv"]):
                below = [p["level"] for p in pools if p["side"] == -s and ((p["level"] < c[i]) if s == 1 else (p["level"] > c[i]))]
                tgt = (max(below) if s == 1 else min(below)) if below else np.nan
                if i + 1 < n:
                    rows.append(lsm_trade(o, h, l, sp, t, i + 1, -s, ch["ext"] + s * ch["buf"], tgt))
                continue
            if i - ch["bar"] >= 10:
                continue
            alive.append(ch)
        chains = alive
        # 감쇠 만료
        pools[:] = [p for p in pools if (i - p["created"]) <= p["exp"]]
        # 스윙 / 같은 고저
        for side, pv in ((1, phv[i]), (-1, plv[i])):
            if pv != pv:
                continue
            cb = i - 3
            best, bd = None, 1e300
            for p in pools:
                if p["side"] == side and p["src"] in ("SW", "EQ") and cb - p["pivot"] <= 100:
                    dd = abs(p["level"] - pv)
                    if dd <= tol and dd <= bd:
                        best, bd = p, dd
            if best is not None:
                best["level"] = max(best["level"], pv) if side == 1 else min(best["level"], pv)
                best.update(src="EQ", created=i, pivot=cb, members=best["members"] + 1)
                best["exp"] = expiry(best)
            else:
                new_pool(pv, side, "SW", i, cb)
        if len(pools) > 150:
            pools.sort(key=lambda p: weight(p, i))
            del pools[:len(pools) - 150]
    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows, columns=["open", "close", "side", "entry", "pct"])


def lsm_trade(o, h, l, sp, t, k, s, sl, tgt):
    """k 봉 시가 진입, 손절 sl, 목표 tgt (없으면 시간만), 100봉 뒤 시가 청산"""
    ent = o[k] + sp[k] + ts.SLIP if s == 1 else o[k] - ts.SLIP
    if (ent - sl) * s <= 0:
        return None
    n = len(o)
    for j in range(k, min(n, k + 100)):
        if s == 1:
            if l[j] <= sl:
                px = min(sl, o[j]) - ts.SLIP
                break
            if tgt == tgt and h[j] >= tgt:
                px = max(tgt, o[j])
                break
        else:
            if h[j] + sp[j] >= sl:
                px = max(sl, o[j] + sp[j]) + ts.SLIP
                break
            if tgt == tgt and l[j] + sp[j] <= tgt:
                px = min(tgt, o[j] + sp[j])
                break
    else:
        j = min(n - 1, k + 100)
        px = o[j] - ts.SLIP if s == 1 else o[j] + sp[j] + ts.SLIP
    pnl = (px - ent) * s + ts.swap_nights(t[k], t[j]) * (ts.SWAP_LONG if s == 1 else ts.SWAP_SHORT)
    return (t[k], t[j], s, ent, pnl / ent * 100)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    m1 = ts.load_m1("XAUUSD")
    vol = tv.load()
    out = []

    def add(ind, tf, how, tr):
        if len(tr) >= 10:
            out.append({"지표": ind, "봉": tf, "방식": how, **ts.summary(tr)})

    for tf, mins in TFS.items():
        d = bars_vol(m1, vol, mins)
        add("Adaptive Donchian", tf, "신호대로 갈아타기", ts.trades(d, adaptive_donchian(d)))
        sa, sta, sall, stall = pa_master(d)
        for rr in (1.0, 2.0, 3.0):
            add("PA Master", tf, f"Grade-A, 익절 {rr:.0f}R", rr_trades(d, sa, sta, rr))
            add("PA Master", tf, f"모든 패턴, 익절 {rr:.0f}R", rr_trades(d, sall, stall, rr))
        if mins <= 60:
            dc = delta_clock(m1, vol, mins)
            if dc is not None:
                for how, sig in zip(("4시간 창 방향", "세 창 모두 같은 방향"), dc):
                    add("Delta Clock", tf, how, ts.trades(d, sig.reindex(d.time).fillna(0).values))
            add("Liquidity Sweep", tf, "스윕 -> FVG 뒤집힘 -> 반대 풀 목표", lsm_trades(d))
        print(f"{tf} 끝", flush=True)
    res = pd.DataFrame(out)
    res.to_csv("results/tv_batch2.csv", index=False, encoding="utf-8-sig")
    print(res.drop(columns=["평균%"]).to_string(index=False))


if __name__ == "__main__":
    main()
