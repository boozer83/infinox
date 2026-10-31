"""SAR 단타 신호를 머신러닝으로 거르기 (2026-10-29 사용자 아이디어)

데이터: results/sar_regime_trades.csv (backtests/sar_regime feat가 만든 것) = 지금 봇 거래 2,441건 + 진입 때 특징
  특징 (진입 순간에 알 수 있는 값만): ext(뚫은 SAR에서 거리/ATR), risk(새 SAR 점까지/ATR), body·range(신호봉 크기/ATR),
  ema(EMA 거리/ATR), side, prev_run(직전 SAR 추세 길이), atr(5분봉 $), rsi_dir((RSI-50) x 방향), h1_run(1시간 SAR 추세 길이),
  hour(서버 시각 sin/cos), dow, prev_win(직전 거래 이김)
  빼는 것: mins·mae·rev (거래가 끝나야 알 수 있음 = 미래 정보)
목표: 이김(oz > 0) 확률
모델 (sklearn이 없어 numpy로 직접): 로지스틱 회귀 (L2), 깊이 2 나무 부스팅 (100그루, 학습률 0.05, 분위수 분할)
검증 (시간순, 앞으로 걷기): 2024 / 2025 / 2026 각각을 그 전 해 전부로 학습해서 시험
  거르기: 학습 기간 점수 분포의 하위 q% (q = 10 / 20 / 30, 미리 정함) 보다 낮은 신호는 쉼
          + "최근": 직전 200개 신호 점수(같은 모델) 중 하위 q%면 쉼 -> 해마다 승률 수준이 달라져도 맞춰짐
  기준 = 그해 전체 거래. 승률·oz 둘 다 나아져야 의미 있음
  (쉰 자리에 다른 신호가 들어가는 효과는 무시)
결과 results/sar_ml_filter.csv

사용법: python -m backtests.sar_ml_filter
"""
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
FEATS = ["ext", "risk", "body", "range", "ema", "side", "prev_run", "atr", "rsi_dir", "h1_run", "dow", "prev_win", "hs", "hc"]
QS = (10, 20, 30)


def load():
    d = pd.read_csv(RES / "sar_regime_trades.csv", parse_dates=["open"])
    d["hs"], d["hc"] = np.sin(d.hour / 24 * 2 * np.pi), np.cos(d.hour / 24 * 2 * np.pi)
    d["y"] = d.open.dt.year
    d["win"] = (d.oz > 0).astype(float)
    return d.dropna(subset=FEATS).reset_index(drop=True)


class Logit:
    def fit(self, X, y, lam=1.0, it=500, lr=0.1):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.w, self.b = np.zeros(Z.shape[1]), 0.0
        for _ in range(it):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
            g = p - y
            self.w -= lr * (Z.T @ g / len(y) + lam * self.w / len(y))
            self.b -= lr * g.mean()
        return self

    def score(self, X):
        return 1 / (1 + np.exp(-(((X - self.mu) / self.sd) @ self.w + self.b)))


class Boost:
    """로지스틱 손실 그래디언트 부스팅, 깊이 2 나무, 분할 후보 = 특징별 분위수 9개"""

    def fit(self, X, y, n=100, lr=0.05, min_leaf=40):
        self.cuts = [np.unique(np.quantile(X[:, j], np.linspace(0.1, 0.9, 9))) for j in range(X.shape[1])]
        self.f0 = np.log(y.mean() / (1 - y.mean()))
        f = np.full(len(y), self.f0)
        self.trees, self.lr, self.min_leaf = [], lr, min_leaf
        for _ in range(n):
            p = 1 / (1 + np.exp(-f))
            g, h = y - p, p * (1 - p)
            t = self._grow(X, g, h, np.arange(len(y)), 2)
            self.trees.append(t)
            f += lr * self._pred(t, X)
        return self

    def _best(self, X, g, h, idx):
        best = None
        G, H = g[idx].sum(), h[idx].sum()
        for j, cs in enumerate(self.cuts):
            x = X[idx, j]
            for c in cs:
                m = x <= c
                nl = m.sum()
                if nl < self.min_leaf or len(idx) - nl < self.min_leaf:
                    continue
                gl, hl = g[idx][m].sum(), h[idx][m].sum()
                gain = gl ** 2 / (hl + 1) + (G - gl) ** 2 / (H - hl + 1) - G ** 2 / (H + 1)
                if best is None or gain > best[0]:
                    best = (gain, j, c)
        return best

    def _grow(self, X, g, h, idx, depth):
        leaf = ("leaf", g[idx].sum() / (h[idx].sum() + 1))
        if depth == 0:
            return leaf
        b = self._best(X, g, h, idx)
        if b is None or b[0] <= 0:
            return leaf
        _, j, c = b
        m = X[idx, j] <= c
        return ("split", j, c, self._grow(X, g, h, idx[m], depth - 1), self._grow(X, g, h, idx[~m], depth - 1))

    def _pred(self, t, X):
        if t[0] == "leaf":
            return np.full(len(X), t[1])
        _, j, c, l, r = t
        m = X[:, j] <= c
        out = np.empty(len(X))
        out[m], out[~m] = self._pred(l, X[m]), self._pred(r, X[~m])
        return out

    def score(self, X):
        f = np.full(len(X), self.f0)
        for t in self.trees:
            f += self.lr * self._pred(t, X)
        return 1 / (1 + np.exp(-f))


def main():
    d = load()
    rows = []
    for test_y in (2024, 2025, 2026):
        tr, te = d[d.y < test_y], d[d.y == test_y]
        Xtr, ytr, Xte = tr[FEATS].values.astype(float), tr.win.values, te[FEATS].values.astype(float)
        base = {"시험 해": test_y, "학습 건수": len(tr), "모델": "-", "거르기": "없음", "건수": len(te),
                "승률%": round(te.win.mean() * 100, 1), "oz": round(te.oz.sum(), 1)}
        rows.append(base)
        for name, model in (("로지스틱", Logit().fit(Xtr, ytr)), ("나무 부스팅", Boost().fit(Xtr, ytr))):
            s_tr, s_te = model.score(Xtr), model.score(Xte)
            # 점수 순위와 실제 결과의 상관 (시험 해)
            auc = (pd.Series(s_te).rank().values[te.win.values == 1].mean() - (te.win.sum() + 1) / 2) / max((1 - te.win).sum(), 1)
            for q in QS:
                cut = np.percentile(s_tr, q)
                k = s_te >= cut
                allsc = np.concatenate([s_tr, s_te])
                n0 = len(s_tr)
                k2 = np.array([s_te[i] >= np.percentile(allsc[n0 + i - 200:n0 + i], q) for i in range(len(s_te))])
                rows.append({"시험 해": test_y, "학습 건수": len(tr), "모델": name, "거르기": f"최근 200개 하위 {q}% 쉼",
                             "건수": int(k2.sum()), "승률%": round(te.win.values[k2].mean() * 100, 1),
                             "oz": round(te.oz.values[k2].sum(), 1), "AUC": round(auc, 3),
                             "쉰 거래 승률%": round(te.win.values[~k2].mean() * 100, 1) if (~k2).any() else None,
                             "쉰 거래 oz": round(te.oz.values[~k2].sum(), 1)})
                rows.append({"시험 해": test_y, "학습 건수": len(tr), "모델": name, "거르기": f"하위 {q}% 쉼",
                             "건수": int(k.sum()), "승률%": round(te.win.values[k].mean() * 100, 1),
                             "oz": round(te.oz.values[k].sum(), 1), "AUC": round(auc, 3),
                             "쉰 거래 승률%": round(te.win.values[~k].mean() * 100, 1) if (~k).any() else None,
                             "쉰 거래 oz": round(te.oz.values[~k].sum(), 1)})
        print("완료", test_y, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "sar_ml_filter.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print(df.to_string(index=False))
    tot = df[df.모델 != "-"].groupby(["모델", "거르기"])[["건수", "oz"]].sum()
    b = df[df.모델 == "-"][["건수", "oz"]].sum()
    print(f"\n2024~26 합계 기준: {int(b.건수)}건 {b.oz:.1f}oz")
    print(tot)


if __name__ == "__main__":
    main()
