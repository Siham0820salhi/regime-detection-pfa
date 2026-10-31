import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

df = pd.read_csv("data/features.csv", index_col=0, parse_dates=True)
r = df["ret"].values
X = (r * 100).reshape(-1, 1)

# ---------- 1. HMM K=2 sur rendements ----------
best, best_ll = None, -np.inf
for seed in range(10):
    m = GaussianHMM(2, covariance_type="full", n_iter=500, tol=1e-3,
                    random_state=seed).fit(X)
    if m.score(X) > best_ll:
        best, best_ll = m, m.score(X)
m = best
crise = int(np.argmax([m.covars_[k][0, 0] for k in range(2)]))   # état agité

# ---------- 2. Probabilités filtrées (forward) ----------
sig = np.sqrt([m.covars_[k][0, 0] for k in range(2)])
B = np.column_stack([norm.pdf(X[:, 0], m.means_[k, 0], sig[k]) for k in range(2)]) + 1e-300
alpha = np.zeros((len(X), 2))
a = m.startprob_ * B[0]; alpha[0] = a / a.sum()
for t in range(1, len(X)):
    a = (alpha[t - 1] @ m.transmat_) * B[t]
    alpha[t] = a / a.sum()
p_crise = alpha[:, crise]

# ---------- 3. Hystérésis : entrée > 0.8, sortie < 0.4 ----------
etat = np.zeros(len(X), dtype=int)
for t in range(1, len(X)):
    if etat[t - 1] == 0 and p_crise[t] > 0.8:
        etat[t] = 1
    elif etat[t - 1] == 1 and p_crise[t] < 0.4:
        etat[t] = 0
    else:
        etat[t] = etat[t - 1]
brut = (p_crise > 0.5).astype(int)
print("Changements filtered brut      :", int((np.diff(brut) != 0).sum()))
print("Changements filtered hystérésis:", int((np.diff(etat) != 0).sum()))

# ---------- 4. Backtest (signal décalé d'1 jour : pas de look-ahead) ----------
cout = 0.0005                                   # 5 points de base par changement
expo = pd.Series(1 - etat, index=df.index).shift(1).fillna(1)   # 0 en crise, 1 sinon
turn = expo.diff().abs().fillna(0)
r_simple = np.exp(df["ret"]) - 1
strat = expo * r_simple - turn * cout
bh = r_simple

def stats(x):
    cum = (1 + x).cumprod()
    ann = cum.iloc[-1] ** (252 / len(x)) - 1
    vol = x.std() * np.sqrt(252)
    mdd = (cum / cum.cummax() - 1).min()
    return {"rend_ann": ann, "vol": vol, "sharpe": ann / vol, "max_drawdown": mdd}

res = pd.DataFrame({"Buy&Hold": stats(bh), "Stratégie HMM": stats(strat)}).T
print("\n=== Backtest (coûts 5 bp, signal décalé 1 jour) ===")
print(res.round(3))
print("Part du temps investi :", round(expo.mean(), 3))

fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
ax[0].plot((1 + bh).cumprod(), label="Buy & Hold", color="black")
ax[0].plot((1 + strat).cumprod(), label="Stratégie HMM", color="green")
ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("Valeur cumulée")
ax[1].plot(df.index, p_crise, color="red", lw=0.5, label="P(crise) filtrée")
ax[1].fill_between(df.index, 0, 1, where=etat == 1, color="red", alpha=0.2)
ax[1].legend(); ax[1].set_title("Signal avec hystérésis")
plt.tight_layout()
plt.savefig("figures/fig6_backtest.png", dpi=200)
plt.show()