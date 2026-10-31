"""
Etape 7 : walk-forward + exposition graduelle + comparaison avec les baselines.

Principe du walk-forward (fenetre expansive) :
  - on entraine le HMM sur le passe seulement (au depart 5 ans),
  - on l'utilise pour la periode suivante (1 an) avec l'algorithme forward,
  - on reentraine chaque annee en ajoutant les nouvelles donnees.
Les parametres du HMM ne contiennent donc plus aucune information du futur.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

# ---------- Parametres ----------
TRAIN_MIN = 1260      # 5 ans de bourse avant la 1re prediction
REFIT = 252           # reentrainement tous les ans
N_SEEDS = 5           # initialisations aleatoires par entrainement
SEUIL_IN, SEUIL_OUT = 0.8, 0.4   # hysteresis (comme a l'etape 6)
COUT = 0.0005         # 5 points de base par changement d'exposition

df = pd.read_csv("data/features.csv", index_col=0, parse_dates=True)
r = df["ret"].values
X = (r * 100).reshape(-1, 1)
T = len(X)


# ---------- Fonctions ----------
def ajuster(Xtrain):
    """HMM K=2 sur rendements, meilleure vraisemblance parmi N_SEEDS."""
    best, best_ll = None, -np.inf
    for seed in range(N_SEEDS):
        m = GaussianHMM(2, covariance_type="full", n_iter=300, tol=1e-3,
                        random_state=seed).fit(Xtrain)
        ll = m.score(Xtrain)
        if ll > best_ll:
            best, best_ll = m, ll
    return best


def forward(m, Xs):
    """Probabilites filtrees P(regime_t | donnees jusqu'a t). Passe seulement."""
    sig = np.sqrt([m.covars_[k][0, 0] for k in range(2)])
    B = np.column_stack([norm.pdf(Xs[:, 0], m.means_[k, 0], sig[k])
                         for k in range(2)]) + 1e-300
    alpha = np.zeros((len(Xs), 2))
    a = m.startprob_ * B[0]
    alpha[0] = a / a.sum()
    for t in range(1, len(Xs)):
        a = (alpha[t - 1] @ m.transmat_) * B[t]
        alpha[t] = a / a.sum()
    return alpha


def crise_idx(m):
    return int(np.argmax([m.covars_[k][0, 0] for k in range(2)]))


def hysteresis(p, s_in=SEUIL_IN, s_out=SEUIL_OUT):
    e = np.zeros(len(p), dtype=int)
    for t in range(1, len(p)):
        if e[t - 1] == 0 and p[t] > s_in:
            e[t] = 1
        elif e[t - 1] == 1 and p[t] < s_out:
            e[t] = 0
        else:
            e[t] = e[t - 1]
    return e


# ---------- 1. Walk-forward : probabilites hors echantillon ----------
p_wf = np.full(T, np.nan)
debuts = list(range(TRAIN_MIN, T, REFIT))
print(f"Walk-forward : {len(debuts)} reentrainements, 1re prediction le "
      f"{df.index[TRAIN_MIN].date()}")
for i, t0 in enumerate(debuts):
    t1 = min(t0 + REFIT, T)
    m = ajuster(X[:t0])                      # entrainement sur le PASSE
    a = forward(m, X[:t1])                   # filtre avec parametres du passe
    p_wf[t0:t1] = a[t0:t1, crise_idx(m)]
    vols = np.sqrt([m.covars_[k][0, 0] for k in range(2)]) * np.sqrt(252) / 100
    print(f"  {i+1:2d}/{len(debuts)} {df.index[t0].date()}  "
          f"vol calme={vols.min():.0%}  vol crise={vols.max():.0%}")

# ---------- 2. Reference "in-sample" (etape 6) pour mesurer l'effet du look-ahead ----------
m_full = ajuster(X)
p_is = forward(m_full, X)[:, crise_idx(m_full)]

# ---------- 3. Signaux -> expositions (0 a 1) ----------
oos = ~np.isnan(p_wf)
idx = df.index
expo = pd.DataFrame(index=idx)
expo["Buy & Hold"] = 1.0
expo["VIX > 30"] = (df["vix"] <= 30).astype(float)
expo["Prix > MA200"] = (df["prix"] >= df["ma200"]).astype(float)
expo["HMM binaire (in-sample)"] = 1 - hysteresis(p_is)
etat_wf = np.zeros(T, dtype=int)
etat_wf[oos] = hysteresis(p_wf[oos])
expo["HMM binaire (walk-forward)"] = 1 - etat_wf
expo["HMM graduel (walk-forward)"] = 1 - np.nan_to_num(p_wf, nan=0.0)

# Decalage d'un jour : decision a la cloture de t, appliquee au rendement de t+1
expo_s = expo.shift(1)
expo_s["Buy & Hold"] = 1.0

# ---------- 4. Backtest sur la MEME periode pour tout le monde ----------
debut_oos = idx[TRAIN_MIN + 1]
r_simple = np.exp(df["ret"]) - 1
rows, courbes = {}, {}
for nom in expo.columns:
    e = expo_s[nom].loc[debut_oos:]
    rs = r_simple.loc[debut_oos:]
    turnover = e.diff().abs().fillna(0)
    x = e * rs - turnover * COUT
    cum = (1 + x).cumprod()
    ann = cum.iloc[-1] ** (252 / len(x)) - 1
    vol = x.std() * np.sqrt(252)
    mdd = (cum / cum.cummax() - 1).min()
    rows[nom] = {
        "rend_ann": ann, "vol": vol, "sharpe": ann / vol,
        "max_dd": mdd, "calmar": ann / abs(mdd),
        "investi": e.mean(), "rotation/an": turnover.sum() / (len(x) / 252),
    }
    courbes[nom] = cum

res = pd.DataFrame(rows).T
print(f"\n=== Backtest du {debut_oos.date()} au {idx[-1].date()} "
      f"(cout {COUT*1e4:.0f} bp, signal decale 1 jour) ===")
print(res.round(3).to_string())
res.round(4).to_csv("data/backtest_etape7.csv")

# ---------- 5. Figures ----------
fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
styles = {"Buy & Hold": ("black", 2), "VIX > 30": ("tab:blue", 1),
          "Prix > MA200": ("tab:purple", 1),
          "HMM binaire (in-sample)": ("tab:gray", 1),
          "HMM binaire (walk-forward)": ("tab:orange", 1.5),
          "HMM graduel (walk-forward)": ("tab:green", 2)}
for nom, (c, lw) in styles.items():
    ax[0].plot(courbes[nom], label=nom, color=c, lw=lw)
ax[0].set_yscale("log")
ax[0].legend()
ax[0].set_title("Valeur cumulee (base 1) : walk-forward vs baselines")
ax[1].plot(idx[oos], p_wf[oos], color="red", lw=0.6,
           label="P(crise) walk-forward")
ax[1].fill_between(idx, 0, 1, where=(etat_wf == 1), color="red", alpha=0.2)
ax[1].legend()
ax[1].set_title("Probabilite de crise hors echantillon et zones d'hysteresis")
plt.tight_layout()
plt.savefig("figures/fig7_walkforward.png", dpi=200)
plt.show()
