import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

df = pd.read_csv("data/features.csv", index_col=0, parse_dates=True)

# ---------- 1. Observations : rendement + volatilité 20j ----------
# On garde peu de variables pour un HMM plus stable et interprétable
obs = df[["ret", "vol20"]].copy()
obs["ret"] = obs["ret"] * 100          # en %
obs["vol20"] = obs["vol20"] * 100
X = obs.values

# ---------- 2. Choix de K par BIC ----------
def bic_hmm(model, X):
    k = model.n_components
    d = X.shape[1]
    n_params = k * (k - 1) + (k - 1) + k * d + k * d * (d + 1) / 2
    return -2 * model.score(X) + n_params * np.log(len(X))

print("=== Choix de K (HMM) ===")
modeles = {}
for k in range(2, 5):
    best, best_ll = None, -np.inf
    for seed in range(10):           # 10 initialisations, on garde la meilleure
        m = GaussianHMM(n_components=k, covariance_type="full",
                        n_iter=500, random_state=seed).fit(X)
        ll = m.score(X)
        if ll > best_ll:
            best, best_ll = m, ll
    modeles[k] = best
    print(f"K={k}  loglik={best_ll:.0f}  BIC={bic_hmm(best, X):.0f}")

# ---------- 3. Modèle final K=3 ----------
K = 3
model = modeles[K]

# Régimes : smoothed (Viterbi) et probabilités filtrées
df["reg_hmm"] = model.predict(X)
proba_smooth = model.predict_proba(X)

# Réordonner par volatilité croissante
ordre = df.groupby("reg_hmm")["vol20"].mean().sort_values().index
mapping = {old: new for new, old in enumerate(ordre)}
df["reg_hmm"] = df["reg_hmm"].map(mapping)

# ---------- 4. Statistiques ----------
g = df.groupby("reg_hmm")["ret"]
res = pd.DataFrame({
    "jours": g.count(),
    "rend_annuel": g.mean() * 252,
    "vol_annuelle": g.std() * np.sqrt(252),
})
print("\n=== HMM (0 = calme, 2 = crise) ===")
print(res.round(3))
nb = (df["reg_hmm"].diff().abs() > 0).sum()
print(f"Nombre de changements : {nb}")
print(f"Durée moyenne d'un régime : {len(df)/(nb+1):.1f} jours")

# ---------- 5. Matrice de transition (réordonnée) ----------
A = model.transmat_
ordre_l = list(ordre)
A = A[np.ix_(ordre_l, ordre_l)]
print("\n=== Matrice de transition ===")
print(pd.DataFrame(A, index=["calme", "moyen", "crise"],
                   columns=["calme", "moyen", "crise"]).round(3))
print("\nDurée moyenne théorique (jours) = 1/(1-aii) :")
for i, n in enumerate(["calme", "moyen", "crise"]):
    print(f"  {n} : {1/(1-A[i,i]):.1f}")

df.to_csv("data/regimes_hmm.csv")

# ---------- 6. Figure ----------
couleurs = {0: "green", 1: "orange", 2: "red"}
fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
ax[0].plot(df["prix"], color="black", lw=0.5)
for r in range(K):
    m = df["reg_hmm"] == r
    ax[0].scatter(df.index[m], df["prix"][m], s=2, color=couleurs[r],
                  label=f"Régime {r}")
ax[0].set_yscale("log")
ax[0].set_title("S&P 500 coloré par régime : HMM gaussien (K=3)")
ax[0].legend(markerscale=5)

for r in range(K):
    ax[1].plot(df.index, proba_smooth[:, ordre_l[r]],
               color=couleurs[r], lw=0.7, label=f"P(régime {r})")
ax[1].set_title("Probabilités de régime (smoothed)")
ax[1].legend()
plt.tight_layout()
plt.savefig("figures/fig4_hmm.png", dpi=200)
plt.show()