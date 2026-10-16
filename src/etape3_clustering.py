import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

df = pd.read_csv("data/features.csv", index_col=0, parse_dates=True)

# ---------- 1. Variables du clustering ----------
cols = ["ret", "vol20", "vol60", "mom60", "drawdown"]
X = StandardScaler().fit_transform(df[cols])

# ---------- 2. Choix du nombre de régimes K (BIC) ----------
print("=== Choix de K (GMM) ===")
bics = {}
for k in range(2, 6):
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          n_init=5, random_state=42).fit(X)
    bics[k] = gmm.bic(X)
    print(f"K={k}  BIC={bics[k]:.0f}  AIC={gmm.aic(X):.0f}")

# ---------- 3. Modèles avec K=3 ----------
# Choix exploratoire de 3 régimes pour faciliter
# l'interprétation financière : calme, intermédiaire et forte volatilité.
# Le BIC sera étudié plus largement pour comparer plusieurs valeurs de K.
K = 3
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
gmm = GaussianMixture(n_components=K, covariance_type="full",
                      n_init=5, random_state=42).fit(X)
df["reg_km"] = km.labels_
df["reg_gmm"] = gmm.predict(X)

# ---------- 4. Réordonner les régimes par volatilité croissante ----------
def reordonner(col):
    ordre = df.groupby(col)["vol20"].mean().sort_values().index
    mapping = {old: new for new, old in enumerate(ordre)}
    df[col] = df[col].map(mapping)

reordonner("reg_km")
reordonner("reg_gmm")

# ---------- 5. Statistiques ----------
for nom in ["reg_km", "reg_gmm"]:
    print(f"\n=== {nom} (0 = calme, {K-1} = crise) ===")
    g = df.groupby(nom)["ret"]
    res = pd.DataFrame({
        "jours": g.count(),
        "rend_annuel": g.mean() * 252,
        "vol_annuelle": g.std() * np.sqrt(252),
    })
    print(res.round(3))
    print("Nombre de changements :", (df[nom].diff().abs() > 0).sum())
    duree = len(df) / ((df[nom].diff().abs() > 0).sum() + 1)
    print(f"Durée moyenne d'un régime : {duree:.1f} jours")

df.to_csv("data/regimes_clustering.csv")

# ---------- 6. Figure ----------
couleurs = {0: "green", 1: "orange", 2: "red"}
fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
for a, col, titre in zip(ax, ["reg_km", "reg_gmm"], ["K-means", "GMM"]):
    a.plot(df["prix"], color="black", lw=0.5)
    for r in range(K):
        a.scatter(df.index[df[col] == r], df["prix"][df[col] == r],
                  s=2, color=couleurs[r], label=f"Régime {r}")
    a.set_yscale("log")
    a.set_title(f"S&P 500 coloré par régime : {titre}")
    a.legend(markerscale=5)
plt.tight_layout()
plt.savefig("figures/fig3_clustering.png", dpi=200)
plt.show()