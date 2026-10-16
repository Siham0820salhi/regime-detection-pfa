import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

# ---------- 1. Données figées ----------
df0 = pd.read_csv("data/sp500_vix_brut.csv", index_col=0, parse_dates=True)
prix = df0["prix"]

# ---------- 2. Rendements log ----------
r = np.log(prix / prix.shift(1)).dropna()

# ---------- 3. Statistiques descriptives ----------
print("\n=== Statistiques des rendements journaliers ===")
print(f"Moyenne annualisée : {r.mean()*252:.2%}")
print(f"Volatilité annualisée : {r.std()*np.sqrt(252):.2%}")
print(f"Skewness (asymétrie) : {stats.skew(r):.3f}")
print(f"Kurtosis (excès) : {stats.kurtosis(r):.3f}   (0 = loi normale)")
jb_stat, jb_p = stats.jarque_bera(r)
print(f"Test Jarque-Bera p-value : {jb_p:.2e}  (< 0.05 => non normal)")

# ---------- 4. Figure des faits stylisés ----------
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# (a) Prix
ax[0, 0].plot(prix, color="navy")
ax[0, 0].set_yscale("log")
ax[0, 0].set_title("(a) S&P 500 (échelle log)")

# (b) Rendements : volatility clustering
ax[0, 1].plot(r, color="steelblue", lw=0.6)
ax[0, 1].set_title("(b) Rendements journaliers : volatility clustering")

# (c) Histogramme vs loi normale : queues épaisses
ax[1, 0].hist(r, bins=150, density=True, alpha=0.6, color="steelblue",
              label="Rendements observés")
x = np.linspace(r.min(), r.max(), 500)
ax[1, 0].plot(x, stats.norm.pdf(x, r.mean(), r.std()), "r-", lw=2,
              label="Loi normale")
ax[1, 0].set_yscale("log")
ax[1, 0].set_ylim(1e-3, None)
ax[1, 0].set_title("(c) Queues épaisses (échelle log)")
ax[1, 0].legend()

# (d) ACF des rendements au carré
plot_acf(r**2, lags=60, ax=ax[1, 1])
ax[1, 1].set_title("(d) ACF des rendements au carré")

plt.tight_layout()
plt.savefig("figures/fig1_faits_stylises.png", dpi=200)
plt.show()