import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal

df = pd.read_csv("data/features.csv", index_col=0, parse_dates=True)
X = (df["ret"] * 100).values.reshape(-1, 1)   # rendements seuls, en %

def ajuster(K, prior_diag=None, n_seeds=10):
    best, best_ll = None, -np.inf
    P = np.ones((K, K))                 # 1.0 = prior neutre (aucun effet)
    if prior_diag is not None:
        np.fill_diagonal(P, prior_diag) # sticky : favorise de rester dans l'état
    for seed in range(n_seeds):
        m = GaussianHMM(n_components=K, covariance_type="full",
                        n_iter=500, random_state=seed, transmat_prior=P)
        m.fit(X)
        ll = m.score(X)
        if ll > best_ll:
            best, best_ll = m, ll
    return best, best_ll

def probas_filtrees(m, X):
    """Algorithme forward : P(régime_t | observations jusqu'à t) seulement."""
    T, K = len(X), m.n_components
    B = np.column_stack([
        multivariate_normal.pdf(X, m.means_[k], m.covars_[k]) for k in range(K)
    ]) + 1e-300
    alpha = np.zeros((T, K))
    a = m.startprob_ * B[0]
    alpha[0] = a / a.sum()
    for t in range(1, T):
        a = (alpha[t - 1] @ m.transmat_) * B[t]
        alpha[t] = a / a.sum()
    return alpha

def nb_changements(s):
    return int((np.diff(s) != 0).sum())

variantes = {
    "ret K=2":        (2, None),
    "ret K=3":        (3, None),
    "ret K=3 sticky": (3, 200),
}

lignes, sauvegarde = [], {}
for nom, (K, prior) in variantes.items():
    m, ll = ajuster(K, prior)
    ordre = np.argsort([m.covars_[k][0, 0] for k in range(K)])   # calme -> agité
    lisse = m.predict_proba(X)[:, ordre]
    filtre = probas_filtrees(m, X)[:, ordre]
    s_lisse = lisse.argmax(axis=1)
    s_filtre = filtre.argmax(axis=1)

    vols = [np.sqrt(m.covars_[k][0, 0]) * np.sqrt(252) / 100 for k in ordre]
    lignes.append({
        "variante": nom,
        "loglik": round(ll),
        "chang. smoothed": nb_changements(s_lisse),
        "chang. filtered": nb_changements(s_filtre),
        "vol par régime": [f"{v:.0%}" for v in vols],
    })
    A = m.transmat_[np.ix_(ordre, ordre)]
    print(f"\n--- {nom} : matrice de transition ---")
    print(pd.DataFrame(A).round(3))
    sauvegarde[nom] = (s_lisse, s_filtre)

print("\n=== Comparaison des variantes ===")
print(pd.DataFrame(lignes).to_string(index=False))

# Figure : K=3 sticky, smoothed vs filtered
s_lisse, s_filtre = sauvegarde["ret K=3 sticky"]
couleurs = np.array(["green", "orange", "red"])
fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
for a, s, t in zip(ax, [s_lisse, s_filtre],
                   ["Smoothed (utilise le futur)", "Filtered (temps réel)"]):
    a.plot(df["prix"], color="black", lw=0.5)
    a.scatter(df.index, df["prix"], s=2, c=couleurs[s])
    a.set_yscale("log")
    a.set_title(f"HMM sur rendements, K=3 sticky : {t}")
plt.tight_layout()
plt.savefig("figures/fig5_hmm_filtre.png", dpi=200)
plt.show()