"""
Etape 9a : comparaison equitable des modeles, stabilite (H2), convergence,
sensibilite au nombre de regimes K.

Tous les modeles (K-means, GMM, HMM, HMM sticky) utilisent les MEMES variables
(rendement + volatilite 20 j, standardises), les MEMES dates et K = 3.
Un HMM sur rendements seuls est ajoute comme specification de reference.
Analyse OFFLINE (parametres estimes sur toute la periode) : descriptive, pas temps reel.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from hmmlearn.hmm import GaussianHMM
from outils import *

K = 3
PRIOR_STICKY = 2000          # 200 etait trop faible a l'etape 5
N_SEEDS_STAB = 20
pd.set_option("display.width", 200)


class Compteur(logging.Handler):
    """Compte les avertissements 'Model is not converging' de hmmlearn."""
    def __init__(self):
        super().__init__()
        self.n = 0

    def emit(self, record):
        if "not converging" in record.getMessage():
            self.n += 1


cpt = Compteur()
logging.getLogger("hmmlearn").addHandler(cpt)

df = charger_features()
ret = df["ret"].values
Xs = StandardScaler().fit_transform(df[["ret", "vol20"]])     # memes variables pour tous
Xr = (df["ret"].values * 100).reshape(-1, 1)                  # rendements seuls

# ---------- 1. Ajustement des modeles (meilleure initialisation) ----------
print("Ajustement des modeles...")
km = KMeans(K, n_init=10, random_state=42).fit(Xs)
gmm = GaussianMixture(K, covariance_type="full", n_init=10, random_state=42).fit(Xs)
h = ajuster_hmm(Xs, K)
h_s = ajuster_hmm(Xs, K, prior_diag=PRIOR_STICKY)
h_r = ajuster_hmm(Xr, K)

noms = {"km": "K-means", "gmm": "GMM", "hmm": "HMM (ret+vol20)",
        "hmm_sticky": f"HMM sticky {PRIOR_STICKY} (ret+vol20)", "hmm_ret3": "HMM (ret seul)"}
brut = {"km": km.labels_, "gmm": gmm.predict(Xs), "hmm": h.predict(Xs),
        "hmm_sticky": h_s.predict(Xs), "hmm_ret3": h_r.predict(Xr)}
labels = {k: reordonner(v, ret)[0] for k, v in brut.items()}

# ---------- 2. Comparaison equitable ----------
print(f"\n=== 1. Comparaison equitable (K={K}, memes dates : {df.index[0].date()} -> "
      f"{df.index[-1].date()}, {len(df)} jours) ===")
tab = pd.DataFrame({noms[k]: metriques_regimes(l, ret) for k, l in labels.items()}).T
print(tab.round(3).to_string())
tab.round(4).to_csv("data/comparaison_equitable.csv")

# ---------- 3. Smoothed vs filtered (HMM K=2 sur rendements) ----------
h2 = ajuster_hmm(Xr, 2)
crise2 = int(np.argmax([h2.covars_[k][0, 0] for k in range(2)]))
lisse = (h2.predict_proba(Xr).argmax(axis=1) == crise2).astype(int)
filtre = (forward_gaussien(h2, Xr).argmax(axis=1) == crise2).astype(int)
print("\n=== 2. HMM K=2 : smoothed vs filtered ===")
print(f"Transitions smoothed : {int((np.diff(lisse) != 0).sum())}   "
      f"filtered : {int((np.diff(filtre) != 0).sum())}")
print(f"Accord jour par jour : {(lisse == filtre).mean():.1%}")
print(f"Jours en crise  smoothed : {lisse.mean():.1%}   filtered : {filtre.mean():.1%}")

out = pd.DataFrame({k: v for k, v in labels.items()}, index=df.index)
out["hmm_k2_smooth"], out["hmm_k2_filt"] = lisse, filtre
out.to_csv("data/regimes_A.csv", index_label="Date")

# ---------- 4. Stabilite entre initialisations (H2) + convergence ----------
print(f"\n=== 3. Stabilite sur {N_SEEDS_STAB} initialisations (une seule initialisation par run) ===")


def run_hmm(X, prior, seed):
    P = np.ones((K, K))
    if prior:
        np.fill_diagonal(P, prior)
    n0 = cpt.n
    m = GaussianHMM(K, covariance_type="full", n_iter=1000, tol=1e-3,
                    random_state=seed, transmat_prior=P).fit(X)
    info = {"converge": bool(m.monitor_.converged), "iter": int(m.monitor_.iter),
            "avertissements": cpt.n - n0}
    return m.predict(X), m.score(X), info


generateurs = {
    "km": lambda s: (KMeans(K, n_init=1, random_state=s).fit(Xs), None),
    "gmm": lambda s: (GaussianMixture(K, n_init=1, random_state=s).fit(Xs), None),
    "hmm": lambda s: run_hmm(Xs, None, s),
    "hmm_sticky": lambda s: run_hmm(Xs, PRIOR_STICKY, s),
    "hmm_ret3": lambda s: run_hmm(Xr, None, s),
}
stab, conv = {}, {}
for cle, gen in generateurs.items():
    labs, lls, infos = [], [], []
    for s in range(N_SEEDS_STAB):
        res = gen(s)
        if cle == "km":
            m = res[0]; l, ll, info = m.labels_, -m.inertia_, None
        elif cle == "gmm":
            m = res[0]; l, ll, info = m.predict(Xs), m.score(Xs) * len(Xs), None
        else:
            l, ll, info = res
        labs.append(reordonner(l, ret)[0]); lls.append(ll); infos.append(info)
    lls = np.array(lls)
    ref = int(np.argmax(lls))
    ari = np.array([adjusted_rand_score(labs[ref], l) for l in labs])
    meme_opt = np.abs(lls - lls[ref]) <= 1e-4 * abs(lls[ref])
    ntr = np.array([int((np.diff(l) != 0).sum()) for l in labs])
    stab[noms[cle]] = {"ARI moyen vs meilleur": ari.mean(), "ARI min": ari.min(),
                       "% runs meme optimum": meme_opt.mean(),
                       "transitions (moy)": ntr.mean(), "transitions (ecart-type)": ntr.std(),
                       "transitions (min-max)": f"{ntr.min()}-{ntr.max()}"}
    if infos[0] is not None:
        conv[noms[cle]] = {"% converges (tol 1e-3)": np.mean([i["converge"] for i in infos]),
                           "iterations moy": np.mean([i["iter"] for i in infos]),
                           "iterations max": max(i["iter"] for i in infos),
                           "avertissements 'not converging'": sum(i["avertissements"] for i in infos)}
tab_s = pd.DataFrame(stab).T
print(tab_s.to_string())
print("Lecture : ARI = 1 -> memes regimes quelle que soit l'initialisation ; "
      "'meme optimum' = part des runs atteignant la meilleure vraisemblance.")
tab_s.to_csv("data/stabilite_initialisations.csv")

print("\n=== 4. Convergence de l'algorithme EM (HMM) ===")
tab_c = pd.DataFrame(conv).T
print(tab_c.to_string())
print("Un avertissement 'not converging' signale une baisse minime de vraisemblance "
      "a une iteration : sans gravite si le run converge quand meme.")
tab_c.to_csv("data/convergence_hmm.csv")

# ---------- 5. Sensibilite au nombre de regimes K ----------
print("\n=== 5. Sensibilite a K (HMM, meilleure initialisation sur 10) ===")
lignes = {}
for nom_spec, X in [("ret seul", Xr), ("ret+vol20", Xs)]:
    for k in range(2, 6):
        runs = ajuster_hmm_runs(X, k, n_seeds=10)
        if not runs:
            continue
        m, ll, _ = max(runs, key=lambda r: r[1])
        l, _ = reordonner(m.predict(X), ret)
        long, _ = segments(l)
        tailles = np.bincount(l, minlength=k) / len(l)
        ari5 = [adjusted_rand_score(l, reordonner(r[0].predict(X), ret)[0])
                for r in runs[:5]]
        vols = [ret[l == j].std() * np.sqrt(252) for j in range(k)]
        lignes[f"{nom_spec} K={k}"] = {
            "loglik": ll, "BIC": bic_hmm(m, X), "transitions": len(long) - 1,
            "duree_moy_j": len(l) / len(long),
            "plus petit regime (%)": tailles.min(),
            "ARI (5 inits)": np.mean(ari5),
            "vols (%)": "/".join(f"{v * 100:.0f}" for v in vols)}
tab_k = pd.DataFrame(lignes).T
print(tab_k.to_string())
print("Lecture : BIC qui baisse sans minimum = K non determine statistiquement ; "
      "on regarde aussi la taille du plus petit regime et la stabilite (ARI).")
tab_k.to_csv("data/sensibilite_K.csv")
print("\nFichiers ecrits : data/regimes_A.csv + 5 tableaux CSV dans data/")
