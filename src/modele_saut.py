"""
Statistical Jump Model (Nystrup et al.) : clustering temporel avec penalite sur les changements.

    minimiser   somme_t ||x_t - mu_{s_t}||^2  +  lambda * (nombre de changements d'etat)

lambda = 0  -> K-means (aucune memoire) ; lambda grand -> peu de changements.
Algorithme : on alterne (1) programmation dynamique (Viterbi avec penalite) a centres fixes,
(2) mise a jour des centres = moyenne des points de chaque etat.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def features_saut(ret, hl=10, hl_long=63):
    """3 variables causales (n'utilisent que le passe) : rendement, risque baissier, volatilite."""
    r = pd.Series(np.asarray(ret))
    f = pd.DataFrame({
        "ret_ewma": r.ewm(halflife=hl).mean(),
        "downside": np.sqrt((np.minimum(r, 0) ** 2).ewm(halflife=hl).mean()),
        "vol_long": np.sqrt((r ** 2).ewm(halflife=hl_long).mean()),
    })
    return f.values


def pertes(X, centres):
    return ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)     # T x K


def dp_saut(L, lam):
    """Sequence d'etats de cout minimal (avec retour arriere : utilise tout l'echantillon)."""
    T, K = L.shape
    V = np.zeros((T, K))
    back = np.zeros((T, K), dtype=int)
    V[0] = L[0]
    ar = np.arange(K)
    for t in range(1, T):
        prev = V[t - 1]
        j = prev.argmin()
        change = prev.min() + lam
        reste = prev <= change
        V[t] = L[t] + np.where(reste, prev, change)
        back[t] = np.where(reste, ar, j)
    s = np.zeros(T, dtype=int)
    s[-1] = V[-1].argmin()
    for t in range(T - 1, 0, -1):
        s[t - 1] = back[t, s[t]]
    return s


def etat_online(L, lam):
    """Etat estime en temps reel : argmin du cout cumule a t (n'utilise que le passe)."""
    T, K = L.shape
    V = L[0].copy()
    out = np.zeros(T, dtype=int)
    out[0] = V.argmin()
    for t in range(1, T):
        V = L[t] + np.minimum(V, V.min() + lam)
        V = V - V.min()                   # evite la derive numerique, ne change pas l'argmin
        out[t] = V.argmin()
    return out


def objectif(L, s, lam):
    return L[np.arange(len(s)), s].sum() + lam * (np.diff(s) != 0).sum()


def ajuster_saut(X, K, lam, n_init=3, max_iter=100, centres_init=None, seed=0):
    """Renvoie (etats, centres, objectif)."""
    meilleur = None
    inits = []
    if centres_init is not None:
        inits.append(np.asarray(centres_init, float))
    for i in range(n_init):
        km = KMeans(K, n_init=1, random_state=seed + i).fit(X)
        inits.append(km.cluster_centers_)
    rng = np.random.default_rng(seed)
    for c in inits:
        s_prec = None
        for _ in range(max_iter):
            s = dp_saut(pertes(X, c), lam)
            if s_prec is not None and np.array_equal(s, s_prec):
                break
            c = c.copy()
            for k in range(K):
                m = s == k
                c[k] = X[m].mean(axis=0) if m.any() else X[rng.integers(len(X))]
            s_prec = s
        obj = objectif(pertes(X, c), s, lam)
        if meilleur is None or obj < meilleur[2]:
            meilleur = (s, c, obj)
    return meilleur
