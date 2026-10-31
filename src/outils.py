"""
Fonctions communes du projet (a importer dans les scripts des etapes 9 et suivantes).
Usage : from outils import *
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal


def charger_features(chemin="data/features.csv"):
    return pd.read_csv(chemin, index_col=0, parse_dates=True)


# ---------- HMM ----------
def ajuster_hmm_runs(X, K, prior_diag=None, n_seeds=10, tol=1e-3, n_iter=1000):
    """Ajuste n_seeds HMM (une par graine). Renvoie la liste (modele, loglik, graine)."""
    P = np.ones((K, K))                       # 1 = prior neutre
    if prior_diag is not None:
        np.fill_diagonal(P, prior_diag)       # sticky : favorise la persistance
    runs = []
    for seed in range(n_seeds):
        try:
            m = GaussianHMM(K, covariance_type="full", n_iter=n_iter, tol=tol,
                            random_state=seed, transmat_prior=P).fit(X)
            runs.append((m, m.score(X), seed))
        except Exception:
            continue                          # ajustement degenere : on l'ignore
    return runs


def ajuster_hmm(X, K, prior_diag=None, n_seeds=10, **kw):
    """Meilleur HMM (vraisemblance maximale) parmi n_seeds initialisations."""
    runs = ajuster_hmm_runs(X, K, prior_diag, n_seeds, **kw)
    return max(runs, key=lambda r: r[1])[0]


def bic_hmm(m, X):
    K, d = m.n_components, X.shape[1]
    n_par = K * (K - 1) + (K - 1) + K * d + K * d * (d + 1) / 2
    return -2 * m.score(X) + n_par * np.log(len(X))


def forward_gaussien(m, X):
    """Probabilites filtrees P(regime_t | donnees jusqu'a t) : passe seulement."""
    K = m.n_components
    B = np.column_stack([multivariate_normal.pdf(X, m.means_[k], m.covars_[k])
                         for k in range(K)]).reshape(len(X), K) + 1e-300
    alpha = np.zeros((len(X), K))
    a = m.startprob_ * B[0]
    alpha[0] = a / a.sum()
    for t in range(1, len(X)):
        a = (alpha[t - 1] @ m.transmat_) * B[t]
        alpha[t] = a / a.sum()
    return alpha


# ---------- Etiquettes de regimes ----------
def reordonner(labels, ret):
    """Renumerote les regimes par volatilite croissante (0 = calme)."""
    lab = np.asarray(labels)
    ks = np.unique(lab)
    ordre = sorted(ks, key=lambda k: np.asarray(ret)[lab == k].std())
    mapping = {k: i for i, k in enumerate(ordre)}
    return np.array([mapping[k] for k in lab]), mapping


def segments(labels):
    """Decoupe en segments consecutifs : (longueurs, regime de chaque segment)."""
    lab = np.asarray(labels)
    bornes = np.r_[0, np.flatnonzero(np.diff(lab) != 0) + 1, len(lab)]
    return np.diff(bornes), lab[bornes[:-1]]


def metriques_regimes(labels, ret):
    lab, ret = np.asarray(labels), np.asarray(ret)
    long, _ = segments(lab)
    K = int(lab.max()) + 1
    vol = [ret[lab == k].std() * np.sqrt(252) if (lab == k).sum() > 1 else np.nan
           for k in range(K)]
    return {"transitions": len(long) - 1,
            "duree_moy_j": len(lab) / len(long),
            "duree_med_j": float(np.median(long)),
            "% segments < 10 j": float((long < 10).mean()),
            "vol_calme": vol[0], "vol_crise": vol[-1],
            "ratio_vol": vol[-1] / vol[0],
            "% jours en crise": float((lab == K - 1).mean())}


def matrice_empirique(labels, K):
    lab = np.asarray(labels)
    A = np.zeros((K, K))
    for a, b in zip(lab[:-1], lab[1:]):
        A[a, b] += 1
    return A / np.maximum(A.sum(axis=1, keepdims=True), 1)


# ---------- Strategie ----------
def hysteresis(p, s_in=0.8, s_out=0.4):
    v = np.asarray(p)
    e = np.zeros(len(v), dtype=int)
    for t in range(1, len(v)):
        if e[t - 1] == 0 and v[t] > s_in:
            e[t] = 1
        elif e[t - 1] == 1 and v[t] < s_out:
            e[t] = 0
        else:
            e[t] = e[t - 1]
    return e


def backtest(expo, ret_log, debut=None, cout=0.0005):
    """expo : Serie 0..1 decidee a la cloture de t, appliquee a t+1 (decalage d'un jour)."""
    r = np.exp(ret_log) - 1
    e = expo.shift(1)
    if debut is not None:
        e, r = e.loc[debut:], r.loc[debut:]
    turn = e.diff().abs().fillna(0)
    x = e * r - turn * cout
    cum = (1 + x).cumprod()
    ann = cum.iloc[-1] ** (252 / len(x)) - 1
    vol = x.std() * np.sqrt(252)
    mdd = (cum / cum.cummax() - 1).min()
    return x, {"rend_ann": ann, "vol": vol, "sharpe": ann / vol, "max_dd": mdd,
               "calmar": ann / abs(mdd), "changements": int((turn > 0).sum())}
