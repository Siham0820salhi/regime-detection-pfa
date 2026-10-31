"""
Etape 11c : les 3 regimes sont-ils bien detectes sur les marches europeens ?
             Combien de temps dure chaque regime ?

Marches : Euro Stoxx 50, DAX (Allemagne), CAC 40 (France), FTSE 100 (Royaume-Uni),
          avec le S&P 500 comme reference. Meme modele partout : HMM gaussien a 3 etats
          sur les rendements journaliers, meilleure vraisemblance parmi 10 initialisations,
          AUCUN reglage specifique a un marche.
Questions : (1) les 3 etats sont-ils distincts, ordonnes en volatilite et stables ?
            (2) quelle est la duree de chaque regime (observee et theorique) ?
            (3) les crises europeennes connues sont-elles classees en regime de crise ?
            (4) les regimes europeens coincident-ils avec ceux du S&P 500 ?
Analyse OFFLINE (descriptive). Les telechargements sont mis en cache dans data/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from outils import *

pd.set_option("display.width", 220)
K = 3
N_SEEDS = int(os.environ.get("N_SEEDS", 10))      # initialisations pour le modele retenu
N_STAB = int(os.environ.get("N_STAB", 20))        # initialisations pour tester la stabilite
REG = ["calme", "moyen", "crise"]
MARCHES = {"S&P 500 (reference)": ("sp", None), "Euro Stoxx 50": ("estoxx", "^STOXX50E"),
           "DAX (Allemagne)": ("dax", "^GDAXI"), "CAC 40 (France)": ("cac", "^FCHI"),
           "FTSE 100 (Royaume-Uni)": ("ftse", "^FTSE")}
FEN = {"Bulle internet 2000-03": ("2000-01-01", "2003-06-30"),
       "Crise 2008": ("2007-06-01", "2009-12-31"),
       "Crise de la zone euro 2011": ("2011-04-01", "2012-03-31"),
       "COVID 2020": ("2020-01-01", "2020-06-30"),
       "Inflation / taux 2022": ("2021-11-01", "2022-12-31"),
       "Chocs 2025": ("2025-01-01", "2025-04-30")}


def charger(cle, symbole):
    if cle == "sp":
        return pd.read_csv("data/sp500_vix_brut.csv", index_col=0, parse_dates=True)["prix"]
    f = f"data/europe_{cle}.csv"
    if os.path.exists(f):
        return pd.read_csv(f, index_col=0, parse_dates=True)["prix"]
    import yfinance as yf
    s = yf.download(symbole, start="2000-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
    if s.empty or len(s) < 1500 or not s.index.is_unique or (s <= 0).any():
        raise ValueError(f"donnees {symbole} vides, trop courtes ou invalides")
    s = s.rename("prix")
    s.to_csv(f, index_label="Date")
    return s


def analyser(prix):
    lr = np.log(prix / prix.shift(1)).dropna()
    ret = lr.values
    X = (ret * 100).reshape(-1, 1)
    runs = ajuster_hmm_runs(X, K, n_seeds=N_SEEDS)
    m, ll, _ = max(runs, key=lambda r: r[1])
    lab, mp = reordonner(m.predict(X), ret)
    # matrice de transition du modele, reordonnee calme -> crise
    A = None
    if len(mp) == K:
        A = np.zeros((K, K))
        for a in range(K):
            for b in range(K):
                A[mp[a], mp[b]] = m.transmat_[a, b]
    # regimes filtres (temps reel)
    filt = forward_gaussien(m, X).argmax(axis=1)
    lab_f = np.array([mp.get(k, k) for k in filt])
    # durees par regime
    long, etats = segments(lab)
    lignes = {}
    for k in range(K):
        d = long[etats == k]
        if len(d) == 0:
            continue
        lignes[REG[k]] = {"% du temps": (lab == k).mean(),
                          "rend. annuel": ret[lab == k].mean() * 252,
                          "vol. annuelle": ret[lab == k].std() * np.sqrt(252),
                          "nb episodes": len(d), "duree moy. (j)": d.mean(),
                          "duree mediane (j)": float(np.median(d)), "duree max (j)": int(d.max()),
                          "duree theorique (j)": 1 / (1 - A[k, k]) if A is not None else np.nan}
    # stabilite entre initialisations
    r20 = ajuster_hmm_runs(X, K, n_seeds=N_STAB)
    lls = np.array([r[1] for r in r20])
    ref = int(np.argmax(lls))
    labs = [reordonner(r[0].predict(X), ret)[0] for r in r20]
    ari = np.array([adjusted_rand_score(labs[ref], l) for l in labs])
    stab = {"ARI moyen": ari.mean(), "ARI min": ari.min(),
            "% runs meme optimum": float((np.abs(lls - lls[ref]) <= 1e-4 * abs(lls[ref])).mean())}
    # choix de K
    bic = {}
    for k in range(2, 5):
        rr = ajuster_hmm_runs(X, k, n_seeds=5)
        if rr:
            mk = max(rr, key=lambda r: r[1])[0]
            lk, _ = reordonner(mk.predict(X), ret)
            bic[k] = {"BIC": bic_hmm(mk, X), "plus petit regime (%)": np.bincount(lk).min() / len(lk) * 100}
    return {"lr": lr, "ret": ret, "lab": pd.Series(lab, index=lr.index),
            "lab_f": lab_f, "A": A, "tab": pd.DataFrame(lignes).T, "stab": stab,
            "bic": pd.DataFrame(bic).T, "prix": prix.loc[lr.index]}


res = {}
for nom, (cle, symbole) in MARCHES.items():
    try:
        prix = charger(cle, symbole)
        res[nom] = analyser(prix)
    except Exception as e:
        print(f"[{nom}] ignore : {e}")
if not res:
    raise SystemExit("Aucun marche exploitable.")

# ---------- 1. Detail par marche ----------
for nom, r in res.items():
    lr = r["lr"]
    print(f"\n{'=' * 78}\n{nom} : {lr.index[0].date()} -> {lr.index[-1].date()} ({len(lr)} jours)\n{'=' * 78}")
    print("Regimes (0 = calme ... 2 = crise) et leur duree :")
    print(r["tab"].round(3).to_string())
    if r["A"] is not None:
        print("\nMatrice de transition du modele (ligne -> colonne) :")
        print(pd.DataFrame(r["A"], index=REG, columns=REG).round(3).to_string())
    nt_s, nt_f = int((np.diff(r["lab"].values) != 0).sum()), int((np.diff(r["lab_f"]) != 0).sum())
    print(f"\nTransitions : smoothed {nt_s} | filtered (temps reel) {nt_f}")
    print("Stabilite sur", N_STAB, "initialisations :", {k: round(v, 3) for k, v in r["stab"].items()})
    print("Choix de K :\n", r["bic"].round(1).to_string())

# ---------- 2. Synthese ----------
print(f"\n{'=' * 78}\nSYNTHESE : 3 regimes sur tous les marches\n{'=' * 78}")
syn = {}
for nom, r in res.items():
    t = r["tab"]
    if len(t) < K:
        continue
    syn[nom] = {"vol calme": t.loc["calme", "vol. annuelle"], "vol moyen": t.loc["moyen", "vol. annuelle"],
                "vol crise": t.loc["crise", "vol. annuelle"], "% crise": t.loc["crise", "% du temps"],
                "dur. med. calme": t.loc["calme", "duree mediane (j)"],
                "dur. med. moyen": t.loc["moyen", "duree mediane (j)"],
                "dur. med. crise": t.loc["crise", "duree mediane (j)"],
                "dur. theo. crise": t.loc["crise", "duree theorique (j)"],
                "ARI moyen": r["stab"]["ARI moyen"],
                "transitions": int((np.diff(r["lab"].values) != 0).sum())}
tab_syn = pd.DataFrame(syn).T
print(tab_syn.round(3).to_string())
tab_syn.round(4).to_csv("data/europe_synthese.csv")

# ---------- 3. Evenements historiques ----------
print("\n=== % des jours classes en CRISE entre le pic et le creux de chaque evenement ===")
h1 = {}
for nom, r in res.items():
    p, lab = r["prix"], r["lab"]
    l = {"base": (lab == K - 1).mean() * 100}
    for ev, (a, b) in FEN.items():
        seg = p.loc[a:b]
        if len(seg) < 20:
            l[ev] = np.nan
            continue
        tp = seg.idxmax()
        tc = seg.loc[tp:].idxmin()
        mk = (lab.index >= tp) & (lab.index <= tc)
        l[ev] = (lab[mk] == K - 1).mean() * 100 if mk.sum() > 5 else np.nan
    h1[nom] = l
tab_h1 = pd.DataFrame(h1).T
print(tab_h1.round(1).to_string())
tab_h1.round(2).to_csv("data/europe_evenements.csv")

# ---------- 4. Les marches sont-ils synchrones ? ----------
if len(res) > 1:
    com = pd.concat({n: r["lab"] for n, r in res.items()}, axis=1, join="inner")
    print(f"\n=== Concordance des regimes entre marches ({len(com)} jours communs) ===")
    noms = list(com.columns)
    ari = pd.DataFrame({b: {a: adjusted_rand_score(com[a], com[b]) for a in noms} for b in noms})
    print("Indice de Rand ajuste (1 = regimes identiques, 0 = independants) :")
    print(ari.round(2).to_string())
    ref = noms[0]
    print(f"\nProbabilite d'etre en crise sur le marche, sachant que le {ref} est en crise :")
    for n in noms[1:]:
        print(f"  {n:26s} {(com[n][com[ref] == K - 1] == K - 1).mean():.0%}"
              f"   (frequence normale de la crise : {(com[n] == K - 1).mean():.0%})")
    ari.round(3).to_csv("data/europe_concordance_ari.csv")

# ---------- 5. Figures ----------
coul = np.array(["green", "orange", "red"])
n = len(res)
fig, ax = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=True)
ax = np.atleast_1d(ax)
for a, (nom, r) in zip(ax, res.items()):
    a.plot(r["prix"], color="black", lw=0.4)
    a.scatter(r["prix"].index, r["prix"], s=2, c=coul[r["lab"].values])
    a.set_yscale("log")
    a.set_title(f"{nom} : HMM 3 regimes (vert calme, orange moyen, rouge crise)", fontsize=10)
plt.tight_layout(); plt.savefig("figures/fig22_regimes_europe.png", dpi=200)

fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 4), squeeze=False)
for a, (nom, r) in zip(ax[0], res.items()):
    if r["A"] is None:
        continue
    im = a.imshow(r["A"], vmin=0, vmax=1, cmap="Blues")
    a.set_xticks(range(K), REG); a.set_yticks(range(K), REG)
    a.set_title(nom.split(" (")[0], fontsize=10)
    for i in range(K):
        for j in range(K):
            a.text(j, i, f"{r['A'][i, j]:.3f}", ha="center", va="center",
                   color="white" if r["A"][i, j] > 0.5 else "black")
plt.suptitle("Matrices de transition (de la ligne vers la colonne)")
plt.tight_layout(); plt.savefig("figures/fig23_transitions_europe.png", dpi=200)

fig, ax = plt.subplots(figsize=(12, 5))
larg = 0.25
for j, (rg, c) in enumerate(zip(REG, coul)):
    v = [r["tab"].loc[rg, "duree mediane (j)"] if rg in r["tab"].index else np.nan for r in res.values()]
    th = [r["tab"].loc[rg, "duree theorique (j)"] if rg in r["tab"].index else np.nan for r in res.values()]
    x = np.arange(len(res)) + (j - 1) * larg
    ax.bar(x, v, larg, color=c, label=f"{rg} (mediane observee)")
    ax.scatter(x, th, color="black", marker="_", s=300, zorder=3,
               label="duree theorique 1/(1-aii)" if j == 0 else None)
ax.set_xticks(range(len(res)), [k.split(" (")[0] for k in res])
ax.set_ylabel("jours de bourse"); ax.legend(fontsize=8)
ax.set_title("Duree des regimes par marche")
plt.tight_layout(); plt.savefig("figures/fig24_durees_europe.png", dpi=200)
plt.show()
print("\nFigures : fig22 (regimes), fig23 (matrices), fig24 (durees)")
