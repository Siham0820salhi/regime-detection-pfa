"""
Prepare les donnees de l'application Streamlit : tout est ecrit dans app/donnees/.
L'application ne lit QUE ce dossier : elle est rapide et deployable telle quelle.

A lancer une fois, apres les etapes 1 a 11 :   python src/preparer_app.py
Temps : 2 a 4 minutes (HMM a 3 regimes sur 5 marches).
"""
import glob
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from outils import *

N_SEEDS = int(os.environ.get("N_SEEDS", 10))
K = 3
OUT = Path("app/donnees")
OUT.mkdir(parents=True, exist_ok=True)

MARCHES = {"sp": "S&P 500", "estoxx": "Euro Stoxx 50", "dax": "DAX (Allemagne)",
           "cac": "CAC 40 (France)", "ftse": "FTSE 100 (Royaume-Uni)"}
ACTIFS = {**MARCHES, "nasdaq": "Nasdaq", "or": "Or", "btc": "Bitcoin"}


def prix_de(cle):
    """Serie de prix d'un actif depuis les fichiers figes de data/ (jamais de telechargement)."""
    if cle == "sp":
        d = pd.read_csv("data/sp500_vix_brut.csv", index_col=0, parse_dates=True)
        return d["prix"], d["vix"] if "vix" in d else None
    for f in (f"data/europe_{cle}.csv", f"data/robustesse_{cle}.csv"):
        if os.path.exists(f):
            return pd.read_csv(f, index_col=0, parse_dates=True)["prix"], None
    return None, None


# ---------------------------------------------------------------- 1. Regimes (K=3)
meta = {}
for cle, nom in MARCHES.items():
    prix, vix = prix_de(cle)
    if prix is None:
        print(f"[{nom}] donnees absentes : lance etape11c_europe.py -> ignore")
        continue
    lr = np.log(prix / prix.shift(1)).dropna()
    ret = lr.values
    X = (ret * 100).reshape(-1, 1)
    runs = ajuster_hmm_runs(X, K, n_seeds=N_SEEDS)
    m, ll, _ = max(runs, key=lambda r: r[1])
    lab_s, mp = reordonner(m.predict(X), ret)
    if len(mp) < K:
        print(f"[{nom}] modele degenere (moins de {K} etats) -> ignore")
        continue
    inv = {v: k for k, v in mp.items()}
    cols = [inv[r] for r in range(K)]                    # ordre calme -> crise
    ps = m.predict_proba(X)[:, cols]                     # probabilites lissees (futur inclus)
    pf = forward_gaussien(m, X)[:, cols]                 # probabilites filtrees (temps reel)
    A = np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            A[mp[a], mp[b]] = m.transmat_[a, b]
    out = pd.DataFrame({"prix": prix.loc[lr.index].values, "ret": ret,
                        "lab_s": lab_s, "lab_f": pf.argmax(axis=1),
                        "ps0": ps[:, 0], "ps1": ps[:, 1], "ps2": ps[:, 2],
                        "pf0": pf[:, 0], "pf1": pf[:, 1], "pf2": pf[:, 2]}, index=lr.index)
    out["vol20"] = out["ret"].rolling(20).std() * np.sqrt(252)
    out["drawdown"] = out["prix"] / out["prix"].cummax() - 1
    if vix is not None:
        out["vix"] = vix.reindex(out.index)
    out.round(5).to_csv(OUT / f"regimes_{cle}.csv", index_label="Date")
    meta[cle] = {"nom": nom, "A": A.round(5).tolist(), "loglik": float(ll), "n": int(len(lr)),
                 "debut": str(lr.index[0].date()), "fin": str(lr.index[-1].date()),
                 "mu_pct": [float(ret[lab_s == k].mean() * 100) for k in range(K)],
                 "vol_ann": [float(ret[lab_s == k].std() * np.sqrt(252)) for k in range(K)]}
    print(f"[{nom}] OK : {len(lr)} jours, vol par regime "
          f"{[round(v * 100) for v in meta[cle]['vol_ann']]} %")
json.dump(meta, open(OUT / "modeles.json", "w"), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- 2. Donnees du backtest
CANDIDATS_P = {"sp": sorted(glob.glob("data/p_hmm_split_*.csv"), reverse=True)
                     + ["data/robustesse_p_sp.csv"]}
bt_meta = {}
for cle, nom in ACTIFS.items():
    prix, vix = prix_de(cle)
    pf = [f for f in CANDIDATS_P.get(cle, [f"data/robustesse_p_{cle}.csv"]) if os.path.exists(f)]
    if prix is None or not pf:
        print(f"[backtest {nom}] prix ou probabilites absents -> ignore")
        continue
    p = pd.read_csv(pf[0], index_col=0, parse_dates=True).iloc[:, 0].rename("p")
    d = pd.concat([prix.rename("prix"), p], axis=1, join="inner").sort_index()
    if vix is not None:
        d["vix"] = vix.reindex(d.index)
    d.round(6).to_csv(OUT / f"bt_{cle}.csv", index_label="Date")
    ppy = 365 if (len(d) / ((d.index[-1] - d.index[0]).days / 365.25)) > 300 else 252
    bt_meta[cle] = {"nom": nom, "ppy": ppy, "source_p": os.path.basename(pf[0])}
    print(f"[backtest {nom}] OK ({os.path.basename(pf[0])})")
json.dump(bt_meta, open(OUT / "bt_meta.json", "w"), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- 3. Tableaux des etapes precedentes
A_COPIER = ["comparaison_equitable.csv", "stabilite_initialisations.csv", "convergence_hmm.csv",
            "sensibilite_K.csv", "jump_balayage_lambda.csv", "concordance_dates_tol10.csv",
            "concordance_dates_tol20.csv", "pelt_segments.csv", "regimes_A.csv", "regimes_B.csv",
            "europe_synthese.csv", "europe_evenements.csv", "europe_concordance_ari.csv",
            "eval_delais.csv", "eval_signaux_nber.csv", "eval_auc.csv", "eval_bootstrap.csv",
            "resultats_finaux_test.csv", "resultats_finaux_dev.csv", "resultats_finaux_couts.csv",
            "resultats_finaux_bootstrap.csv", "robustesse_synthese.csv", "h1_evenements.csv",
            "sensibilite_seuils_sharpe.csv", "sensibilite_couts.csv", "transitions_hmm.csv",
            "episodes_crise.csv", "choix_figes.json"]
manquants = []
for f in A_COPIER:
    if os.path.exists(f"data/{f}"):
        shutil.copy(f"data/{f}", OUT / f)
    else:
        manquants.append(f)
if os.path.exists("data/features.csv"):
    shutil.copy("data/features.csv", OUT / "features_sp.csv")
else:
    manquants.append("features.csv")
if os.path.exists("data/usrec.csv"):
    u = pd.read_csv("data/usrec.csv", index_col=0, parse_dates=True)
    u.loc["1999":].to_csv(OUT / "usrec.csv", index_label="Date")
else:
    manquants.append("usrec.csv")
if manquants:
    print("\nFichiers optionnels absents (les pages concernees afficheront un message) :")
    for f in manquants:
        print("  -", f)
print(f"\nTermine : {len(list(OUT.glob('*')))} fichiers dans {OUT}/")
