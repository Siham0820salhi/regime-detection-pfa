"""
Etape 11b : robustesse sur d'autres actifs (Nasdaq, or, Bitcoin, MASI du Maroc).

Les hyperparametres (seuils d'hysteresis) sont ceux FIGES a l'etape 11a sur le S&P 500 :
AUCUN reglage specifique a chaque actif. Le HMM K=2 est reentraine chaque annee sur le
passe de CHAQUE actif (walk-forward), signal decale d'un jour, couts de transaction.

MASI : pas de telechargement automatique fiable. Exporte l'historique en CSV depuis
Investing.com (MASI Index, Historical Data, depuis 2005 ou avant) et enregistre-le sous
data/masi.csv. Le script lit aussi les formats "Price" ou "Close"/"Dernier", dates jj/mm/aaaa
ou mm/jj/aaaa. Si le fichier est absent, le MASI est simplement ignore.
Prerequis : etape11a_test_final.py (data/choix_figes.json).
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from outils import *

pd.set_option("display.width", 200)
COUTS = [0.0005, 0.002]

ACTIFS = {"S&P 500": ("sp", None), "Nasdaq": ("nasdaq", "^IXIC"),
          "Euro Stoxx 50": ("estoxx", "^STOXX50E"), "DAX": ("dax", "^GDAXI"),
          "CAC 40": ("cac", "^FCHI"), "FTSE 100": ("ftse", "^FTSE"),
          "Or": ("or", "GC=F"), "Bitcoin": ("btc", "BTC-USD"),
          "MASI (Maroc)": ("masi", None)}

try:
    ch = json.load(open("data/choix_figes.json"))
    S_IN, S_OUT = ch["seuil_entree"], ch["seuil_sortie"]
    print(f"Seuils figes a l'etape 11a : entree {S_IN}, sortie {S_OUT}")
except FileNotFoundError:
    S_IN, S_OUT = 0.8, 0.4
    print("data/choix_figes.json absent : seuils par defaut 0,8 / 0,4 (lance d'abord l'etape 11a)")


# ---------- Chargement des prix ----------
def telecharger(cle, symbole):
    f = f"data/robustesse_{cle}.csv"
    if os.path.exists(f):
        return pd.read_csv(f, index_col=0, parse_dates=True)["prix"]
    import yfinance as yf
    s = yf.download(symbole, start="2000-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
    if s.empty or len(s) < 1500 or not s.index.is_unique or (s <= 0).any():
        raise ValueError(f"donnees {symbole} invalides ou incompletes")
    s = s.rename("prix")
    s.to_csv(f, index_label="Date")
    return s


def lire_masi(chemin="data/masi.csv"):
    d = pd.read_csv(chemin)
    cols = {c.lower().strip(): c for c in d.columns}
    c_date = next(cols[c] for c in cols if c in ("date", "dates"))
    c_prix = next(cols[c] for c in cols if c in ("price", "close", "dernier", "clôture", "cloture", "adj close"))
    prix = d[c_prix].astype(str).str.replace(r"[^\d,.\-]", "", regex=True)
    if prix.str.contains(",").any() and prix.str.contains(r"\.").any():
        prix = prix.str.replace(",", "", regex=False)           # 13,456.78
    else:
        prix = prix.str.replace(",", ".", regex=False)          # 13456,78
    dates = pd.to_datetime(d[c_date], format="%m/%d/%Y", errors="coerce")
    if dates.isna().mean() > 0.2:
        dates = pd.to_datetime(d[c_date], dayfirst=True, errors="coerce")
    s = pd.Series(prix.astype(float).values, index=dates, name="prix").dropna()
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if len(s) < 1500 or (s <= 0).any():
        raise ValueError("fichier MASI trop court ou invalide")
    return s


def charger(nom, cle, symbole):
    if cle == "sp":
        return pd.read_csv("data/sp500_vix_brut.csv", index_col=0, parse_dates=True)["prix"]
    if cle == "masi":
        if not os.path.exists("data/masi.csv"):
            raise FileNotFoundError("data/masi.csv absent")
        return lire_masi()
    return telecharger(cle, symbole)


# ---------- Walk-forward + backtest pour un actif ----------
def analyser(nom, cle, prix):
    ppy = 365 if len(prix) / ((prix.index[-1] - prix.index[0]).days / 365.25) > 300 else 252
    lr = np.log(prix / prix.shift(1)).dropna()
    prix = prix.loc[lr.index]
    ma = prix.rolling(200).mean()
    T = len(lr)
    X = (lr.values * 100).reshape(-1, 1)
    train, refit = 5 * ppy, ppy
    if T < train + 2 * refit:
        raise ValueError(f"historique insuffisant ({T} jours)")
    f = f"data/robustesse_p_{cle}.csv"
    if os.path.exists(f) and len(pd.read_csv(f)) == T:
        p = pd.read_csv(f, index_col=0, parse_dates=True)["p"]
    else:
        print(f"  walk-forward {nom} ({T} jours)...")
        pv = np.full(T, np.nan)
        for t0 in range(train, T, refit):
            t1 = min(t0 + refit, T)
            m = ajuster_hmm(X[:t0], 2, n_seeds=5, n_iter=300)
            crise = int(np.argmax([m.covars_[k][0, 0] for k in range(2)]))
            pv[t0:t1] = forward_gaussien(m, X[:t1])[t0:t1, crise]
        p = pd.Series(pv, index=lr.index, name="p")
        p.to_csv(f, index_label="Date")
    expo = {"Buy & Hold": pd.Series(1.0, index=lr.index),
            "Prix > MA200": (prix >= ma).astype(float),
            f"HMM K=2 ({S_IN}/{S_OUT})": 1 - pd.Series(hysteresis(p.values, S_IN, S_OUT), index=lr.index)}
    debut = lr.index[train + 1]
    r = np.exp(lr) - 1
    out, cums = {}, {}
    for n, e in expo.items():
        for c in COUTS:
            ee = e.shift(1).loc[debut:]
            rr = r.loc[debut:]
            turn = ee.diff().abs().fillna(0)
            x = ee * rr - turn * c
            cum = (1 + x).cumprod()
            ann = cum.iloc[-1] ** (ppy / len(x)) - 1
            vol = x.std() * np.sqrt(ppy)
            mdd = (cum / cum.cummax() - 1).min()
            if c == COUTS[0]:
                out[n] = {"rend_ann": ann, "vol": vol, "sharpe": ann / vol, "max_dd": mdd,
                          "calmar": ann / abs(mdd), "investi": ee.mean(),
                          "chgts/an": (turn > 0).sum() / (len(x) / ppy)}
                cums[n] = cum
            else:
                out[n][f"sharpe @{c * 1e4:.0f}pb"] = ann / vol
    return pd.DataFrame(out).T, cums, debut, lr.index[-1], ppy


resultats, courbes = {}, {}
for nom, (cle, symbole) in ACTIFS.items():
    try:
        prix = charger(nom, cle, symbole)
        tab, cums, d0, d1, ppy = analyser(nom, cle, prix)
    except Exception as e:
        print(f"\n[{nom}] ignore : {e}")
        continue
    resultats[nom], courbes[nom] = tab, cums
    print(f"\n=== {nom} : {d0.date()} -> {d1.date()} ({ppy} periodes/an, cout {COUTS[0] * 1e4:.0f} pb) ===")
    print(tab.round(3).to_string())

if resultats:
    print("\n=== Synthese : le HMM reduit-il le risque ? (HMM contre Buy & Hold) ===")
    lignes = {}
    for nom, tab in resultats.items():
        h = tab.index[2]
        lignes[nom] = {"drawdown B&H": tab.loc["Buy & Hold", "max_dd"], "drawdown HMM": tab.loc[h, "max_dd"],
                       "vol B&H": tab.loc["Buy & Hold", "vol"], "vol HMM": tab.loc[h, "vol"],
                       "Sharpe B&H": tab.loc["Buy & Hold", "sharpe"], "Sharpe HMM": tab.loc[h, "sharpe"],
                       "Sharpe MA200": tab.loc["Prix > MA200", "sharpe"],
                       "Calmar B&H": tab.loc["Buy & Hold", "calmar"], "Calmar HMM": tab.loc[h, "calmar"]}
    syn = pd.DataFrame(lignes).T
    print(syn.round(3).to_string())
    syn.round(4).to_csv("data/robustesse_synthese.csv")
    n = len(resultats)
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for a, (nom, cums) in zip(ax[0], courbes.items()):
        for k, c in zip(cums, ["black", "tab:purple", "tab:orange"]):
            a.plot(cums[k], color=c, lw=1.4, label=k)
        a.set_yscale("log"); a.set_title(nom); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig("figures/fig21_robustesse.png", dpi=200)
    plt.show()
    print("\nFigure : fig21_robustesse.png ; tableaux : data/robustesse_synthese.csv")
