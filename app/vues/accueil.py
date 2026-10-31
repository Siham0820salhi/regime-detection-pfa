import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import streamlit as st
from outils_app import *
from outils_app import json_

entete("Detection des changements de regime des marches financiers par IA", "Hidden Markov Models, jump models et detection de ruptures appliques a 5 marches : combien de temps durent les regimes, et que change leur detection pour la gestion du risque ?", "PFA | Sujet 5 | Finance quantitative et IA")
st.markdown("Un **regime de marche** est une periode ou le marche garde des proprietes statistiques "
            "stables (niveau de risque, rendement moyen). Cette application montre comment des modeles "
            "d'intelligence artificielle (**HMM**, jump model, detection de ruptures) identifient "
            "ces regimes, combien de temps ils durent, et ce que cela change pour la gestion du risque.")

d = serie_temp("regimes_sp.csv")
if d is None:
    absent("regimes_sp.csv")
    st.stop()

# ----------------------------------------------------------------- regime actuel
st.subheader("Regime actuel du S&P 500")
dernier = d.iloc[-1]
k = int(dernier["lab_f"])
couleur, nom_reg = COUL[k], REG[k]
long, etats = segments(d["lab_f"].values)
depuis = int(long[-1])
mediane_hist = stats_regimes(d["ret"], d["lab_s"]).loc[nom_reg, "Duree mediane (j)"]
descr = {0: "Marche calme : faible volatilite, tendance generalement haussiere.",
         1: "Marche agite : volatilite intermediaire, phase de correction ou d'incertitude.",
         2: "Marche en crise : volatilite tres elevee, risque de pertes importantes."}[k]

c1, c2 = st.columns([1.35, 1])
with c1:
    st.markdown(
        f"""<div style='display:flex;align-items:center;gap:22px;padding:14px 18px;border-radius:14px;
        background:linear-gradient(145deg,rgba(30,41,66,.78),rgba(15,22,38,.78));border:1px solid rgba(148,163,184,.18);box-shadow:0 10px 28px rgba(0,0,0,.35)'>
        <div style='min-width:92px;height:92px;border-radius:50%;background:{couleur};
        box-shadow:0 0 28px {couleur},inset 0 0 18px rgba(255,255,255,.25)'></div>
        <div><div style='font-size:2.1rem;font-weight:700;line-height:1.1'>Regime {nom_reg.lower()}</div>
        <div style='margin-top:6px'>{descr}</div>
        <div style='margin-top:6px;opacity:0.75'>Donnees au <b>{d.index[-1]:%d/%m/%Y}</b> |
        regime en cours depuis <b>{depuis} jours de bourse</b> (mediane historique de ce regime :
        {mediane_hist:.0f} jours)</div></div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown("**Probabilites du regime (estimees en temps reel)**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Calme", f"{dernier['pf0']:.0%}")
    m2.metric("Moyen", f"{dernier['pf1']:.0%}")
    m3.metric("Crise", f"{dernier['pf2']:.0%}")
    st.caption("Probabilites *filtrees* : elles n'utilisent que le passe. Un regime est reellement "
               "confirme apres plusieurs jours : ne pas sur-interpreter un seul jour.")

# ----------------------------------------------------------------- indicateurs
st.subheader("Indicateurs de marche")
prix = d["prix"]
ma200 = prix.rolling(200).mean().iloc[-1]
fin_annee = prix[prix.index.year < d.index[-1].year]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Dernier cours", f"{prix.iloc[-1]:,.0f}", f"{prix.iloc[-1] / prix.iloc[-2] - 1:+.2%} (1 jour)")
k2.metric("Sur 1 mois", f"{prix.iloc[-1] / prix.iloc[-22] - 1:+.1%}")
k3.metric("Depuis le 1er janvier",
          f"{prix.iloc[-1] / fin_annee.iloc[-1] - 1:+.1%}" if len(fin_annee) else "n/d")
k4.metric("Volatilite 20 j (annualisee)", f"{dernier['vol20']:.1%}")
k5.metric("Distance au plus haut", f"{dernier['drawdown']:.1%}")
k6, k7, k8, k9, k10 = st.columns(5)
if "vix" in d and pd.notna(dernier.get("vix")):
    k6.metric("VIX (indice de la peur)", f"{dernier['vix']:.1f}")
k7.metric("Ecart a la MA200", f"{prix.iloc[-1] / ma200 - 1:+.1%}" if pd.notna(ma200) else "n/d")

# ----------------------------------------------------------------- signal de strategie
bt = serie_temp("bt_sp.csv")
choix = json_("choix_figes.json") or {}
s_in, s_out = choix.get("seuil_entree", 0.9), choix.get("seuil_sortie", 0.3)
if bt is not None and bt["p"].notna().any():
    et = hysteresis(bt["p"].values, s_in, s_out)
    p_der = bt["p"].dropna().iloc[-1]
    st.subheader("Signal de la strategie de gestion du risque")
    s1, s2 = st.columns([1, 2])
    if et[-1] == 1:
        s1.error("**HORS MARCHE** (exposition 0 %)")
    else:
        s1.success("**INVESTI** (exposition 100 %)")
    s2.markdown(f"Le HMM a deux regimes estime une probabilite de crise de **{p_der:.0%}** "
                f"(derniere date : {bt['p'].dropna().index[-1]:%d/%m/%Y}). Regle d'hysteresis figee sur "
                f"2005-2014 : sortie du marche si P > **{s_in}**, retour si P < **{s_out}**. "
                "La decision est appliquee a la seance suivante, avec des couts de transaction.")

# ----------------------------------------------------------------- graphique
st.subheader("Deux dernieres annees, colorees par regime")
r = d.iloc[-504:]
afficher(fig_prix_regimes(r.index, r["prix"].values, r["lab_f"].values,
                          "S&P 500 : regimes estimes en temps reel (filtres)", log=False, hauteur=380))

# ----------------------------------------------------------------- resultats cles
st.subheader("Resultats cles du projet")
res = table("resultats_finaux_test.csv")
if res is not None:
    lig_h = [i for i in res.index if i.startswith("HMM K=2 (seuils choisis")]
    if lig_h and "Buy & Hold" in res.index:
        h, b = res.loc[lig_h[0]], res.loc["Buy & Hold"]
        st.caption("Test chronologique independant : hyperparametres figes sur 2005-2014, "
                   "evalue une seule fois sur la periode suivante, en temps reel, avec couts de transaction.")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Perte maximale (drawdown)", f"{h['max_dd']:.1%}", f"{h['max_dd'] - b['max_dd']:+.1%} vs Buy & Hold",
                  delta_color="normal")
        a2.metric("Volatilite", f"{h['vol']:.1%}", f"{h['vol'] - b['vol']:+.1%} vs Buy & Hold",
                  delta_color="inverse")
        a3.metric("Sharpe", f"{h['sharpe']:.2f}", f"{h['sharpe'] - b['sharpe']:+.2f} vs Buy & Hold")
        a4.metric("Rendement annuel", f"{h['rend_ann']:.1%}", f"{h['rend_ann'] - b['rend_ann']:+.1%} vs Buy & Hold")
else:
    absent("resultats_finaux_test.csv")

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
cc = st.columns(4)
contenus = [
    ("Les regimes existent", "Trois niveaux de risque nets (volatilite d'environ 9-11 %, 18-21 % et 41-50 %) retrouves sur cinq marches.", "couches", "#22D3EE"),
    ("Ils sont persistants", "Un regime dure de quelques semaines a quelques mois ; les modeles sans memoire (K-means, GMM) changent presque chaque jour.", "horloge", "#60A5FA"),
    ("Le temps reel est plus dur", "Sans precaution, le signal du HMM change 3 a 4 fois plus souvent qu'en retrospectif : d'ou le filtrage et l'hysteresis.", "activite", "#F59E0B"),
    ("Un benefice sur le risque", "Reduction du risque extreme (drawdown, volatilite), pas de surperformance prouvee : voir la page Backtest.", "bouclier", "#22C55E"),
]
for col, (t, tx, ic, co) in zip(cc, contenus):
    col.markdown(carte(t, tx, ic, co), unsafe_allow_html=True)

st.divider()
st.caption(AVERTISSEMENT)
