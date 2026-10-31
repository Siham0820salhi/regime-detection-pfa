import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from outils_app import *
from outils_app import json_

entete("Durees des regimes et transitions", "Combien de temps dure chaque regime, et par quel chemin passe-t-on de l'un a l'autre ?", "Etape 2 | Persistance")
st.markdown("Combien de temps un marche reste-t-il calme, agite ou en crise ? Et par quel chemin passe-t-on "
            "d'un regime a l'autre ? Ces deux questions mesurent la **persistance** des regimes, "
            "propriete que les modeles sans memoire (K-means, GMM) ne retrouvent pas.")

marches = marches_disponibles()
modeles = json_("modeles.json")
if not marches or modeles is None:
    absent("regimes_*.csv / modeles.json")
    st.stop()

nom = st.selectbox("Marche", list(marches))
cle = marches[nom]
d = serie_temp(f"regimes_{cle}.csv")
mod = modeles[cle]
A = np.array(mod["A"])
lab = d["lab_s"].values
long, etats = segments(lab)

# ----------------------------------------------------------------- matrices
st.subheader("Matrice de transition")
c1, c2 = st.columns(2)
with c1:
    afficher(fig_heatmap(A, "Estimee par le modele (HMM)"), "hm1")
with c2:
    emp = np.zeros((3, 3))
    for a, b in zip(lab[:-1], lab[1:]):
        emp[a, b] += 1
    emp = emp / np.maximum(emp.sum(axis=1, keepdims=True), 1)
    afficher(fig_heatmap(emp, "Observee sur les regimes detectes"), "hm2")
st.markdown(f"""
**Comment lire.** La ligne est le regime d'aujourd'hui, la colonne celui de demain. La diagonale
({A[0, 0]:.3f}, {A[1, 1]:.3f}, {A[2, 2]:.3f}) est la probabilite de *rester* : elle est proche de 1, donc les
regimes sont **persistants**. Les cases *calme -> crise* ({A[0, 2]:.3f}) et *crise -> calme* ({A[2, 0]:.3f}) sont
quasi nulles : **le marche passe presque toujours par le regime intermediaire**. La degradation comme la
reprise sont progressives.""")

# ----------------------------------------------------------------- durees
st.subheader("Duree de chaque regime")
rows = {}
for k in range(3):
    dd = long[etats == k]
    if len(dd) == 0:
        continue
    rows[REG[k]] = {"Episodes": len(dd), "Duree mediane (j)": float(np.median(dd)),
                    "Duree moyenne (j)": float(dd.mean()), "Duree max (j)": int(dd.max()),
                    "Duree theorique 1/(1-aii) (j)": 1 / (1 - A[k, k]),
                    "Part du temps (%)": (lab == k).mean() * 100,
                    "Volatilite annualisee (%)": mod["vol_ann"][k] * 100}
tab = pd.DataFrame(rows).T
tableau(tab.round(1))
m1, m2, m3 = st.columns(3)
for col, k in zip((m1, m2, m3), range(3)):
    if REG[k] in tab.index:
        col.metric(f"Regime {REG[k].lower()} : duree mediane", f"{tab.loc[REG[k], 'Duree mediane (j)']:.0f} jours",
                   f"{tab.loc[REG[k], 'Episodes']:.0f} episodes", delta_color="off")
st.markdown("""
**Mediane ou moyenne ?** La moyenne est tiree vers le haut par quelques tres longs episodes (par exemple
2008-2009). La **mediane** decrit l'episode typique : c'est elle qu'il faut citer.
**Duree theorique** : un HMM implique des durees geometriques de moyenne 1/(1-aii). Les durees observees sont
souvent plus longues car l'algorithme de Viterbi lisse les petits aller-retours et les vrais regimes ont des
durees a queue epaisse. On compare donc surtout les medianes entre marches.""")

# ----------------------------------------------------------------- boxplot + survie
g1, g2 = st.columns(2)
with g1:
    fig = go.Figure()
    for k in range(3):
        dd = long[etats == k]
        if len(dd):
            fig.add_trace(go.Box(y=dd, name=REG[k], marker_color=COUL[k], boxmean=True))
    fig.update_yaxes(type="log", title="duree d'un episode (jours, echelle log)")
    fig.update_layout(title="Distribution des durees d'episodes", height=400,
                      margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    afficher(fig, "box")
with g2:
    reg_k = st.radio("Regime pour la courbe de survie", REG, index=2, horizontal=True)
    k = REG.index(reg_k)
    dd = np.sort(long[etats == k])
    if len(dd) >= 3:
        n = np.arange(0, int(dd.max()) + 1)
        surv = np.array([(dd > x).mean() for x in n])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=n, y=surv, mode="lines", name="Observee", line=dict(color=COUL[k], width=3)))
        fig.add_trace(go.Scatter(x=n, y=A[k, k] ** n, mode="lines", name="Modele (geometrique)",
                                 line=dict(color=BLANC, dash="dash")))
        fig.update_layout(title=f"Probabilite qu'un episode {reg_k.lower()} dure plus de n jours", height=400,
                          xaxis_title="n (jours de bourse)", yaxis=dict(range=[0, 1.02]),
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.12))
        afficher(fig, "surv")
    else:
        st.info("Trop peu d'episodes pour tracer la courbe de survie.")
st.caption("La courbe de survie compare la duree reelle des episodes au modele du HMM. Un ecart indique que la "
           "vraie duree d'un regime n'est pas exactement sans memoire.")

# ----------------------------------------------------------------- comparaison entre marches
st.subheader("Comparaison entre marches")
lignes = {}
for nm, c in marches.items():
    dm = serie_temp(f"regimes_{c}.csv")
    lb = dm["lab_s"].values
    lg, et = segments(lb)
    ligne = {}
    for k in range(3):
        dk = lg[et == k]
        ligne[f"Duree mediane {REG[k].lower()} (j)"] = float(np.median(dk)) if len(dk) else np.nan
    for k in range(3):
        m = lb == k
        ligne[f"Vol. {REG[k].lower()} (%)"] = dm["ret"].values[m].std() * np.sqrt(252) * 100 if m.sum() > 1 else np.nan
    ligne["% crise"] = (lb == 2).mean() * 100
    ligne["Episodes de crise"] = int((et == 2).sum())
    lignes[nm] = ligne
cmp = pd.DataFrame(lignes).T
tableau(cmp.round(1))
fig = go.Figure()
for k in range(3):
    fig.add_trace(go.Bar(x=cmp.index, y=cmp[f"Duree mediane {REG[k].lower()} (j)"], name=REG[k],
                         marker_color=COUL[k]))
fig.update_layout(barmode="group", title="Duree mediane d'un episode selon le regime et le marche",
                  yaxis_title="jours de bourse", height=380, margin=dict(l=10, r=10, t=50, b=10),
                  legend=dict(orientation="h", y=1.12))
afficher(fig, "cmp")
st.markdown("""
**Conclusion.** L'ordre *calme > moyen > crise* est le meme sur tous les marches : un regime de crise
dure typiquement **2 a 5 semaines**, un regime calme **2 a 3 mois**. La structure est quasi identique d'un marche
a l'autre : le modele capte une propriete generale des marches d'actions.""")

# ----------------------------------------------------------------- stabilite et choix de K
with st.expander("Stabilite du modele et choix du nombre de regimes K"):
    syn = table("europe_synthese.csv")
    if syn is not None:
        st.markdown("**Stabilite entre initialisations** (ARI = 1 : memes regimes quelle que soit l'initialisation)")
        tableau(syn[["ARI moyen", "transitions", "dur. theo. crise", "dur. med. crise"]].round(3))
    sk = table("sensibilite_K.csv")
    if sk is not None:
        st.markdown("**Sensibilite a K (S&P 500, HMM)**")
        tableau(sk.round(3))
        st.caption("Le BIC baisse presque toujours quand K augmente ; on retient K = 3 pour la description "
                   "(gain de BIC net de K = 2 a K = 3, regimes stables, aucun regime minuscule) et K = 2 pour la strategie.")
    if syn is None and sk is None:
        absent("europe_synthese.csv / sensibilite_K.csv")
