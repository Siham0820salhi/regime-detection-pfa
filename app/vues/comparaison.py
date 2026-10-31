import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from outils_app import *

entete("Comparaison des modeles", "Memes donnees, memes conditions : qui detecte le mieux, et qui est le plus stable ?", "Etape 3 | Benchmark")
st.markdown("Peut-on faire aussi bien avec des methodes plus simples ? Cette page compare, **sur les memes "
            "donnees (S&P 500) et dans les memes conditions**, le clustering sans memoire (K-means, GMM), "
            "les modeles a memoire (HMM, HMM sticky), la detection de ruptures (PELT) et le jump model.")

NOMS = {"km": "K-means", "gmm": "GMM", "hmm": "HMM (rend. + vol20)", "hmm_sticky": "HMM sticky (rend. + vol20)",
        "hmm_ret3": "HMM (rendements seuls)", "hmm_k2_smooth": "HMM K=2 lisse", "hmm_k2_filt": "HMM K=2 filtre",
        "pelt": "PELT (2 niveaux)", "jump2": "Jump model K=2", "jump2_online": "Jump model K=2 temps reel",
        "jump3": "Jump model K=3"}
DEUX_ETATS = {"hmm_k2_smooth", "hmm_k2_filt", "pelt", "jump2", "jump2_online"}


@st.cache_data(show_spinner=False)
def regimes_et_prix():
    a, b, f = serie_temp("regimes_A.csv"), serie_temp("regimes_B.csv"), serie_temp("features_sp.csv")
    if a is None or f is None:
        return None
    d = a.join(b, how="left") if b is not None else a
    return d.join(f[["prix"]], how="left")


def panneaux(d, cles, a, b, titre):
    r = d.loc[a:b]
    fig = make_subplots(rows=len(cles), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=[f"{NOMS[c]} : {int((np.diff(r[c].values) != 0).sum())} transitions" for c in cles])
    for i, c in enumerate(cles, start=1):
        lab = r[c].fillna(0).astype(int).values
        if c in DEUX_ETATS:
            lab = np.where(lab > 0, 2, 0)
        fig.add_trace(go.Scattergl(x=r.index, y=r["prix"], mode="lines", showlegend=False,
                                   line=dict(color=PRIX_LIGNE, width=1)), row=i, col=1)
        for k in range(3):
            m = lab == k
            if m.any():
                fig.add_trace(go.Scattergl(x=r.index[m], y=r["prix"].values[m], mode="markers", showlegend=False,
                                           marker=dict(size=4, color=COUL[k])), row=i, col=1)
    fig.update_layout(height=max(260, 175 * len(cles)), title=titre, margin=dict(l=10, r=10, t=70, b=10))
    return fig


d = regimes_et_prix()
t1, t2, t3, t4, t5, t6 = st.tabs(["Comparaison equitable", "Stabilite et convergence", "Visuel",
                                  "Lisse contre filtre", "PELT et jump model", "Choix de K"])

# ----------------------------------------------------------------- 1
with t1:
    ce = table("comparaison_equitable.csv")
    if ce is None:
        absent("comparaison_equitable.csv")
    else:
        st.markdown("**Protocole** : memes variables (rendement et volatilite 20 j standardises), memes dates, "
                    "K = 3 regimes pour tous ; un HMM sur rendements seuls est ajoute comme reference. Analyse retrospective.")
        tableau(ce.round(3))
        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure(go.Bar(x=ce.index, y=ce["transitions"], marker_color=BLEU))
            fig.update_yaxes(type="log", title="transitions (echelle log)")
            fig.update_layout(title="Nombre de changements de regime", height=360, margin=dict(l=10, r=10, t=50, b=10))
            afficher(fig, "ce1")
        with g2:
            fig = go.Figure(go.Bar(x=ce.index, y=ce["% segments < 10 j"] * 100, marker_color=COUL[2]))
            fig.update_layout(title="Part des segments de moins de 10 jours (%)", height=360,
                              margin=dict(l=10, r=10, t=50, b=10))
            afficher(fig, "ce2")
        st.markdown("""
**Ce que montre le tableau (H2).** Les modeles sans memoire alternent de regime presque *tous les jours*
(duree mediane de 1 jour) : un gestionnaire ne peut rien en faire, les couts de transaction seraient prohibitifs.
Les HMM sont **un ordre de grandeur plus stables**. Le prior *sticky* ne change presque pas le nombre de
transitions, mais reduit la part de segments tres courts.
**Nuance** : une partie de l'instabilite du clustering vient du choix de variables tres bruitees. Avec des
variables lentes (drawdown, volatilite 60 j), K-means et GMM font environ 230 a 330 transitions : mieux, mais
encore 1,5 a 2 fois plus que le HMM, et sans probabilites.""")

# ----------------------------------------------------------------- 2
with t2:
    st_ = table("stabilite_initialisations.csv")
    cv = table("convergence_hmm.csv")
    if st_ is None:
        absent("stabilite_initialisations.csv")
    else:
        st.markdown("**Stabilite sur 20 initialisations aleatoires.** ARI = 1 signifie que l'on retrouve les "
                    "memes regimes quelle que soit l'initialisation ; *runs au meme optimum* = part des essais "
                    "qui atteignent la meilleure vraisemblance.")
        tableau(st_.round(3))
        st.markdown("""
Un **GMM** peut donner des regimes tres differents d'un essai a l'autre ; un **HMM sur rendements seuls** tombe
parfois dans un mauvais optimum local. C'est pourquoi tous nos HMM retiennent *la meilleure vraisemblance parmi
plusieurs initialisations* (10 en general).""")
    if cv is not None:
        st.markdown("**Convergence de l'algorithme EM (HMM)**")
        tableau(cv.round(2))
        st.caption("Tous les essais convergent bien avant la limite d'iterations. Les rares avertissements "
                   "« not converging » correspondent a des baisses infimes de vraisemblance (probablement dues "
                   "au prior pour le HMM sticky) et sont sans consequence.")

# ----------------------------------------------------------------- 3
with t3:
    if d is None:
        absent("regimes_A.csv / features_sp.csv")
    else:
        presets = {"Crise financiere 2007-2010": ("2007-01-01", "2010-12-31"),
                   "Inflation et taux 2018-2023": ("2018-01-01", "2023-12-31"),
                   "Covid 2019-2021": ("2019-06-01", "2021-06-30"),
                   "Toute la periode": (str(d.index[0].date()), str(d.index[-1].date()))}
        c1, c2 = st.columns([1, 2])
        p = c1.selectbox("Periode", list(presets))
        dispo = [c for c in NOMS if c in d.columns]
        defaut = [c for c in ["km", "gmm", "hmm", "hmm_ret3", "jump2_online", "pelt"] if c in dispo]
        sel = c2.multiselect("Modeles a comparer", dispo, default=defaut, format_func=lambda c: NOMS[c])
        if sel:
            a, b = presets[p]
            afficher(panneaux(d, sel, a, b, f"Regimes detectes, {p.lower()} (vert calme, orange moyen, rouge crise)"), "vis")
            st.caption("Les modeles a memoire produisent de grands blocs lisibles ; K-means et GMM produisent un melange de "
                       "couleurs jour apres jour. PELT et le jump model ne distinguent que deux niveaux (calme/crise).")

# ----------------------------------------------------------------- 4
with t4:
    if d is None or "hmm_k2_smooth" not in d:
        absent("regimes_A.csv")
    else:
        s, f = d["hmm_k2_smooth"].values, d["hmm_k2_filt"].values
        c1, c2, c3 = st.columns(3)
        c1.metric("Transitions (lisse, utilise le futur)", int((np.diff(s) != 0).sum()))
        c2.metric("Transitions (filtre, temps reel)", int((np.diff(f) != 0).sum()),
                  f"x{(np.diff(f) != 0).sum() / max((np.diff(s) != 0).sum(), 1):.1f}", delta_color="inverse")
        c3.metric("Accord jour par jour", f"{(s == f).mean():.1%}")
        st.markdown("""
**Le biais d'anticipation (look-ahead bias).** Les probabilites *lissees* utilisent toute l'histoire, futur
compris : elles produisent de belles figures mais sont **inutilisables en temps reel**. Les probabilites *filtrees*
n'utilisent que le passe. Les deux sequences sont d'accord presque tout le temps, mais le filtre hesite aux moments de
bascule et change de regime **3 a 4 fois plus souvent**. Ignorer cette difference revient a surestimer ce qu'un
investisseur pourrait vraiment faire.""")
        c = st.selectbox("Zone a observer", ["2007-2010", "2018-2020", "2020-2023"], key="zoom_lf")
        a, b = {"2007-2010": ("2007-01-01", "2010-12-31"), "2018-2020": ("2018-01-01", "2020-12-31"),
                "2020-2023": ("2020-01-01", "2023-12-31")}[c]
        afficher(panneaux(d, ["hmm_k2_smooth", "hmm_k2_filt"], a, b, f"HMM K=2 : lisse (haut) et filtre (bas), {c}"), "lf")
        st.caption("Pendant la baisse de 2008, le filtre insere de courts retours au vert pendant les rebonds : "
                   "d'ou la regle d'hysteresis utilisee dans la strategie.")

# ----------------------------------------------------------------- 5
with t5:
    jb = table("jump_balayage_lambda.csv")
    if jb is None:
        absent("jump_balayage_lambda.csv")
    else:
        st.markdown("""
**Jump model** : on minimise la somme des distances aux centres de regime **plus une penalite lambda a chaque
changement d'etat**. lambda = 0 redonne un K-means (aucune memoire) ; plus lambda augmente, moins il y a de
changements. Le reglage est donc explicite, contrairement au prior sticky du HMM.""")
        tableau(jb.round(3))
        try:
            jb2 = jb.copy()
            jb2["K"] = [i.split(" ")[0] for i in jb2.index]
            jb2["lambda"] = [float(i.split("lambda=")[1]) for i in jb2.index]
            fig = go.Figure()
            for k, g in jb2.groupby("K"):
                fig.add_trace(go.Scatter(x=g["lambda"] + 1, y=g["transitions"], mode="lines+markers", name=k))
            fig.update_xaxes(type="log", title="lambda + 1 (echelle log)")
            fig.update_yaxes(type="log", title="transitions")
            fig.update_layout(title="Nombre de transitions selon lambda", height=360,
                              margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.12))
            afficher(fig, "jl")
        except Exception:
            pass
    cd = table("concordance_dates_tol10.csv")
    if cd is not None:
        st.markdown("**Les dates de rupture se correspondent-elles ?** Part des dates de la ligne situees a moins de "
                    "10 jours de bourse d'une date de la colonne.")
        z = cd.values.astype(float)
        fig = go.Figure(go.Heatmap(z=z, x=list(cd.columns), y=list(cd.index), zmin=0, zmax=1, colorscale=ECHELLE_BLEUE,
                                   text=[["" if np.isnan(v) else f"{v:.0%}" for v in l] for l in z],
                                   texttemplate="%{text}", textfont=dict(color="white", size=14), showscale=False))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        afficher(fig, "cd")
        st.markdown("""
Les ruptures **PELT** retrouvent une large part des transitions du HMM, bien au-dessus du hasard (test de hasard
realise dans l'etape 10) : deux methodes de conception differente voient les memes changements. La relation n'est
pas symetrique : le HMM detecte aussi des changements mineurs (calme/moyen) que PELT ignore.
**PELT** est un outil *retrospectif* (il ne donne pas de signal en temps reel), et le **jump model** est plus stable en
temps reel mais plus lent a detecter une chute rapide : voir la page *Backtest*.""")
    ps = table("pelt_segments.csv", index_col=None)
    if ps is not None:
        with st.expander("Segments detectes par PELT"):
            tableau(ps)

# ----------------------------------------------------------------- 6
with t6:
    sk = table("sensibilite_K.csv")
    if sk is None:
        absent("sensibilite_K.csv")
    else:
        st.markdown("**Combien de regimes ?** Le critere BIC penalise la complexite, mais ici il baisse presque "
                    "toujours quand K augmente : les etats supplementaires absorbent les queues epaisses plutot que de "
                    "vrais regimes economiques. On regarde donc aussi la **taille du plus petit regime** et la "
                    "**stabilite** (ARI).")
        tableau(sk.round(3))
        st.markdown("""
**Decision.** K = 3 pour decrire (calme, moyen, crise : gain de BIC net par rapport a K = 2, regimes stables, aucun regime
minuscule) ; K = 2 pour agir (strategie simple et robuste). Au-dela, les regimes deviennent trop petits ou instables.""")
