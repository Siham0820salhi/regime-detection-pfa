import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from outils_app import *
from outils_app import json_

entete("Detection des regimes", "Quel regime traverse chaque marche, en temps reel ou avec le recul ?", "Etape 1 | Detection")
st.markdown("Le modele est un **HMM gaussien a 3 etats** estime sur les rendements journaliers : "
            "chaque jour appartient a un regime *calme*, *moyen* ou *crise*, classes par volatilite croissante. "
            "Le meme modele, sans reglage specifique, est applique a chaque marche.")

marches = marches_disponibles()
if not marches:
    absent("regimes_*.csv")
    st.stop()

# ----------------------------------------------------------------- controles
c1, c2 = st.columns([1.2, 1.8])
nom = c1.selectbox("Marche", list(marches))
mode = c2.radio("Type de regimes", ["Temps reel (filtre)", "Retrospectif (lisse)"], horizontal=True,
                help="Filtre : utilise uniquement le passe, comme un investisseur reel. "
                     "Lisse : utilise aussi le futur, ce qui est impossible en pratique (biais d'anticipation).")
cle = marches[nom]
d = serie_temp(f"regimes_{cle}.csv")
suf = "f" if mode.startswith("Temps") else "s"
lab_col, p_cols = f"lab_{suf}", [f"p{suf}{k}" for k in range(3)]

dmin, dmax = d.index[0].date(), d.index[-1].date()
periode = st.slider("Periode affichee", min_value=dmin, max_value=dmax, value=(dmin, dmax), format="DD/MM/YYYY")
o1, o2, o3, o4 = st.columns(4)
log = o1.checkbox("Echelle logarithmique", True)
show_nber = o2.checkbox("Recessions americaines (NBER)", cle == "sp")
show_ev = o3.checkbox("Pics -> creux historiques", True)
show_ind = o4.checkbox("Indicateurs de risque", True)

r = d.loc[str(periode[0]):str(periode[1])]
if len(r) < 30:
    st.warning("Periode trop courte.")
    st.stop()

# ----------------------------------------------------------------- indicateurs cles
long, etats = segments(r[lab_col].values)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Transitions sur la periode", int(len(long) - 1))
k2.metric("Duree mediane d'un segment", f"{np.median(long):.0f} jours")
k3.metric("Part du temps en crise", f"{(r[lab_col] == 2).mean():.1%}")
k4.metric("Jours de bourse", len(r))

# ----------------------------------------------------------------- graphique principal
zones = []
if show_nber:
    for a, b in periodes_nber(serie_temp("usrec.csv")):
        if b >= r.index[0] and a <= r.index[-1]:
            zones.append((a, b, "#94a3b8", 0.22))
ev = calculer_evenements(d["prix"])
if show_ev:
    for nm, (tp, tc, _) in ev.items():
        if tc >= r.index[0] and tp <= r.index[-1]:
            zones.append((tp, tc, "#60a5fa", 0.12))
afficher(fig_prix_regimes(r.index, r["prix"].values, r[lab_col].values,
                          f"{nom} : regimes ({mode.lower()})", log=log, zones=zones, hauteur=470), "prix")
leg = []
if show_nber:
    leg.append("bandes grises = recessions NBER")
if show_ev:
    leg.append("bandes bleues = baisses historiques (pic -> creux)")
if leg:
    st.caption(" ; ".join(leg).capitalize() + ".")

afficher(fig_probas(r.index, [r[c].values for c in p_cols],
                    f"Probabilite de chaque regime ({mode.lower()})"), "probas")
if suf == "f":
    st.caption("En temps reel, la probabilite oscille aux moments de bascule : c'est le prix du "
               "filtrage sans connaitre le futur. Pour la gestion du risque, on la stabilise avec une "
               "regle d'hysteresis (voir page *Backtest*).")

# ----------------------------------------------------------------- indicateurs
if show_ind:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("VIX (indice de la peur)" if "vix" in r and r["vix"].notna().any()
                                        else "Volatilite realisee 20 jours (annualisee)", "Drawdown (baisse depuis le plus haut)"))
    serie = r["vix"] if "vix" in r and r["vix"].notna().any() else r["vol20"] * 100
    for k in range(3):
        m = r[lab_col].values == k
        fig.add_trace(go.Scattergl(x=r.index[m], y=serie.values[m], mode="markers", name=REG[k],
                                   marker=dict(size=3, color=COUL[k])), row=1, col=1)
        fig.add_trace(go.Scattergl(x=r.index[m], y=r["drawdown"].values[m] * 100, mode="markers",
                                   marker=dict(size=3, color=COUL[k]), showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="%" if "vix" not in r else "points", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1))
    afficher(fig, "ind")
    st.caption("Les points sont colores par regime : le regime de crise correspond aux pics de volatilite "
               "et aux drawdowns profonds, ce qui valide economiquement la classification.")

# ----------------------------------------------------------------- tableau par regime
st.subheader("Caracteristiques de chaque regime")
tab = stats_regimes(r["ret"], r[lab_col])
tableau(tab.round(1))
st.markdown("""
**Lecture financiere.** Le rendement moyen *diminue* quand la volatilite augmente (effet de levier :
les marches baissiers sont plus risques). Le regime de crise est rare (moins de 10 % du temps) mais concentre
les pires jours. La notion de regime est donc une **mesure de risque dependante du temps**,
pas une prevision du rendement.""")

# ----------------------------------------------------------------- validation par evenements (H1)
st.subheader("Validation (H1) : les baisses historiques sont-elles reconnues ?")
lignes = []
for nm, (tp, tc, perte) in ev.items():
    m = (d.index >= tp) & (d.index <= tc)
    if m.sum() < 5:
        continue
    lab = d[lab_col].values
    lignes.append({"Evenement": nm, "Pic": tp.date(), "Creux": tc.date(), "Perte pic->creux": f"{perte:.1%}",
                   "% jours en CRISE": round((lab[m] == 2).mean() * 100, 1),
                   "% jours en CALME": round((lab[m] == 0).mean() * 100, 1)})
if lignes:
    base = (d[lab_col] == 2).mean() * 100
    tableau(pd.DataFrame(lignes))
    st.caption(f"Repere : le marche est en regime de crise {base:.1f} % du temps sur toute la periode. "
               "Si la part pendant une baisse est bien superieure a ce repere, le modele reconnait l'evenement. "
               "Le regime de crise capte surtout les *chocs de volatilite* (2008, 2020) et moins les baisses lentes (2022).")

# ----------------------------------------------------------------- dernieres transitions
with st.expander("Dernieres transitions de regime"):
    lab = d[lab_col].values
    ch = np.flatnonzero(np.diff(lab) != 0) + 1
    t = pd.DataFrame({"Date": d.index[ch].date, "De": [REG[i] for i in lab[ch - 1]],
                      "Vers": [REG[i] for i in lab[ch]], "Cours": d["prix"].values[ch].round(0),
                      "Volatilite 20 j": (d["vol20"].values[ch] * 100).round(1),
                      "Drawdown (%)": (d["drawdown"].values[ch] * 100).round(1)})
    tableau(t.tail(25).iloc[::-1].reset_index(drop=True))
    st.caption(f"{len(t)} transitions au total sur la periode complete.")

# ----------------------------------------------------------------- validation externe NBER
if cle == "sp":
    with st.expander("Validation externe : recessions officielles NBER et delais de detection"):
        a, b = table("eval_signaux_nber.csv"), table("eval_delais.csv")
        if a is not None:
            st.markdown("**Concordance des alertes avec les recessions NBER** (precision : part des alertes "
                        "en recession ; rappel : part des jours de recession detectes)")
            tableau(a.round(3))
        if b is not None:
            st.markdown("**Delai avant l'alerte depuis le pic du marche** (perte deja subie entre parentheses)")
            tableau(b)
        if a is None and b is None:
            absent("eval_signaux_nber.csv")
        st.caption("Le HMM identifie un regime de *risque de marche*, pas un cycle economique : "
                   "il declenche aussi pendant des corrections hors recession (2011, 2018, 2022).")
