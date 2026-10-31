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

entete("Backtest et gestion du risque", "Que se passe-t-il si l'on reduit l'exposition quand le modele detecte une crise ? Test en temps reel simule, couts inclus.", "Etape 4 | Strategie")
st.markdown("Et si un investisseur reduisait son exposition quand le modele detecte un regime de crise ? "
            "Le backtest repond en **temps reel simule** : probabilites *filtrees*, modele **reentraine chaque annee "
            "sur le passe seul** (walk-forward), decision appliquee a la **seance suivante**, **couts de transaction** "
            "a chaque changement d'exposition.")

actifs, meta = actifs_backtest(), json_("bt_meta.json")
if not actifs:
    absent("bt_*.csv")
    st.stop()
choix = json_("choix_figes.json") or {}
DEF_IN, DEF_OUT = float(choix.get("seuil_entree", 0.9)), float(choix.get("seuil_sortie", 0.3))
SPLIT = pd.Timestamp(choix.get("split", "2015-01-01"))

# ----------------------------------------------------------------- controles
c1, c2 = st.columns([1.1, 1.9])
nom = c1.selectbox("Actif", list(actifs))
cle = actifs[nom]
d = serie_temp(f"bt_{cle}.csv").sort_index()
ppy = meta[cle]["ppy"]
ret = np.log(d["prix"] / d["prix"].shift(1))
p_debut = d["p"].first_valid_index()
debut_oos = d.index[d.index.get_loc(p_debut) + 1]

periode = c2.radio("Periode d'evaluation",
                   ["Toute la periode hors echantillon", f"Test independant (depuis {SPLIT:%Y})", "Personnalisee"],
                   horizontal=True,
                   help="Le HMM n'est evalue que apres une periode initiale d'entrainement de 5 ans. "
                        "Les hyperparametres par defaut ont ete figes sur la periode precedant le test.")
fin = d.index[-1]
if periode.startswith("Test"):
    debut = max(SPLIT, debut_oos)
elif periode.startswith("Pers"):
    plage = st.date_input("Dates", value=(debut_oos.date(), fin.date()), min_value=debut_oos.date(),
                          max_value=fin.date(), format="DD/MM/YYYY")
    if len(plage) != 2:
        st.info("Selectionnez une date de debut et une date de fin.")
        st.stop()
    a, b = plage
    debut, fin = pd.Timestamp(a), pd.Timestamp(b)
else:
    debut = debut_oos

s1, s2, s3, s4 = st.columns(4)
s_in = s1.slider("Seuil d'entree en crise (P >)", 0.50, 0.99, DEF_IN, 0.01,
                 help="La strategie sort du marche quand la probabilite de crise depasse ce seuil.")
s_out = s2.slider("Seuil de sortie de crise (P <)", 0.05, 0.80, DEF_OUT, 0.01,
                  help="La strategie revient quand la probabilite repasse sous ce seuil (hysteresis).")
cout_pb = s3.slider("Cout de transaction (points de base)", 0, 100, 5, 1,
                    help="5 pb = 0,05 % du montant echange a chaque changement d'exposition.")
decal = s4.checkbox("Decaler la decision d'une seance", True,
                    help="Decision a la cloture de t appliquee au rendement de t+1. "
                         "Decocher montre le biais d'anticipation : resultat IRREALISTE.")
if s_out >= s_in:
    st.error("Le seuil de sortie doit etre inferieur au seuil d'entree.")
    st.stop()
if not decal:
    st.error("⚠️ Sans decalage, le signal utilise l'information du jour pour decider ce meme jour : "
             "les performances affichees sont **irrealistes** (biais d'anticipation).")
if periode.startswith("Test") and (abs(s_in - DEF_IN) > 1e-9 or abs(s_out - DEF_OUT) > 1e-9):
    st.warning("Vous modifiez les seuils **en regardant la periode de test** : ce n'est plus un test independant "
               f"(les seuils par defaut {DEF_IN}/{DEF_OUT} ont ete figes avant le test). Utile pour explorer, pas pour conclure.")
if cle == "btc":
    st.info("Bitcoin : marche ouvert 7 jours sur 7 (365 periodes/an) et couts reels superieurs a 5 pb : "
            "augmentez le curseur de couts.")

expo_tout = construire_strategies(d, s_in, s_out)
choisies = st.multiselect("Strategies", list(expo_tout), default=list(expo_tout))
if not choisies:
    st.stop()
cout = cout_pb / 1e4

res, cums, xs = {}, {}, {}
for n in choisies:
    xs[n], cums[n], res[n] = evaluer(expo_tout[n], ret, debut, fin, cout, ppy, 1 if decal else 0)
tab = pd.DataFrame(res).T
st.caption(f"Periode evaluee : **{debut:%d/%m/%Y} -> {fin:%d/%m/%Y}** ({len(xs[choisies[0]])} observations) | "
           f"probabilites : `{meta[cle]['source_p']}`")

# ----------------------------------------------------------------- resultats
st.subheader("Performances")
fmt = {"Rendement annualise": "{:.1%}", "Volatilite": "{:.1%}", "Sharpe": "{:.2f}", "Max drawdown": "{:.1%}",
       "Calmar": "{:.2f}", "Changements de position": "{:.0f}", "Rotation / an": "{:.1f}", "Part investie": "{:.0%}"}
try:
    tableau(tab.style.format(fmt))
except Exception:
    tableau(tab.round(3))

if "Buy & Hold" in res and "HMM binaire" in res:
    h, b = res["HMM binaire"], res["Buy & Hold"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Perte maximale : HMM", f"{h['Max drawdown']:.1%}", f"{h['Max drawdown'] - b['Max drawdown']:+.1%} vs Buy & Hold")
    m2.metric("Volatilite : HMM", f"{h['Volatilite']:.1%}", f"{h['Volatilite'] - b['Volatilite']:+.1%} vs Buy & Hold", delta_color="inverse")
    m3.metric("Sharpe : HMM", f"{h['Sharpe']:.2f}", f"{h['Sharpe'] - b['Sharpe']:+.2f} vs Buy & Hold")
    m4.metric("Rendement annuel : HMM", f"{h['Rendement annualise']:.1%}", f"{h['Rendement annualise'] - b['Rendement annualise']:+.1%} vs Buy & Hold")

# ----------------------------------------------------------------- graphiques
couleurs = {"Buy & Hold": BLANC, "Prix > MA200": "#a78bfa", "VIX > 30": "#60a5fa",
            "HMM binaire": "#fb923c", "HMM graduel (1 - P crise)": "#34d399"}
log = st.checkbox("Echelle logarithmique", True)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.6, 0.4],
                    subplot_titles=("Valeur cumulee d'un investissement de 1 (nette de couts)", "Drawdown : baisse depuis le plus haut"))
for n in choisies:
    c = couleurs.get(n, None)
    fig.add_trace(go.Scatter(x=cums[n].index, y=cums[n].values, mode="lines", name=n,
                             line=dict(color=c, width=2.4 if n in ("Buy & Hold", "HMM binaire") else 1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=cums[n].index, y=(cums[n] / cums[n].cummax() - 1).values * 100, mode="lines",
                             line=dict(color=c, width=1.4), showlegend=False), row=2, col=1)
if log:
    fig.update_yaxes(type="log", row=1, col=1)
fig.update_yaxes(title_text="%", row=2, col=1)
fig.update_layout(height=620, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.1))
afficher(fig, "cum")

# signal
r = d.loc[debut:fin]
etat = hysteresis(d["p"].values, s_in, s_out)
etat = pd.Series(etat, index=d.index).loc[debut:fin]
fig = go.Figure()
fig.add_trace(go.Scatter(x=r.index, y=r["p"], mode="lines", name="P(crise) filtree", line=dict(color=COUL[2], width=1)))
fig.add_hline(y=s_in, line_dash="dash", line_color="#cbd5e1", annotation_text="entree", annotation_position="top left")
fig.add_hline(y=s_out, line_dash="dot", line_color="#cbd5e1", annotation_text="sortie", annotation_position="bottom left")
fig.add_trace(go.Scatter(x=r.index, y=etat.values, mode="lines", name="Hors marche (1) / investi (0)",
                         line=dict(color="rgba(214,39,40,0.0)"), fill="tozeroy", fillcolor="rgba(239,68,68,0.20)"))
fig.update_layout(title="Signal : probabilite de crise et periodes hors marche (HMM binaire)", height=300,
                  margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(range=[0, 1.02]), legend=dict(orientation="h", y=1.2))
afficher(fig, "sig")

# ----------------------------------------------------------------- analyses de robustesse
st.subheader("Robustesse")
t1, t2, t3, t4 = st.tabs(["Couts de transaction", "Seuils d'hysteresis", "Biais d'anticipation", "Test de significativite"])

with t1:
    couts = [0, 5, 10, 20, 50, 100]
    lignes = {n: [evaluer(expo_tout[n], ret, debut, fin, c / 1e4, ppy, 1 if decal else 0)[2]["Sharpe"] for c in couts]
              for n in choisies}
    fig = go.Figure()
    for n, v in lignes.items():
        fig.add_trace(go.Scatter(x=couts, y=v, mode="lines+markers", name=n, line=dict(color=couleurs.get(n))))
    fig.update_layout(title="Sharpe selon le cout de transaction", xaxis_title="points de base par changement",
                      yaxis_title="Sharpe", height=380, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.15))
    afficher(fig, "couts")
    st.markdown("Une strategie qui change souvent d'exposition (colonne *Rotation / an*) est plus sensible aux frais : "
                "l'exposition graduelle, qui reequilibre chaque jour, perd son avantage des que les couts augmentent.")

with t2:
    @st.cache_data(show_spinner=False)
    def grille(cle_, debut_, fin_, cout_, decal_):
        dd = serie_temp(f"bt_{cle_}.csv").sort_index()
        rr = np.log(dd["prix"] / dd["prix"].shift(1))
        ins, outs = [0.6, 0.7, 0.8, 0.9, 0.95], [0.1, 0.2, 0.3, 0.4, 0.5]
        sh = pd.DataFrame(np.nan, index=outs, columns=ins)
        for si in ins:
            for so in outs:
                if so < si:
                    e = pd.Series(1 - hysteresis(dd["p"].values, si, so), index=dd.index)
                    sh.loc[so, si] = evaluer(e, rr, debut_, fin_, cout_, ppy, decal_)[2]["Sharpe"]
        return sh

    sh = grille(cle, debut, fin, cout, 1 if decal else 0)
    fig = go.Figure(go.Heatmap(z=sh.values, x=[str(c) for c in sh.columns], y=[str(i) for i in sh.index],
                               colorscale="Viridis", text=[[f"{v:.2f}" for v in l] for l in sh.values],
                               texttemplate="%{text}", colorbar=dict(title="Sharpe")))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Sharpe du HMM binaire selon les seuils (sur la periode evaluee)", height=380,
                      xaxis_title="seuil d'entree", yaxis_title="seuil de sortie", margin=dict(l=10, r=10, t=50, b=10))
    afficher(fig, "grille")
    st.markdown("""
**Attention : choisir ici la meilleure case puis la presenter comme resultat serait du sur-ajustement.**
Cette carte sert a juger si le resultat est *robuste* aux seuils (couleurs voisines) ou *fragile* (grandes variations).
Dans le protocole du projet, les seuils sont choisis **uniquement sur la periode de developpement (2005-2014)**,
puis evalues une seule fois sur le test.""")

with t3:
    if "HMM binaire" in expo_tout:
        a0 = evaluer(expo_tout["HMM binaire"], ret, debut, fin, cout, ppy, 0)[2]
        a1 = evaluer(expo_tout["HMM binaire"], ret, debut, fin, cout, ppy, 1)[2]
        x1, x2, x3 = st.columns(3)
        x1.metric("Sharpe sans decalage (irrealiste)", f"{a0['Sharpe']:.2f}")
        x2.metric("Sharpe avec decalage d'une seance", f"{a1['Sharpe']:.2f}")
        x3.metric("Max drawdown : sans / avec decalage", f"{a0['Max drawdown']:.1%} / {a1['Max drawdown']:.1%}")
        st.markdown("""
Sans decalage, la strategie « voit » la baisse du jour avant d'en subir la perte : c'est le **biais d'anticipation**.
L'ecart entre les deux colonnes mesure ce qu'une analyse naive surestimerait. Ce projet applique **toujours** le decalage,
l'estimation filtree (sans futur) et le reentrainement annuel sur le passe seul.""")

with t4:
    if len(choisies) >= 2:
        k1, k2 = st.columns(2)
        A_ = k1.selectbox("Strategie A", choisies, index=min(len(choisies) - 1, choisies.index("HMM binaire")) if "HMM binaire" in choisies else 0)
        B_ = k2.selectbox("Strategie B (reference)", choisies, index=0)
        if A_ == B_:
            st.info("Choisissez deux strategies differentes.")
        elif st.button("Lancer le test (bootstrap par blocs, 1000 tirages)"):
            diff, ic, pv = bootstrap_diff(xs[A_].values, xs[B_].values, ppy)
            r1, r2, r3 = st.columns(3)
            r1.metric("Difference de Sharpe (A - B)", f"{diff:+.2f}")
            r2.metric("Intervalle de confiance a 95 %", f"[{ic[0]:+.2f} ; {ic[1]:+.2f}]")
            r3.metric("P(difference <= 0)", f"{pv:.2f}")
            if ic[0] <= 0 <= ic[1]:
                st.warning("L'intervalle contient 0 : la difference de Sharpe n'est **pas statistiquement significative**.")
            else:
                st.success("L'intervalle ne contient pas 0 : la difference est statistiquement significative.")
            st.caption("Bootstrap par blocs de 20 jours (conserve l'autocorrelation de la volatilite). "
                       "Sharpe arithmetique, sans taux sans risque.")
    else:
        st.info("Selectionnez au moins deux strategies.")

st.markdown("""
### Comment lire ces resultats
- **Max drawdown** : plus forte baisse depuis un plus haut. C'est la mesure de risque la plus parlante pour un investisseur.
- **Sharpe** : rendement annuel divise par la volatilite annuelle (sans taux sans risque). **Calmar** : rendement annuel divise par le drawdown maximal.
- **Rotation / an** : somme des variations d'exposition par an ; elle determine l'impact des couts.
- La strategie **reduit nettement le risque extreme** (drawdown, volatilite) au prix d'un rendement plus faible : c'est une *assurance*.
  Sur les indices actions americains, le Sharpe s'ameliore ; **cette amelioration n'est pas statistiquement prouvee**
  et ne se retrouve pas sur tous les actifs (voir *Methodologie et limites*).
""")
st.caption(AVERTISSEMENT)
