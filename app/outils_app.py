"""Fonctions communes de l'application : lecture des donnees, graphiques, backtest."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

DONNEES = Path(__file__).resolve().parent / "donnees"
COUL = ["#22c55e", "#f59e0b", "#ef4444"]              # calme, moyen, crise (vif sur fond sombre)
TEXTE, SOUS_TEXTE = "#e6edf7", "#94a3b8"
GRILLE = "rgba(148,163,184,0.12)"
PRIX_LIGNE = "rgba(203,213,225,0.55)"
BLEU, BLANC = "#60a5fa", "#e5e7eb"
ECHELLE_BLEUE = [[0, "#0f1b33"], [0.5, "#1d4ed8"], [1, "#60a5fa"]]
REG = ["Calme", "Moyen", "Crise"]
AVERTISSEMENT = ("Projet pedagogique de recherche : aucune information presentee ici ne constitue "
                 "un conseil en investissement. Les performances passees ne prejugent pas des "
                 "performances futures.")
FENETRES_EVENEMENTS = {"Bulle internet": ("2000-01-01", "2003-06-30"),
                       "Crise 2008": ("2007-06-01", "2009-12-31"),
                       "Zone euro 2011": ("2011-04-01", "2011-12-31"),
                       "COVID 2020": ("2020-01-01", "2020-06-30"),
                       "Inflation 2022": ("2021-11-01", "2022-12-31"),
                       "Chocs 2025": ("2025-01-01", "2025-04-30")}


# ------------------------------------------------------------------ style moderne (theme sombre)
# icones vectorielles (style Lucide, licence MIT), dessinees en ligne
_ICONES = {
    "activite": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "bouclier": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "horloge": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "couches": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "baisse": '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    "cible": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
}


def icone(nom, taille=22, couleur="#22D3EE"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{taille}" height="{taille}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{couleur}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'{_ICONES.get(nom, "")}</svg>')


def carte(titre, texte, icon="activite", couleur="#22D3EE", grand=None):
    g = f'<div class="grand" style="color:{couleur}">{grand}</div>' if grand else ""
    return (f'<div class="carte"><div class="ico" style="background:{couleur}22;border:1px solid {couleur}55">'
            f'{icone(icon, 22, couleur)}</div>{g}<div class="titre">{titre}</div>'
            f'<div class="texte">{texte}</div></div>')


# ------------------------------------------------------------------ affichage
def style_fig(fig):
    """Theme sombre commun a tous les graphiques."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, Segoe UI, sans-serif", color=TEXTE, size=13),
                      title=dict(font=dict(size=15, color=TEXTE), x=0.01),
                      hoverlabel=dict(bgcolor="#162036", font_color=TEXTE, bordercolor="#334155"))
    fig.update_xaxes(gridcolor=GRILLE, zeroline=False, linecolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(gridcolor=GRILLE, zeroline=False, linecolor="rgba(148,163,184,0.25)")
    return fig


def afficher(fig, cle=None):
    style_fig(fig)
    try:
        st.plotly_chart(fig, width="stretch", key=cle, theme=None)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=cle, theme=None)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important; }
.stApp { background: radial-gradient(1100px 560px at 8% -8%, #16264a 0%, #0b1220 58%) fixed; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1380px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f1a30 0%, #0a1120 100%);
    border-right: 1px solid rgba(148,163,184,.12); }
[data-testid="stMetric"] { background: linear-gradient(145deg, rgba(30,41,66,.78), rgba(15,22,38,.78));
    border: 1px solid rgba(148,163,184,.16); border-radius: 16px; padding: 14px 18px;
    box-shadow: 0 8px 22px rgba(0,0,0,.28); }
[data-testid="stMetricLabel"] p { color: #94a3b8; font-size: .8rem; letter-spacing: .02em; }
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -.01em; }
h2, h3 { letter-spacing: -.015em; font-weight: 700; }
.hero { padding: 22px 26px; border-radius: 20px; margin: 4px 0 18px 0;
    background: linear-gradient(120deg, rgba(37,99,235,.28) 0%, rgba(14,165,233,.12) 45%, rgba(15,22,38,.6) 100%);
    border: 1px solid rgba(96,165,250,.28); box-shadow: 0 12px 34px rgba(0,0,0,.35); }
.hero h1 { margin: 0; font-size: 2.05rem; font-weight: 800; letter-spacing: -.02em; padding: 0;
    background: linear-gradient(90deg, #ffffff 0%, #93c5fd 100%); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; }
.hero p { margin: 8px 0 0 0; color: #b6c2d6; font-size: 1.02rem; }
.pill { display: inline-block; padding: 3px 12px; margin: 0 8px 10px 0; border-radius: 999px; font-size: .74rem;
    font-weight: 600; letter-spacing: .04em; text-transform: uppercase; color: #93c5fd;
    background: rgba(59,130,246,.14); border: 1px solid rgba(96,165,250,.35); }
.carte { padding: 16px 20px; border-radius: 18px; border: 1px solid rgba(148,163,184,.16);
    background: linear-gradient(145deg, rgba(30,41,66,.7), rgba(15,22,38,.7)); box-shadow: 0 8px 24px rgba(0,0,0,.3); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid rgba(148,163,184,.15); }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 16px; font-weight: 600; }
[data-testid="stDataFrame"] { border: 1px solid rgba(148,163,184,.16); border-radius: 12px; overflow: hidden; }
div[data-testid="stExpander"] { border: 1px solid rgba(148,163,184,.16); border-radius: 14px; background: rgba(15,22,38,.55); }
.stButton > button, .stDownloadButton > button { border-radius: 12px; border: 1px solid rgba(96,165,250,.55);
    background: linear-gradient(135deg, #2563eb, #1d4ed8); color: #fff; font-weight: 600; }
.stButton > button:hover { border-color: #93c5fd; box-shadow: 0 0 0 3px rgba(96,165,250,.2); }
div[data-baseweb="select"] > div, .stTextInput input, .stDateInput input { border-radius: 12px !important; }
hr { border-color: rgba(148,163,184,.15) !important; }
footer, #MainMenu { visibility: hidden; }
.carte { height: 100%; }
.carte .ico { width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center;
    justify-content: center; margin-bottom: 10px; }
.carte .titre { font-weight: 700; font-size: 1.02rem; margin-bottom: 4px; }
.carte .texte { opacity: .80; font-size: .92rem; line-height: 1.45; }
.carte .grand { font-size: 1.9rem; font-weight: 800; line-height: 1.1; }
</style>
"""


def appliquer_style():
    st.markdown(CSS, unsafe_allow_html=True)


def entete(titre, sous_titre="", etiquette=""):
    pill = f"<span class='pill'>{etiquette}</span><br>" if etiquette else ""
    sous = f"<p>{sous_titre}</p>" if sous_titre else ""
    st.markdown(f"<div class='hero'>{pill}<h1>{titre}</h1>{sous}</div>", unsafe_allow_html=True)


def tableau(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def absent(nom):
    st.info(f"Fichier `{nom}` absent de `app/donnees/`. Relance `python src/preparer_app.py` "
            "apres avoir execute l'etape correspondante.")


# ------------------------------------------------------------------ lecture
@st.cache_data(show_spinner=False)
def serie_temp(nom):
    f = DONNEES / nom
    if not f.exists():
        return None
    return pd.read_csv(f, index_col=0, parse_dates=True)


@st.cache_data(show_spinner=False)
def table(nom, index_col=0):
    f = DONNEES / nom
    if not f.exists():
        return None
    return pd.read_csv(f, index_col=index_col)


@st.cache_data(show_spinner=False)
def json_(nom):
    f = DONNEES / nom
    return json.load(open(f, encoding="utf-8")) if f.exists() else None


def marches_disponibles():
    m = json_("modeles.json") or {}
    return {v["nom"]: k for k, v in m.items() if (DONNEES / f"regimes_{k}.csv").exists()}


def actifs_backtest():
    m = json_("bt_meta.json") or {}
    return {v["nom"]: k for k, v in m.items() if (DONNEES / f"bt_{k}.csv").exists()}


# ------------------------------------------------------------------ analyse des regimes
def segments(lab):
    lab = np.asarray(lab)
    bornes = np.r_[0, np.flatnonzero(np.diff(lab) != 0) + 1, len(lab)]
    return np.diff(bornes), lab[bornes[:-1]]


def stats_regimes(ret, lab, ppy=252):
    """Tableau descriptif par regime."""
    ret, lab = np.asarray(ret), np.asarray(lab)
    long, etats = segments(lab)
    lignes = {}
    for k in range(3):
        m = lab == k
        if not m.any():
            continue
        d = long[etats == k]
        lignes[REG[k]] = {"Jours": int(m.sum()), "% du temps": m.mean() * 100,
                          "Rendement annualise (%)": ret[m].mean() * ppy * 100,
                          "Volatilite annualisee (%)": ret[m].std() * np.sqrt(ppy) * 100,
                          "Pire jour (%)": ret[m].min() * 100, "Meilleur jour (%)": ret[m].max() * 100,
                          "Episodes": len(d), "Duree mediane (j)": float(np.median(d)),
                          "Duree moyenne (j)": float(d.mean()), "Duree max (j)": int(d.max())}
    return pd.DataFrame(lignes).T


def calculer_evenements(prix):
    """Pic -> creux de chaque evenement, calcules sur les donnees."""
    ev = {}
    for nom, (a, b) in FENETRES_EVENEMENTS.items():
        seg = prix.loc[a:b]
        if len(seg) < 20:
            continue
        tp = seg.idxmax()
        tc = seg.loc[tp:].idxmin()
        if tc > tp:
            ev[nom] = (tp, tc, float(seg.loc[tc] / seg.loc[tp] - 1))
    return ev


def periodes_nber(usrec):
    if usrec is None:
        return []
    u = usrec["USREC"] if "USREC" in usrec else usrec.iloc[:, 0]
    out, debut = [], None
    for d, v in u.items():
        if v == 1 and debut is None:
            debut = d
        if v == 0 and debut is not None:
            out.append((debut, d))
            debut = None
    if debut is not None:
        out.append((debut, u.index[-1] + pd.offsets.MonthEnd(1)))
    return out


# ------------------------------------------------------------------ graphiques
def fig_prix_regimes(idx, prix, lab, titre, log=True, zones=None, hauteur=430):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=idx, y=prix, mode="lines", showlegend=False,
                               line=dict(color=PRIX_LIGNE, width=1), hoverinfo="skip"))
    lab = np.asarray(lab)
    for k in range(3):
        m = lab == k
        if m.any():
            fig.add_trace(go.Scattergl(x=idx[m], y=np.asarray(prix)[m], mode="markers", name=REG[k],
                                       marker=dict(size=4, color=COUL[k]),
                                       hovertemplate="%{x|%d/%m/%Y}<br>%{y:,.0f}<extra>" + REG[k] + "</extra>"))
    for z in (zones or []):
        fig.add_vrect(x0=z[0], x1=z[1], fillcolor=z[2], opacity=z[3], line_width=0, layer="below")
    fig.update_layout(title=titre, height=hauteur, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="closest")
    if log:
        fig.update_yaxes(type="log")
    return fig


def fig_probas(idx, probas, titre, hauteur=260):
    fig = go.Figure()
    for k in range(3):
        fig.add_trace(go.Scatter(x=idx, y=probas[k], mode="lines", stackgroup="p", name=REG[k],
                                 line=dict(width=0.3, color=COUL[k]), fillcolor=COUL[k]))
    fig.update_layout(title=titre, height=hauteur, margin=dict(l=10, r=10, t=45, b=10),
                      yaxis=dict(range=[0, 1]), legend=dict(orientation="h", y=1.15, x=0))
    return fig


def fig_heatmap(A, titre, hauteur=330):
    A = np.asarray(A)
    fig = go.Figure(go.Heatmap(z=A, x=REG, y=REG, zmin=0, zmax=1, colorscale=ECHELLE_BLEUE,
                               text=[[f"{v:.3f}" for v in l] for l in A], texttemplate="%{text}",
                               textfont=dict(color="white", size=15), showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=titre, height=hauteur, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="Vers", yaxis_title="De")
    return fig


# ------------------------------------------------------------------ backtest
def hysteresis(p, s_in, s_out):
    v = np.asarray(p, dtype=float)
    e = np.zeros(len(v), dtype=int)
    for t in range(1, len(v)):
        if e[t - 1] == 0 and v[t] > s_in:
            e[t] = 1
        elif e[t - 1] == 1 and v[t] < s_out:
            e[t] = 0
        else:
            e[t] = e[t - 1]
    return e


def construire_strategies(d, s_in, s_out):
    """d : DataFrame (prix, p, [vix]). Renvoie {nom: exposition 0..1 decidee a la cloture de t}."""
    idx = d.index
    ma = d["prix"].rolling(200).mean()
    expo = {"Buy & Hold": pd.Series(1.0, index=idx),
            "Prix > MA200": ((d["prix"] >= ma) | ma.isna()).astype(float)}
    if "vix" in d and d["vix"].notna().any():
        expo["VIX > 30"] = (d["vix"].fillna(0) <= 30).astype(float)
    expo["HMM binaire"] = pd.Series(1 - hysteresis(d["p"].values, s_in, s_out), index=idx)
    expo["HMM graduel (1 - P crise)"] = 1 - d["p"].fillna(0)
    return expo


def evaluer(expo, ret_log, debut, fin, cout, ppy=252, decalage=1):
    e = expo.shift(decalage) if decalage else expo
    e, r = e.loc[debut:fin], (np.exp(ret_log) - 1).loc[debut:fin]
    turn = e.diff().abs().fillna(0)
    x = e * r - turn * cout
    cum = (1 + x).cumprod()
    n = len(x)
    ann = cum.iloc[-1] ** (ppy / n) - 1
    vol = x.std() * np.sqrt(ppy)
    mdd = (cum / cum.cummax() - 1).min()
    return x, cum, {"Rendement annualise": ann, "Volatilite": vol, "Sharpe": ann / vol if vol else np.nan,
                    "Max drawdown": mdd, "Calmar": ann / abs(mdd) if mdd else np.nan,
                    "Changements de position": int((turn > 1e-9).sum()),
                    "Rotation / an": turn.sum() / (n / ppy), "Part investie": e.mean()}


def bootstrap_diff(xa, xb, ppy=252, B=1000, bloc=20, graine=42):
    rng = np.random.default_rng(graine)
    a, b = np.asarray(xa), np.asarray(xb)
    n = len(a)
    nb = int(np.ceil(n / bloc))

    def sh(x):
        return x.mean() / x.std(ddof=1) * np.sqrt(ppy)

    d = np.empty(B)
    for i in range(B):
        st_ = rng.integers(0, n - bloc, nb)
        ix = (st_[:, None] + np.arange(bloc)).ravel()[:n]
        d[i] = sh(a[ix]) - sh(b[ix])
    return sh(a) - sh(b), np.percentile(d, [2.5, 97.5]), float((d <= 0).mean())
