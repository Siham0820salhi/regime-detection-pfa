"""
Etape 9b : validation de H1 sur les evenements, dates de transition,
et figures 9 a 15 (regimes, transitions, comparaison visuelle, smoothed vs filtered,
heatmaps, durees, volatilites). Prerequis : etape9a_comparaison.py.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from outils import *

pd.set_option("display.width", 220)
REF = "hmm_ret3"                       # modele de reference pour l'analyse des transitions
NOMS = {"km": "K-means", "gmm": "GMM", "hmm": "HMM (ret+vol20)",
        "hmm_sticky": "HMM sticky (ret+vol20)", "hmm_ret3": "HMM (ret seul)"}
COUL3 = np.array(["green", "orange", "red"])
COUL2 = np.array(["green", "red"])
ZOOM = ("2007-01-01", "2010-12-31")

df = charger_features()
reg = pd.read_csv("data/regimes_A.csv", index_col=0, parse_dates=True)
ret = df["ret"].values

# ---------- 1. Evenements historiques (pic -> creux calcules sur les donnees) ----------
FENETRES = {"Bulle internet (2000-2002)": ("2000-10-01", "2003-01-01"),
            "Crise financiere 2008": ("2007-06-01", "2009-12-31"),
            "COVID 2020": ("2020-01-01", "2020-06-30"),
            "Inflation / taux 2022": ("2021-12-01", "2022-12-31"),
            "Chocs 2025": ("2025-01-01", "2025-06-30")}
events = {}
for nom, (a, b) in FENETRES.items():
    seg = df["prix"].loc[a:b]
    if len(seg) < 20:
        continue
    t_pic = seg.idxmax()
    t_creux = seg.loc[t_pic:].idxmin()
    events[nom] = (t_pic, t_creux, seg.loc[t_creux] / seg.loc[t_pic] - 1)
print("=== 1. Evenements : pic -> creux (calcules sur tes donnees) ===")
print(pd.DataFrame({n: {"pic": e[0].date(), "creux": e[1].date(),
                        "perte": f"{e[2]:.1%}", "jours": int((e[1] - e[0]).days)}
                    for n, e in events.items()}).T.to_string())

modeles_h1 = {**{k: NOMS[k] for k in ["km", "gmm", "hmm", "hmm_sticky", "hmm_ret3"]}}
lignes = {}
for cle, nom in modeles_h1.items():
    lab = reg[cle].values
    crise = lab.max()
    ligne = {"base (toute la periode)": (lab == crise).mean()}
    for ev, (tp, tc, _) in events.items():
        m = (df.index >= tp) & (df.index <= tc)
        ligne[ev] = (lab[m] == crise).mean()
    lignes[nom] = ligne
for cle, nom in [("hmm_k2_smooth", "HMM K=2 smoothed"), ("hmm_k2_filt", "HMM K=2 filtered")]:
    lab = reg[cle].values
    ligne = {"base (toute la periode)": lab.mean()}
    for ev, (tp, tc, _) in events.items():
        m = (df.index >= tp) & (df.index <= tc)
        ligne[ev] = lab[m].mean()
    lignes[nom] = ligne
h1 = pd.DataFrame(lignes).T
print("\nPart des jours classes en regime de CRISE entre le pic et le creux de chaque evenement")
print("(H1 est soutenue si ces parts sont bien superieures a la colonne 'base') :")
print((h1 * 100).round(1).to_string())
h1.round(4).to_csv("data/h1_evenements.csv")

# ---------- 2. Dates de transition du modele de reference ----------
lab = reg[REF].values
K = int(lab.max()) + 1
idx_ch = np.flatnonzero(np.diff(lab) != 0) + 1
rows = []
for i in idx_ch:
    d = df.index[i]
    av = df.iloc[max(0, i - 10):i]
    ap = df.iloc[i:i + 10]
    rows.append({"date": d.date(), "de": int(lab[i - 1]), "vers": int(lab[i]),
                 "prix": round(df["prix"].iloc[i], 1), "vix": round(df["vix"].iloc[i], 1),
                 "drawdown": round(df["drawdown"].iloc[i], 3),
                 "vol20": round(df["vol20"].iloc[i], 3),
                 "vix_moy_-10j": round(av["vix"].mean(), 1),
                 "vix_moy_+10j": round(ap["vix"].mean(), 1),
                 "vol20_moy_-10j": round(av["vol20"].mean(), 3),
                 "vol20_moy_+10j": round(ap["vol20"].mean(), 3)})
trans = pd.DataFrame(rows)
trans.to_csv("data/transitions_hmm.csv", index=False)
print(f"\n=== 2. Transitions du modele de reference ({NOMS[REF]}) : {len(trans)} au total ===")
print("Liste complete : data/transitions_hmm.csv. Entrees en regime de crise :")
ent = trans[trans["vers"] == K - 1]
print(ent.head(15).to_string(index=False))
print(f"... {len(ent)} entrees en crise.  VIX moyen a l'entree : {ent['vix'].mean():.1f}  "
      f"| vol20 moyenne a l'entree : {ent['vol20'].mean():.2f}")

long, etats = segments(lab)
debuts = np.r_[0, idx_ch]
ep = pd.DataFrame({"debut": df.index[debuts], "duree_j": long, "regime": etats})
ep = ep[(ep["regime"] == K - 1) & (ep["duree_j"] >= 20)].copy()
ep["fin"] = [df.index[min(df.index.get_loc(d) + n - 1, len(df) - 1)]
             for d, n in zip(ep["debut"], ep["duree_j"])]
ep["drawdown_min"] = [df["drawdown"].loc[d:f].min() for d, f in zip(ep["debut"], ep["fin"])]
ep["vix_max"] = [df["vix"].loc[d:f].max() for d, f in zip(ep["debut"], ep["fin"])]
print("\nEpisodes de crise (>= 20 jours) :")
aff = ep.assign(debut=ep["debut"].dt.date, fin=ep["fin"].dt.date)
print(aff[["debut", "fin", "duree_j", "drawdown_min", "vix_max"]].round(2).to_string(index=False))
ep.to_csv("data/episodes_crise.csv", index=False)

# ---------- 3. Figures ----------
def couleurs(l):
    return (COUL3 if int(np.max(l)) == 2 else COUL2)[np.asarray(l)]


def bandes(ax):
    for nom, (tp, tc, _) in events.items():
        ax.axvspan(tp, tc, color="grey", alpha=0.25)
        ax.text(tp, ax.get_ylim()[1], nom.split(" (")[0], fontsize=7, va="top", rotation=90)


# Fig 9 : regimes + evenements
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df["prix"], color="black", lw=0.4)
ax.scatter(df.index, df["prix"], s=3, c=couleurs(lab))
ax.set_yscale("log")
bandes(ax)
ax.set_title(f"S&P 500 : regimes detectes ({NOMS[REF]}) et evenements historiques (zones grises)")
plt.tight_layout(); plt.savefig("figures/fig9_regimes_evenements.png", dpi=200)

# Fig 10 : transitions sur prix et indicateurs
fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
for a, col, titre in zip(ax, ["prix", "vix", "vol20", "drawdown"],
                         ["Prix (log)", "VIX", "Volatilite 20 j", "Drawdown"]):
    a.plot(df[col], color="black", lw=0.6)
    a.set_title(titre)
    for _, t in trans.iterrows():
        if t["vers"] == K - 1:
            a.axvline(pd.Timestamp(t["date"]), color="red", lw=0.5, alpha=0.5)
        elif t["de"] == K - 1:
            a.axvline(pd.Timestamp(t["date"]), color="green", lw=0.5, alpha=0.5)
ax[0].set_yscale("log")
ax[0].set_title("Prix (log) : traits rouges = entree en crise, verts = sortie de crise")
plt.tight_layout(); plt.savefig("figures/fig10_transitions.png", dpi=200)

# Fig 11 : comparaison visuelle des modeles sur 2007-2010
d0, d1 = pd.Timestamp(ZOOM[0]), pd.Timestamp(ZOOM[1])
m = (df.index >= d0) & (df.index <= d1)
fig, ax = plt.subplots(len(NOMS), 1, figsize=(14, 14), sharex=True)
for a, (cle, nom) in zip(ax, NOMS.items()):
    a.plot(df.index[m], df["prix"][m], color="black", lw=0.4)
    a.scatter(df.index[m], df["prix"][m], s=5, c=couleurs(reg[cle].values)[m])
    a.set_title(f"{nom} : {int((np.diff(reg[cle].values[m]) != 0).sum())} transitions sur la periode")
plt.suptitle(f"Comparaison des modeles, {ZOOM[0][:4]}-{ZOOM[1][:4]}")
plt.tight_layout(); plt.savefig("figures/fig11_comparaison_modeles.png", dpi=200)

# Fig 12 : smoothed vs filtered
fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for a, cle, t in zip(ax, ["hmm_k2_smooth", "hmm_k2_filt"],
                     ["Smoothed (utilise le futur)", "Filtered (temps reel)"]):
    a.plot(df.index[m], df["prix"][m], color="black", lw=0.4)
    a.scatter(df.index[m], df["prix"][m], s=5, c=COUL2[reg[cle].values][m])
    a.set_title(f"HMM K=2 : {t}, {int((np.diff(reg[cle].values[m]) != 0).sum())} transitions")
plt.tight_layout(); plt.savefig("figures/fig12_smoothed_vs_filtered.png", dpi=200)
dif = (reg["hmm_k2_smooth"] != reg["hmm_k2_filt"])
print(f"\n=== Smoothed vs filtered (HMM K=2) ===")
print(f"Jours en desaccord : {dif.mean():.1%} sur toute la periode ; {dif[m].mean():.1%} sur {ZOOM[0][:4]}-{ZOOM[1][:4]}")
print("Les desaccords sont concentres autour des changements de regime : le filtre "
      "reagit avec retard puis oscille, le lissage 'voit' la suite.")

# Fig 13 : heatmaps de transition (matrices empiriques)
fig, ax = plt.subplots(1, len(NOMS), figsize=(20, 4.2))
for a, (cle, nom) in zip(ax, NOMS.items()):
    k = int(reg[cle].max()) + 1
    A = matrice_empirique(reg[cle].values, k)
    im = a.imshow(A, vmin=0, vmax=1, cmap="Blues")
    a.set_xticks(range(k), ["calme", "moyen", "crise"][:k] if k == 3 else range(k))
    a.set_yticks(range(k), ["calme", "moyen", "crise"][:k] if k == 3 else range(k))
    a.set_title(nom, fontsize=10)
    for i in range(k):
        for j in range(k):
            a.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center",
                   color="white" if A[i, j] > 0.5 else "black")
plt.suptitle("Matrices de transition empiriques (de la ligne vers la colonne)")
plt.tight_layout(); plt.savefig("figures/fig13_heatmaps_transition.png", dpi=200)

# Fig 14 : durees des regimes
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
data = [segments(reg[c].values)[0] for c in NOMS]
ax[0].boxplot(data, tick_labels=[n.replace(" (", "\n(") for n in NOMS.values()])
ax[0].set_yscale("log"); ax[0].set_ylabel("duree d'un segment (jours, log)")
ax[0].set_title("Duree des segments de regime")
ax[1].bar(range(len(NOMS)), [(d < 10).mean() for d in data], color="tab:red")
ax[1].set_xticks(range(len(NOMS)), [n.replace(" (", "\n(") for n in NOMS.values()])
ax[1].set_title("Part des segments de moins de 10 jours (plus bas = plus stable)")
plt.tight_layout(); plt.savefig("figures/fig14_durees.png", dpi=200)

# Fig 15 : volatilite par regime
fig, ax = plt.subplots(figsize=(11, 5))
larg = 0.25
for j, (nom_r, c) in enumerate(zip(["calme", "moyen", "crise"], COUL3)):
    v = [ret[reg[cle].values == j].std() * np.sqrt(252) * 100 if (reg[cle].values == j).any()
         else np.nan for cle in NOMS]
    ax.bar(np.arange(len(NOMS)) + (j - 1) * larg, v, larg, color=c, label=nom_r)
ax.set_xticks(range(len(NOMS)), [n.replace(" (", "\n(") for n in NOMS.values()])
ax.set_ylabel("volatilite annualisee (%)")
ax.set_title("Volatilite par regime et par modele")
ax.legend()
plt.tight_layout(); plt.savefig("figures/fig15_vol_par_regime.png", dpi=200)
print("\nFigures : fig9 a fig15 dans figures/")
plt.show()
