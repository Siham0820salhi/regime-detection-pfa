"""
Etape 10 : detection de ruptures (PELT) et jump model, compares au HMM.

- PELT (ruptures) : trouve les dates de rupture de la distribution des rendements (OFFLINE).
- Jump model : clustering avec penalite lambda sur les changements d'etat.
  Version offline (DP complet) et version online (etat estime avec le passe seulement).
Prerequis : etape9a_comparaison.py (fichier data/regimes_A.csv).
Les analyses sont OFFLINE (parametres sur toute la periode) : descriptives, pas temps reel.
La comparaison temps reel de tous les modeles se fait a l'etape 11.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import ruptures as rpt
warnings.filterwarnings("ignore", message="New behaviour in v1.1.5")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from outils import *
from modele_saut import *

pd.set_option("display.width", 220)
rng = np.random.default_rng(42)
TOL = [10, 20]                        # tolerance de comparaison des dates (jours de bourse)
MIN_SIZE = 20                         # segment PELT minimum (20 jours)

df = charger_features()
idx = df.index
ret = df["ret"].values
T = len(df)
regA = pd.read_csv("data/regimes_A.csv", index_col=0, parse_dates=True)


def positions(lab):
    return np.flatnonzero(np.diff(np.asarray(lab)) != 0) + 1


# =====================================================================
# 1. PELT
# =====================================================================
sig = (ret * 100).reshape(-1, 1)
pen_bic = 3 * np.log(T)               # 3 parametres par segment (moyenne, variance, position)
penalites = {"10": 10, "20": 20, f"BIC ({pen_bic:.0f})": pen_bic, "50": 50, "100": 100, "200": 200}
print("=== 1. PELT : sensibilite a la penalite (modele 'normal' : moyenne et variance) ===")
bk_all, lignes = {}, {}
for nom, pen in penalites.items():
    b = rpt.Pelt(model="normal", min_size=MIN_SIZE, jump=1).fit(sig).predict(pen=pen)[:-1]
    bk_all[nom] = np.array(b)
    lignes[f"pen={nom}"] = {"ruptures": len(b),
                            "duree moyenne segment (j)": T / (len(b) + 1)}
print(pd.DataFrame(lignes).T.round(1).to_string())
nom_ref = [n for n in penalites if n.startswith("BIC")][0]
bk = bk_all[nom_ref]
print(f"\nPenalite retenue : critere BIC -> {len(bk)} ruptures")

# segments, volatilite de chaque segment, classification calme/crise (2 groupes)
bornes = np.r_[0, bk, T]
seg_vol = np.array([ret[a:b].std() * np.sqrt(252) for a, b in zip(bornes[:-1], bornes[1:])])
seg_len = np.diff(bornes)
if len(seg_vol) >= 2:
    km = KMeans(2, n_init=10, random_state=0).fit(np.log(seg_vol).reshape(-1, 1),
                                                  sample_weight=seg_len)
    haut = int(np.argmax(km.cluster_centers_.ravel()))
    seg_crise = (km.labels_ == haut).astype(int)
else:
    seg_crise = np.zeros(len(seg_vol), dtype=int)
lab_pelt = np.repeat(seg_crise, seg_len)
print(f"Segments : {len(seg_vol)} ; en regime de crise : {seg_crise.sum()} "
      f"({lab_pelt.mean():.1%} des jours) ; vol segments calmes {seg_vol[seg_crise == 0].mean():.0%}"
      f" / crise {seg_vol[seg_crise == 1].mean():.0%}")
pd.DataFrame({"debut": idx[bornes[:-1]].date, "fin": idx[bornes[1:] - 1].date, "jours": seg_len,
              "vol_annualisee": seg_vol.round(3), "crise": seg_crise}).to_csv(
    "data/pelt_segments.csv", index=False)

# =====================================================================
# 2. Jump model
# =====================================================================
F = StandardScaler().fit_transform(features_saut(ret))
print("\n=== 2. Jump model : balayage de lambda (variables causales : rendement, risque baissier, vol) ===")


def balayage(K, lams):
    lignes, modeles, c_prec = {}, {}, None
    for lam in lams:
        s, c, _ = ajuster_saut(F, K, lam, centres_init=c_prec)
        lab, mapping = reordonner(s, ret)
        c_prec = c                                       # demarrage a chaud pour le lambda suivant
        m = metriques_regimes(lab, ret)
        lignes[f"K={K} lambda={lam}"] = {"transitions": m["transitions"],
                                         "duree_med_j": m["duree_med_j"],
                                         "% segments < 10 j": m["% segments < 10 j"],
                                         "vol_calme": m["vol_calme"], "vol_crise": m["vol_crise"],
                                         "% jours en crise": m["% jours en crise"]}
        modeles[lam] = (lab, c)
    return pd.DataFrame(lignes).T, modeles


LAMS = [0, 5, 10, 20, 50, 100, 200, 400]
tab2, mod2 = balayage(2, LAMS)
tab3, mod3 = balayage(3, LAMS)
tab_l = pd.concat([tab2, tab3])
print(tab_l.round(3).to_string())
tab_l.round(4).to_csv("data/jump_balayage_lambda.csv")


def choisir_lambda(tab, K):
    """Regle sans regarder la performance : plus petit lambda avec < 5 % de segments courts."""
    for lam in LAMS:
        if tab.loc[f"K={K} lambda={lam}", "% segments < 10 j"] < 0.05:
            return lam
    return LAMS[-1]


lam2, lam3 = choisir_lambda(tab2, 2), choisir_lambda(tab3, 3)
print(f"\nlambda retenu (regle : < 5 % de segments courts, choisi sans regarder la performance) : "
      f"K=2 -> {lam2} ; K=3 -> {lam3}")
lab_j2, c2 = mod2[lam2]
lab_j3, c3 = mod3[lam3]

# version online (temps reel) du jump model K=2
s_on = etat_online(pertes(F, c2), lam2)
# aligner les etiquettes de l'online sur celles de l'offline (meme ordre de volatilite)
_, mp = reordonner(dp_saut(pertes(F, c2), lam2), ret)
lab_on = np.array([mp[k] for k in s_on])
print(f"\nJump model K=2 : transitions offline {len(positions(lab_j2))}  |  online {len(positions(lab_on))}"
      f"  |  accord {np.mean(lab_j2 == lab_on):.1%}")
print("(rappel HMM K=2 : smoothed 89, filtered 355)")

reg = pd.DataFrame({"pelt": lab_pelt, "jump2": lab_j2, "jump3": lab_j3, "jump2_online": lab_on},
                   index=idx)
reg.to_csv("data/regimes_B.csv", index_label="Date")

# comparaison avec les autres modeles sur les memes dates
cmp_ = {"PELT (BIC)": lab_pelt, f"Jump K=3 (l={lam3})": lab_j3,
        "HMM K=3 (ret seul)": regA["hmm_ret3"].values}
print("\n=== Comparaison a K=3 / K=2 (memes dates) ===")
rows = {n: metriques_regimes(l, ret) for n, l in cmp_.items() if int(np.max(l)) >= 1}
rows[f"Jump K=2 (l={lam2})"] = metriques_regimes(lab_j2, ret)
rows["HMM K=2 smoothed"] = metriques_regimes(regA["hmm_k2_smooth"].values, ret)
print(pd.DataFrame(rows).T[["transitions", "duree_moy_j", "duree_med_j", "% segments < 10 j",
                            "vol_calme", "vol_crise", "% jours en crise"]].round(3).to_string())

# =====================================================================
# 3. Dates : PELT / Jump / HMM
# =====================================================================
dates = {"PELT": bk, f"Jump K=2": positions(lab_j2), "HMM K=2 smoothed": positions(regA["hmm_k2_smooth"]),
         "HMM K=3 (ret seul)": positions(regA["hmm_ret3"])}


def part_proches(A, B, tol):
    return float(np.mean([np.abs(B - a).min() <= tol for a in A])) if len(A) and len(B) else np.nan


print("\n=== 3. Les dates se correspondent-elles ? (part des dates de la ligne a moins de tol jours "
      "d'une date de la colonne) ===")
for tol in TOL:
    mat = pd.DataFrame({b: {a: part_proches(dates[a], dates[b], tol) if a != b else np.nan
                            for a in dates} for b in dates})
    print(f"\ntolerance +/- {tol} jours de bourse :")
    print((mat * 100).round(0).to_string(na_rep="-"))
    mat.round(3).to_csv(f"data/concordance_dates_tol{tol}.csv")

print("\nTest de hasard (mêmes nombre de dates tirees au hasard, 2000 tirages), tol = 10 j :")
for a in ["PELT", "Jump K=2"]:
    for b in ["HMM K=2 smoothed", "HMM K=3 (ret seul)"]:
        obs = part_proches(dates[a], dates[b], 10)
        sim = np.array([part_proches(np.sort(rng.integers(1, T, len(dates[a]))), dates[b], 10)
                        for _ in range(2000)])
        print(f"  {a:9s} -> {b:20s} observe {obs:.0%} | hasard {sim.mean():.0%} "
              f"(95e percentile {np.percentile(sim, 95):.0%}) | p = {(sim >= obs).mean():.3f}")

# =====================================================================
# 4. H1 : part des jours classes en crise pendant les baisses connues
# =====================================================================
FEN = {"Bulle internet": ("2000-10-01", "2003-01-01"), "Crise 2008": ("2007-06-01", "2009-12-31"),
       "COVID 2020": ("2020-01-01", "2020-06-30"), "Inflation 2022": ("2021-12-01", "2022-12-31"),
       "Chocs 2025": ("2025-01-01", "2025-04-30")}
ev = {}
for n, (a, b) in FEN.items():
    seg = df["prix"].loc[a:b]
    if len(seg) > 20:
        tp = seg.idxmax()
        ev[n] = (tp, seg.loc[tp:].idxmin())
print("\n=== 4. H1 : % de jours en regime de crise entre le pic et le creux ===")
lab_h = {"PELT": lab_pelt, f"Jump K=2 offline": lab_j2, "Jump K=2 online": lab_on,
         "HMM K=2 smoothed": regA["hmm_k2_smooth"].values, "HMM K=2 filtered": regA["hmm_k2_filt"].values}
h1 = {}
for n, l in lab_h.items():
    crise = int(np.max(l))
    ligne = {"base": np.mean(np.asarray(l) == crise) * 100}
    for e, (tp, tc) in ev.items():
        mk = (idx >= tp) & (idx <= tc)
        ligne[e] = np.mean(np.asarray(l)[mk] == crise) * 100
    h1[n] = ligne
print(pd.DataFrame(h1).T.round(1).to_string())

# =====================================================================
# 5. Figures
# =====================================================================
col2, col3 = np.array(["green", "red"]), np.array(["green", "orange", "red"])

fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
ax[0].plot(df["prix"], color="black", lw=0.4)
ax[0].scatter(idx, df["prix"], s=2, c=col2[lab_pelt])
ax[0].set_yscale("log")
ax[0].set_title(f"PELT (penalite BIC) : {len(bk)} ruptures, segments colores par niveau de volatilite")
ax[1].plot(df["vol20"], color="black", lw=0.5)
for p in bk:
    ax[1].axvline(idx[p], color="blue", lw=0.5, alpha=0.6)
ax[1].set_title("Volatilite 20 j et ruptures PELT (traits bleus)")
ax[2].plot(df["vol20"], color="black", lw=0.5)
for p in dates["HMM K=2 smoothed"]:
    ax[2].axvline(idx[p], color="red", lw=0.4, alpha=0.5)
ax[2].set_title("Volatilite 20 j et transitions HMM K=2 smoothed (traits rouges)")
plt.tight_layout(); plt.savefig("figures/fig16_pelt.png", dpi=200)

fig, ax = plt.subplots(3, 1, figsize=(14, 11))
ax[0].plot(df["prix"], color="black", lw=0.4)
ax[0].scatter(idx, df["prix"], s=2, c=col3[lab_j3])
ax[0].set_yscale("log")
ax[0].set_title(f"Jump model K=3, lambda={lam3} : {len(positions(lab_j3))} transitions")
z = (idx >= "2007-01-01") & (idx <= "2010-12-31")
ax[1].plot(idx[z], df["prix"][z], color="black", lw=0.4)
ax[1].scatter(idx[z], df["prix"][z], s=6, c=col2[lab_j2][z])
ax[1].scatter(idx[z], df["prix"][z] * 0.93, s=6, c=col2[lab_on][z], marker="s")
ax[1].set_title("Jump K=2, 2007-2010 : offline (courbe) et online (carres, decales vers le bas)")
for K, tab, c in [(2, tab2, "tab:blue"), (3, tab3, "tab:orange")]:
    ax[2].plot(LAMS, tab["transitions"].values, marker="o", color=c, label=f"K={K}")
ax[2].set_xscale("symlog"); ax[2].set_yscale("log"); ax[2].legend()
ax[2].axhline(125, color="grey", ls="--", lw=0.8)
ax[2].set_xlabel("lambda"); ax[2].set_ylabel("transitions")
ax[2].set_title("Nombre de transitions selon lambda (tirets : HMM K=3 ret seul = 125)")
plt.tight_layout(h_pad=3); plt.savefig("figures/fig17_jump_model.png", dpi=200)

fig, ax = plt.subplots(figsize=(14, 4))
couleurs = ["tab:blue", "tab:green", "tab:red", "tab:orange"]
for i, (n, p) in enumerate(dates.items()):
    ax.eventplot(idx[p], lineoffsets=i, linelengths=0.8, colors=couleurs[i], linewidths=0.8)
ax.set_yticks(range(len(dates)), list(dates.keys()))
ax.set_title("Dates de changement de regime par methode")
plt.tight_layout(); plt.savefig("figures/fig18_dates_methodes.png", dpi=200)
plt.show()
print("\nFigures : fig16 a fig18. Fichiers : data/regimes_B.csv, pelt_segments.csv, "
      "jump_balayage_lambda.csv, concordance_dates_tol*.csv")
