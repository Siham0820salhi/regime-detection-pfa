"""
Etape 11a : TEST CHRONOLOGIQUE INDEPENDANT (temps reel, sans fuite d'information).

Protocole
  1. Periode de DEVELOPPEMENT : de la 1re prediction (apres 5 ans d'historique) au 31/12/2014.
     Seuls les hyperparametres sont choisis ici : seuils d'hysteresis du HMM, lambda du jump model.
  2. Periode de TEST : a partir du 01/01/2015, evaluee UNE SEULE FOIS avec les choix figes.
  3. Walk-forward : tous les modeles sont reentraines chaque annee sur le PASSE SEUL ; les
     etats sont estimes en temps reel (filtre forward / etat online). Signal decale d'un jour,
     couts de transaction a chaque changement d'exposition.
  4. Regles sans parametre (VIX > 30, prix > MA200) : aucun reglage.
Usage : python src/etape11a_test_final.py            (separation au 2015-01-01)
        python src/etape11a_test_final.py 2010-01-01  (autre date, pour tester)
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from outils import *
from modele_saut import *

pd.set_option("display.width", 220)
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "2015-01-01"
TRAIN_MIN, REFIT, COUT = 1260, 252, 0.0005
G_IN, G_OUT = [0.6, 0.7, 0.8, 0.9], [0.2, 0.3, 0.4, 0.5]
LAMS = [5, 10, 20, 50, 100]
DEF_IN, DEF_OUT = 0.8, 0.4
rng = np.random.default_rng(42)

df = charger_features()
idx, T = df.index, len(df)
ret = df["ret"].values
X = (ret * 100).reshape(-1, 1)
S = int(idx.searchsorted(pd.Timestamp(SPLIT)))
refits = sorted(set(range(TRAIN_MIN, S, REFIT)) | set(range(S, T, REFIT)))
blocs = list(zip(refits, refits[1:] + [T]))
blocs_dev = [b for b in blocs if b[0] < S]
blocs_test = [b for b in blocs if b[0] >= S]
a_dev, b_dev = idx[TRAIN_MIN + 1], idx[S - 1]
a_test, b_test = idx[S], idx[-1]
print(f"Developpement : {a_dev.date()} -> {b_dev.date()}  ({len(blocs_dev)} reentrainements)")
print(f"Test          : {a_test.date()} -> {b_test.date()}  ({len(blocs_test)} reentrainements)")


# ---------- Evaluation ----------
def evaluer(expo, a, b, cout=COUT):
    e = expo.shift(1).loc[a:b]
    r = (np.exp(df["ret"]) - 1).loc[a:b]
    turn = e.diff().abs().fillna(0)
    x = e * r - turn * cout
    cum = (1 + x).cumprod()
    ann = cum.iloc[-1] ** (252 / len(x)) - 1
    vol = x.std() * np.sqrt(252)
    mdd = (cum / cum.cummax() - 1).min()
    return x, cum, {"rend_ann": ann, "vol": vol, "sharpe": ann / vol, "max_dd": mdd,
                    "calmar": ann / abs(mdd), "changements": int((turn > 0).sum()),
                    "rotation/an": turn.sum() / (len(x) / 252), "investi": e.mean()}


# ---------- HMM walk-forward (K=2, rendements, filtre forward) ----------
def p_hmm():
    f = f"data/p_hmm_split_{SPLIT}.csv"
    if os.path.exists(f):
        s = pd.read_csv(f, index_col=0, parse_dates=True)["p"]
        if len(s) == T:
            return s
    print("Walk-forward HMM (premiere fois : quelques minutes)...")
    p = np.full(T, np.nan)
    for t0, t1 in blocs:
        m = ajuster_hmm(X[:t0], 2, n_seeds=5, n_iter=300)
        crise = int(np.argmax([m.covars_[k][0, 0] for k in range(2)]))
        p[t0:t1] = forward_gaussien(m, X[:t1])[t0:t1, crise]
    s = pd.Series(p, index=idx, name="p")
    s.to_csv(f, index_label="Date")
    return s


# ---------- Jump model walk-forward (K=2, etat online) ----------
Fraw = features_saut(ret)          # variables causales (moyennes exponentielles du passe)


def etat_jump(lam, bl):
    et = np.zeros(T, dtype=int)
    c_prec = None
    for t0, t1 in bl:
        sc = StandardScaler().fit(Fraw[:t0])
        s, c, _ = ajuster_saut(sc.transform(Fraw[:t0]), 2, lam, n_init=2, centres_init=c_prec)
        c_prec = c
        _, mp = reordonner(s, ret[:t0])
        st = etat_online(pertes(sc.transform(Fraw[:t1]), c), lam)
        et[t0:t1] = np.array([mp[k] for k in st])[t0:t1]
    return pd.Series(et, index=idx)


p = p_hmm()

# =====================================================================
# 1. CHOIX DES HYPERPARAMETRES SUR LE DEVELOPPEMENT UNIQUEMENT
# =====================================================================
sh_dev = pd.DataFrame(np.nan, index=G_OUT, columns=G_IN)
sh_test_diag = sh_dev.copy()                      # diagnostic, NON utilise pour choisir
for si in G_IN:
    for so in G_OUT:
        if so >= si:
            continue
        e = 1 - pd.Series(hysteresis(p.values, si, so), index=idx)
        sh_dev.loc[so, si] = evaluer(e, a_dev, b_dev)[2]["sharpe"]
        sh_test_diag.loc[so, si] = evaluer(e, a_test, b_test)[2]["sharpe"]
print("\n=== 1a. HMM : Sharpe sur le DEVELOPPEMENT selon les seuils (lignes : sortie, colonnes : entree) ===")
print(sh_dev.round(3).to_string())
val = sh_dev.stack()
so_c, si_c = val.idxmax()
print(f"\nSeuils retenus (Sharpe maximal en developpement) : entree {si_c}, sortie {so_c}")
print("Les 3 meilleurs :", [(f"{a}/{b}", round(v, 3)) for (b, a), v in val.nlargest(3).items()])

print("\n=== 1b. Jump model : Sharpe sur le DEVELOPPEMENT selon lambda ===")
dev_jump, sh_j = {}, {}
for lam in LAMS:
    dev_jump[lam] = etat_jump(lam, blocs_dev)
    r_ = evaluer(1 - dev_jump[lam], a_dev, b_dev)[2]
    sh_j[lam] = r_["sharpe"]
    print(f"  lambda={lam:4d}  Sharpe={r_['sharpe']:.3f}  max_dd={r_['max_dd']:.3f}  "
          f"changements={r_['changements']}")
lam_c = max(sh_j, key=sh_j.get)
print(f"lambda retenu : {lam_c}")

json.dump({"seuil_entree": float(si_c), "seuil_sortie": float(so_c), "lambda": int(lam_c),
           "split": SPLIT}, open("data/choix_figes.json", "w"))
print("Choix figes enregistres dans data/choix_figes.json (utilises par l'etape 11b).")

# =====================================================================
# 2. TEST : UNE SEULE EVALUATION AVEC LES CHOIX FIGES
# =====================================================================
et_test = etat_jump(lam_c, blocs_test)
et_j = pd.Series(np.where(idx < idx[S], dev_jump[lam_c].values, et_test.values), index=idx)

strat = {
    "Buy & Hold": pd.Series(1.0, index=idx),
    "VIX > 30": (df["vix"] <= 30).astype(float),
    "Prix > MA200": (df["prix"] >= df["ma200"]).astype(float),
    f"HMM K=2 (seuils a priori {DEF_IN}/{DEF_OUT})": 1 - pd.Series(hysteresis(p.values, DEF_IN, DEF_OUT), index=idx),
    f"HMM K=2 (seuils choisis {si_c}/{so_c})": 1 - pd.Series(hysteresis(p.values, si_c, so_c), index=idx),
    f"Jump K=2 online (lambda={lam_c})": 1 - et_j,
}
res_test, res_dev, cums, xs = {}, {}, {}, {}
for n, e in strat.items():
    xs[n], cums[n], res_test[n] = evaluer(e, a_test, b_test)
    res_dev[n] = evaluer(e, a_dev, b_dev)[2]
tab_test, tab_dev = pd.DataFrame(res_test).T, pd.DataFrame(res_dev).T
print(f"\n=== 2. RESULTATS SUR LE TEST ({a_test.date()} -> {b_test.date()}), cout {COUT * 1e4:.0f} pb, signal decale 1 jour ===")
print(tab_test.round(3).to_string())
print(f"\n(Rappel : developpement {a_dev.date()} -> {b_dev.date()})")
print(tab_dev.round(3).to_string())
tab_test.round(4).to_csv("data/resultats_finaux_test.csv")
tab_dev.round(4).to_csv("data/resultats_finaux_dev.csv")

print("\n=== 2b. Sharpe sur le test selon les couts de transaction ===")
couts = [0, 0.0005, 0.001, 0.002, 0.005]
cs = pd.DataFrame({n: [evaluer(e, a_test, b_test, c)[2]["sharpe"] for c in couts]
                   for n, e in strat.items()}, index=[f"{c * 1e4:.0f} pb" for c in couts])
print(cs.round(3).to_string())
cs.round(4).to_csv("data/resultats_finaux_couts.csv")

print("\n=== 2c. Diagnostic (non utilise pour choisir) : Sharpe du HMM sur le TEST selon les seuils ===")
print(sh_test_diag.round(3).to_string())
print(f"Cellule choisie en developpement ({si_c}/{so_c}) : Sharpe test = {sh_test_diag.loc[so_c, si_c]:.3f} ; "
      f"meilleure cellule du test : {sh_test_diag.stack().max():.3f}")


# =====================================================================
# 3. BOOTSTRAP PAR BLOCS (periode de test)
# =====================================================================
def sharpe_a(x):
    return x.mean() / x.std(ddof=1) * np.sqrt(252)


def boot(xa, xb, B=2000, bloc=20):
    a, b = xa.values, xb.values
    n = len(a)
    nb = int(np.ceil(n / bloc))
    d = np.empty(B)
    for i in range(B):
        st = rng.integers(0, n - bloc, nb)
        ix = (st[:, None] + np.arange(bloc)).ravel()[:n]
        d[i] = sharpe_a(a[ix]) - sharpe_a(b[ix])
    return sharpe_a(a) - sharpe_a(b), np.percentile(d, [2.5, 97.5]), (d <= 0).mean()


noms = list(strat)
bh, ma = noms[0], noms[2]
hmm_c, jmp = noms[4], noms[5]
print("\n=== 3. Bootstrap par blocs (test) : difference de Sharpe ===")
rows = {}
for a, b in [(hmm_c, bh), (hmm_c, ma), (jmp, bh), (jmp, ma), (jmp, hmm_c), (ma, bh)]:
    o, ic, pv = boot(xs[a], xs[b])
    rows[f"{a.split(' (')[0]} - {b.split(' (')[0]}"] = {"diff Sharpe": o, "IC95 bas": ic[0],
                                                      "IC95 haut": ic[1], "P(diff<=0)": pv}
tb = pd.DataFrame(rows).T
print(tb.round(3).to_string())
print("Lecture : IC95 contenant 0 => difference non significative.")
tb.round(4).to_csv("data/resultats_finaux_bootstrap.csv")

# =====================================================================
# 4. FIGURES
# =====================================================================
coul = {bh: ("black", 2), noms[1]: ("tab:blue", 1), ma: ("tab:purple", 1),
        noms[3]: ("tab:gray", 1), hmm_c: ("tab:orange", 1.8), jmp: ("tab:green", 1.8)}
fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
for n, (c, lw) in coul.items():
    ax[0].plot(cums[n], color=c, lw=lw, label=n)
    ax[1].plot(cums[n] / cums[n].cummax() - 1, color=c, lw=lw)
ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
ax[0].set_title(f"TEST {a_test.year}-{b_test.year} : valeur cumulee (base 1), signal decale, couts {COUT * 1e4:.0f} pb")
ax[1].set_title("Drawdown de chaque strategie")
plt.tight_layout(); plt.savefig("figures/fig19_test_final.png", dpi=200)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for a, tabh, t in [(ax[0], sh_dev, "Sharpe DEVELOPPEMENT (selection)"),
                   (ax[1], sh_test_diag, "Sharpe TEST (diagnostic, non utilise)")]:
    im = a.imshow(np.ma.masked_invalid(tabh.values), cmap="viridis")
    a.set_xticks(range(len(G_IN)), G_IN); a.set_yticks(range(len(G_OUT)), G_OUT)
    a.set_xlabel("seuil d'entree"); a.set_ylabel("seuil de sortie"); a.set_title(t)
    for i in range(len(G_OUT)):
        for j in range(len(G_IN)):
            v = tabh.values[i, j]
            if not np.isnan(v):
                a.text(j, i, f"{v:.2f}", ha="center", va="center", color="white")
    plt.colorbar(im, ax=a, fraction=0.046)
ax[2].plot(LAMS, [sh_j[l] for l in LAMS], marker="o")
ax[2].set_xscale("log"); ax[2].set_xlabel("lambda"); ax[2].set_ylabel("Sharpe")
ax[2].set_title("Jump model : Sharpe en developpement")
plt.tight_layout(); plt.savefig("figures/fig20_choix_hyperparametres.png", dpi=200)
plt.show()
print("\nFigures : fig19, fig20.  Resultats : data/resultats_finaux_*.csv")
