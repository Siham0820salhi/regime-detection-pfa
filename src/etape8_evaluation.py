"""
Etape 8 : evaluation externe (NBER), delais de detection, sensibilite, bootstrap.

Repond a la question du jury : "comment sais-tu que ton modele a raison
alors qu'il n'existe pas de verite terrain ?"

Prerequis : python src/telecharger_nber.py  (une seule fois)
Premier lancement long (3 walk-forward). Les probabilites sont mises en cache
dans data/p_wf_*.csv. Si tu relances les etapes 1-2, supprime ces fichiers.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score)

# ---------- Parametres ----------
TRAIN_MIN, REFIT, N_SEEDS = 1260, 252, 5
SEUIL_IN, SEUIL_OUT = 0.8, 0.4
COUT = 0.0005
FENETRES = [756, 1260, 2016]       # 3, 5 et 8 ans d'entrainement initial
rng = np.random.default_rng(42)

df = pd.read_csv("data/features.csv", index_col=0, parse_dates=True)
idx = df.index
X = (df["ret"].values * 100).reshape(-1, 1)
T = len(X)
r_simple = np.exp(df["ret"]) - 1


# ---------- Outils ----------
def ajuster(Xtrain):
    best, best_ll = None, -np.inf
    for seed in range(N_SEEDS):
        m = GaussianHMM(2, covariance_type="full", n_iter=300, tol=1e-3,
                        random_state=seed).fit(Xtrain)
        ll = m.score(Xtrain)
        if ll > best_ll:
            best, best_ll = m, ll
    return best


def forward(m, Xs):
    sig = np.sqrt([m.covars_[k][0, 0] for k in range(2)])
    B = np.column_stack([norm.pdf(Xs[:, 0], m.means_[k, 0], sig[k])
                         for k in range(2)]) + 1e-300
    alpha = np.zeros((len(Xs), 2))
    a = m.startprob_ * B[0]
    alpha[0] = a / a.sum()
    for t in range(1, len(Xs)):
        a = (alpha[t - 1] @ m.transmat_) * B[t]
        alpha[t] = a / a.sum()
    return alpha


def crise_idx(m):
    return int(np.argmax([m.covars_[k][0, 0] for k in range(2)]))


def p_walkforward(train_min):
    """P(crise) hors echantillon (walk-forward), avec cache sur disque."""
    f = f"data/p_wf_{train_min}.csv"
    if os.path.exists(f):
        s = pd.read_csv(f, index_col=0, parse_dates=True)["p"]
        if len(s) == T:
            return s
    print(f"  calcul du walk-forward (fenetre initiale {train_min} j)...")
    p = np.full(T, np.nan)
    for t0 in range(train_min, T, REFIT):
        t1 = min(t0 + REFIT, T)
        m = ajuster(X[:t0])
        p[t0:t1] = forward(m, X[:t1])[t0:t1, crise_idx(m)]
    s = pd.Series(p, index=idx, name="p")
    s.to_csv(f, index_label="Date")
    return s


def etat_hyst(p, s_in, s_out):
    """Hysteresis : entree si P > s_in, sortie si P < s_out."""
    v = p.values
    e = np.zeros(len(v), dtype=int)
    for t in range(1, len(v)):
        if e[t - 1] == 0 and v[t] > s_in:
            e[t] = 1
        elif e[t - 1] == 1 and v[t] < s_out:
            e[t] = 0
        else:
            e[t] = e[t - 1]
    return pd.Series(e, index=p.index)


def backtest(expo, debut, cout=COUT):
    """Signal decale d'un jour, couts a chaque changement d'exposition."""
    e = expo.shift(1).loc[debut:]
    rs = r_simple.loc[debut:]
    turn = e.diff().abs().fillna(0)
    x = e * rs - turn * cout
    cum = (1 + x).cumprod()
    ann = cum.iloc[-1] ** (252 / len(x)) - 1
    vol = x.std() * np.sqrt(252)
    mdd = (cum / cum.cummax() - 1).min()
    return x, {"rend_ann": ann, "vol": vol, "sharpe": ann / vol,
               "max_dd": mdd, "calmar": ann / abs(mdd)}


# ---------- 0. Probabilites principales ----------
print("Chargement / calcul des probabilites walk-forward...")
p_wf = p_walkforward(TRAIN_MIN)
etat_main = etat_hyst(p_wf, SEUIL_IN, SEUIL_OUT)
m_full = ajuster(X)
p_is = pd.Series(forward(m_full, X)[:, crise_idx(m_full)], index=idx)  # in-sample

signaux = {
    "HMM walk-forward + hysteresis": etat_main,
    "VIX > 30": (df["vix"] > 30).astype(int),
    "Prix < MA200": (df["prix"] < df["ma200"]).astype(int),
    "Drawdown < -20 %": (df["drawdown"] < -0.20).astype(int),
}
debut = idx[TRAIN_MIN + 1]
expo_hmm_bin = 1 - etat_main
expo_hmm_gr = 1 - p_wf.fillna(0)
expo_ma = 1 - signaux["Prix < MA200"]
expo_bh = pd.Series(1.0, index=idx)

print("\n=== Rappel etape 7 (controle de coherence) ===")
for nom, e in [("Buy & Hold", expo_bh), ("HMM binaire WF", expo_hmm_bin),
               ("HMM graduel WF", expo_hmm_gr), ("Prix > MA200", expo_ma)]:
    print(f"{nom:16s}", {k: round(float(v), 3) for k, v in backtest(e, debut)[1].items()})

# ---------- 1. Concordance avec les recessions NBER ----------
usrec = pd.read_csv("data/usrec.csv", index_col=0, parse_dates=True)["USREC"]
fin_nber = usrec.index[-1] + pd.offsets.MonthEnd(1)
rec = usrec.reindex(idx, method="ffill")
valid = ((idx >= usrec.index[0]) & (idx <= fin_nber)).astype(bool)
print(f"\nRecessions NBER disponibles jusqu'au {fin_nber.date()} "
      f"(evaluation limitee a cette date)")


def concordance(mask, scores, binaires, titre):
    y = rec[mask].values.astype(int)
    print(f"\n=== {titre} ===")
    print(f"Jours : {mask.sum()}  |  jours de recession NBER : {y.mean():.1%}")
    auc = {n: roc_auc_score(y, s[mask].values) for n, s in scores.items()}
    print("\nAUC (1 = parfait, 0,5 = hasard) :")
    print(pd.Series(auc).round(3).to_string())
    lignes = {}
    for n, s in binaires.items():
        b = s[mask].values
        lignes[n] = {"% jours en alerte": b.mean(),
                     "precision": precision_score(y, b, zero_division=0),
                     "rappel": recall_score(y, b, zero_division=0),
                     "F1": f1_score(y, b, zero_division=0)}
    print("\nSignaux binaires :")
    tab = pd.DataFrame(lignes).T
    print(tab.round(3).to_string())
    return pd.Series(auc), tab


scores_oos = {
    "P(crise) HMM walk-forward": p_wf, "VIX": df["vix"],
    "Volatilite 20j": df["vol20"], "Drawdown (inverse)": -df["drawdown"],
    "Distance sous MA200": -(df["prix"] / df["ma200"] - 1),
}
mask_oos = valid & p_wf.notna().values
auc_oos, tab_oos = concordance(mask_oos, scores_oos, signaux,
                               "A. Hors echantillon (walk-forward)")

scores_is = dict(scores_oos)
scores_is.pop("P(crise) HMM walk-forward")
scores_is["P(crise) HMM (parametres sur toute la periode)"] = p_is
auc_is, _ = concordance(valid, scores_is,
                        {"HMM in-sample + hysteresis": etat_hyst(p_is, SEUIL_IN, SEUIL_OUT),
                         **{k: v for k, v in signaux.items() if k.startswith(("VIX", "Prix", "Draw"))}},
                        "B. Toute la periode (inclut la recession de 2001, parametres in-sample)")
pd.concat({"oos": auc_oos, "insample": auc_is}).to_csv("data/eval_auc.csv")
tab_oos.round(4).to_csv("data/eval_signaux_nber.csv")

# ---------- 2. Delais de detection ----------
pics = {"Pic oct. 2007 (crise 2008)": "2007-10-09",
        "Pic fev. 2020 (COVID)": "2020-02-19",
        "Pic janv. 2022 (inflation)": "2022-01-03"}


def alerte(sig, date_pic, horizon=250):
    i0 = idx.searchsorted(pd.Timestamp(date_pic))
    seg = sig.iloc[i0:i0 + horizon].values
    if seg.size == 0:
        return "n/d"
    if seg[0] == 1:
        return "deja actif"
    if not seg.any():
        return "aucune alerte"
    k = int(np.argmax(seg == 1))
    perte = df["prix"].iloc[i0 + k] / df["prix"].iloc[i0] - 1
    return f"{k} j ({perte:.0%})"


print("\n=== Delai avant alerte depuis le pic (perte deja subie depuis le pic) ===")
tab = pd.DataFrame({ev: {n: alerte(s, d) for n, s in signaux.items()}
                    for ev, d in pics.items()})
print(tab.to_string())
print("\nPerte maximale du pic au creux (250 j) :")
for ev, d in pics.items():
    i0 = idx.searchsorted(pd.Timestamp(d))
    if i0 < T:
        seg = df["prix"].iloc[i0:i0 + 250]
        print(f"  {ev} : {seg.min() / seg.iloc[0] - 1:.1%}")
tab.to_csv("data/eval_delais.csv")

# ---------- 3. Sensibilite ----------
print("\n=== Sensibilite A : seuils d'hysteresis (HMM binaire WF) ===")
ins, outs = [0.6, 0.7, 0.8, 0.9], [0.2, 0.3, 0.4, 0.5]
sh = pd.DataFrame(np.nan, index=outs, columns=ins)
dd, ch = sh.copy(), sh.copy()
for si in ins:
    for so in outs:
        if so >= si:
            continue
        e = etat_hyst(p_wf, si, so)
        s = backtest(1 - e, debut)[1]
        sh.loc[so, si], dd.loc[so, si] = s["sharpe"], s["max_dd"]
        ch.loc[so, si] = int((np.diff(e.values) != 0).sum())
sh.index.name = "sortie \\ entree"
print("Sharpe :\n", sh.round(3).to_string())
print("\nMax drawdown :\n", dd.round(3).to_string())
print("\nNombre de changements :\n", ch.round(0).to_string())
sh.to_csv("data/sensibilite_seuils_sharpe.csv")

print("\n=== Sensibilite B : couts de transaction (Sharpe) ===")
couts = [0, 0.0005, 0.001, 0.002, 0.005]
strat = {"Buy & Hold": expo_bh, "Prix > MA200": expo_ma,
         "HMM binaire WF": expo_hmm_bin, "HMM graduel WF": expo_hmm_gr}
cs = pd.DataFrame({n: [backtest(e, debut, c)[1]["sharpe"] for c in couts]
                   for n, e in strat.items()},
                  index=[f"{c * 1e4:.0f} pb" for c in couts])
print(cs.round(3).to_string())
cs.to_csv("data/sensibilite_couts.csv")

print("\n=== Sensibilite C : fenetre d'entrainement initiale (meme periode de test) ===")
debut_c = idx[max(FENETRES) + 1]
lignes = {"Buy & Hold": backtest(expo_bh, debut_c)[1],
          "Prix > MA200": backtest(expo_ma, debut_c)[1]}
for f in FENETRES:
    pf = p_walkforward(f)
    lignes[f"HMM binaire, fenetre {f / 252:.0f} ans"] = backtest(
        1 - etat_hyst(pf, SEUIL_IN, SEUIL_OUT), debut_c)[1]
print(f"Periode commune : {debut_c.date()} -> {idx[-1].date()}")
tab_c = pd.DataFrame(lignes).T
print(tab_c.round(3).to_string())
tab_c.round(4).to_csv("data/sensibilite_fenetres.csv")

# ---------- 4. Bootstrap par blocs : la difference de Sharpe est-elle reelle ? ----------
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


x = {n: backtest(e, debut)[0] for n, e in strat.items()}
print("\n=== Bootstrap par blocs (20 j, 2000 tirages) : difference de Sharpe ===")
res_b = {}
for a, b in [("HMM binaire WF", "Buy & Hold"), ("HMM binaire WF", "Prix > MA200"),
             ("HMM graduel WF", "Buy & Hold"), ("HMM graduel WF", "Prix > MA200"),
             ("HMM graduel WF", "HMM binaire WF")]:
    obs, ic, p = boot(x[a], x[b])
    res_b[f"{a} - {b}"] = {"diff Sharpe": obs, "IC95 bas": ic[0],
                           "IC95 haut": ic[1], "P(diff<=0)": p}
tab_b = pd.DataFrame(res_b).T
print(tab_b.round(3).to_string())
print("Lecture : si l'IC95 contient 0 (ou P(diff<=0) > 0,05), la difference n'est pas significative.")
tab_b.round(4).to_csv("data/eval_bootstrap.csv")

# ---------- 5. Figures ----------
fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
rec_v = (rec.fillna(0).values == 1) & valid
ax[0].plot(df["prix"], color="black", lw=0.7)
ax[0].fill_between(idx, df["prix"].min(), df["prix"].max(), where=rec_v,
                   color="grey", alpha=0.35, label="Recession NBER")
ax[0].fill_between(idx, df["prix"].min(), df["prix"].max(),
                   where=(etat_main.values == 1), color="red", alpha=0.25,
                   label="Alerte HMM walk-forward")
ax[0].set_yscale("log")
ax[0].legend(loc="upper left")
ax[0].set_title("S&P 500 : recessions NBER (gris) et alertes du HMM (rouge)")
ax[1].plot(p_wf, color="red", lw=0.6)
ax[1].axhline(SEUIL_IN, color="black", ls="--", lw=0.8)
ax[1].axhline(SEUIL_OUT, color="black", ls=":", lw=0.8)
ax[1].fill_between(idx, 0, 1, where=rec_v, color="grey", alpha=0.35)
ax[1].set_title("P(crise) hors echantillon (seuils 0,8 / 0,4)")
plt.tight_layout()
plt.savefig("figures/fig8_nber.png", dpi=200)

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for a, tabh, titre in [(ax[0], sh, "Sharpe"), (ax[1], dd, "Max drawdown")]:
    im = a.imshow(np.ma.masked_invalid(tabh.values), cmap="viridis")
    a.set_xticks(range(len(ins)), ins)
    a.set_yticks(range(len(outs)), outs)
    a.set_xlabel("seuil d'entree")
    a.set_ylabel("seuil de sortie")
    a.set_title(f"{titre} selon les seuils")
    for i in range(len(outs)):
        for j in range(len(ins)):
            v = tabh.values[i, j]
            if not np.isnan(v):
                a.text(j, i, f"{v:.2f}", ha="center", va="center", color="white")
    plt.colorbar(im, ax=a, fraction=0.046)
for n in cs.columns:
    ax[2].plot(couts, cs[n].values, marker="o", label=n)
ax[2].set_xlabel("cout par changement (fraction)")
ax[2].set_ylabel("Sharpe")
ax[2].set_title("Sharpe selon les couts de transaction")
ax[2].legend()
plt.tight_layout()
plt.savefig("figures/fig8_sensibilite.png", dpi=200)
plt.show()
