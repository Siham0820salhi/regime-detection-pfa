import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ---------- 1. Données : lues depuis le fichier figé ----------
df = pd.read_csv("data/sp500_vix_brut.csv", index_col=0, parse_dates=True)


# ---------- 2. Features ----------
df["ret"] = np.log(df["prix"] / df["prix"].shift(1))
df["vol20"] = df["ret"].rolling(20).std() * np.sqrt(252)
df["vol60"] = df["ret"].rolling(60).std() * np.sqrt(252)
df["mom60"] = df["prix"].pct_change(60)
df["max_cum"] = df["prix"].cummax()
df["drawdown"] = df["prix"] / df["max_cum"] - 1
df["ma200"] = df["prix"].rolling(200).mean()
df = df.dropna()
df.to_csv("data/features.csv")

# ---------- 3. Baselines (règles simples) ----------
# Règle 1 : bear market si drawdown < -20 %
df["reg_dd"] = (df["drawdown"] < -0.20).astype(int)
# Règle 2 : crise si VIX > 30
df["reg_vix"] = (df["vix"] > 30).astype(int)
# Règle 3 : sous la moyenne mobile 200 jours
df["reg_ma"] = (df["prix"] < df["ma200"]).astype(int)

# ---------- 4. Statistiques par règle ----------
for nom in ["reg_dd", "reg_vix", "reg_ma"]:
    print(f"\n=== {nom} ===")
    g = df.groupby(nom)["ret"]
    res = pd.DataFrame({
        "jours": g.count(),
        "rend_annuel": g.mean() * 252,
        "vol_annuelle": g.std() * np.sqrt(252),
    })
    print(res.round(3))
    changements = (df[nom].diff().abs() == 1).sum()
    print(f"Nombre de changements de régime : {changements}")

# ---------- 5. Figure : prix coloré par régime (VIX > 30) ----------
fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

ax[0].plot(df["prix"], color="navy", lw=0.8)
ax[0].fill_between(df.index, df["prix"].min(), df["prix"].max(),
                   where=df["reg_vix"] == 1, color="red", alpha=0.3)
ax[0].set_yscale("log")
ax[0].set_title("S&P 500 : zones rouges = VIX > 30 (régime de crise)")

ax[1].plot(df["vol20"], color="orange", label="Volatilité 20j")
ax[1].plot(df["vol60"], color="brown", label="Volatilité 60j")
ax[1].legend()
ax[1].set_title("Volatilité réalisée annualisée")

ax[2].plot(df["drawdown"], color="crimson")
ax[2].axhline(-0.20, color="black", ls="--", label="Seuil bear (-20 %)")
ax[2].legend()
ax[2].set_title("Drawdown")

plt.tight_layout()
plt.savefig("figures/fig2_features_baseline.png", dpi=200)
plt.show()