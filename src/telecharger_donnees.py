from pathlib import Path
import pandas as pd
import yfinance as yf

# 1. Télécharger
sp = yf.download("^GSPC", start="2000-01-01",
                 auto_adjust=True, progress=False)["Close"].squeeze()
vix = yf.download("^VIX", start="2000-01-01",
                  auto_adjust=True, progress=False)["Close"].squeeze()

# 2. Vérifier
if sp.empty or vix.empty:
    raise SystemExit("Téléchargement échoué : données vides. Aucun fichier enregistré.")

# 3. Combiner
df = pd.concat([sp.rename("prix"), vix.rename("vix")], axis=1, sort=True).dropna()

if df.empty:
    raise SystemExit("Aucune date commune. Aucun fichier enregistré.")
if len(df) < 5000:
    raise SystemExit("Trop peu de lignes : téléchargement incomplet.")
if not df.index.is_unique:
    raise SystemExit("Dates dupliquées : fichier non enregistré.")
if (df["prix"] <= 0).any() or (df["vix"] <= 0).any():
    raise SystemExit("Valeurs non positives : fichier non enregistré.")

# 4. Enregistrer
output = Path("data/sp500_vix_brut.csv")
output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output, index_label="Date")

print(f"Fichier créé : {output}")
print(f"Nombre de lignes : {len(df)}")
print(f"Période : {df.index.min().date()} à {df.index.max().date()}")
print(df.tail())