
from pathlib import Path
import pandas as pd
import numpy as np

# 1. Chemins du projet
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sp500.csv"

# 2. Chargement des données
data = pd.read_csv(DATA_PATH, parse_dates=["Date"])

# 3. Vérification de la structure
print("=== APERÇU DES DONNÉES ===")
print(data.head())
print("\nDimensions :", data.shape)
print("\nColonnes :", data.columns.tolist())
print("\nTypes des colonnes :")
print(data.dtypes)

# 4. Vérification des valeurs manquantes
print("\n=== VALEURS MANQUANTES ===")
print(data.isna().sum())

# 5. Vérification des doublons
print("\n=== DOUBLONS ===")
print("Dates dupliquées :", data["Date"].duplicated().sum())
print("Lignes dupliquées :", data.duplicated().sum())

# 6. Vérification des dates et des prix
prix_col = "^GSPC"
data[prix_col] = pd.to_numeric(data[prix_col], errors="coerce")

print("\n=== VALIDITÉ ===")
print("Dates triées :", data["Date"].is_monotonic_increasing)
print("Dates uniques :", data["Date"].is_unique)
print("Prix manquants après conversion :", data[prix_col].isna().sum())
print("Prix non positifs :", (data[prix_col] <= 0).sum())

# 7. Vérification des variations extrêmes
rendements = np.log(data[prix_col] / data[prix_col].shift(1))
print("\n=== RENDEMENTS EXTRÊMES À EXAMINER ===")
print(rendements.abs().nlargest(10))

print("\nVérification terminée.")
