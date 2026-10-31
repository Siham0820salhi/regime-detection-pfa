"""
Telecharge UNE FOIS les recessions officielles NBER (serie USREC de FRED).
USREC : serie mensuelle, 1 = mois de recession, 0 = expansion.
Le fichier est valide puis fige dans data/usrec.csv.
"""
from pathlib import Path
import pandas as pd

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"
output = Path("data/usrec.csv")

try:
    s = pd.read_csv(URL, parse_dates=[0], index_col=0).iloc[:, 0]
    source = "FRED"
except Exception as e:
    print("Telechargement FRED impossible :", e)
    print("-> Repli sur les dates officielles NBER (jusqu'a fin 2024).")
    mois = pd.date_range("2000-01-01", "2024-12-01", freq="MS")
    s = pd.Series(0, index=mois)
    for debut, fin in [("2001-04", "2001-11"), ("2008-01", "2009-06"),
                       ("2020-03", "2020-04")]:
        s.loc[debut:fin] = 1
    source = "repli NBER"

s = s.dropna().astype(int).rename("USREC")
s.index.name = "Date"

if len(s) < 250:
    raise SystemExit("Trop peu de lignes : fichier non enregistre.")
if not s.index.is_unique:
    raise SystemExit("Dates dupliquees : fichier non enregistre.")
if not set(s.unique()) <= {0, 1}:
    raise SystemExit("Valeurs differentes de 0/1 : fichier non enregistre.")
if s.loc["2000":].sum() == 0:
    raise SystemExit("Aucune recession apres 2000 : donnees suspectes.")

output.parent.mkdir(parents=True, exist_ok=True)
s.to_csv(output)
print(f"Source : {source}")
print(f"Fichier cree : {output}  ({len(s)} mois, {s.index[0].date()} -> {s.index[-1].date()})")
print("Mois de recession depuis 2000 :", int(s.loc['2000':].sum()))
