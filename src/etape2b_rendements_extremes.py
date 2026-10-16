
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "data" / "sp500.csv", parse_dates=["Date"])

prix = data["^GSPC"]
rendements = np.log(prix / prix.shift(1))

# Afficher les 10 plus fortes variations absolues,
# avec leurs dates et leurs rendements signés.
resultats = pd.DataFrame({
    "Date": data["Date"],
    "Prix": prix,
    "Rendement_log": rendements,
    "Variation_simple_pct": (prix / prix.shift(1) - 1) * 100
})

resultats["Abs_rendement"] = resultats["Rendement_log"].abs()

print(
    resultats.nlargest(10, "Abs_rendement")[
        ["Date", "Prix", "Rendement_log", "Variation_simple_pct"]
    ].to_string(index=False)
)
