# Détection des changements de régime des marchés financiers par intelligence artificielle

Projet de fin d'année (PFA), Sujet 5 : Master DSBD, Faculté des Sciences Ben M'Sik, Université Hassan II de Casablanca, 2026.

**Auteure :** SALHI Siham
**Encadrant :** Pr. SAID Nouh | **Co-encadrante :** SAIFI Wafae


> **Avertissement :** projet pédagogique de recherche. Rien ici ne constitue un conseil en investissement.

---

## 1. En bref

Un **régime de marché** est une période où le marché garde des propriétés statistiques stables (niveau de risque, rendement moyen) : calme, agité ou en crise. Ce projet utilise des modèles d'intelligence artificielle (surtout un **HMM**, modèle de Markov caché) pour **détecter ces régimes en temps réel**, mesurer **leur durée**, et évaluer leur utilité pour la **gestion du risque**.

**Question de recherche.** Dans quelle mesure les méthodes d'IA (HMM, jump models, détection de ruptures) détectent-elles de façon stable et exploitable les changements de régime, et quelle valeur ajoutée apportent-elles à la gestion du risque ?

| Hypothèse | Énoncé | Verdict |
|---|---|---|
| **H1** | Les régimes détectés correspondent aux crises connues (2002, 2008, 2020, 2022, 2025) | Confirmée, avec nuances |
| **H2** | Les modèles à mémoire (HMM, jump model) sont plus stables que le clustering (K-means, GMM) | Confirmée |
| **H3** | Une stratégie par régime réduit le drawdown par rapport au Buy & Hold | Confirmée sur les indices actions |

## 2. Résultats clés

| Résultat | Valeur |
|---|---|
| Régimes retrouvés sur 5 marchés (S&P 500, Euro Stoxx 50, DAX, CAC 40, FTSE 100) | 3 niveaux de volatilité : environ 9-11 %, 18-21 %, 41-50 % |
| Durée médiane d'un épisode (calme / moyen / crise) | 38-69 j / 22-42 j / 12,5-26 j |
| Changements de régime sur 26 ans (mêmes variables) | K-means 1554, GMM 739, **HMM 125-159** |
| Biais d'anticipation (lissé contre filtré, temps réel) | 89 contre 355 transitions : **×4** |
| Concordance avec les récessions NBER | AUC **0,93** (la règle prix/MA200 fait 0,96) |
| Délai d'alerte après le pic du marché | 3 à 17 jours (2008, COVID, 2022) |
| Test indépendant 2015-2026 : perte maximale | **−33,9 % → −16,9 %** |
| Test indépendant : volatilité | 17,6 % → 10,8 % |
| Test indépendant : Sharpe | 0,67 → 0,77, **non significatif** (IC95 bootstrap [−0,41 ; +0,54]) |

**Ce qui est prouvé :** détection robuste de régimes persistants sur plusieurs marchés et réduction du risque extrême des indices actions.
**Ce qui ne l'est pas :** un gain de performance ajustée du risque, et la généralisation à tous les actifs (échec sur le Bitcoin, pas d'amélioration du Sharpe sur les indices européens).

## 3. Structure du dépôt

```
regime-detection-pfa/
├── src/                     Scripts d'analyse (un par étape)
│   ├── outils.py            Fonctions communes : HMM, filtre forward, segments, backtest
│   ├── modele_saut.py       Jump model (programmation dynamique avec pénalité)
│   ├── telecharger_donnees.py   Téléchargement unique et validation (S&P 500, VIX)
│   ├── telecharger_nber.py      Récessions NBER (FRED)
│   ├── etape1_faits_stylises.py ... etape11c_europe.py
│   └── preparer_app.py      Prépare les données de l'application
├── data/                    Données figées et résultats intermédiaires (CSV)
├── figures/                 Figures produites par les scripts (fig1 à fig24)
├── app/                     Application Streamlit (thème sombre)
│   ├── app.py               Point d'entrée
│   ├── outils_app.py        Fonctions de l'application
│   ├── vues/                Les 6 pages
│   └── donnees/             Données lues par l'application
├── report/                  Cahier de bord et rapport
├── .streamlit/config.toml   Thème de l'application
└── requirements.txt
```

### Les scripts, étape par étape

| Script | Rôle |
|---|---|
| `telecharger_donnees.py` | Télécharge S&P 500 et VIX **une fois**, valide et fige dans `data/sp500_vix_brut.csv` |
| `etape1_faits_stylises.py` | Faits stylisés : kurtosis 10,7, volatilité par vagues |
| `etape2_features_baseline.py` | Variables (rendement, volatilités, drawdown, MA200) et règles simples |
| `etape3_clustering.py` | K-means et GMM |
| `etape4_hmm.py`, `etape5_hmm_ameliore.py` | HMM, HMM sur rendements seuls, lissé contre filtré |
| `etape6_backtest.py`, `etape7_walkforward.py` | Hystérésis, backtest, walk-forward, exposition graduelle |
| `etape8_evaluation.py` | Récessions NBER, délais de détection, sensibilité, bootstrap |
| `etape9a_comparaison.py`, `etape9b_transitions_figures.py` | Comparaison équitable, stabilité, convergence, choix de K, transitions |
| `etape10_pelt_jump.py` | PELT et jump model |
| `etape11a_test_final.py` | Test chronologique indépendant (choix figés sur 2005-2014, test 2015-2026) |
| `etape11b_robustesse.py`, `etape11c_europe.py` | Robustesse multi-actifs et marchés européens |
| `preparer_app.py` | Prépare `app/donnees/` |

## 4. Installation

Python 3.10 ou plus.

```bash
git clone https://github.com/Siham0820salhi/regime-detection-pfa.git
cd regime-detection-pfa
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux
pip install -r requirements.txt
```

## 5. Reproduire les résultats

Les données sont **déjà dans le dépôt** (`data/`). Pour tout refaire, depuis la racine du projet :

```bash
python src/telecharger_donnees.py     # une seule fois (écrase la base figée : à éviter si elle existe déjà)
python src/telecharger_nber.py
python src/etape1_faits_stylises.py
python src/etape2_features_baseline.py
python src/etape3_clustering.py
python src/etape4_hmm.py
python src/etape5_hmm_ameliore.py
python src/etape6_backtest.py
python src/etape7_walkforward.py
python src/etape8_evaluation.py
python src/etape9a_comparaison.py
python src/etape9b_transitions_figures.py
python src/etape10_pelt_jump.py
python src/etape11a_test_final.py
python src/etape11b_robustesse.py
python src/etape11c_europe.py
python src/preparer_app.py
```

> Les walk-forward (étapes 8, 11a, 11b, 11c) sont longs à la première exécution ; leurs probabilités sont mises en cache dans `data/p_*.csv`. **Si tu relances les étapes 1 et 2, supprime ces caches.**
> Les prix Yahoo Finance peuvent être légèrement révisés entre deux téléchargements : c'est pourquoi la base est **figée** (16/10/2026).

## 6. Lancer l'application

```bash
pip install plotly            # si ce n'est pas déjà fait
streamlit run app/app.py
```

Lancer la commande **depuis la racine** du dépôt, pour que le thème (`.streamlit/config.toml`) soit pris en compte.

| Page | Contenu |
|---|---|
| Accueil | Feu vert / orange / rouge du régime actuel du S&P 500, indicateurs, signal de la stratégie, résultats clés |
| Détection des régimes | 5 marchés, régimes filtrés ou lissés, récessions NBER, tableau par régime, validation des événements |
| Durées et transitions | Matrices de transition, durée de chaque régime, courbe de survie, comparaison des marchés |
| Comparaison des modèles | Comparaison équitable, stabilité, PELT et jump model, lissé contre filtré, choix de K |
| Backtest et risque | Seuils, coûts et décalage réglables, 5 stratégies, bootstrap, démonstration du biais d'anticipation |
| Méthodologie et limites | Protocole, verdict sur les hypothèses, limites, glossaire |

**Déploiement (Streamlit Community Cloud) :** fichier principal `app/app.py`. Le dossier `app/donnees/` doit être dans le dépôt. Le fichier `app/requirements.txt` (léger) évite d'installer les bibliothèques d'analyse.

## 7. Données

| Source | Contenu |
|---|---|
| Yahoo Finance | S&P 500 (`^GSPC`), VIX (`^VIX`), Euro Stoxx 50 (`^STOXX50E`), DAX (`^GDAXI`), CAC 40 (`^FCHI`), FTSE 100 (`^FTSE`), Nasdaq (`^IXIC`), or (`GC=F`), Bitcoin (`BTC-USD`) |
| FRED | Récessions officielles du NBER (série `USREC`) |

Période : 2000-2026, données quotidiennes. Rendements logarithmiques. Le MASI (Bourse de Casablanca) n'a pas pu être obtenu sous forme de série exploitable ; la validation hors États-Unis porte sur des indices européens.

## 8. Précautions méthodologiques (absence de fuite d'information)

- Probabilités **filtrées** (passé seul) pour toute décision, jamais les probabilités lissées.
- **Walk-forward** : réentraînement annuel sur le passé seul (fenêtre croissante).
- Signal **décalé d'une séance** et **coûts de transaction** (5 points de base par défaut).
- Hyperparamètres choisis sur **2005-2014** ; **test unique** sur 2015-2026 ; règles sans paramètre non réglées (`data/choix_figes.json`).
- Incertitude mesurée par **bootstrap par blocs**.

## 9. Limites

1. Pas de vérité terrain : l'évaluation est indirecte.
2. Le modèle capte un risque de marché (volatilité), pas un cycle économique ; il rate les baisses lentes (2022 en Europe).
3. Un seul découpage de test, dominé par 2020 et 2022.
4. Le gain de Sharpe n'est pas statistiquement prouvé ; il dépend des seuils.
5. Échec sur le Bitcoin ; pas d'amélioration du Sharpe sur les indices européens.
6. Hypothèse gaussienne face à des queues épaisses ; K choisi par interprétabilité.

**Perspectives :** loi de Student, détection de ruptures en ligne (BOCPD), exposition par paliers.

## 10. Références (indicatives : à vérifier avant citation)

- Hamilton (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.*
- Rabiner (1989). *A Tutorial on Hidden Markov Models.*
- Cont (2001). *Empirical properties of asset returns: stylized facts and statistical issues.*
- Ang et Bekaert (2002). *International Asset Allocation with Regime Shifts.*
- Kritzman, Page et Turkington (2012). *Regime Shifts: Implications for Dynamic Strategies.*
- Killick, Fearnhead et Eckley (2012). *Optimal detection of changepoints with a linear computational cost.*
- Nystrup et al. : travaux sur les *statistical jump models*.

## 11. Licence

MIT (voir `LICENSE`).
