## Jour 1
- Environnement + GitHub OK
- Données S&P 500 : 6738 jours (2000-2026)
- Volatilité annualisée 19,23 %, skewness -0,348, kurtosis 10,69
- JB p-value = 0 : normalité rejetée
- Fig 1 produite : clustering de volatilité et queues épaisses confirmés

## Jour 1 (suite) : Étape 2
- Features construites : ret, vol20, vol60, mom60, drawdown, ma200 + VIX
- 3 baselines : drawdown<-20%, VIX>30, prix<MA200
- Changements : 82 / 142 / 171 => règles instables
- Régime crise : vol 2-3x plus élevée, rendement négatif
- Limite : retard du drawdown, allers-retours du VIX

## Étape 3 : Clustering
- K-means et GMM, K=3 (BIC décroissant sans minimum, choix justifié par interprétabilité)
- K-means : 227 changements, durée moy. 28,7 j
- GMM : 329 changements, durée moy. 19,8 j
- Vol par régime : ~10 %, 16-21 %, 46-50 %
- Limite : instabilité, fit offline (look-ahead)
- Référence H2 : à battre avec le HMM (< 50 changements)

## Reproductibilité
- Données téléchargées une fois (16/10/2026), validées, figées : sp500_vix_brut.csv (6738 lignes)
- Étapes 1 à 4 relancées depuis ce fichier
- Chaîne : telecharger_donnees -> etape1 -> etape2 -> etape3 -> etape4

## Étape 4 : HMM
- K=3, observations (ret, vol20). BIC décroissant (58256 / 52712 / 49362) : K fixé par interprétabilité
- 159 changements, durée moy. 40,9 j (vs GMM 325, K-means 227)
- Matrice : diagonale 0,98 / 0,965 / 0,978, calme<->crise quasi nul
- Limites : vol20 autocorrélée, probabilités smoothed, 159 changements encore élevé
- À faire : observation sur les rendements seuls, prior "sticky", filtered vs smoothed

## Étape 5 : HMM sur rendements, filtered vs smoothed
- K=2 : 87 chang. smoothed / 355 filtered. K=3 : 168 / 512. K=3 sticky(200) : 159 / 459
- Look-ahead bias : changements x3 à x4 en temps réel
- Vol par régime K=3 : 9 %, 18 %, 45 %
- Prior sticky 200 : effet marginal (trop faible face à 6539 obs.)
- À faire : hystérésis + persistance minimale ; backtest avec régime décalé d'un jour

## Étape 6 : backtest HMM K=2 filtered
- Hystérésis : 355 -> 204 changements
- B&H : 6,8 % / vol 19,1 % / Sharpe 0,358 / MDD -56,8 %
- HMM : 3,9 % / vol 10,9 % / Sharpe 0,358 / MDD -38,0 %, investi 73 %
- H3 confirmée (risque), pas de surperformance ajustée du risque
- Bug figure corrigé : tracer avec df.index
- À faire : exposition graduelle, jump model, MASI/Bitcoin, walk-forward

## Étape 7 : walk-forward + baselines
- 21 réentraînements annuels, 1re prédiction le 2005-10-21
- B&H : 9,3 % / vol 19,2 % / Sharpe 0,486 / MDD -56,8 %
- HMM walk-forward : 6,6 % / 10,9 % / 0,606 / -22,1 % / Calmar 0,300
- HMM graduel : 6,5 % / 10,2 % / 0,639 / -22,8 % / rotation 11,9 par an
- MA200 proche du HMM : apport de l'IA à relativiser
- Walk-forward non dégradé : pas de look-ahead qui flatte le HMM

## Étape 8 : évaluation externe
- NBER (FRED, 28 mois de récession depuis 2000) : AUC HMM WF 0,932 ; MA200 0,960 ; VIX 0,913
- Rappel 93 %, précision 30 % : régime de risque de marché, pas de cycle
- Délais : 17 / 3 / 13 j (perte 4 / 5 / 8 %), plus rapide que VIX, MA200, drawdown
- Seuils : tous battent le B&H ; meilleur = entrée 0,9 sortie 0,3, à ne pas retenir (sélection sur 2005-2014)
- Coûts : binaire > B&H jusqu'à ~30 pb ; graduel fragile (> B&H jusqu'à ~10-20 pb)
- Bootstrap : aucune différence de Sharpe significative
- Sous-période 2008-10 -> 2026 : Sharpe HMM 0,630 < B&H 0,668, MDD -18,9 % vs -33,9 %
- Erreur : test fenêtre (3/5/8 ans) non informatif, à refaire (fenêtre glissante)

## Étape 9 : comparaison équitable, stabilité, convergence, K, H1
- Mêmes variables (ret+vol20), K=3 : transitions K-means 1554, GMM 739, HMM 159, sticky 152, HMM ret seul 125
- Durée médiane : 1 j (K-means, GMM) contre 26-36 j (HMM) -> H2 confirmée
- Stabilité 20 inits : ARI HMM 0,999 ; GMM 0,70 ; HMM ret seul min 0,35 (30 % mauvais optimum)
- Convergence 100 % ; 4 avertissements sur sticky seulement (MAP-EM probable)
- K : BIC ret seul minimum à K=4 mais instable (ARI 0,41) ; K=3 description, K=2 décision
- Sticky 2000 : effet faible (159 -> 152) -> motivation du jump model
- Bug : fenêtre 2025 (corriger 2025-04-30)

## Étape 9 : figures
- fig9 : le rouge couvre 2001-02, 2008-09, 2020, 2022, 2025 ; bug 2025 à corriger
- fig11 (2007-2010) : K-means 413 transitions, GMM 134, HMM 24, sticky 26, ret seul 16
- fig12 : lissé 22 contre filtré 94 ; îlots verts du filtré pendant les rebonds de 2008
- fig13 : diagonales HMM 0,97-0,985 ; K-means 0,29-0,37 (états interchangeables)
- fig14 : segments < 10 j : K-means 95 %, GMM 81 %, HMM 5,6 %, sticky 3,3 %, ret seul 13,5 %
- fig15 : K-means moyen et crise non séparés en volatilité (20,4 % / 21,1 %)

## Étape 10 : PELT et jump model
- PELT (BIC, 26 ruptures) : 65 % des dates à ±10 j d'une transition HMM K=2 (hasard 22 %, p<0,001)
- Jump K=2, lambda=10 : 34 transitions offline, 54 online (HMM filtered 355, avec hystérésis 204)
- Plateau lambda 10-100 stable ; lambda>=200 dégénère ; jump K=3 dégénéré (2,5 % de jours en crise)
- H1 : PELT large (base 36 %) ; jump sélectif (2008 : 47 %, 2025 raté) ; HMM K=2 le plus complet
- Limites : offline, variables lissées pour le jump, PELT non temps réel
- À faire à l'étape 11 : comparer tous les modèles en temps réel avec choix figés sur 2005-2014

## Étape 11 : test chronologique indépendant + robustesse
- Dev 2005-2014 / test 2015-2026 ; seuils HMM choisis 0,9/0,3 ; jump lambda 100 (bord de grille)
- Test : B&H Sharpe 0,669, MDD -33,9 % ; HMM 0,765, MDD -16,9 %, 58 changements ; jump 0,529, MDD -30,1 %
- Seuils a priori 0,8/0,4 : Sharpe 0,549 < B&H -> dépendance aux seuils
- Bootstrap : aucune différence significative ; jump presque significativement pire que B&H
- Robustesse : Nasdaq bon, or neutre, Bitcoin échec (33 chgts/an) ; S&P et Nasdaq non indépendants (2005-2014)
- À faire : data/masi.csv puis relancer 11b

## Étape 11b/11c : Europe
- 3 régimes sur S&P, Euro Stoxx, DAX, CAC, FTSE : vol 9-11 / 18-21 / 41-50 %, diag 0,95-0,977, calme<->crise = 0
- Durées médianes : calme 38-69 j, moyen 22-42 j, crise 12,5-26 j ; 10-16 épisodes de crise par marché
- K=3 justifié (BIC min à K=3 pour le CAC) ; ARI 0,92-0,996 (meilleure init sur 10)
- Événements : 2008, 2011, COVID reconnus (3-11x la base) ; 2022 non détecté en Europe
- P(crise Europe | crise S&P) = 70-80 % ; ARI S&P-Europe 0,24-0,30 (décalage horaire probable)
- Stratégie : vol réduite sur 8/8 actifs, drawdown réduit sur 7/8, Sharpe amélioré sur 2-3/8 ; MA200 >= HMM en Europe
