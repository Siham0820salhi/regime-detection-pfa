import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import streamlit as st
from outils_app import *

entete("Methodologie, resultats et limites", "La demarche scientifique, ce que les resultats prouvent... et surtout ce qu'ils ne prouvent pas.", "Rigueur | Transparence")
st.markdown("Cette page resume la demarche scientifique du projet, ses resultats par rapport aux hypotheses de depart, "
            "et surtout **ce que les resultats ne permettent pas de conclure**.")

t1, t2, t3, t4, t5, t6, t7 = st.tabs(["Probleme et donnees", "Modeles", "Protocole d'evaluation",
                                      "Resultats et hypotheses", "Limites", "Glossaire", "Reproductibilite et references"])

# ----------------------------------------------------------------- 1
with t1:
    st.subheader("Problematique")
    st.markdown("""
> *Dans quelle mesure les methodes d'IA (HMM, jump models, detection de ruptures) detectent-elles de facon stable
> et exploitable les changements de regime des marches financiers, et quelle valeur ajoutee apportent-elles a la
> gestion du risque ?*

**Pourquoi c'est difficile.** Il n'existe **aucune verite terrain** : personne ne fournit la liste des jours « calmes »
et « en crise ». Le probleme est **non supervise**, et l'evaluation doit se faire indirectement (evenements connus,
recessions officielles, stabilite, utilite economique d'une strategie).

**Pourquoi c'est utile.** Les modeles classiques supposent des parametres constants, ce qui est contredit par les
*faits styliques* des marches : queues epaisses, regroupement de la volatilite, asymetrie des rendements (Cont, 2001).
Un melange de regimes (un calme, un agite) reproduit naturellement ces faits.""")
    st.subheader("Hypotheses")
    st.markdown("""
- **H1** : les regimes detectes correspondent a des evenements economiques connus (2002, 2008, 2020, 2022, 2025).
- **H2** : les modeles a memoire (HMM, jump) sont plus stables que le clustering sans memoire (GMM, K-means).
- **H3** : une strategie qui adapte son exposition au regime reduit le drawdown par rapport au Buy & Hold.""")
    st.subheader("Donnees")
    st.markdown("""
| Source | Contenu |
|---|---|
| Yahoo Finance | S&P 500 (^GSPC), VIX (^VIX), Euro Stoxx 50 (^STOXX50E), DAX (^GDAXI), CAC 40 (^FCHI), FTSE 100 (^FTSE), Nasdaq (^IXIC), or (GC=F), Bitcoin (BTC-USD) |
| FRED (Federal Reserve de St. Louis) | Recessions officielles du NBER (serie USREC) |

- Periode : janvier 2000 a octobre 2026 ; donnees **telechargees une seule fois, validees automatiquement**
  (absence de valeurs vides, de doublons, de valeurs non positives) puis **figees** pour garantir la reproductibilite.
- Rendements **logarithmiques** r(t) = ln(P(t)/P(t-1)) : additifs dans le temps et approximativement stationnaires.
- Le MASI (Bourse de Casablanca) n'est pas disponible gratuitement sous forme de serie historique exploitable :
  sur recommandation de l'encadrant, la validation de robustesse porte sur les principaux indices europeens.""")

# ----------------------------------------------------------------- 2
with t2:
    st.subheader("Famille 1 : regles simples (references a battre)")
    st.markdown("Drawdown inferieur a -20 % (definition du bear market), VIX superieur a 30, prix sous la moyenne mobile "
                "200 jours. Elles sont **instables** (82 a 171 changements en 26 ans) et pour certaines **en retard**.")
    st.subheader("Famille 2 : clustering sans memoire")
    st.markdown("**K-means** et **GMM** regroupent les jours selon leurs caracteristiques, **sans notion de temps** : le regime peut "
                "changer chaque jour.")
    st.subheader("Famille 3 : Hidden Markov Model (HMM)")
    st.markdown("Le regime est un **etat cache** S(t) qui evolue selon une chaine de Markov ; on observe seulement les rendements.")
    st.latex(r"r_t \mid S_t = k \;\sim\; \mathcal{N}(\mu_k,\ \sigma_k^2), \qquad "
             r"A_{ij} = \mathbb{P}(S_t = j \mid S_{t-1} = i)")
    st.latex(r"\text{duree moyenne du regime } i = \frac{1}{1 - A_{ii}}")
    st.markdown("""
- **Estimation** : algorithme EM (Baum-Welch), meilleure vraisemblance parmi plusieurs initialisations.
- **Filtrage (forward)** : probabilite du regime a t connaissant le passe seul -> *utilisable en temps reel*.
- **Lissage (forward-backward) et Viterbi** : utilisent aussi le futur -> *descriptif seulement*.
- **HMM sticky** : prior qui favorise la persistance des regimes.""")
    st.subheader("Famille 4 : detection de ruptures (PELT)")
    st.markdown("Cherche les dates ou la distribution change, en minimisant un cout plus une penalite par rupture (critere BIC). "
                "Methode *retrospective* : pas de signal en temps reel.")
    st.subheader("Famille 5 : jump model")
    st.latex(r"\min_{\{s_t\},\{\mu_k\}} \ \sum_{t} \lVert x_t - \mu_{s_t} \rVert^2 \;+\; \lambda \sum_{t} \mathbb{1}\{ s_t \neq s_{t-1} \}")
    st.markdown("Comme un K-means avec une **penalite lambda par changement d'etat** : lambda = 0 redonne le K-means, "
                "lambda grand donne des regimes tres stables. Variables causales (rendement, risque baissier, volatilite lissees).")
    st.subheader("Strategie de gestion du risque")
    st.markdown("""
Sortie du marche si P(crise) > seuil d'entree, retour si P(crise) < seuil de sortie (**hysteresis**, qui evite les
aller-retours). Decision **decalee d'une seance**, **couts de transaction** a chaque changement.""")

# ----------------------------------------------------------------- 3
with t3:
    st.subheader("Eviter les fuites d'information")
    st.markdown("""
| Piege | Protection mise en place |
|---|---|
| **Look-ahead sur les probabilites** (lissage) | Probabilites **filtrees** uniquement pour toute decision |
| **Look-ahead sur les parametres** (modele estime sur toute la periode) | **Walk-forward** : reentrainement annuel sur le passe seul, fenetre croissante |
| **Execution irrealiste** | Signal decale d'une seance + couts de transaction |
| **Sur-ajustement des reglages** | Hyperparametres choisis sur **2005-2014 seulement** ; **test unique 2015-2026** ; regles sans parametre non reglees |
| **Resultats fragiles** | Sensibilite (seuils, couts), bootstrap par blocs, autres marches |""")
    st.subheader("Evaluer sans verite terrain : quatre axes")
    st.markdown("""
1. **Concordance externe** avec les recessions NBER et le VIX (AUC, precision, rappel).
2. **Coherence evenementielle** : reconnaissance de 2002, 2008, 2011, 2020, 2022, 2025 et delai de detection.
3. **Qualite statistique** : nombre de transitions, duree des segments, stabilite entre initialisations (ARI), convergence.
4. **Utilite economique** : backtest (rendement, volatilite, Sharpe, drawdown, Calmar) contre des references simples.""")

# ----------------------------------------------------------------- 4
with t4:
    st.caption("Chiffres de reference du projet (etapes 8 a 11, donnees figees au 16/10/2026).")
    st.subheader("Verdict sur les hypotheses")
    st.markdown("""
| Hypothese | Verdict | Elements |
|---|---|---|
| **H1** regimes = evenements connus | **Confirmee**, avec nuances | 2008, zone euro 2011 et COVID reconnus (3 a 11 fois la frequence de base) ; alerte moyenne apres 3 a 17 jours de bourse, avec 4 a 8 % de perte deja subie ; la baisse lente de 2022 n'est pas classee « crise » en Europe |
| **H2** memoire = stabilite | **Confirmee** | 1554 (K-means) et 739 (GMM) transitions contre 125 a 159 (HMM) a variables egales ; duree mediane 1 jour contre 26 a 36 jours ; en temps reel le HMM brut change 3 a 4 fois plus souvent qu'en retrospectif |
| **H3** reduction du risque | **Confirmee** sur les indices actions | Test independant 2015-2026 : drawdown -33,9 % -> -16,9 %, volatilite 17,6 % -> 10,8 % |""")
    st.subheader("Ce qui n'est pas prouve")
    st.markdown("""
- **Le gain de Sharpe** (0,67 -> 0,77 sur le test) **n'est pas statistiquement significatif** (intervalle de confiance a 95 % [-0,41 ; +0,54]).
- Il **depend des seuils** : avec des seuils choisis a priori (0,8/0,4), le Sharpe tombe a 0,55, sous celui du Buy & Hold.
- Le HMM **ne bat pas de facon significative** la regle simple prix/MA200.
- Le **jump model**, tres stable (4 changements de position), detecte la chute du COVID trop tard et sous-performe (Sharpe 0,53).
- Sur d'autres actifs : la **volatilite baisse partout** (8 actifs sur 8) et le drawdown sur 7 sur 8, mais le **Sharpe ne s'ameliore que sur les indices americains** ; la regle MA200 fait aussi bien ou mieux en Europe ; **echec sur le Bitcoin** (33 changements de position par an).""")
    st.subheader("Conclusion")
    st.info("Un HMM filtre, stabilise par hysteresis et valide en walk-forward, **reduit de moitie le risque extreme** des indices "
            "actions, sans que l'on puisse prouver un gain de performance ajustee du risque, et sans que la strategie se "
            "generalise a tous les actifs. La *detection* des regimes, elle, est robuste d'un marche a l'autre.")
    res = table("resultats_finaux_test.csv")
    if res is not None:
        with st.expander("Tableau complet du test independant"):
            tableau(res.round(3))
    rs = table("robustesse_synthese.csv")
    if rs is not None:
        with st.expander("Robustesse sur d'autres actifs (seuils figes sur le S&P 500)"):
            tableau(rs.round(3))

# ----------------------------------------------------------------- 5
with t5:
    st.subheader("Limites a garder en tete")
    st.markdown("""
1. **Pas de verite terrain** : l'evaluation reste indirecte ; les regimes sont des constructions statistiques.
2. **Risque de marche, pas cycle economique** : le HMM capte la volatilite ; il declenche aussi hors recession et rate les baisses lentes.
3. **Un seul decoupage chronologique** : le test 2015-2026 est domine par 2020 et 2022 ; peu d'evenements majeurs.
4. **Choix de K** : le BIC baisse presque toujours ; K = 3 est retenu par interpretabilite et stabilite, pas par un optimum statistique.
5. **Hypothese gaussienne** : les rendements ont des queues epaisses ; les etats supplementaires l'absorbent en partie.
6. **Hyperparametres** (seuils, lambda, prior) : choisis sur 2005-2014 (dominee par 2008), certains en bord de grille.
7. **Couts de transaction** : 5 pb realistes pour le S&P 500, optimistes pour le Bitcoin ; ni dividendes ni taux sans risque.
8. **Signal retarde** : un regime est confirme apres plusieurs jours ; les chocs tres rapides (COVID) sont subis avant detection.
9. **Marches correles** : les indices europeens ne sont pas des tests totalement independants du S&P 500.
10. **Non-stationnarite** : un modele estime sur le passe peut perdre en pertinence si la structure des marches change.""")
    st.warning(AVERTISSEMENT)

# ----------------------------------------------------------------- 6
with t6:
    glossaire = {
        "Regime de marche": "Periode ou le marche a des proprietes statistiques (risque, rendement) stables et differentes des autres periodes.",
        "Rendement logarithmique": "ln(P_t / P_(t-1)) ; additif dans le temps.",
        "Volatilite": "Ecart-type des rendements, mesure du risque ; annualisee en multipliant par la racine de 252.",
        "Drawdown": "Baisse depuis le plus haut precedent. Le drawdown maximal est la pire baisse sur la periode.",
        "VIX": "Indice de volatilite implicite du S&P 500, dit « indice de la peur ».",
        "Sharpe": "Rendement annuel divise par la volatilite annuelle.",
        "Calmar": "Rendement annuel divise par la valeur absolue du drawdown maximal.",
        "Buy & Hold": "Acheter et conserver sans jamais vendre : reference de comparaison.",
        "Kurtosis": "Epaisseur des queues de la distribution ; 0 pour une loi normale (ici environ 10,7 pour le S&P 500).",
        "Volatility clustering": "La volatilite arrive par vagues : une journee agitee est suivie d'autres.",
        "HMM": "Modele a etats caches dont l'etat evolue selon une chaine de Markov.",
        "Matrice de transition": "Probabilites de passer d'un regime a l'autre ; la diagonale mesure la persistance.",
        "Filtrage / lissage": "Probabilite d'un regime avec le passe seul (temps reel) ou avec tout l'historique (retrospectif).",
        "Look-ahead bias": "Utiliser sans le vouloir de l'information du futur : surestime les performances.",
        "Walk-forward": "Reentrainer periodiquement un modele sur le passe seul puis l'evaluer sur la periode suivante.",
        "Hysteresis": "Seuils differents pour entrer et sortir d'un etat, afin d'eviter les aller-retours.",
        "BIC / AIC": "Criteres de selection de modele : ajustement moins penalite de complexite (plus bas = mieux).",
        "ARI": "Indice de Rand ajuste : 1 = deux classifications identiques, 0 = independantes.",
        "PELT": "Algorithme exact de detection de ruptures avec cout lineaire en moyenne.",
        "Jump model": "Clustering avec penalite lambda sur les changements d'etat.",
        "Bootstrap par blocs": "Reechantillonnage de blocs consecutifs pour estimer l'incertitude en conservant l'autocorrelation.",
        "NBER": "Organisme americain qui date officiellement les recessions."}
    tableau(pd.DataFrame({"Terme": list(glossaire), "Definition": list(glossaire.values())}).set_index("Terme"))

# ----------------------------------------------------------------- 7
with t7:
    st.subheader("Reproduire le projet")
    st.code("""python src/telecharger_donnees.py      # donnees figees (une seule fois)
python src/telecharger_nber.py         # recessions NBER
python src/etape1_faits_stylises.py ... etape11c_europe.py
python src/preparer_app.py             # prepare les donnees de l'application
streamlit run app/app.py""", language="bash")
    st.subheader("Bibliographie indicative (a verifier avant citation)")
    st.markdown("""
- Hamilton, J. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle.* Econometrica.
- Rabiner, L. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.* Proceedings of the IEEE.
- Cont, R. (2001). *Empirical properties of asset returns: stylized facts and statistical issues.* Quantitative Finance.
- Ang, A. et Bekaert, G. (2002). *International Asset Allocation with Regime Shifts.* Review of Financial Studies.
- Kritzman, M., Page, S. et Turkington, D. (2012). *Regime Shifts: Implications for Dynamic Strategies.* Financial Analysts Journal.
- Nystrup, P., Kolm, P. et Lindstrom, E. : travaux sur les *statistical jump models* et l'allocation d'actifs dynamique.
- Killick, R., Fearnhead, P. et Eckley, I. (2012). *Optimal detection of changepoints with a linear computational cost.* JASA.
- Truong, C., Oudre, L. et Vayatis, N. (2020). *Selective review of offline change point detection methods.* Signal Processing.
- Adams, R. et MacKay, D. (2007). *Bayesian Online Changepoint Detection.* arXiv.""")
    st.caption("Verifier titres exacts, annees et references sur Google Scholar avant de les citer dans le rapport.")
