"""
Application Streamlit : detection des changements de regime des marches financiers par IA.
Lancer (depuis la racine du projet) :   streamlit run app/app.py
"""
import streamlit as st
from outils_app import appliquer_style, icone

st.set_page_config(page_title="Regimes de marche par IA", page_icon=":material/query_stats:", layout="wide",
                   initial_sidebar_state="expanded")

pages = [
    st.Page("vues/accueil.py", title="Accueil", icon=":material/dashboard:", default=True),
    st.Page("vues/detection.py", title="Detection des regimes", icon=":material/monitoring:"),
    st.Page("vues/durees.py", title="Durees et transitions", icon=":material/timer:"),
    st.Page("vues/comparaison.py", title="Comparaison des modeles", icon=":material/compare_arrows:"),
    st.Page("vues/backtest.py", title="Backtest et risque", icon=":material/account_balance_wallet:"),
    st.Page("vues/methodologie.py", title="Methodologie et limites", icon=":material/menu_book:"),
]
pg = st.navigation(pages)
appliquer_style()

with st.sidebar:
    st.markdown(("<div style='font-size:1.35rem;font-weight:800;letter-spacing:-.02em'>{}&nbsp; Regimes de marche</div>"
                "<div style='color:#93c5fd;font-weight:600;margin-bottom:6px'>Detection par intelligence artificielle</div>").replace("{}", icone("activite", 26, "#22D3EE")),
                unsafe_allow_html=True)
    st.caption("PFA - Sujet 5 : detection des changements de regime des marches financiers "
               "par intelligence artificielle")
    st.divider()
    st.caption("Donnees figees au 16/10/2026 (Yahoo Finance, FRED). Toutes les analyses sont "
               "reproductibles depuis le depot GitHub.")
    st.caption("Projet pedagogique : pas un conseil en investissement.")

pg.run()
