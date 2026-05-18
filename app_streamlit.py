import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from dotenv import load_dotenv


# =====================================================
# CONFIGURATION
# =====================================================

load_dotenv()

PROJECT_DIR = Path(__file__).parent
GREEN_IT_DIR = PROJECT_DIR / "green_it_results"
EMOTION_RESULTS_DIR = PROJECT_DIR / "emotion_analysis_results"
CLASSIFICATION_RESULTS_FILE = PROJECT_DIR / "classification_results.csv"

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB", "fake_news_project"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

APP_TITLE = "Thumalien"
APP_SUBTITLE = "Bluesky Fake News Detection & Green IT Monitoring"

BLUE = "#1185FE"
BLUE_DARK = "#0A4DA3"
SKY = "#56CCF2"
GREEN = "#22C55E"
ORANGE = "#F59E0B"
RED = "#EF4444"
PURPLE = "#8B5CF6"
TEXT = "#0F172A"
MUTED = "#64748B"
CARD_BG = "rgba(255,255,255,0.92)"

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [BLUE, SKY, PURPLE, GREEN, ORANGE, RED]

st.set_page_config(
    page_title="Thumalien - Bluesky Fake News Detection",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# CUSTOM CSS - BLUESKY THEME
# =====================================================

def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(86, 204, 242, 0.25), transparent 32%),
                radial-gradient(circle at top right, rgba(17, 133, 254, 0.20), transparent 35%),
                linear-gradient(135deg, #F8FBFF 0%, #EAF6FF 45%, #F8FAFC 100%);
            color: {TEXT};
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #071A3A 0%, #0A4DA3 55%, #1185FE 100%);
            border-right: 1px solid rgba(255,255,255,0.2);
        }}

        [data-testid="stSidebar"] * {{
            color: white !important;
        }}

        [data-testid="stSidebar"] .stRadio label {{
            padding: 0.35rem 0;
            border-radius: 14px;
        }}

        .block-container {{
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }}

        .hero {{
            position: relative;
            overflow: hidden;
            border-radius: 30px;
            padding: 2.3rem 2.4rem;
            background: linear-gradient(120deg, #0A4DA3 0%, #1185FE 48%, #56CCF2 100%);
            color: white;
            box-shadow: 0 25px 70px rgba(17, 133, 254, 0.28);
            margin-bottom: 1.5rem;
        }}

        .hero::before {{
            content: "";
            position: absolute;
            width: 270px;
            height: 270px;
            right: -70px;
            top: -85px;
            border-radius: 50%;
            background: rgba(255,255,255,0.20);
            animation: floatBubble 7s ease-in-out infinite;
        }}

        .hero::after {{
            content: "";
            position: absolute;
            width: 150px;
            height: 150px;
            right: 150px;
            bottom: -80px;
            border-radius: 50%;
            background: rgba(255,255,255,0.13);
            animation: floatBubble 9s ease-in-out infinite reverse;
        }}

        @keyframes floatBubble {{
            0% {{ transform: translateY(0px) scale(1); }}
            50% {{ transform: translateY(18px) scale(1.04); }}
            100% {{ transform: translateY(0px) scale(1); }}
        }}

        .hero-content {{
            position: relative;
            z-index: 2;
        }}

        .hero h1 {{
            font-size: 2.7rem;
            line-height: 1.05;
            margin: 0 0 0.7rem 0;
            font-weight: 800;
            letter-spacing: -0.04em;
        }}

        .hero p {{
            font-size: 1.05rem;
            max-width: 780px;
            color: rgba(255,255,255,0.9);
            margin-bottom: 1.1rem;
        }}

        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.25);
            font-size: 0.83rem;
            font-weight: 600;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
            backdrop-filter: blur(8px);
        }}

        .glass-card {{
            background: {CARD_BG};
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 24px;
            padding: 1.25rem;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(12px);
            margin-bottom: 1rem;
            transition: all 0.22s ease;
        }}

        .glass-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 22px 60px rgba(17, 133, 254, 0.14);
        }}

        .metric-card {{
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(240,249,255,0.96));
            border: 1px solid rgba(17,133,254,0.15);
            border-radius: 24px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 16px 40px rgba(17,133,254,0.10);
            min-height: 145px;
            position: relative;
            overflow: hidden;
        }}

        .metric-card::after {{
            content: "";
            position: absolute;
            width: 90px;
            height: 90px;
            border-radius: 50%;
            background: rgba(17,133,254,0.10);
            right: -30px;
            top: -30px;
        }}

        .metric-label {{
            font-size: 0.85rem;
            color: {MUTED};
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        .metric-value {{
            font-size: 2rem;
            color: {TEXT};
            font-weight: 800;
            margin-top: 0.35rem;
        }}

        .metric-note {{
            font-size: 0.85rem;
            color: {MUTED};
            margin-top: 0.35rem;
        }}

        .section-title {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 1.2rem 0 0.8rem 0;
        }}

        .section-title h2 {{
            font-size: 1.45rem;
            font-weight: 800;
            color: {TEXT};
            margin: 0;
        }}

        .step-badge {{
            width: 35px;
            height: 35px;
            border-radius: 12px;
            background: linear-gradient(135deg, {BLUE}, {SKY});
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            box-shadow: 0 10px 30px rgba(17,133,254,0.25);
        }}

        .pipeline {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.85rem;
            margin-top: 0.8rem;
        }}

        .pipe-step {{
            padding: 1rem;
            border-radius: 20px;
            background: white;
            border: 1px solid rgba(17,133,254,0.14);
            text-align: center;
            box-shadow: 0 14px 40px rgba(15,23,42,0.06);
        }}

        .pipe-icon {{
            font-size: 1.7rem;
            margin-bottom: 0.35rem;
            animation: pulseIcon 2.6s ease-in-out infinite;
        }}

        @keyframes pulseIcon {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.08); }}
        }}

        .pipe-title {{
            font-size: 0.9rem;
            font-weight: 800;
            color: {TEXT};
        }}

        .pipe-subtitle {{
            font-size: 0.78rem;
            color: {MUTED};
            margin-top: 0.2rem;
        }}

        .status-ok {{
            display: inline-block;
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            background: rgba(34,197,94,0.12);
            color: #166534;
            font-size: 0.8rem;
            font-weight: 700;
        }}

        .status-warn {{
            display: inline-block;
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            background: rgba(245,158,11,0.13);
            color: #92400E;
            font-size: 0.8rem;
            font-weight: 700;
        }}

        .stButton > button {{
            border: 0 !important;
            background: linear-gradient(135deg, {BLUE_DARK}, {BLUE}, {SKY}) !important;
            color: white !important;
            font-weight: 800 !important;
            border-radius: 16px !important;
            padding: 0.75rem 1.1rem !important;
            box-shadow: 0 14px 35px rgba(17,133,254,0.25) !important;
            transition: all 0.2s ease !important;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 18px 45px rgba(17,133,254,0.32) !important;
        }}

        .stDataFrame, .stPlotlyChart {{
            border-radius: 18px;
            overflow: hidden;
        }}

        code, pre {{
            border-radius: 16px !important;
        }}

        .footer-note {{
            color: {MUTED};
            font-size: 0.82rem;
            text-align: center;
            padding: 1rem;
        }}

        @media (max-width: 900px) {{
            .pipeline {{ grid-template-columns: 1fr; }}
            .hero h1 {{ font-size: 2rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

@st.cache_data(ttl=20)
def load_table_count(table_name: str) -> int:
    try:
        conn = get_connection()
        query = f"SELECT COUNT(*) FROM {table_name};"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return int(df.iloc[0, 0])
    except Exception:
        return 0


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def run_python_script(script_name: str):
    """
    Lance un script Python du projet et retourne stdout / stderr.
    Compatible Windows UTF-8 pour éviter les erreurs d'encodage avec les emojis.
    """
    script_path = PROJECT_DIR / script_name

    if not script_path.exists():
        return "", f"Script introuvable : {script_path}"

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=PROJECT_DIR,
        env=env,
    )

    st.cache_data.clear()
    return result.stdout, result.stderr


def safe_sql(query: str) -> pd.DataFrame:
    try:
        conn = get_connection()
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=20)
def load_processed_posts():
    query = """
        SELECT
            id,
            lang,
            cleaned_text,
            word_count,
            like_count,
            repost_count,
            reply_count,
            created_at
        FROM processed_posts
        ORDER BY created_at DESC;
    """
    return safe_sql(query)


@st.cache_data(ttl=20)
def load_raw_posts():
    query = """
        SELECT
            id,
            lang,
            text,
            like_count,
            repost_count,
            reply_count,
            created_at
        FROM raw_posts
        ORDER BY created_at DESC;
    """
    return safe_sql(query)


@st.cache_data(ttl=20)
def load_green_it_files():
    files = []
    if GREEN_IT_DIR.exists():
        for file in GREEN_IT_DIR.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                df["source_file"] = file.name
                files.append(df)
            except Exception:
                pass

    if files:
        return pd.concat(files, ignore_index=True)
    return pd.DataFrame()


@st.cache_data(ttl=20)
def load_latest_emotion_file():
    if not EMOTION_RESULTS_DIR.exists():
        return pd.DataFrame()

    files = sorted(
        EMOTION_RESULTS_DIR.glob("emotions_*.csv"),
        key=os.path.getmtime,
        reverse=True,
    )

    if not files:
        return pd.DataFrame()

    return pd.read_csv(files[0])


@st.cache_data(ttl=20)
def load_classification_results():
    if CLASSIFICATION_RESULTS_FILE.exists():
        return pd.read_csv(CLASSIFICATION_RESULTS_FILE)
    return pd.DataFrame()


@st.cache_data(ttl=20)
def load_raw_id_uri_mapping():
    """Mapping permettant de relier classification_results.csv aux résultats d'émotions.

    classification_results.csv contient généralement l'id de raw_posts.
    Les résultats émotionnels contiennent raw_uri depuis processed_posts.
    Cette table permet donc de faire : raw_posts.id -> raw_posts.uri -> processed_posts.raw_uri.
    """
    query = """
        SELECT
            id AS raw_post_id,
            uri AS raw_uri
        FROM raw_posts;
    """
    return safe_sql(query)


@st.cache_data(ttl=20)
def load_fake_emotion_joined():
    """Jointure entre résultats fake news et résultats émotionnels.

    Stratégie principale :
    1. classification_results.csv : id = raw_posts.id
    2. raw_posts : id -> uri
    3. emotion_analysis_results : raw_uri
    4. jointure finale sur raw_uri

    Fallback : jointure directe sur id si les deux fichiers utilisent le même identifiant.
    """
    classification_df = load_classification_results()
    emotion_df = load_latest_emotion_file()

    if classification_df.empty or emotion_df.empty:
        return pd.DataFrame()

    classification_df = classification_df.copy()
    emotion_df = emotion_df.copy()

    # Harmoniser les noms pour éviter les collisions
    if "confidence" in classification_df.columns:
        classification_df = classification_df.rename(columns={"confidence": "fake_confidence"})
    if "label" in classification_df.columns:
        classification_df = classification_df.rename(columns={"label": "fake_label"})
    if "prediction" in classification_df.columns:
        classification_df = classification_df.rename(columns={"prediction": "fake_prediction"})

    # Jointure recommandée via raw_uri
    mapping_df = load_raw_id_uri_mapping()
    if not mapping_df.empty and "id" in classification_df.columns and "raw_uri" in emotion_df.columns:
        classification_df["id"] = pd.to_numeric(classification_df["id"], errors="coerce")
        mapping_df["raw_post_id"] = pd.to_numeric(mapping_df["raw_post_id"], errors="coerce")

        classification_with_uri = classification_df.merge(
            mapping_df,
            left_on="id",
            right_on="raw_post_id",
            how="left",
        )

        joined_df = classification_with_uri.merge(
            emotion_df,
            on="raw_uri",
            how="inner",
            suffixes=("_classification", "_emotion"),
        )

        if not joined_df.empty:
            return joined_df

    # Fallback si les ids correspondent directement
    if "id" in classification_df.columns and "id" in emotion_df.columns:
        joined_df = classification_df.merge(
            emotion_df,
            on="id",
            how="inner",
            suffixes=("_classification", "_emotion"),
        )
        return joined_df

    return pd.DataFrame()


def metric_card(label: str, value, note: str = "", icon: str = "📊"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:1.6rem; margin-bottom:0.4rem;">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(number: str, title: str):
    st.markdown(
        f"""
        <div class="section-title">
            <div class="step-badge">{number}</div>
            <h2>{title}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def script_runner_card(script_name: str, button_label: str, spinner_text: str):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write(f"Script cible : `{script_name}`")

    if st.button(button_label, use_container_width=True):
        started_at = datetime.now()
        progress = st.progress(0, text="Initialisation...")
        with st.spinner(spinner_text):
            for pct in [15, 35, 55, 75]:
                time.sleep(0.15)
                progress.progress(pct, text=f"Exécution de {script_name}...")
            stdout, stderr = run_python_script(script_name)
            progress.progress(100, text="Terminé")

        elapsed = (datetime.now() - started_at).total_seconds()
        st.success(f"Étape terminée en {elapsed:.1f} secondes")

        with st.expander("Voir la sortie du script", expanded=True):
            st.code(stdout or "Aucune sortie standard.")

        if stderr:
            with st.expander("Voir les erreurs / warnings", expanded=True):
                st.code(stderr)

        st.toast("Étape exécutée avec succès", icon="✅")
    st.markdown('</div>', unsafe_allow_html=True)


def style_plot(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=TEXT),
        title_font=dict(size=19, color=TEXT),
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.18)")
    return fig


def render_hero():
    raw_count = load_table_count("raw_posts")
    processed_count = load_table_count("processed_posts")
    classification_df = load_classification_results()
    emotion_df = load_latest_emotion_file()

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-content">
                <div class="pill">🦋 Bluesky API</div>
                <div class="pill">🤖 RoBERTa IA</div>
                <div class="pill">🌱 Green IT</div>
                <div class="pill">📊 Streamlit Dashboard</div>
                <h1>{APP_TITLE}</h1>
                <p>{APP_SUBTITLE}</p>
                <p style="font-size:0.92rem; opacity:0.86;">
                    Interface avancée de pilotage : collecte, ETL, classification fake news,
                    analyse émotionnelle et monitoring environnemental.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Posts bruts", f"{raw_count:,}".replace(",", " "), "Table raw_posts", "📥")
    with c2:
        metric_card("Posts nettoyés", f"{processed_count:,}".replace(",", " "), "Table processed_posts", "🧹")
    with c3:
        metric_card("Posts classifiés", len(classification_df) if not classification_df.empty else 0, "Fake News / Not Fake News", "🤖")
    with c4:
        metric_card("Posts émotionnels", len(emotion_df) if not emotion_df.empty else 0, "Émotions détectées", "🎭")


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("# 🦋 Thumalien")
    st.markdown("Pilotage IA & Green IT")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Vue d'ensemble",
            "1. Collecte des données",
            "2. ETL & Nettoyage",
            "3. Classification Fake News",
            "4. Analyse émotionnelle",
            "5. Fake News x Émotions",
            "6. Green IT",
        ],
    )

    st.markdown("---")
    st.markdown("### État rapide")
    raw_count_side = load_table_count("raw_posts")
    processed_count_side = load_table_count("processed_posts")
    st.markdown(f"<span class='status-ok'>Raw posts : {raw_count_side}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='status-ok'>Processed : {processed_count_side}</span>", unsafe_allow_html=True)
    st.caption("Les données sont rafraîchies automatiquement après chaque exécution de script.")


render_hero()


# =====================================================
# PAGE 0 - OVERVIEW
# =====================================================

if page == "Vue d'ensemble":
    section_title("0", "Vue d'ensemble du pipeline")

    st.markdown(
        """
        <div class="glass-card">
            <b>Objectif :</b> suivre le cycle complet de traitement des posts Bluesky,
            depuis la collecte jusqu'à l'analyse Green IT.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pipeline">
            <div class="pipe-step"><div class="pipe-icon">🦋</div><div class="pipe-title">Collecte</div><div class="pipe-subtitle">Bluesky API → raw_posts</div></div>
            <div class="pipe-step"><div class="pipe-icon">🧹</div><div class="pipe-title">ETL</div><div class="pipe-subtitle">Nettoyage → processed_posts</div></div>
            <div class="pipe-step"><div class="pipe-icon">🤖</div><div class="pipe-title">Classification</div><div class="pipe-subtitle">RoBERTa Fake News</div></div>
            <div class="pipe-step"><div class="pipe-icon">🎭</div><div class="pipe-title">Émotions</div><div class="pipe-subtitle">DistilRoBERTa Emotion</div></div>
            <div class="pipe-step"><div class="pipe-icon">🌱</div><div class="pipe-title">Green IT</div><div class="pipe-subtitle">CodeCarbon metrics</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    processed_df = load_processed_posts()

    if not processed_df.empty:
        left, right = st.columns([1.05, 1])

        with left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Répartition des langues collectées")
            lang_counts = processed_df["lang"].value_counts().reset_index()
            lang_counts.columns = ["Langue", "Nombre de posts"]
            fig = px.bar(
                lang_counts,
                x="Langue",
                y="Nombre de posts",
                color="Langue",
                title="Posts nettoyés par langue",
            )
            st.plotly_chart(style_plot(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Volume de mots après nettoyage")
            fig = px.histogram(
                processed_df,
                x="word_count",
                nbins=30,
                title="Distribution des longueurs de posts",
                color_discrete_sequence=[BLUE],
            )
            st.plotly_chart(style_plot(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Aperçu des posts nettoyés")
        st.dataframe(processed_df.head(20), use_container_width=True, height=390)
    else:
        st.info("Aucune donnée nettoyée disponible pour le moment.")


# =====================================================
# PAGE 1 - COLLECTE
# =====================================================

elif page == "1. Collecte des données":
    section_title("1", "Collecte des données Bluesky")

    st.markdown(
        """
        <div class="glass-card">
        Cette étape lance <code>collect.py</code>. Elle récupère des posts depuis l'API Bluesky,
        filtre les langues FR/EN et stocke les résultats dans PostgreSQL, table <code>raw_posts</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    script_runner_card("collect.py", "🚀 Lancer la collecte Bluesky", "Collecte Bluesky en cours...")

    raw_df = load_raw_posts()

    if not raw_df.empty:
        raw_df = raw_df.copy()
        raw_df["engagement"] = raw_df["like_count"].fillna(0) + raw_df["repost_count"].fillna(0) + raw_df["reply_count"].fillna(0)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Total posts bruts", len(raw_df), "Données collectées", "📥")
        with c2:
            metric_card("Likes cumulés", int(raw_df["like_count"].sum()), "Engagement", "💙")
        with c3:
            metric_card("Reposts cumulés", int(raw_df["repost_count"].sum()), "Amplification", "🔁")
        with c4:
            metric_card("Réponses cumulées", int(raw_df["reply_count"].sum()), "Conversation", "💬")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Répartition par langue")
            lang_counts = raw_df["lang"].value_counts().reset_index()
            lang_counts.columns = ["Langue", "Nombre de posts"]
            fig = px.pie(
                lang_counts,
                names="Langue",
                values="Nombre de posts",
                hole=0.45,
                title="Langues des posts bruts",
            )
            st.plotly_chart(style_plot(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Engagement par langue")
            eng_lang = raw_df.groupby("lang", as_index=False)["engagement"].sum()
            fig = px.bar(
                eng_lang,
                x="lang",
                y="engagement",
                color="lang",
                title="Engagement total par langue",
            )
            st.plotly_chart(style_plot(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Top posts collectés par engagement")
        st.dataframe(raw_df.sort_values("engagement", ascending=False).head(20), use_container_width=True, height=420)
    else:
        st.info("Aucun post brut disponible. Lance d'abord la collecte.")


# =====================================================
# PAGE 2 - ETL
# =====================================================

elif page == "2. ETL & Nettoyage":
    section_title("2", "ETL & Nettoyage")

    st.markdown(
        """
        <div class="glass-card">
        Cette étape lance <code>etl_bluesky.py</code>. Elle nettoie les posts de <code>raw_posts</code>,
        filtre les textes trop courts et insère les résultats dans <code>processed_posts</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    script_runner_card("etl_bluesky.py", "🧹 Lancer l'ETL / Nettoyage", "Nettoyage des posts en cours...")

    processed_df = load_processed_posts()

    if not processed_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Posts nettoyés", len(processed_df), "Données utiles", "🧹")
        with c2:
            metric_card("Mots moyens", round(processed_df["word_count"].mean(), 2), "Par post", "📝")
        with c3:
            metric_card("Posts EN", int((processed_df["lang"] == "en").sum()), "Anglais", "🇬🇧")
        with c4:
            metric_card("Posts FR", int((processed_df["lang"] == "fr").sum()), "Français", "🇫🇷")

        col_a, col_b = st.columns([1.1, 1])
        with col_a:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Distribution du nombre de mots")
            fig = px.histogram(
                processed_df,
                x="word_count",
                nbins=35,
                title="Longueur des textes après nettoyage",
                color_discrete_sequence=[SKY],
            )
            st.plotly_chart(style_plot(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Mots moyens par langue")
            words_by_lang = processed_df.groupby("lang", as_index=False)["word_count"].mean()
            fig = px.bar(
                words_by_lang,
                x="lang",
                y="word_count",
                color="lang",
                title="Moyenne de mots par langue",
            )
            st.plotly_chart(style_plot(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Aperçu des textes nettoyés")
        st.dataframe(processed_df[["id", "lang", "cleaned_text", "word_count", "created_at"]].head(20), use_container_width=True, height=420)
    else:
        st.info("Aucune donnée nettoyée disponible. Lance d'abord l'ETL.")


# =====================================================
# PAGE 3 - CLASSIFICATION
# =====================================================

elif page == "3. Classification Fake News":
    section_title("3", "Classification Fake News")

    st.markdown(
        """
        <div class="glass-card">
        Cette étape lance <code>classification.py</code>. Elle applique le modèle RoBERTa fine-tuné
        pour classifier les posts en <b>Fake News</b> ou <b>Not Fake News</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    script_runner_card("classification.py", "🤖 Lancer la classification fake news", "Classification IA en cours...")

    classification_df = load_classification_results()

    if not classification_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Posts classifiés", len(classification_df), "Résultats IA", "🤖")

        if "confidence" in classification_df.columns:
            with c2:
                metric_card("Confiance moyenne", round(classification_df["confidence"].mean(), 3), "Score modèle", "🎯")

        if "label" in classification_df.columns:
            fake_count = int((classification_df["label"] == "Fake News").sum())
            not_fake_count = int((classification_df["label"] == "Not Fake News").sum())
            with c3:
                metric_card("Fake News", fake_count, "Détectées", "🚨")
            with c4:
                metric_card("Not Fake News", not_fake_count, "Détectées", "✅")

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Distribution des prédictions")
                label_counts = classification_df["label"].value_counts().reset_index()
                label_counts.columns = ["Label", "Nombre"]
                fig = px.bar(label_counts, x="Label", y="Nombre", color="Label", title="Fake News vs Not Fake News")
                st.plotly_chart(style_plot(fig), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_b:
                if "confidence" in classification_df.columns:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("Distribution de la confiance")
                    fig = px.histogram(
                        classification_df,
                        x="confidence",
                        color="label" if "label" in classification_df.columns else None,
                        nbins=25,
                        title="Scores de confiance du modèle",
                    )
                    st.plotly_chart(style_plot(fig), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Aperçu des résultats")
        st.dataframe(classification_df.head(30), use_container_width=True, height=430)
    else:
        st.info("Aucun résultat de classification trouvé. Lance d'abord la classification.")


# =====================================================
# PAGE 4 - EMOTIONS
# =====================================================

elif page == "4. Analyse émotionnelle":
    section_title("4", "Analyse émotionnelle")

    st.markdown(
        """
        <div class="glass-card">
        Cette étape lance <code>emotion.py</code>. Elle détecte l'émotion dominante des posts :
        colère, peur, joie, tristesse, surprise, neutralité, etc.
        </div>
        """,
        unsafe_allow_html=True,
    )

    script_runner_card("emotion.py", "🎭 Lancer l'analyse émotionnelle", "Analyse émotionnelle en cours...")

    emotion_df = load_latest_emotion_file()

    if not emotion_df.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Posts analysés", len(emotion_df), "Émotions calculées", "🎭")

        if "dominant_score" in emotion_df.columns:
            with c2:
                metric_card("Confiance moyenne", round(emotion_df["dominant_score"].mean(), 3), "Score dominant", "🎯")

        if "dominant_emotion" in emotion_df.columns:
            top_emotion = emotion_df["dominant_emotion"].mode()[0]
            with c3:
                metric_card("Émotion principale", top_emotion, "Mode du dataset", "💫")

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Distribution des émotions dominantes")
                emotion_counts = emotion_df["dominant_emotion"].value_counts().reset_index()
                emotion_counts.columns = ["Émotion", "Nombre"]
                fig = px.bar(emotion_counts, x="Émotion", y="Nombre", color="Émotion", title="Émotions dominantes")
                st.plotly_chart(style_plot(fig), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            emotion_cols = [col for col in emotion_df.columns if col.startswith("emotion_")]
            with col_b:
                if emotion_cols:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("Scores moyens par émotion")
                    avg_emotions = emotion_df[emotion_cols].mean().reset_index()
                    avg_emotions.columns = ["Émotion", "Score moyen"]
                    avg_emotions["Émotion"] = avg_emotions["Émotion"].str.replace("emotion_", "")
                    fig = px.bar(avg_emotions, x="Émotion", y="Score moyen", color="Émotion", title="Score moyen par émotion")
                    st.plotly_chart(style_plot(fig), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Aperçu des résultats")
        st.dataframe(emotion_df.head(30), use_container_width=True, height=430)
    else:
        st.info("Aucun résultat émotionnel trouvé. Lance d'abord l'analyse émotionnelle.")


# =====================================================
# PAGE 5 - FAKE NEWS X EMOTIONS
# =====================================================

elif page == "5. Fake News x Émotions":
    section_title("5", "Lien entre Fake News et Émotions")

    st.markdown(
        """
        <div class="glass-card">
        Cette page croise les résultats de <code>classification.py</code> avec ceux de <code>emotion.py</code>.
        L'objectif est d'identifier si les contenus classifiés comme <b>Fake News</b> sont associés à des émotions
        particulières comme la peur, la colère ou la surprise.
        </div>
        """,
        unsafe_allow_html=True,
    )

    joined_df = load_fake_emotion_joined()

    if joined_df.empty:
        st.warning(
            "Aucune jointure disponible. Lance d'abord la classification fake news puis l'analyse émotionnelle. "
            "Vérifie aussi que les fichiers classification_results.csv et emotion_analysis_results/emotions_*.csv existent."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Résultats classification")
            classification_df = load_classification_results()
            if classification_df.empty:
                st.info("Aucun fichier classification_results.csv trouvé.")
            else:
                st.dataframe(classification_df.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Résultats émotions")
            emotion_df = load_latest_emotion_file()
            if emotion_df.empty:
                st.info("Aucun fichier d'émotions trouvé.")
            else:
                st.dataframe(emotion_df.head(10), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Nettoyage léger des colonnes utiles
        if "fake_label" in joined_df.columns:
            joined_df["fake_label"] = joined_df["fake_label"].fillna("Unknown")
        if "dominant_emotion" in joined_df.columns:
            joined_df["dominant_emotion"] = joined_df["dominant_emotion"].fillna("unknown")

        total_joined = len(joined_df)
        fake_count = int((joined_df["fake_label"] == "Fake News").sum()) if "fake_label" in joined_df.columns else 0
        not_fake_count = int((joined_df["fake_label"] == "Not Fake News").sum()) if "fake_label" in joined_df.columns else 0
        avg_fake_conf = round(joined_df["fake_confidence"].mean(), 3) if "fake_confidence" in joined_df.columns else "N/A"
        avg_emotion_conf = round(joined_df["dominant_score"].mean(), 3) if "dominant_score" in joined_df.columns else "N/A"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Posts croisés", total_joined, "Classification + émotion", "🔗")
        with c2:
            metric_card("Fake News", fake_count, "Dans l'échantillon joint", "🚨")
        with c3:
            metric_card("Not Fake News", not_fake_count, "Dans l'échantillon joint", "✅")
        with c4:
            metric_card("Confiance moyenne", avg_fake_conf, "Classification", "🎯")

        if "fake_label" in joined_df.columns and "dominant_emotion" in joined_df.columns:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Matrice Fake News x Émotions")

            matrix_df = (
                joined_df.groupby(["fake_label", "dominant_emotion"])
                .size()
                .reset_index(name="Nombre de posts")
            )

            fig = px.density_heatmap(
                matrix_df,
                x="dominant_emotion",
                y="fake_label",
                z="Nombre de posts",
                color_continuous_scale="Blues",
                title="Volume de posts par classe fake news et émotion dominante",
                labels={
                    "dominant_emotion": "Émotion dominante",
                    "fake_label": "Classe Fake News",
                    "Nombre de posts": "Nombre de posts",
                },
            )
            st.plotly_chart(style_plot(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns([1, 1])

            with col_a:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Émotions par classe")
                fig = px.bar(
                    matrix_df,
                    x="dominant_emotion",
                    y="Nombre de posts",
                    color="fake_label",
                    barmode="group",
                    title="Comparaison des émotions entre Fake News et Not Fake News",
                    labels={
                        "dominant_emotion": "Émotion",
                        "fake_label": "Classe",
                    },
                )
                st.plotly_chart(style_plot(fig), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_b:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Répartition émotionnelle des Fake News")
                fake_only = joined_df[joined_df["fake_label"] == "Fake News"]
                if not fake_only.empty:
                    fake_emotions = fake_only["dominant_emotion"].value_counts().reset_index()
                    fake_emotions.columns = ["Émotion", "Nombre"]
                    fig = px.pie(
                        fake_emotions,
                        names="Émotion",
                        values="Nombre",
                        hole=0.45,
                        title="Émotions dominantes dans les Fake News",
                    )
                    st.plotly_chart(style_plot(fig), use_container_width=True)
                else:
                    st.info("Aucune Fake News détectée dans la jointure actuelle.")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Taux de Fake News par émotion")
            rate_df = (
                joined_df.assign(is_fake=(joined_df["fake_label"] == "Fake News").astype(int))
                .groupby("dominant_emotion", as_index=False)
                .agg(
                    total_posts=("is_fake", "count"),
                    fake_posts=("is_fake", "sum"),
                    fake_rate=("is_fake", "mean"),
                )
            )
            rate_df["fake_rate_pct"] = rate_df["fake_rate"] * 100

            fig = px.bar(
                rate_df.sort_values("fake_rate_pct", ascending=False),
                x="dominant_emotion",
                y="fake_rate_pct",
                color="dominant_emotion",
                title="Pourcentage de Fake News par émotion dominante",
                labels={
                    "dominant_emotion": "Émotion dominante",
                    "fake_rate_pct": "% Fake News",
                },
                text="fake_rate_pct",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(style_plot(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        score_cols = [col for col in joined_df.columns if col.startswith("emotion_")]
        if score_cols and "fake_label" in joined_df.columns:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Scores émotionnels moyens par classe")
            score_df = joined_df.groupby("fake_label")[score_cols].mean().reset_index()
            score_long = score_df.melt(
                id_vars="fake_label",
                value_vars=score_cols,
                var_name="Émotion",
                value_name="Score moyen",
            )
            score_long["Émotion"] = score_long["Émotion"].str.replace("emotion_", "")

            fig = px.line(
                score_long,
                x="Émotion",
                y="Score moyen",
                color="fake_label",
                markers=True,
                title="Profil émotionnel moyen : Fake News vs Not Fake News",
            )
            st.plotly_chart(style_plot(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if "fake_confidence" in joined_df.columns and "dominant_score" in joined_df.columns:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Relation confiance IA x intensité émotionnelle")
            fig = px.scatter(
                joined_df,
                x="fake_confidence",
                y="dominant_score",
                color="fake_label" if "fake_label" in joined_df.columns else None,
                hover_data=["dominant_emotion"] if "dominant_emotion" in joined_df.columns else None,
                title="Confiance fake news vs score d'émotion dominante",
                labels={
                    "fake_confidence": "Confiance classification fake news",
                    "dominant_score": "Score émotion dominante",
                },
            )
            st.plotly_chart(style_plot(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Voir le dataset joint", expanded=False):
            display_cols = [
                col for col in [
                    "id",
                    "raw_post_id",
                    "raw_uri",
                    "fake_label",
                    "fake_confidence",
                    "dominant_emotion",
                    "dominant_score",
                    "cleaned_text",
                    "text",
                ]
                if col in joined_df.columns
            ]
            st.dataframe(joined_df[display_cols].head(100), use_container_width=True, height=520)

        st.markdown(
            """
            <div class="glass-card">
            <h3>Lecture métier</h3>
            <p>
            Ces visualisations permettent d'observer si les contenus détectés comme Fake News mobilisent davantage certaines émotions.
            Une forte présence de <b>fear</b>, <b>anger</b> ou <b>surprise</b> peut indiquer des contenus émotionnellement polarisants,
            ce qui est utile pour l'analyse des dynamiques de désinformation.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =====================================================
# PAGE 6 - GREEN IT
# =====================================================

elif page == "6. Green IT":
    section_title("6", "Suivi Green IT")

    st.markdown(
        """
        <div class="glass-card">
        Cette section affiche les mesures générées par <b>CodeCarbon</b> :
        temps d'exécution, énergie consommée et émissions CO₂ estimées.
        </div>
        """,
        unsafe_allow_html=True,
    )

    green_df = load_green_it_files()

    if not green_df.empty:
        if "project_name" in green_df.columns and "emissions" in green_df.columns:
            emissions_total = green_df["emissions"].sum()
        else:
            emissions_total = 0

        duration_total = green_df["duration"].sum() if "duration" in green_df.columns else 0
        energy_total = green_df["energy_consumed"].sum() if "energy_consumed" in green_df.columns else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Émissions totales", f"{emissions_total:.6f}", "kg CO₂eq", "🌱")
        with c2:
            metric_card("Temps mesuré", f"{duration_total:.1f}", "secondes", "⏱️")
        with c3:
            metric_card("Énergie totale", f"{energy_total:.6f}", "kWh", "⚡")

        tabs = st.tabs(["📊 Visualisations", "📄 Données brutes", "💡 Recommandations"])

        with tabs[0]:
            if "project_name" in green_df.columns and "emissions" in green_df.columns:
                col_a, col_b = st.columns([1, 1])

                with col_a:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("Émissions CO₂ par étape")
                    emissions_by_project = green_df.groupby("project_name", as_index=False)["emissions"].sum()
                    fig = px.bar(
                        emissions_by_project,
                        x="project_name",
                        y="emissions",
                        color="project_name",
                        title="Émissions CO₂ estimées",
                        labels={"project_name": "Étape", "emissions": "kg CO₂eq"},
                    )
                    st.plotly_chart(style_plot(fig), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_b:
                    if "duration" in green_df.columns:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.subheader("Temps d'exécution par étape")
                        duration_by_project = green_df.groupby("project_name", as_index=False)["duration"].sum()
                        fig = px.bar(
                            duration_by_project,
                            x="project_name",
                            y="duration",
                            color="project_name",
                            title="Temps d'exécution",
                            labels={"project_name": "Étape", "duration": "Secondes"},
                        )
                        st.plotly_chart(style_plot(fig), use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                if "energy_consumed" in green_df.columns:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("Énergie consommée")
                    energy_by_project = green_df.groupby("project_name", as_index=False)["energy_consumed"].sum()
                    fig = px.area(
                        energy_by_project,
                        x="project_name",
                        y="energy_consumed",
                        title="Énergie consommée par étape",
                        labels={"project_name": "Étape", "energy_consumed": "kWh"},
                    )
                    st.plotly_chart(style_plot(fig), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        with tabs[1]:
            st.subheader("Données CodeCarbon")
            st.dataframe(green_df, use_container_width=True, height=520)

        with tabs[2]:
            st.markdown(
                """
                <div class="glass-card">
                <h3>Recommandations Green IT</h3>
                <ul>
                    <li>Réduire <code>MAX_LENGTH</code> à 128 ou 256 pour les posts courts.</li>
                    <li>Ne classifier que les nouveaux posts non encore traités.</li>
                    <li>Utiliser <code>processed_posts.cleaned_text</code> pour éviter le bruit textuel.</li>
                    <li>Comparer RoBERTa avec une baseline plus légère : TF-IDF + Logistic Regression.</li>
                    <li>Stocker les résultats IA en base pour éviter de relancer les mêmes prédictions.</li>
                    <li>Limiter les traitements en démo avec un paramètre <code>limit</code>.</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        st.info("Aucune donnée Green IT trouvée. Lance d'abord classification.py ou emotion.py avec CodeCarbon activé.")


st.markdown(
    """
    <div class="footer-note">
        Thumalien · Bluesky Fake News Detection · Streamlit Advanced UI · Green IT Monitoring
    </div>
    """,
    unsafe_allow_html=True,
)
