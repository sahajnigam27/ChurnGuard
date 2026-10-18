from pathlib import Path
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# APPLICATION SETUP
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring import (
    initialize_database,
    get_monitoring_stats,
    get_prediction_history,
    get_drift_alert_history,
)

from src.drift_detection import detect_drift


st.set_page_config(
    page_title="ChurnGuard — Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN TOKENS
# ============================================================

BG = "#080A0F"
BG_SOFT = "#0B0F16"
PANEL = "#10151F"
PANEL_2 = "#141B27"
BORDER = "#252F3F"
TEXT = "#F4F7FB"
MUTED = "#8B98AB"
GREEN = "#43F0AD"
GREEN_SOFT = "#153D32"
BLUE = "#7AA7FF"
AMBER = "#FFC875"
RED = "#FF7188"
PURPLE = "#B6A0FF"

FONT = "Inter, sans-serif"
DISPLAY_FONT = "'Space Grotesk', sans-serif"


# ============================================================
# LUXURY UI STYLES
# ============================================================

st.markdown(
    f"""
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap'
    );

    :root {{
        color-scheme: dark;
        --bg: {BG};
        --panel: {PANEL};
        --border: {BORDER};
        --text: {TEXT};
        --muted: {MUTED};
        --green: {GREEN};
    }}

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(
                ellipse at 10% -10%,
                rgba(67,240,173,0.105),
                transparent 33%
            ),
            radial-gradient(
                ellipse at 100% 5%,
                rgba(122,167,255,0.075),
                transparent 28%
            ),
            linear-gradient(150deg, #080A0F 0%, #0A0E15 55%, #080B11 100%);
        color: {TEXT};
    }}

    .main .block-container {{
        max-width: 1600px;
        padding-top: 2.1rem;
        padding-bottom: 2.5rem;
    }}

    [data-testid="stHeader"] {{
        background: rgba(8,10,15,0.78);
        backdrop-filter: blur(18px);
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0D121B 0%, #090D14 100%);
        border-right: 1px solid rgba(139,152,171,0.16);
    }}

    [data-testid="stSidebar"] > div:first-child {{
        padding: 1.5rem 1rem;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {MUTED};
    }}

    .brand {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -1.4px;
        color: {TEXT};
        line-height: 1.1;
    }}

    .brand-accent {{
        color: {GREEN};
    }}

    .brand-symbol {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border: 1px solid rgba(67,240,173,0.4);
        border-radius: 12px;
        margin-right: 9px;
        color: {GREEN};
        background: rgba(67,240,173,0.08);
        font-size: 1.2rem;
    }}

    .brand-caption {{
        margin-top: 9px;
        margin-bottom: 26px;
        font-size: 0.67rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: {MUTED};
    }}

    .workspace-label {{
        color: #66758B;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin: 20px 0 8px;
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        padding: 27px 30px 25px;
        border: 1px solid rgba(139,152,171,0.19);
        border-radius: 22px;
        background:
            radial-gradient(
                ellipse at 90% 0%,
                rgba(67,240,173,0.12),
                transparent 42%
            ),
            linear-gradient(135deg, rgba(20,27,39,0.97), rgba(11,15,23,0.96));
        box-shadow: 0 18px 60px rgba(0,0,0,0.20);
        margin-bottom: 22px;
    }}

    .hero-eyebrow {{
        color: {GREEN};
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2.4px;
        margin-bottom: 13px;
    }}

    .hero-title {{
        color: {TEXT};
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.1rem, 4vw, 3.25rem);
        font-weight: 700;
        line-height: 1.05;
        letter-spacing: -2px;
        margin: 0;
    }}

    .hero-description {{
        color: {MUTED};
        max-width: 760px;
        font-size: 0.92rem;
        line-height: 1.8;
        margin-top: 14px;
    }}

    .hero-bottom {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
        margin-top: 22px;
    }}

    .pill {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 11px;
        border-radius: 999px;
        border: 1px solid rgba(139,152,171,0.20);
        background: rgba(8,10,15,0.38);
        color: #C2CCDA;
        font-size: 0.72rem;
        font-weight: 600;
    }}

    .pill-green {{
        border-color: rgba(67,240,173,0.35);
        background: rgba(67,240,173,0.075);
        color: {GREEN};
    }}

    .section-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.28rem;
        font-weight: 600;
        letter-spacing: -0.45px;
        color: {TEXT};
        margin: 22px 0 4px;
    }}

    .section-description {{
        font-size: 0.82rem;
        line-height: 1.6;
        color: {MUTED};
        margin-bottom: 15px;
    }}

    .metric-card {{
        position: relative;
        overflow: hidden;
        min-height: 151px;
        padding: 21px 20px 17px;
        border-radius: 17px;
        border: 1px solid rgba(139,152,171,0.17);
        background:
            linear-gradient(145deg, rgba(21,29,42,0.98), rgba(13,18,28,0.98));
        box-shadow: 0 12px 36px rgba(0,0,0,0.13);
        transition: transform 180ms ease, border-color 180ms ease,
                    box-shadow 180ms ease;
    }}

    .metric-card:hover {{
        transform: translateY(-3px);
        border-color: rgba(67,240,173,0.43);
        box-shadow: 0 18px 45px rgba(0,0,0,0.22);
    }}

    .metric-card::after {{
        content: "";
        position: absolute;
        width: 115px;
        height: 115px;
        right: -50px;
        top: -60px;
        border-radius: 50%;
        background: var(--metric-glow, rgba(67,240,173,0.09));
        filter: blur(5px);
    }}

    .metric-top {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
    }}

    .metric-label {{
        color: {MUTED};
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 1.15px;
        text-transform: uppercase;
    }}

    .metric-icon {{
        color: var(--metric-accent, {GREEN});
        font-size: 1.15rem;
        opacity: 0.95;
    }}

    .metric-value {{
        position: relative;
        z-index: 1;
        color: var(--metric-accent, {TEXT});
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.75rem, 2.3vw, 2.35rem);
        font-weight: 700;
        letter-spacing: -1.3px;
        margin-top: 17px;
        line-height: 1.08;
        overflow-wrap: anywhere;
    }}

    .metric-note {{
        color: #738197;
        font-size: 0.72rem;
        margin-top: 12px;
        line-height: 1.5;
    }}

    .surface {{
        border: 1px solid rgba(139,152,171,0.17);
        border-radius: 18px;
        padding: 20px;
        background: linear-gradient(
            145deg,
            rgba(17,23,34,0.94),
            rgba(12,17,26,0.96)
        );
        box-shadow: 0 12px 35px rgba(0,0,0,0.11);
    }}

    .surface-label {{
        font-size: 0.69rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: {MUTED};
    }}

    .surface-value {{
        color: {TEXT};
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 8px;
    }}

    .result-card {{
        border-radius: 19px;
        border: 1px solid rgba(67,240,173,0.3);
        background:
            radial-gradient(ellipse at 100% 0%, rgba(67,240,173,0.10), transparent 52%),
            linear-gradient(145deg, #121C25, #0D131D);
        padding: 24px;
        margin-top: 15px;
    }}

    .result-kicker {{
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 1.6px;
        font-size: 0.68rem;
        font-weight: 700;
    }}

    .result-main {{
        color: {TEXT};
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.7px;
        margin-top: 8px;
    }}

    .result-caption {{
        color: {MUTED};
        font-size: 0.8rem;
        line-height: 1.7;
        margin-top: 10px;
    }}

    div[data-testid="stButton"] > button,
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stDownloadButton"] > button {{
        border-radius: 11px;
        font-weight: 600;
        min-height: 42px;
        transition: all 180ms ease;
    }}

    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button[kind="primary"] {{
        background: {GREEN};
        color: #06120D;
        border: 1px solid {GREEN};
    }}

    div[data-testid="stButton"] > button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {{
        background: #7AF7C3;
        border-color: #7AF7C3;
        box-shadow: 0 0 25px rgba(67,240,173,0.15);
        color: #06120D;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 13px;
        overflow: hidden;
    }}

    div[data-testid="stTabs"] button {{
        border-radius: 9px;
    }}

    [data-testid="stMetric"] {{
        background: rgba(16,21,31,0.8);
        border: 1px solid rgba(139,152,171,0.16);
        border-radius: 13px;
        padding: 13px;
    }}

    input, textarea {{
        border-radius: 9px !important;
    }}

    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] > div > div {{
        border-radius: 10px;
    }}

    hr {{
        border-color: rgba(139,152,171,0.18);
    }}

    #MainMenu, footer {{
        visibility: hidden;
    }}

    .footer {{
        text-align: center;
        color: #637187;
        font-size: 0.70rem;
        letter-spacing: 0.65px;
        line-height: 1.9;
        padding: 16px 0 4px;
    }}

    .footer strong {{
        color: {GREEN};
        letter-spacing: 1.8px;
    }}

    @media (max-width: 700px) {{
        .main .block-container {{
            padding: 1rem 0.8rem 2rem;
        }}

        .hero {{
            padding: 22px 19px;
        }}

        .hero-title {{
            font-size: 2rem;
        }}

        .metric-card {{
            padding: 15px;
            min-height: 130px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section_heading(title, description=None):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )
    if description:
        st.markdown(
            f'<div class="section-description">{description}</div>',
            unsafe_allow_html=True,
        )


def metric_card(label, value, note, accent=GREEN, icon="◈"):
    st.markdown(
        f"""
        <div class="metric-card"
             style="--metric-accent:{accent};
                    --metric-glow:{accent}18;">
            <div class="metric-top">
                <div class="metric-label">{label}</div>
                <div class="metric-icon">{icon}</div>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig, height=340):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans, sans-serif",
            color=TEXT,
            size=12,
        ),
        margin=dict(l=12, r=12, t=35, b=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(color=MUTED),
        ),
        hoverlabel=dict(
            bgcolor=PANEL_2,
            bordercolor=BORDER,
            font_color=TEXT,
            font_family="DM Sans, sans-serif",
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(color=MUTED),
    )

    fig.update_yaxes(
        gridcolor="rgba(139,152,171,0.12)",
        zeroline=False,
        linecolor=BORDER,
        tickfont=dict(color=MUTED),
    )

    return fig


def format_probability(value):
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "N/A"


def normalize_history(data):
    if data is None:
        return pd.DataFrame()

    if isinstance(data, pd.DataFrame):
        return data.copy()

    try:
        return pd.DataFrame(data)
    except (TypeError, ValueError):
        return pd.DataFrame()


def fetch_api_health(base_url):
    if not base_url:
        return False, {}

    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/health",
            timeout=3,
        )
        response.raise_for_status()
        return True, response.json()
    except (requests.RequestException, ValueError):
        return False, {}


def safe_number(value, default=0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def add_center_label(fig, value, label):
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text=(
            f"<b>{value:,}</b>"
            f"<br><span style='font-size:11px;color:{MUTED}'>{label}</span>"
        ),
        showarrow=False,
        font=dict(size=23, color=TEXT),
    )


# ============================================================
# INITIALIZE DATABASE AND LOAD LIVE DATA
# ============================================================

try:
    initialize_database()
except Exception as error:
    st.error(f"Database initialization failed: {error}")

try:
    stats = get_monitoring_stats() or {}
except Exception as error:
    stats = {}
    st.error(f"Could not load monitoring statistics: {error}")

try:
    history = normalize_history(get_prediction_history(limit=500))
except Exception as error:
    history = pd.DataFrame()
    st.error(f"Could not load prediction history: {error}")

try:
    drift_alerts = normalize_history(get_drift_alert_history(limit=200))
except Exception as error:
    drift_alerts = pd.DataFrame()
    st.error(f"Could not load drift history: {error}")


total_predictions = int(stats.get("total_predictions", 0) or 0)
predicted_churn = int(stats.get("predicted_churn", 0) or 0)
predicted_no_churn = int(stats.get("predicted_no_churn", 0) or 0)
churn_rate = safe_number(stats.get("predicted_churn_rate", 0))
average_probability = safe_number(
    stats.get("average_churn_probability", 0)
)
total_drift_alerts = int(stats.get("total_drift_alerts", 0) or 0)
active_drift_alerts = int(stats.get("active_drift_alerts", 0) or 0)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand">
            <span class="brand-symbol">◈</span>Churn<span class="brand-accent">Guard</span>
        </div>
        <div class="brand-caption">Customer intelligence platform</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="workspace-label">Workspace</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Live Prediction",
            "Prediction Explorer",
            "Drift Intelligence",
            "Platform Info",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown('<div class="workspace-label">Connectivity</div>', unsafe_allow_html=True)

    api_url = st.text_input(
        "Prediction API URL",
        value="http://127.0.0.1:8000",
        help="The FastAPI server must be running for live predictions.",
    ).strip().rstrip("/")

    api_online, api_health = fetch_api_health(api_url)

    if api_online:
        st.markdown(
            f'<div class="pill pill-green">● API CONNECTED</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="pill">● API OFFLINE</div>',
            unsafe_allow_html=True,
        )
        st.caption("Start the FastAPI server to enable predictions.")

    st.divider()

    if st.button("↻ Refresh workspace", use_container_width=True):
        st.rerun()

    st.markdown('<div class="workspace-label">Environment</div>', unsafe_allow_html=True)
    st.caption("Local ML monitoring")
    st.caption(f"Updated {datetime.now().strftime('%d %b %Y · %H:%M:%S')}")


# ============================================================
# GLOBAL HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-eyebrow">◈ &nbsp; Customer intelligence / ML operations</div>
        <h1 class="hero-title">ChurnGuard<span style="color:{GREEN}">.</span></h1>
        <div class="hero-description">
            A unified intelligence workspace for customer churn predictions,
            risk exploration and statistical data drift monitoring.
            Turn model outputs into a clearer operational picture.
        </div>
        <div class="hero-bottom">
            <span class="pill pill-green">● MONITORING WORKSPACE</span>
            <span class="pill">◈ &nbsp; MACHINE LEARNING</span>
            <span class="pill">↗ &nbsp; INTERACTIVE ANALYTICS</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    section_heading(
        "Executive overview",
        "Your live snapshot of recorded model predictions and monitoring activity.",
    )

    cards = st.columns(4, gap="medium")

    with cards[0]:
        metric_card(
            "Total predictions",
            f"{total_predictions:,}",
            "Recorded API predictions",
            BLUE,
            "↗",
        )

    with cards[1]:
        metric_card(
            "Predicted churn",
            f"{predicted_churn:,}",
            "Classified as likely to churn",
            AMBER,
            "◉",
        )

    with cards[2]:
        metric_card(
            "Churn classification rate",
            format_probability(churn_rate),
            "Share classified as churn",
            GREEN,
            "◌",
        )

    with cards[3]:
        metric_card(
            "Average churn probability",
            format_probability(average_probability),
            "Mean model-estimated probability",
            PURPLE,
            "◎",
        )

    left, right = st.columns([1.25, 1], gap="large")

    with left:
        section_heading(
            "Prediction composition",
            "Distribution of model classifications in recorded predictions.",
        )

        if total_predictions > 0:
            distribution = pd.DataFrame({
                "Classification": ["Predicted churn", "Predicted no churn"],
                "Count": [predicted_churn, predicted_no_churn],
            })

            fig = px.pie(
                distribution,
                names="Classification",
                values="Count",
                hole=0.76,
                color="Classification",
                color_discrete_map={
                    "Predicted churn": AMBER,
                    "Predicted no churn": GREEN,
                },
            )

            fig.update_traces(
                textposition="outside",
                textinfo="label+percent",
                marker=dict(line=dict(color=BG, width=4)),
                hovertemplate=(
                    "%{label}<br>Count: %{value:,}"
                    "<br>Share: %{percent}<extra></extra>"
                ),
            )

            add_center_label(fig, total_predictions, "TOTAL SCORED")
            style_figure(fig, 370)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions yet. Generate one from Live Prediction.")

    with right:
        section_heading(
            "Monitoring control",
            "Status indicators from the local monitoring database.",
        )

        st.markdown(
            f"""
            <div class="surface">
                <div class="surface-label">Drift alerts recorded</div>
                <div class="surface-value">{total_drift_alerts:,}</div>
                <div class="metric-note">Total drift records stored in SQLite</div>
                <hr>
                <div class="surface-label">Active drift alerts</div>
                <div class="surface-value" style="color:{RED if active_drift_alerts else GREEN}">
                    {active_drift_alerts:,}
                </div>
                <div class="metric-note">
                    No active alerts does not mean a fresh drift evaluation has run.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        section_heading("Classification volume")

        fig = go.Figure(
            go.Bar(
                x=["Predicted churn", "Predicted no churn"],
                y=[predicted_churn, predicted_no_churn],
                marker=dict(
                    color=[AMBER, GREEN],
                    line=dict(width=0),
                ),
                text=[predicted_churn, predicted_no_churn],
                textposition="outside",
                hovertemplate="%{x}<br>Count: %{y:,}<extra></extra>",
            )
        )

        fig.update_layout(showlegend=False)
        style_figure(fig, 250)
        st.plotly_chart(fig, use_container_width=True)

    section_heading(
        "Prediction activity",
        "Latest records received by the monitoring database.",
    )

    if not history.empty:
        recent = history.head(10).copy()

        if "churn_probability" in recent.columns:
            recent["churn_probability"] = recent["churn_probability"].apply(
                format_probability
            )

        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Prediction history will appear here after your first API request.")

    st.caption(
        "All displayed counts are based on recorded model predictions, "
        "not verified customer outcomes."
    )


# ============================================================
# LIVE PREDICTION
# ============================================================

elif page == "Live Prediction":

    section_heading(
        "Live risk intelligence",
        "Submit a customer profile to your trained FastAPI prediction service.",
    )

    if not api_online:
        st.warning(
            "The prediction API is currently offline. Start FastAPI in a "
            "separate terminal before submitting a customer profile."
        )

    st.markdown(
        """
        <div class="surface">
            <div class="surface-label">Prediction workflow</div>
            <div style="color:#F4F7FB;font-size:1rem;font-weight:600;margin-top:9px">
                Customer profile → ML pipeline → Churn probability
            </div>
            <div class="metric-note">
                The result comes from the trained model behind your API.
                No simulated predictions are displayed.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("customer_prediction_form"):

        section_heading("01 / Customer profile")

        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1])
            partner = st.selectbox("Has partner", ["Yes", "No"])
            dependents = st.selectbox("Has dependents", ["Yes", "No"])

        with c2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple lines",
                ["No", "Yes", "No phone service"],
            )
            internet = st.selectbox(
                "Internet service",
                ["DSL", "Fiber optic", "No"],
            )

        with c3:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        st.divider()
        section_heading("02 / Service configuration")

        s1, s2, s3 = st.columns(3)

        if internet == "No":
            internet_options = ["No internet service", "No", "Yes"]
        else:
            internet_options = ["Yes", "No", "No internet service"]

        with s1:
            online_security = st.selectbox(
                "Online security",
                internet_options,
            )
            online_backup = st.selectbox(
                "Online backup",
                internet_options,
            )
            device_protection = st.selectbox(
                "Device protection",
                internet_options,
            )

        with s2:
            tech_support = st.selectbox("Tech support", internet_options)
            streaming_tv = st.selectbox("Streaming TV", internet_options)
            streaming_movies = st.selectbox(
                "Streaming movies",
                internet_options,
            )

        with s3:
            monthly_charges = st.number_input(
                "Monthly charges",
                min_value=0.0,
                max_value=10000.0,
                value=70.0,
                step=5.0,
            )
            total_charges = st.number_input(
                "Total charges",
                min_value=0.0,
                max_value=1000000.0,
                value=840.0,
                step=10.0,
            )
            st.caption("Use the currency units represented in the source dataset.")

        submitted = st.form_submit_button(
            "◈  Analyze customer risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        if not api_url:
            st.error("Enter a valid API URL in the sidebar.")
        else:
            try:
                with st.spinner("Scoring customer with the trained model..."):
                    response = requests.post(
                        f"{api_url}/predict",
                        json=customer,
                        timeout=30,
                    )

                if response.status_code >= 400:
                    st.error(
                        f"API error {response.status_code}: "
                        f"{response.text[:1000]}"
                    )
                else:
                    result = response.json()

                    if result.get("predicted_churn") is None:
                        st.error(
                            "The API response is missing 'predicted_churn'. "
                            "Check the response schema in src/api.py."
                        )
                    else:
                        st.session_state["churnguard_last_prediction"] = {
                            "customer": customer,
                            "result": result,
                        }
                        st.success("Customer profile scored successfully.")

            except (requests.RequestException, ValueError) as error:
                st.error(f"Prediction request failed: {error}")

    last_prediction = st.session_state.get("churnguard_last_prediction")

    if last_prediction:
        result = last_prediction["result"]
        customer = last_prediction["customer"]

        prediction = result.get("predicted_churn")
        probability = result.get("churn_probability")

        st.divider()
        section_heading(
            "Assessment report",
            "Results returned by your current model endpoint.",
        )

        if prediction == 1:
            st.error(
                "PREDICTED CHURN — the model classifies this profile "
                "as likely to churn."
            )
        else:
            st.success(
                "PREDICTED NO CHURN — the model classifies this profile "
                "as likely to stay."
            )

        result_col1, result_col2 = st.columns([1, 1.3], gap="large")

        with result_col1:
            outcome = (
                "Predicted churn"
                if prediction == 1
                else "Predicted no churn"
            )

            accent = AMBER if prediction == 1 else GREEN

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-kicker">Model classification</div>
                    <div class="result-main" style="color:{accent}">
                        {outcome}
                    </div>
                    <div class="result-caption">
                        This is a statistical estimate from the trained
                        model, not confirmation of a future customer outcome.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            try:
                p = float(probability)
                metric_card(
                    "Estimated churn probability",
                    f"{p:.1%}",
                    "Model output; calibration is not guaranteed",
                    accent,
                    "◎",
                )
            except (TypeError, ValueError):
                st.info("The API did not return a numeric probability.")

        with result_col2:
            try:
                p = max(0.0, min(1.0, float(probability)))

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=p * 100,
                        number={
                            "suffix": "%",
                            "font": {"size": 42, "color": TEXT},
                        },
                        title={
                            "text": "CHURN PROBABILITY",
                            "font": {"size": 13, "color": MUTED},
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100],
                                "ticksuffix": "%",
                                "tickcolor": MUTED,
                                "tickfont": {"color": MUTED},
                            },
                            "bar": {
                                "color": AMBER if prediction == 1 else GREEN,
                                "thickness": 0.28,
                            },
                            "bgcolor": PANEL,
                            "bordercolor": BORDER,
                            "borderwidth": 1,
                            "steps": [
                                {"range": [0, 35], "color": "#15382F"},
                                {"range": [35, 65], "color": "#40351F"},
                                {"range": [65, 100], "color": "#40232D"},
                            ],
                        },
                    )
                )

                style_figure(gauge, 330)
                st.plotly_chart(gauge, use_container_width=True)

            except (TypeError, ValueError):
                st.info("No numeric probability is available to visualize.")

        with st.expander("Inspect submitted customer profile"):
            st.json(customer)

        with st.expander("Inspect raw API response"):
            st.json(result)

        st.download_button(
            "Download assessment JSON",
            data=pd.Series(
                {
                    "customer": customer,
                    "result": result,
                }
            ).to_json(indent=2),
            file_name="churnguard_assessment.json",
            mime="application/json",
        )

        st.caption(
            "The prediction probability is not necessarily calibrated. "
            "It should not be interpreted as a verified real-world churn rate."
        )


# ============================================================
# PREDICTION EXPLORER
# ============================================================

elif page == "Prediction Explorer":

    section_heading(
        "Prediction explorer",
        "A filterable workspace for reviewing and exporting recorded predictions.",
    )

    if history.empty:
        st.info(
            "No prediction records found. Generate predictions through "
            "Live Prediction or the API first."
        )
    else:
        filtered = history.copy()

        f1, f2, f3 = st.columns(3)

        with f1:
            if "predicted_churn" in filtered.columns:
                outcomes = st.multiselect(
                    "Model classification",
                    options=[0, 1],
                    default=[0, 1],
                    format_func=lambda v: (
                        "Predicted churn" if v == 1
                        else "Predicted no churn"
                    ),
                )
                filtered = filtered[
                    filtered["predicted_churn"].isin(outcomes)
                ]

        with f2:
            min_p = st.slider(
                "Minimum probability",
                0.0,
                1.0,
                0.0,
                0.05,
            )

        with f3:
            max_p = st.slider(
                "Maximum probability",
                0.0,
                1.0,
                1.0,
                0.05,
            )

        if min_p > max_p:
            st.warning("Minimum probability exceeds maximum.")
            filtered = filtered.iloc[0:0]
        elif "churn_probability" in filtered.columns:
            filtered = filtered[
                pd.to_numeric(
                    filtered["churn_probability"],
                    errors="coerce",
                ).between(min_p, max_p)
            ]

        search = st.text_input(
            "Search all visible fields",
            placeholder="Search timestamp, model, or customer details...",
        )

        if search:
            search_mask = filtered.astype(str).apply(
                lambda col: col.str.contains(
                    search,
                    case=False,
                    regex=False,
                    na=False,
                )
            ).any(axis=1)
            filtered = filtered[search_mask]

        m1, m2, m3 = st.columns(3)

        m1.metric("Matching records", f"{len(filtered):,}")

        if "predicted_churn" in filtered.columns:
            m2.metric(
                "Predicted churn",
                int((filtered["predicted_churn"] == 1).sum()),
            )
        else:
            m2.metric("Predicted churn", "N/A")

        if "churn_probability" in filtered.columns and not filtered.empty:
            avg = pd.to_numeric(
                filtered["churn_probability"],
                errors="coerce",
            ).mean()
            m3.metric("Mean probability", format_probability(avg))
        else:
            m3.metric("Mean probability", "N/A")

        if not filtered.empty and "churn_probability" in filtered.columns:
            chart_data = filtered.copy()
            chart_data["churn_probability"] = pd.to_numeric(
                chart_data["churn_probability"],
                errors="coerce",
            )

            if "predicted_churn" in chart_data.columns:
                chart_data["Classification"] = chart_data[
                    "predicted_churn"
                ].map({
                    0: "Predicted no churn",
                    1: "Predicted churn",
                })

                fig = px.histogram(
                    chart_data,
                    x="churn_probability",
                    color="Classification",
                    nbins=20,
                    color_discrete_map={
                        "Predicted churn": AMBER,
                        "Predicted no churn": GREEN,
                    },
                    labels={
                        "churn_probability": "Churn probability",
                        "count": "Number of records",
                    },
                )
            else:
                fig = px.histogram(
                    chart_data,
                    x="churn_probability",
                    nbins=20,
                )

            style_figure(fig, 320)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "↓  Export filtered records as CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="churnguard_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# DRIFT INTELLIGENCE
# ============================================================

elif page == "Drift Intelligence":

    section_heading(
        "Drift intelligence",
        "Compare incoming prediction features with the reference distribution.",
    )

    st.markdown(
        f"""
        <div class="surface">
            <div class="surface-label">Population Stability Index · PSI</div>
            <div style="margin-top:12px;line-height:2.0;color:{MUTED};font-size:0.87rem">
                <span style="color:{GREEN};font-weight:700">STABLE</span>
                &nbsp; PSI below 0.10<br>
                <span style="color:{AMBER};font-weight:700">WARNING</span>
                &nbsp; PSI from 0.10 to below 0.25<br>
                <span style="color:{RED};font-weight:700">CRITICAL</span>
                &nbsp; PSI of 0.25 or higher
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "These are practical thresholds, not universal statistical laws. "
        "Feature drift alone does not prove reduced model accuracy."
    )

    if st.button(
        "◈  Run drift evaluation",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Comparing current and reference distributions..."):
            try:
                report = detect_drift()
                st.session_state["churnguard_drift_report"] = report
            except Exception as error:
                st.error(f"Drift evaluation failed: {error}")

    report = st.session_state.get("churnguard_drift_report")

    if report:
        status = str(report.get("status", "UNKNOWN")).upper()

        if status == "STABLE":
            st.success("DRIFT STATUS · STABLE")
        elif status == "WARNING":
            st.warning("DRIFT STATUS · WARNING")
        elif status == "CRITICAL":
            st.error("DRIFT STATUS · CRITICAL")
        elif status == "INSUFFICIENT_DATA":
            st.info("DRIFT STATUS · INSUFFICIENT DATA")
        else:
            st.info(f"DRIFT STATUS · {status}")

        d1, d2, d3 = st.columns(3)

        d1.metric("Reference rows", report.get("reference_rows", 0))
        d2.metric("Current rows", report.get("current_rows", 0))
        d3.metric("Features evaluated", report.get("features_checked", 0))

        if report.get("message"):
            st.info(report["message"])

        results = report.get("results", [])

        if results:
            result_df = pd.DataFrame(results)

            if "psi" in result_df.columns:
                result_df["psi"] = pd.to_numeric(
                    result_df["psi"],
                    errors="coerce",
                )

            section_heading(
                "Feature-level drift scores",
                "Inspect PSI values and compare them with practical thresholds.",
            )

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True,
            )

            if "psi" in result_df.columns and "feature" in result_df.columns:
                plot_df = result_df.dropna(subset=["psi"]).copy()

                if not plot_df.empty:
                    plot_df["Severity"] = plot_df["psi"].apply(
                        lambda value: (
                            "Critical" if value >= 0.25
                            else "Warning" if value >= 0.10
                            else "Stable"
                        )
                    )

                    fig = px.bar(
                        plot_df.sort_values("psi"),
                        x="psi",
                        y="feature",
                        orientation="h",
                        color="Severity",
                        color_discrete_map={
                            "Stable": GREEN,
                            "Warning": AMBER,
                            "Critical": RED,
                        },
                        hover_data=["psi", "Severity"],
                        labels={
                            "psi": "Population Stability Index",
                            "feature": "Feature",
                        },
                    )

                    fig.add_vline(
                        x=0.10,
                        line_dash="dash",
                        line_color=AMBER,
                        annotation_text="Warning",
                    )

                    fig.add_vline(
                        x=0.25,
                        line_dash="dash",
                        line_color=RED,
                        annotation_text="Critical",
                    )

                    style_figure(fig, max(300, len(plot_df) * 34))
                    st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "↓  Export current drift report",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="churnguard_drift_report.csv",
                mime="text/csv",
            )

    st.divider()

    section_heading(
        "Historical monitoring records",
        "Drift checks and alerts recorded by the monitoring system.",
    )

    try:
        drift_alerts = normalize_history(get_drift_alert_history(limit=200))
    except Exception as error:
        drift_alerts = pd.DataFrame()
        st.error(f"Could not load drift history: {error}")

    if drift_alerts.empty:
        st.info(
            "No historical drift records are available yet. "
            "Run an evaluation when sufficient prediction data exists."
        )
    else:
        st.dataframe(
            drift_alerts,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "↓  Export drift history",
            data=drift_alerts.to_csv(index=False).encode("utf-8"),
            file_name="churnguard_drift_history.csv",
            mime="text/csv",
        )


# ============================================================
# PLATFORM INFO
# ============================================================

elif page == "Platform Info":

    section_heading(
        "Platform architecture",
        "The components powering the ChurnGuard ML monitoring workflow.",
    )

    components = [
        (
            "01",
            "Customer data",
            "IBM Telco Customer Churn dataset",
            BLUE,
        ),
        (
            "02",
            "Machine learning",
            "scikit-learn preprocessing and classification pipeline",
            PURPLE,
        ),
        (
            "03",
            "Prediction service",
            "FastAPI REST endpoints",
            GREEN,
        ),
        (
            "04",
            "Monitoring layer",
            "SQLite prediction history and PSI drift checks",
            AMBER,
        ),
        (
            "05",
            "Interface",
            "Streamlit workspace with interactive Plotly charts",
            BLUE,
        ),
    ]

    for code, title, description, accent in components:
        st.markdown(
            f"""
            <div class="surface" style="margin-bottom:12px;display:flex;gap:18px;align-items:flex-start">
                <div style="min-width:44px;height:44px;display:flex;align-items:center;
                            justify-content:center;border-radius:12px;
                            color:{accent};background:{accent}14;
                            border:1px solid {accent}35;font-weight:700">
                    {code}
                </div>
                <div>
                    <div style="font-weight:700;color:{TEXT};font-size:1rem">
                        {title}
                    </div>
                    <div style="color:{MUTED};font-size:0.83rem;margin-top:6px;line-height:1.6">
                        {description}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_heading("Workspace metrics")

    p1, p2, p3 = st.columns(3)

    p1.metric("Recorded predictions", f"{total_predictions:,}")
    p2.metric("Drift records", f"{total_drift_alerts:,}")
    p3.metric("Active alerts", f"{active_drift_alerts:,}")

    section_heading("Technology stack")

    st.markdown(
        """
        `Python` · `Pandas` · `NumPy` · `scikit-learn` ·
        `FastAPI` · `SQLite` · `Streamlit` · `Plotly` · `pytest`
        """
    )

    section_heading("Important limitations")

    st.warning(
        "Predictions are not verified customer outcomes. Churn probability "
        "may not be calibrated, and distribution drift does not necessarily "
        "mean model performance has declined. Actual outcome labels are "
        "needed to measure real-world performance."
    )

    if api_online:
        st.success("Prediction API is reachable.")
        with st.expander("Inspect API health response"):
            st.json(api_health)
    else:
        st.warning("Prediction API is not reachable from this dashboard.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    f"""
    <div class="footer">
        <strong>CHURNGUARD</strong><br>
        CUSTOMER INTELLIGENCE &nbsp;·&nbsp; ML OPERATIONS<br>
        <span style="color:#465367">
            Model estimates and monitoring insights for informed decisions.
            &nbsp;·&nbsp; {datetime.now().strftime('%Y')}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)