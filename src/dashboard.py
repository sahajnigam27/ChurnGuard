from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ChurnGuard | ML Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #263244;
        padding: 18px;
        border-radius: 12px;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #f8fafc;
    }

    .subtitle {
        color: #94a3b8;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE
# ============================================================

initialize_database()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ ChurnGuard")
st.sidebar.caption("Customer Churn Prediction & ML Monitoring")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Prediction History",
        "Drift Monitoring",
        "About",
    ],
)

st.sidebar.divider()

if st.sidebar.button("🔄 Refresh dashboard", use_container_width=True):
    st.rerun()

st.sidebar.caption("Live data from the local SQLite database.")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">ChurnGuard Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Customer churn prediction, analytics and model input monitoring.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MONITORING DATA
# ============================================================

stats = get_monitoring_stats()
history = get_prediction_history(limit=500)
drift_alerts = get_drift_alert_history(limit=200)

if history is None:
    history = pd.DataFrame()

if drift_alerts is None:
    drift_alerts = pd.DataFrame()


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader("Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Predictions",
        f"{stats.get('total_predictions', 0):,}",
    )

    col2.metric(
        "Predicted Churn",
        f"{stats.get('predicted_churn', 0):,}",
    )

    col3.metric(
        "Predicted No Churn",
        f"{stats.get('predicted_no_churn', 0):,}",
    )

    col4.metric(
        "Average Churn Probability",
        f"{stats.get('average_churn_probability', 0):.1%}",
    )

    st.subheader("Prediction Analytics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Prediction Distribution")

        churn_count = stats.get("predicted_churn", 0)
        no_churn_count = stats.get("predicted_no_churn", 0)

        if churn_count + no_churn_count > 0:
            fig, ax = plt.subplots(figsize=(7, 4))

            ax.bar(
                ["Predicted Churn", "Predicted No Churn"],
                [churn_count, no_churn_count],
            )

            ax.set_ylabel("Number of predictions")
            ax.set_title("Predicted Customer Outcomes")
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        else:
            st.info("No predictions recorded yet.")

    with chart_col2:
        st.markdown("#### Monitoring Summary")

        st.metric(
            "Predicted Churn Rate",
            f"{stats.get('predicted_churn_rate', 0):.1%}",
        )

        st.metric(
            "Total Drift Alerts",
            f"{stats.get('total_drift_alerts', 0):,}",
        )

        st.metric(
            "Active Drift Alerts",
            f"{stats.get('active_drift_alerts', 0):,}",
        )

        st.caption(
            "These are model predictions, not verified customer outcomes."
        )

    st.subheader("Recent Predictions")

    if not history.empty:
        st.dataframe(
            history.head(10),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Prediction history will appear here after API requests.")


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "Prediction History":

    st.subheader("Prediction History")

    if history.empty:
        st.info("No prediction records found.")

    else:
        filtered_history = history.copy()

        if "predicted_churn" in filtered_history.columns:

            outcomes = st.multiselect(
                "Filter by prediction",
                options=[0, 1],
                default=[0, 1],
                format_func=lambda value: (
                    "Predicted Churn"
                    if value == 1
                    else "Predicted No Churn"
                ),
            )

            filtered_history = filtered_history[
                filtered_history["predicted_churn"].isin(outcomes)
            ]

        if "churn_probability" in filtered_history.columns:

            min_probability = st.slider(
                "Minimum churn probability",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
            )

            filtered_history = filtered_history[
                filtered_history["churn_probability"] >= min_probability
            ]

        st.caption(f"{len(filtered_history)} matching records")

        st.dataframe(
            filtered_history,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = filtered_history.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download filtered history as CSV",
            data=csv_data,
            file_name="churnguard_prediction_history.csv",
            mime="text/csv",
        )


# ============================================================
# DRIFT MONITORING
# ============================================================

elif page == "Drift Monitoring":

    st.subheader("Statistical Data Drift Monitoring")

    st.write(
        "Compare incoming customer features with the original "
        "Telco customer dataset using Population Stability Index (PSI)."
    )

    st.info(
        "At least 30 valid prediction records are required before "
        "this detector evaluates feature distributions."
    )

    st.markdown(
        """
        **PSI interpretation**

        - **STABLE:** PSI below 0.10
        - **WARNING:** PSI from 0.10 to below 0.25
        - **CRITICAL:** PSI of 0.25 or higher
        """
    )

    st.caption(
        "These are practical thresholds, not universal statistical laws. "
        "Drift does not by itself prove reduced model accuracy."
    )

    if st.button("🔍 Run Drift Check", type="primary"):

        with st.spinner("Comparing reference and current data..."):

            try:
                report = detect_drift()
                st.session_state["churnguard_drift_report"] = report

            except Exception as error:
                st.error(f"Drift detection failed: {error}")

    report = st.session_state.get("churnguard_drift_report")

    if report:

        status = report.get("status", "UNKNOWN")

        if status == "STABLE":
            st.success("Drift status: STABLE")

        elif status == "WARNING":
            st.warning("Drift status: WARNING")

        elif status == "CRITICAL":
            st.error("Drift status: CRITICAL")

        elif status == "INSUFFICIENT_DATA":
            st.info("Drift status: INSUFFICIENT DATA")

        else:
            st.info(f"Drift status: {status}")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Reference Rows",
            report.get("reference_rows", 0),
        )

        metric2.metric(
            "Current Rows",
            report.get("current_rows", 0),
        )

        metric3.metric(
            "Features Evaluated",
            report.get("features_checked", 0),
        )

        if report.get("message"):
            st.write(report["message"])

        results = report.get("results", [])

        if results:

            results_df = pd.DataFrame(results)

            st.markdown("### Feature-Level Drift Results")

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
            )

            evaluated_df = results_df.dropna(subset=["psi"])

            if not evaluated_df.empty:

                fig, ax = plt.subplots(figsize=(10, 5))

                ax.bar(
                    evaluated_df["feature"],
                    evaluated_df["psi"],
                )

                ax.axhline(
                    y=0.10,
                    linestyle="--",
                    label="Warning threshold (0.10)",
                )

                ax.axhline(
                    y=0.25,
                    linestyle="--",
                    label="Critical threshold (0.25)",
                )

                ax.set_ylabel("Population Stability Index")
                ax.set_title("PSI by Feature")
                ax.tick_params(axis="x", rotation=35)
                ax.legend()
                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

        st.caption(
            "Each successful drift check is recorded in the monitoring "
            "database for historical review."
        )

    st.markdown("### Recorded Drift Alerts")

    drift_alerts = get_drift_alert_history(limit=200)

    if drift_alerts is not None and not drift_alerts.empty:

        st.dataframe(
            drift_alerts,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No drift evaluation records are available yet. "
            "Run a check after enough prediction data has accumulated."
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.subheader("About ChurnGuard")

    st.write(
        "ChurnGuard is an end-to-end machine learning project that "
        "predicts customer churn and monitors prediction activity "
        "and incoming feature distributions."
    )

    st.markdown(
        """
        **Technology Stack**

        - Python, Pandas and NumPy
        - Scikit-learn
        - FastAPI
        - SQLite
        - Streamlit
        - Matplotlib
        - Population Stability Index (PSI)
        """
    )

    st.info(
        "Actual customer outcomes are required to evaluate real-world "
        "model accuracy and performance over time."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ChurnGuard | Customer Churn Prediction & ML Monitoring Platform"
)