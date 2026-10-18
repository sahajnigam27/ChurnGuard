"""
ChurnGuard — Monitoring Database

Responsibilities:
1. Initialize the SQLite database.
2. Store customer predictions.
3. Retrieve prediction history.
4. Calculate monitoring statistics.
5. Store data drift alerts.
"""

import sqlite3
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATABASE_DIR = ROOT / "data" / "monitoring"
DATABASE_PATH = DATABASE_DIR / "churnguard.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """Create a connection to the SQLite database."""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():
    """Create database tables if they do not already exist."""

    with get_connection() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_churn INTEGER NOT NULL
                    CHECK (predicted_churn IN (0, 1)),
                churn_probability REAL NOT NULL
                    CHECK (
                        churn_probability >= 0
                        AND churn_probability <= 1
                    ),
                model_name TEXT NOT NULL,
                customer_data TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT NOT NULL,
                drift_score REAL,
                threshold REAL,
                detected INTEGER NOT NULL
                    CHECK (detected IN (0, 1)),
                details TEXT,
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                )
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_predictions_created_at
            ON predictions(created_at)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_drift_alerts_created_at
            ON drift_alerts(created_at)
            """
        )


# ============================================================
# PREDICTION LOGGING
# ============================================================

def log_prediction(
    customer_data,
    predicted_churn,
    churn_probability,
    model_name,
):
    """Save one prediction and its associated customer features."""

    import json

    if predicted_churn not in (0, 1):
        raise ValueError("predicted_churn must be 0 or 1.")

    if not 0 <= churn_probability <= 1:
        raise ValueError(
            "churn_probability must be between 0 and 1."
        )

    with get_connection() as connection:

        cursor = connection.execute(
            """
            INSERT INTO predictions (
                predicted_churn,
                churn_probability,
                model_name,
                customer_data
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                int(predicted_churn),
                float(churn_probability),
                str(model_name),
                json.dumps(customer_data),
            ),
        )

        return cursor.lastrowid


# ============================================================
# PREDICTION HISTORY
# ============================================================

def get_prediction_history(limit=100):
    """Retrieve the most recent predictions."""

    limit = max(1, min(int(limit), 10000))

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT
                id,
                predicted_churn,
                churn_probability,
                model_name,
                customer_data,
                created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return pd.DataFrame(
        [dict(row) for row in rows]
    )


# ============================================================
# MONITORING STATISTICS
# ============================================================

def get_monitoring_stats():
    """Calculate summary statistics from recorded predictions."""

    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_predictions,
                SUM(predicted_churn) AS predicted_churn,
                AVG(churn_probability) AS average_churn_probability,
                MAX(created_at) AS latest_prediction
            FROM predictions
            """
        ).fetchone()

        alert_row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_alerts,
                SUM(detected) AS detected_alerts
            FROM drift_alerts
            """
        ).fetchone()

    total = row["total_predictions"] or 0
    predicted_churn = row["predicted_churn"] or 0

    return {
        "total_predictions": int(total),
        "predicted_churn": int(predicted_churn),
        "predicted_no_churn": int(total - predicted_churn),
        "predicted_churn_rate": (
            float(predicted_churn / total)
            if total
            else 0.0
        ),
        "average_churn_probability": (
            float(row["average_churn_probability"])
            if row["average_churn_probability"] is not None
            else 0.0
        ),
        "latest_prediction": row["latest_prediction"],
        "total_drift_alerts": int(
            alert_row["total_alerts"] or 0
        ),
        "active_drift_alerts": int(
            alert_row["detected_alerts"] or 0
        ),
    }


# ============================================================
# DRIFT ALERT STORAGE
# ============================================================

def log_drift_alert(
    feature_name,
    drift_score,
    threshold,
    detected,
    details="",
):
    """Store a data drift monitoring result."""

    with get_connection() as connection:

        cursor = connection.execute(
            """
            INSERT INTO drift_alerts (
                feature_name,
                drift_score,
                threshold,
                detected,
                details
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(feature_name),
                (
                    float(drift_score)
                    if drift_score is not None
                    else None
                ),
                (
                    float(threshold)
                    if threshold is not None
                    else None
                ),
                int(bool(detected)),
                str(details),
            ),
        )

        return cursor.lastrowid


def get_drift_alert_history(limit=100):
    """Retrieve the most recent drift checks and alerts."""

    limit = max(1, min(int(limit), 10000))

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT
                id,
                feature_name,
                drift_score,
                threshold,
                detected,
                details,
                created_at
            FROM drift_alerts
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return pd.DataFrame(
        [dict(row) for row in rows]
    )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

if __name__ == "__main__":

    initialize_database()

    print("ChurnGuard monitoring database initialized.")
    print(f"Database location: {DATABASE_PATH}")

    print("\nInitial monitoring statistics:")

    for key, value in get_monitoring_stats().items():
        print(f"{key}: {value}")