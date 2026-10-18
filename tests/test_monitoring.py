import json
import sqlite3

from src.monitoring import (
    initialize_database,
    log_prediction,
    get_prediction_history,
    get_monitoring_stats,
)


def test_prediction_is_logged_and_retrievable():
    # Initialize the actual monitoring database.
    initialize_database()

    customer_data = {
        "gender": "Female",
        "tenure": 12,
        "Contract": "Month-to-month",
    }

    prediction_id = log_prediction(
        customer_data=customer_data,
        predicted_churn=1,
        churn_probability=0.88,
        model_name="MonitoringTestModel",
    )

    assert prediction_id is not None

    history = get_prediction_history(limit=100)

    assert not history.empty

    matching_rows = history[
        history["id"] == prediction_id
    ]

    assert len(matching_rows) == 1

    row = matching_rows.iloc[0]

    assert row["predicted_churn"] == 1
    assert abs(row["churn_probability"] - 0.88) < 1e-6
    assert row["model_name"] == "MonitoringTestModel"

    saved_customer = json.loads(row["customer_data"])

    assert saved_customer["tenure"] == 12
    assert saved_customer["Contract"] == "Month-to-month"


def test_monitoring_statistics_have_expected_fields():
    initialize_database()

    stats = get_monitoring_stats()

    required_fields = [
        "total_predictions",
        "predicted_churn",
        "predicted_no_churn",
        "predicted_churn_rate",
        "average_churn_probability",
        "latest_prediction",
        "total_drift_alerts",
        "active_drift_alerts",
    ]

    for field in required_fields:
        assert field in stats

    assert stats["total_predictions"] >= 0
    assert stats["predicted_churn"] >= 0
    assert stats["predicted_no_churn"] >= 0