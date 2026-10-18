import pytest
from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1026.0,
}


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["project"] == "ChurnGuard"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200
    assert "model_name" in response.json()


def test_prediction_endpoint():
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_churn"] in [0, 1]
    assert result["churn_label"] in ["Churn", "No Churn"]
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert "prediction_id" in result


def test_invalid_customer_is_rejected():
    invalid_customer = VALID_CUSTOMER.copy()
    invalid_customer["tenure"] = 500

    response = client.post("/predict", json=invalid_customer)

    assert response.status_code == 422


def test_batch_prediction_endpoint():
    response = client.post(
        "/predict/batch",
        json=[VALID_CUSTOMER, VALID_CUSTOMER],
    )

    assert response.status_code == 200

    result = response.json()

    assert result["total_customers"] == 2
    assert len(result["results"]) == 2
    assert all(
        item["predicted_churn"] in [0, 1]
        for item in result["results"]
    )


def test_empty_batch_is_rejected():
    response = client.post("/predict/batch", json=[])

    assert response.status_code == 400