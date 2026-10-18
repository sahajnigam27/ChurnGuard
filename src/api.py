from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.monitoring import initialize_database, log_prediction


# ============================================================
# PATHS AND MODEL LOADING
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

import json

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. "
        "Run python src/train.py first."
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Model metadata not found at {METADATA_PATH}. "
        "Run python src/train.py first."
    )

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

MODEL_NAME = metadata.get("best_model", metadata.get("model_name", "ChurnGuard model"))
EXPECTED_FEATURES = metadata.get("features", [])


# ============================================================
# APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="ChurnGuard API",
    description=(
        "Customer churn prediction API with SQLite-based "
        "prediction monitoring."
    ),
    version="1.0.0",
)

# Create the monitoring tables if they do not exist.
initialize_database()


# ============================================================
# INPUT SCHEMA
# ============================================================

class CustomerInput(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120)

    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]

    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]

    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    MonthlyCharges: float = Field(ge=0, le=10000)
    TotalCharges: float = Field(ge=0, le=1000000)


# ============================================================
# PREDICTION HELPERS
# ============================================================

def predict_customer(customer: CustomerInput) -> dict:
    """
    Predict churn for one customer and record the prediction
    in the monitoring database.
    """

    customer_data = customer.model_dump()

    input_df = pd.DataFrame([customer_data])

    if EXPECTED_FEATURES:
        missing_features = [
            feature
            for feature in EXPECTED_FEATURES
            if feature not in input_df.columns
        ]

        if missing_features:
            raise HTTPException(
                status_code=500,
                detail=f"Missing model features: {missing_features}",
            )

        input_df = input_df[EXPECTED_FEATURES]

    try:
        predicted_class = int(model.predict(input_df)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 in classes:
                churn_probability = float(
                    probabilities[classes.index(1)]
                )
            else:
                churn_probability = 0.0
        else:
            raise HTTPException(
                status_code=500,
                detail="The loaded model does not support predict_proba.",
            )

        # Record the prediction in SQLite.
        prediction_id = log_prediction(
            customer_data=customer_data,
            predicted_churn=predicted_class,
            churn_probability=churn_probability,
            model_name=MODEL_NAME,
        )

        return {
            "prediction_id": prediction_id,
            "model_name": MODEL_NAME,
            "predicted_churn": predicted_class,
            "churn_label": (
                "Churn" if predicted_class == 1 else "No Churn"
            ),
            "churn_probability": round(churn_probability, 4),
            "message": (
                "Prediction recorded in the monitoring database."
            ),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        ) from error


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "project": "ChurnGuard",
        "description": "Customer Churn Prediction & ML Monitoring Platform",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "monitoring_database": "initialized",
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "features": EXPECTED_FEATURES,
        "metadata": metadata,
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    return predict_customer(customer)


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerInput]):
    if not customers:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one customer.",
        )

    if len(customers) > 500:
        raise HTTPException(
            status_code=400,
            detail="A maximum of 500 customers is allowed per batch.",
        )

    results = [
        predict_customer(customer)
        for customer in customers
    ]

    churn_count = sum(
        result["predicted_churn"] == 1
        for result in results
    )

    return {
        "total_customers": len(results),
        "predicted_churn": churn_count,
        "predicted_no_churn": len(results) - churn_count,
        "results": results,
    }