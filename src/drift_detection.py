from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

# Allow imports when running this file directly.
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring import (
    initialize_database,
    get_prediction_history,
    log_drift_alert,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

# PSI thresholds are practical heuristics, not universal rules.
PSI_WARNING_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25

# Avoid drawing conclusions from extremely small samples.
MIN_REFERENCE_ROWS = 100
MIN_CURRENT_ROWS = 30

# Features to monitor.
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "PaperlessBilling",
    "TechSupport",
]


# ============================================================
# LOAD REFERENCE DATA
# ============================================================

def load_reference_data():
    """Load the original customer dataset used for model development."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {DATASET_PATH}"
        )

    data = pd.read_csv(DATASET_PATH)

    # Remove identifier and target columns from the drift comparison.
    # We compare customer input features, not the churn label.
    columns_to_remove = [
        column
        for column in ["customerID", "Churn"]
        if column in data.columns
    ]

    data = data.drop(columns=columns_to_remove)

    # Convert TotalCharges to numeric where necessary.
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce",
        )

    return data


# ============================================================
# PSI CALCULATION
# ============================================================

def calculate_psi(reference, current, feature_name):
    """
    Calculate PSI between reference and current observations.

    Numeric features use reference-data quantile bins.
    Categorical features use categories observed in either sample.
    A small smoothing value prevents division by zero.
    """

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if reference.empty or current.empty:
        return {
            "feature": feature_name,
            "psi": None,
            "status": "INSUFFICIENT_DATA",
            "reference_count": int(len(reference)),
            "current_count": int(len(current)),
            "message": "No usable observations for comparison.",
        }

    epsilon = 1e-6

    numeric = (
        pd.api.types.is_numeric_dtype(reference)
        and pd.api.types.is_numeric_dtype(current)
    )

    if numeric:
        # Constant reference features cannot form useful quantile bins.
        if reference.nunique() < 2:
            return {
                "feature": feature_name,
                "psi": None,
                "status": "NOT_EVALUATED",
                "reference_count": int(len(reference)),
                "current_count": int(len(current)),
                "message": "Reference feature has fewer than two unique values.",
            }

        quantiles = np.linspace(0, 1, 11)
        edges = np.unique(reference.quantile(quantiles).to_numpy())

        if len(edges) < 2:
            return {
                "feature": feature_name,
                "psi": None,
                "status": "NOT_EVALUATED",
                "reference_count": int(len(reference)),
                "current_count": int(len(current)),
                "message": "Could not create useful reference bins.",
            }

        # Extend outer edges so new values outside the reference range
        # are counted in the first or last bin.
        edges[0] = -np.inf
        edges[-1] = np.inf

        reference_bins = pd.cut(
            reference,
            bins=edges,
            include_lowest=True,
        )

        current_bins = pd.cut(
            current,
            bins=edges,
            include_lowest=True,
        )

        categories = reference_bins.cat.categories

        reference_counts = (
            reference_bins.value_counts(sort=False)
            .reindex(categories, fill_value=0)
            .to_numpy(dtype=float)
        )

        current_counts = (
            current_bins.value_counts(sort=False)
            .reindex(categories, fill_value=0)
            .to_numpy(dtype=float)
        )

    else:
        # Treat categories as strings so values are compared consistently.
        reference = reference.astype(str)
        current = current.astype(str)

        categories = sorted(set(reference.unique()) | set(current.unique()))

        reference_counts = (
            reference.value_counts()
            .reindex(categories, fill_value=0)
            .to_numpy(dtype=float)
        )

        current_counts = (
            current.value_counts()
            .reindex(categories, fill_value=0)
            .to_numpy(dtype=float)
        )

    reference_proportions = (
        reference_counts + epsilon
    ) / (
        reference_counts.sum() + epsilon * len(reference_counts)
    )

    current_proportions = (
        current_counts + epsilon
    ) / (
        current_counts.sum() + epsilon * len(current_counts)
    )

    psi_value = float(
        np.sum(
            (current_proportions - reference_proportions)
            * np.log(current_proportions / reference_proportions)
        )
    )

    if psi_value < PSI_WARNING_THRESHOLD:
        status = "STABLE"
    elif psi_value < PSI_CRITICAL_THRESHOLD:
        status = "WARNING"
    else:
        status = "CRITICAL"

    return {
        "feature": feature_name,
        "psi": round(psi_value, 6),
        "status": status,
        "reference_count": int(len(reference)),
        "current_count": int(len(current)),
        "message": (
            "No PSI threshold exceeded."
            if status == "STABLE"
            else f"PSI exceeded the {status.lower()} threshold."
        ),
    }


# ============================================================
# DRIFT DETECTION
# ============================================================

def detect_drift():
    """
    Compare logged prediction inputs with the original dataset.

    Results are stored in SQLite when a feature is evaluated.
    Insufficient samples are reported without creating drift alerts.
    """

    initialize_database()

    reference = load_reference_data()
    history = get_prediction_history(limit=10000)

    if len(reference) < MIN_REFERENCE_ROWS:
        raise ValueError(
            f"Need at least {MIN_REFERENCE_ROWS} reference rows; "
            f"found {len(reference)}."
        )

    if history is None or history.empty:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": "No predictions have been logged yet.",
            "reference_rows": int(len(reference)),
            "current_rows": 0,
            "features_checked": 0,
            "results": [],
        }

    # The monitoring history stores incoming feature values in JSON.
    if "customer_data" not in history.columns:
        return {
            "status": "ERROR",
            "message": "Prediction history has no customer_data column.",
            "reference_rows": int(len(reference)),
            "current_rows": int(len(history)),
            "features_checked": 0,
            "results": [],
        }

    parsed_rows = []

    for value in history["customer_data"]:
        try:
            parsed = json.loads(value) if isinstance(value, str) else value

            if isinstance(parsed, dict):
                parsed_rows.append(parsed)

        except (json.JSONDecodeError, TypeError):
            continue

    current = pd.DataFrame(parsed_rows)

    if len(current) < MIN_CURRENT_ROWS:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": (
                f"At least {MIN_CURRENT_ROWS} logged predictions are "
                f"required. Currently available: {len(current)}."
            ),
            "reference_rows": int(len(reference)),
            "current_rows": int(len(current)),
            "features_checked": 0,
            "results": [],
        }

    results = []

    for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            results.append({
                "feature": feature,
                "psi": None,
                "status": "NOT_EVALUATED",
                "reference_count": 0,
                "current_count": 0,
                "message": "Feature missing from reference or current data.",
            })
            continue

        reference_values = reference[feature]
        current_values = current[feature]

        if feature in NUMERIC_FEATURES:
            reference_values = pd.to_numeric(
                reference_values,
                errors="coerce",
            )
            current_values = pd.to_numeric(
                current_values,
                errors="coerce",
            )

        result = calculate_psi(
            reference_values,
            current_values,
            feature,
        )

        results.append(result)

        if result["psi"] is not None:
            detected = result["status"] in {"WARNING", "CRITICAL"}

            # Record evaluated results in the existing drift-alert table.
            log_drift_alert(
                feature_name=feature,
                drift_score=result["psi"],
                threshold=PSI_WARNING_THRESHOLD,
                detected=detected,
                details=(
                    f"Status: {result['status']}; "
                    f"reference rows: {result['reference_count']}; "
                    f"current rows: {result['current_count']}; "
                    f"message: {result['message']}"
                ),
            )

    evaluated = [
        result for result in results
        if result["psi"] is not None
    ]

    if any(result["status"] == "CRITICAL" for result in evaluated):
        overall_status = "CRITICAL"
    elif any(result["status"] == "WARNING" for result in evaluated):
        overall_status = "WARNING"
    elif evaluated:
        overall_status = "STABLE"
    else:
        overall_status = "NOT_EVALUATED"

    return {
        "status": overall_status,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features_checked": len(evaluated),
        "results": results,
        "thresholds": {
            "warning": PSI_WARNING_THRESHOLD,
            "critical": PSI_CRITICAL_THRESHOLD,
        },
    }


# ============================================================
# RUN FROM TERMINAL
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CHURNGUARD - DATA DRIFT DETECTION")
    print("=" * 60)

    try:
        report = detect_drift()
        print(json.dumps(report, indent=2, default=str))

    except Exception as error:
        print(f"Drift detection failed: {error}")
        raise