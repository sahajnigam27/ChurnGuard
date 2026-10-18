"""
ChurnGuard
Customer Churn Prediction & ML Model Monitoring Platform

Module: Model Training and Evaluation

Models:
    1. Logistic Regression
    2. Random Forest

Outputs:
    models/best_model.joblib
    models/model_metadata.json
    reports/model_evaluation.json
    reports/confusion_matrix.png
    reports/feature_importance.png
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load and validate the dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {TARGET}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    # Remove accidental whitespace from column names.
    df.columns = df.columns.str.strip()

    # Remove duplicate records if any exist.
    duplicates = int(df.duplicated().sum())

    if duplicates:
        print(f"Removing {duplicates} duplicate rows.")
        df = df.drop_duplicates().copy()

    # Convert TotalCharges to numeric when present.
    # Blank strings become NaN and are imputed later.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    # Validate target labels.
    df = df[df[TARGET].isin(["Yes", "No"])].copy()

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "Expected both 'Yes' and 'No' churn labels."
        )

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor(X):
    """Create preprocessing pipelines for numerical and categorical data."""

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_columns, categorical_columns


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_models():
    """Return the candidate models."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Calculate metrics from held-out test data."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "predictions": predictions,
        "probabilities": probabilities,
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(model):
    """Save feature importance for the fitted winning model."""

    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_

    elif hasattr(classifier, "coef_"):
        importance = np.abs(classifier.coef_[0])

    else:
        print("This model does not expose feature importance.")
        return

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    ).sort_values(
        "importance",
        ascending=False,
    ).head(20)

    importance_df = importance_df.sort_values(
        "importance",
        ascending=True,
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        importance_df["feature"],
        importance_df["importance"],
    )

    plt.title("Top 20 Model Features")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    output_path = REPORT_DIR / "feature_importance.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Feature importance saved to: {output_path}")


# ============================================================
# TRAINING WORKFLOW
# ============================================================

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # Customer ID identifies a record but is not a useful
    # generalizable predictive feature.
    drop_columns = [TARGET]

    if ID_COLUMN in df.columns:
        drop_columns.append(ID_COLUMN)

    X = df.drop(columns=drop_columns)
    y = df[TARGET].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining split:", X_train.shape)
    print("Testing split:", X_test.shape)
    print("\nTraining churn distribution:")
    print(y_train.value_counts(normalize=True).round(3))

    results = {}
    fitted_models = {}

    for name, classifier in build_models().items():
        print("\n" + "=" * 60)
        print(f"TRAINING: {name}")
        print("=" * 60)

        preprocessor, _, _ = build_preprocessor(X_train)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test,
        )

        results[name] = {
            key: value
            for key, value in metrics.items()
            if key not in {"predictions", "probabilities"}
        }

        fitted_models[name] = pipeline

        print("\nEvaluation metrics:")

        for metric, value in results[name].items():
            print(f"{metric}: {value:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                metrics["predictions"],
                target_names=["Stayed", "Churned"],
                zero_division=0,
            )
        )

    # Select the candidate with the highest F1-score.
    # Recall and ROC-AUC are also reported for comparison.
    best_name = max(
        results,
        key=lambda name: results[name]["f1_score"],
    )

    best_model = fitted_models[best_name]

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)
    print(f"Selected model: {best_name}")
    print(
        "Selection criterion: highest test-set F1-score "
        "among the evaluated candidates."
    )

    # Save the selected pipeline.
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)

    # Save confusion matrix.
    best_predictions = best_model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        best_predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stayed", "Churned"],
    )

    display.plot(values_format="d")

    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()

    cm_path = REPORT_DIR / "confusion_matrix.png"

    plt.savefig(
        cm_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # Save feature importance.
    save_feature_importance(best_model)

    # Save model evaluation report.
    evaluation = {
        "project": "ChurnGuard",
        "dataset": DATA_PATH.name,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "selection_metric": "f1_score",
        "selected_model": best_name,
        "class_mapping": {
            "Stayed": 0,
            "Churned": 1,
        },
        "dataset_rows_after_cleaning": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": {
            "stayed": int((y == 0).sum()),
            "churned": int((y == 1).sum()),
        },
        "model_results": results,
        "confusion_matrix": cm.tolist(),
    }

    evaluation_path = REPORT_DIR / "model_evaluation.json"

    with evaluation_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation,
            file,
            indent=4,
        )

    metadata = {
        "project": "ChurnGuard",
        "model_name": best_name,
        "target_column": TARGET,
        "positive_class": 1,
        "features": X.columns.tolist(),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "created_at_utc": evaluation["created_at_utc"],
    }

    metadata_path = MODEL_DIR / "model_metadata.json"

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Selected model: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Evaluation report: {evaluation_path}")
    print(f"Confusion matrix: {cm_path}")

    print("\nCandidate comparison:")

    comparison = pd.DataFrame(results).T
    print(comparison.round(4).to_string())


if __name__ == "__main__":
    main()