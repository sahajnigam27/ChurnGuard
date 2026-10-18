import numpy as np
import pandas as pd

from src.drift_detection import calculate_psi


def test_identical_numeric_distributions_are_stable():
    reference = pd.Series([10, 20, 30, 40, 50, 60] * 20)
    current = reference.copy()

    result = calculate_psi(reference, current, "tenure")

    assert result["psi"] is not None
    assert result["psi"] < 0.10
    assert result["status"] == "STABLE"


def test_shifted_numeric_distribution_increases_psi():
    reference = pd.Series(np.linspace(0, 100, 1000))
    current = pd.Series(np.linspace(200, 300, 1000))

    result = calculate_psi(reference, current, "MonthlyCharges")

    assert result["psi"] is not None
    assert result["psi"] >= 0.25
    assert result["status"] == "CRITICAL"


def test_identical_categorical_distributions_are_stable():
    reference = pd.Series(["Month-to-month", "One year", "Two year"] * 100)
    current = reference.copy()

    result = calculate_psi(reference, current, "Contract")

    assert result["psi"] is not None
    assert result["psi"] < 0.10
    assert result["status"] == "STABLE"


def test_shifted_categorical_distribution_is_detected():
    reference = pd.Series(["Month-to-month"] * 900 + ["Two year"] * 100)
    current = pd.Series(["Two year"] * 900 + ["Month-to-month"] * 100)

    result = calculate_psi(reference, current, "Contract")

    assert result["psi"] is not None
    assert result["psi"] >= 0.25
    assert result["status"] == "CRITICAL"


def test_empty_data_returns_insufficient_data():
    reference = pd.Series([], dtype=float)
    current = pd.Series([10, 20, 30], dtype=float)

    result = calculate_psi(reference, current, "tenure")

    assert result["psi"] is None
    assert result["status"] == "INSUFFICIENT_DATA"