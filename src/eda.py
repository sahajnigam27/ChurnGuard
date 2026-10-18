"""
ChurnGuard
Customer Churn Prediction & ML Model Monitoring Platform

Module: Exploratory Data Analysis (EDA)
Purpose:
    1. Inspect dataset structure and data types.
    2. Identify missing values and duplicate records.
    3. Analyze customer churn distribution.
    4. Examine numerical features.
    5. Analyze churn patterns across customer segments.
    6. Generate visualizations and save an analysis report.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "eda"
PLOT_DIR = REPORT_DIR / "plots"

TARGET_COLUMN = "Churn"

sns.set_theme(style="whitegrid", context="notebook")


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():
    """Load the customer churn dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Ensure the CSV file exists in data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    return df


# ============================================================
# DATA QUALITY ANALYSIS
# ============================================================

def analyze_data_quality(df):
    """Inspect missing values, duplicates, and data types."""

    print("\n" + "=" * 65)
    print("1. DATASET OVERVIEW")
    print("=" * 65)

    print(f"Number of records: {df.shape[0]:,}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\n" + "=" * 65)
    print("2. COLUMN DATA TYPES")
    print("=" * 65)

    print(df.dtypes.to_string())

    print("\n" + "=" * 65)
    print("3. MISSING VALUES")
    print("=" * 65)

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No standard missing values detected.")
    else:
        print(missing.to_string())

    print("\n" + "=" * 65)
    print("4. DUPLICATE RECORDS")
    print("=" * 65)

    print(f"Duplicate rows: {df.duplicated().sum()}")

    if "customerID" in df.columns:
        print(
            "Duplicate customer IDs:",
            df["customerID"].duplicated().sum()
        )

    print("\n" + "=" * 65)
    print("5. NUMERIC FEATURE SUMMARY")
    print("=" * 65)

    print(df.describe(include="number").round(2).to_string())

    print("\n" + "=" * 65)
    print("6. CATEGORICAL FEATURE SUMMARY")
    print("=" * 65)

    categorical = df.select_dtypes(
        include=["object", "category"]
    )

    print(categorical.describe().to_string())


# ============================================================
# CHURN ANALYSIS
# ============================================================

def analyze_churn(df):
    """Analyze overall churn and customer segment churn rates."""

    print("\n" + "=" * 65)
    print("7. OVERALL CHURN DISTRIBUTION")
    print("=" * 65)

    counts = df[TARGET_COLUMN].value_counts(dropna=False)
    percentages = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
    )

    churn_summary = pd.DataFrame({
        "Customer Count": counts,
        "Percentage": percentages.round(2)
    })

    print(churn_summary.to_string())

    print("\n" + "=" * 65)
    print("8. CHURN BY CONTRACT TYPE")
    print("=" * 65)

    if "Contract" in df.columns:
        contract_analysis = pd.crosstab(
            df["Contract"],
            df[TARGET_COLUMN],
            normalize="index"
        ).mul(100)

        print(contract_analysis.round(2).to_string())

    print("\n" + "=" * 65)
    print("9. CHURN BY INTERNET SERVICE")
    print("=" * 65)

    if "InternetService" in df.columns:
        internet_analysis = pd.crosstab(
            df["InternetService"],
            df[TARGET_COLUMN],
            normalize="index"
        ).mul(100)

        print(internet_analysis.round(2).to_string())

    print("\n" + "=" * 65)
    print("10. CHURN BY CUSTOMER TENURE")
    print("=" * 65)

    if "tenure" in df.columns:
        tenure_groups = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 60, 72],
            labels=[
                "0-12 months",
                "13-24 months",
                "25-48 months",
                "49-60 months",
                "61-72 months"
            ]
        )

        tenure_analysis = pd.crosstab(
            tenure_groups,
            df[TARGET_COLUMN],
            normalize="index"
        ).mul(100)

        print(tenure_analysis.round(2).to_string())

    print("\n" + "=" * 65)
    print("11. CHURN BY PAYMENT METHOD")
    print("=" * 65)

    if "PaymentMethod" in df.columns:
        payment_analysis = pd.crosstab(
            df["PaymentMethod"],
            df[TARGET_COLUMN],
            normalize="index"
        ).mul(100)

        print(payment_analysis.round(2).to_string())


# ============================================================
# VISUALIZATIONS
# ============================================================

def save_plots(df):
    """Create and save EDA visualizations."""

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Plot 1: Overall churn distribution
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=TARGET_COLUMN,
        hue=TARGET_COLUMN,
        order=["No", "Yes"],
        palette="Set2",
        legend=False
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Customer Churn")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "churn_distribution.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 2: Churn rate by contract type
    # --------------------------------------------------------

    if "Contract" in df.columns:
        contract_rates = (
            df.groupby("Contract", observed=True)[TARGET_COLUMN]
            .apply(lambda values: (values == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(9, 5))

        sns.barplot(
            x=contract_rates.index,
            y=contract_rates.values,
            hue=contract_rates.index,
            palette="viridis",
            legend=False
        )

        plt.title("Customer Churn Rate by Contract Type")
        plt.xlabel("Contract Type")
        plt.ylabel("Churn Rate (%)")
        plt.tight_layout()

        plt.savefig(
            PLOT_DIR / "churn_by_contract.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

    # --------------------------------------------------------
    # Plot 3: Churn rate by internet service
    # --------------------------------------------------------

    if "InternetService" in df.columns:
        internet_rates = (
            df.groupby("InternetService", observed=True)[TARGET_COLUMN]
            .apply(lambda values: (values == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(9, 5))

        sns.barplot(
            x=internet_rates.index,
            y=internet_rates.values,
            hue=internet_rates.index,
            palette="magma",
            legend=False
        )

        plt.title("Customer Churn Rate by Internet Service")
        plt.xlabel("Internet Service")
        plt.ylabel("Churn Rate (%)")
        plt.tight_layout()

        plt.savefig(
            PLOT_DIR / "churn_by_internet_service.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

    # --------------------------------------------------------
    # Plot 4: Tenure distribution by churn status
    # --------------------------------------------------------

    if "tenure" in df.columns:
        plt.figure(figsize=(9, 5))

        sns.histplot(
            data=df,
            x="tenure",
            hue=TARGET_COLUMN,
            bins=30,
            multiple="stack",
            element="bars"
        )

        plt.title("Customer Tenure Distribution by Churn Status")
        plt.xlabel("Tenure (Months)")
        plt.ylabel("Number of Customers")
        plt.tight_layout()

        plt.savefig(
            PLOT_DIR / "tenure_distribution.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

    # --------------------------------------------------------
    # Plot 5: Monthly charges by churn status
    # --------------------------------------------------------

    if "MonthlyCharges" in df.columns:
        plt.figure(figsize=(9, 5))

        sns.boxplot(
            data=df,
            x=TARGET_COLUMN,
            y="MonthlyCharges",
            hue=TARGET_COLUMN,
            order=["No", "Yes"],
            palette="Set2",
            legend=False
        )

        plt.title("Monthly Charges by Customer Churn Status")
        plt.xlabel("Customer Churn")
        plt.ylabel("Monthly Charges")
        plt.tight_layout()

        plt.savefig(
            PLOT_DIR / "monthly_charges_by_churn.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

    print(f"\nVisualizations saved to:\n{PLOT_DIR}")


# ============================================================
# REPORT GENERATION
# ============================================================

def save_report(df):
    """Save a concise text report containing actual dataset results."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "eda_summary.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("CHURNGUARD - EDA SUMMARY\n")
        report.write("=" * 60 + "\n\n")

        report.write(f"Dataset rows: {df.shape[0]}\n")
        report.write(f"Dataset columns: {df.shape[1]}\n\n")

        report.write("DATA TYPES\n")
        report.write(df.dtypes.to_string())
        report.write("\n\n")

        report.write("MISSING VALUES\n")
        report.write(df.isnull().sum().to_string())
        report.write("\n\n")

        report.write("DUPLICATE ROWS\n")
        report.write(str(df.duplicated().sum()))
        report.write("\n\n")

        report.write("CHURN DISTRIBUTION\n")
        report.write(
            df[TARGET_COLUMN]
            .value_counts(dropna=False)
            .to_string()
        )
        report.write("\n\n")

        report.write("CHURN PERCENTAGES\n")
        report.write(
            (
                df[TARGET_COLUMN]
                .value_counts(normalize=True, dropna=False)
                .mul(100)
                .round(2)
                .to_string()
            )
        )
        report.write("\n\n")

        if "Contract" in df.columns:
            report.write("CHURN BY CONTRACT TYPE (%)\n")

            report.write(
                (
                    pd.crosstab(
                        df["Contract"],
                        df[TARGET_COLUMN],
                        normalize="index"
                    )
                    .mul(100)
                    .round(2)
                    .to_string()
                )
            )

            report.write("\n\n")

        if "InternetService" in df.columns:
            report.write("CHURN BY INTERNET SERVICE (%)\n")

            report.write(
                (
                    pd.crosstab(
                        df["InternetService"],
                        df[TARGET_COLUMN],
                        normalize="index"
                    )
                    .mul(100)
                    .round(2)
                    .to_string()
                )
            )

            report.write("\n\n")

    print(f"EDA report saved to:\n{report_path}")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Run the complete EDA workflow."""

    print("\nLoading ChurnGuard dataset...")

    df = load_dataset()

    analyze_data_quality(df)
    analyze_churn(df)
    save_plots(df)
    save_report(df)

    print("\n" + "=" * 65)
    print("EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print(f"\nReports directory: {REPORT_DIR}")
    print(f"Plots directory: {PLOT_DIR}")


if __name__ == "__main__":
    main()