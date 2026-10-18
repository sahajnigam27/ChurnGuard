# 🛡️ ChurnGuard
### Customer Churn Prediction & ML Model Monitoring Platform

ChurnGuard is an end-to-end machine learning project that predicts customer churn and provides a monitoring interface for prediction history, model inputs, and data drift.

The project combines a trained machine learning pipeline, a REST API, a local monitoring database, and an interactive dashboard.

## Key Features

- **Customer Churn Prediction:** Predicts whether a customer is likely to leave a service.
- **Machine Learning Pipeline:** Trains and evaluates Logistic Regression and Random Forest models.
- **REST API:** Serves predictions using FastAPI.
- **Batch Prediction:** Supports predictions for multiple customers.
- **Prediction Logging:** Stores prediction records in SQLite.
- **Interactive Dashboard:** Displays prediction statistics and monitoring history using Streamlit.
- **Data Drift Monitoring:** Uses Population Stability Index (PSI) to compare reference and incoming feature distributions.
- **Automated Testing:** Tests API endpoints, PSI calculations, and database monitoring.

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Database | SQLite |
| Visualization | Matplotlib, Seaborn |
| Model Persistence | joblib |
| Testing | pytest |

## Project Architecture

```text
Customer Dataset
       |
       v
Exploratory Data Analysis
       |
       v
Data Preprocessing and Model Training
       |
       v
Saved ML Pipeline
       |
       v
FastAPI Prediction Service
       |
       +--------------------+
       |                    |
       v                    v
Prediction Logging     Prediction Response
       |
       v
SQLite Monitoring Database
       |
       +--------------------+
       |                    |
       v                    v
Streamlit Dashboard   PSI Data Drift Checks