# G193_MP029 – Warehouse Picking Time Prediction

## Project Information

- **Group ID:** G193
- **Problem ID:** MP029
- **Project Title:** Warehouse Picking Time (Lucknow)
- **Faculty:** Dr. Saurabh Shrivastava
- **Batch:** 28

## Group Members & Roles

| Name | Enrollment Number | Role | Key Responsibilities |
|---|---|---|---|
| Sambhavi Verma | S24CSEU0620 | Data Lead | DVC setup, raw data auditing, cleaning pipeline |
| Rohit Mukhopadhyay | S24CSEU0375 | Model Development & MLOps Lead | Model training (V1/V2), MLflow tracking & registry, time-based comparison, holdout predictions, drift monitoring |
| Sarthak Sharma | S24CSEU0473 | Service & Deployment Lead | FastAPI prediction service, Docker, CI/CD, Kubernetes manifests |

## Project Description

This project focuses on predicting warehouse order picking time in seconds using machine learning regression for the **Lucknow Fulfilment Centre**.

The goal is to help the warehouse identify orders that may take longer to pick and improve operational planning so that late orders do not miss the evening dispatch truck.

## Technologies Used

- Python (pandas, numpy, scikit-learn)
- DVC (Data Version Control)
- MLflow (Experiment tracking and Model Registry)
- FastAPI & Uvicorn (REST API prediction service)
- PyTest (Automated testing suite)
- Docker (Containerization)
- GitHub Actions (CI pipeline)
- Kubernetes (Production deployment manifests)

## Model Development & MLOps Pipeline Summary

### 1. Model Training & MLflow Tracking (Lead: Rohit Mukhopadhyay)
- **Model V1 (Random Forest Regressor):** Trained on clean `train.csv`. Tracked across multiple hyperparameter runs with data SHA-256 fingerprinting. Initial validation MAE: **23.77s**.
- **Model V2 (Gradient Boosting Regressor):** Retrained on clean `train.csv` + first 70% of chronological `later.csv`. Initial validation MAE: **18.05s**.

### 2. Time-Based Validation & Promotion
- Evaluated on the unseen final 30% of `later.csv`:
  - **Model V1 MAE:** **89.02 seconds**
  - **Model V2 MAE:** **33.64 seconds**
  - **Error Reduction:** **62.21%** (satisfies client rule of >= 5% reduction).
  - **Decision:** Model V2 promoted to **`@champion`** in the MLflow Model Registry.

### 3. Drift & Subgroup Performance Monitoring
- **10-bin Population Stability Index (PSI):**
  - `shelf_height_m`: **3.5330** (Severe drift)
  - `pick_seconds`: **1.9516** (Severe drift)
  - `shift_hour`: **0.2764** (Severe drift)
- **Subgroup MAE across Shifts:**
  - Evening Shift: **146.90s** (Worst-performing group)
  - Morning Shift: **53.91s**
  - Night Shift: **51.77s**

### 4. Holdout Predictions
- Champion model used to generate predictions for all 250 records of `holdout_features.csv` into `predictions/holdout_predictions.csv` with zero missing values and zero negative values.

## Project Structure

```text
G193_MP029/
├── data/
│   └── raw/
├── findings.yaml
├── GROUP.md
├── PROBLEM.md
├── STUDENT_HANDBOOK.md
└── README.md
```

## Current Progress

- Git repository initialized.
- DVC initialized.
- Training and later datasets tracked using DVC.
- GitHub repository connected.
- Initial project setup pushed to GitHub.