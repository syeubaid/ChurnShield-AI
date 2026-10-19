"""
Customer Churn Prediction - Model Trainer
Trains a Random Forest classifier and saves model artifacts.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "customer_churn_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CATEGORICAL_COLS = [
    "Gender", "ContractType", "HasInternet", "InternetType",
    "OnlineSecurity", "TechSupport", "StreamingTV", "StreamingMovies",
    "PaymentMethod", "PaperlessBilling", "PromotionOffered"
]

NUMERIC_COLS = [
    "Age", "TenureMonths", "MonthlyCharges", "TotalCharges", "NumProducts",
    "NumSupportCalls", "AvgCallDurationMin", "SatisfactionScore",
    "LastInteractionDays", "ReferralsGiven", "LatePayments",
    "DeviceCount", "DataUsageGB", "PlanUpgrades", "PlanDowngrades"
]

TARGET_COL = "Churn"
ID_COL = "CustomerID"

FEATURE_DESCRIPTIONS = {
    "Age": "Customer Age",
    "TenureMonths": "Months with Company",
    "MonthlyCharges": "Monthly Bill ($)",
    "TotalCharges": "Total Billed ($)",
    "NumProducts": "Products Subscribed",
    "NumSupportCalls": "Support Calls Made",
    "AvgCallDurationMin": "Avg Call Duration (min)",
    "SatisfactionScore": "Satisfaction Score (1-5)",
    "LastInteractionDays": "Days Since Last Contact",
    "ReferralsGiven": "Referrals Given",
    "LatePayments": "Late Payments",
    "DeviceCount": "Devices Connected",
    "DataUsageGB": "Monthly Data Usage (GB)",
    "PlanUpgrades": "Plan Upgrades",
    "PlanDowngrades": "Plan Downgrades",
    "Gender": "Gender",
    "ContractType": "Contract Type",
    "HasInternet": "Has Internet Service",
    "InternetType": "Internet Type",
    "OnlineSecurity": "Online Security",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "PaymentMethod": "Payment Method",
    "PaperlessBilling": "Paperless Billing",
    "PromotionOffered": "Promotion Offered"
}

def preprocess_data(df, encoders=None, fit=True):
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            df[col] = "Unknown"
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le:
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
            else:
                df[col] = 0
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    available = [c for c in feature_cols if c in df.columns]
    return df[available], encoders


def train_model():
    print("=" * 60)
    print("  Customer Churn Prediction - Model Training")
    print("=" * 60)

    df = pd.read_csv(DATASET_PATH)
    print(f"\n Loaded dataset: {len(df)} rows")

    X, encoders = preprocess_data(df, fit=True)
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    num_available = [c for c in NUMERIC_COLS if c in X_train.columns]

    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n Accuracy:  {acc*100:.1f}%")
    print(f" Precision: {prec*100:.1f}%")
    print(f" Recall:    {rec*100:.1f}%")
    print(f" F1-Score:  {f1*100:.1f}%")
    print(f" ROC-AUC:   {auc*100:.1f}%")

    feature_names = X_train.columns.tolist()
    importances = rf_model.feature_importances_
    feature_importance = dict(zip(feature_names, importances.tolist()))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='roc_auc')
    cm = confusion_matrix(y_test, y_pred).tolist()

    model_artifacts = {
        "model": rf_model,
        "encoders": encoders,
        "scaler": scaler,
        "feature_cols": feature_names,
        "numeric_cols": num_available,
    }
    joblib.dump(model_artifacts, os.path.join(MODELS_DIR, "churn_model.pkl"))

    metadata = {
        "model_type": "Random Forest Classifier",
        "n_estimators": 200,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": len(feature_names),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "cv_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_auc_std": round(float(cv_scores.std()), 4)
        },
        "confusion_matrix": cm,
        "feature_importance": sorted_importance,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "class_distribution": {
            "churned": int(y.sum()),
            "retained": int((y == 0).sum()),
            "churn_rate": round(float(y.mean()), 4)
        }
    }

    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n Model saved: {MODELS_DIR}/churn_model.pkl")
    print(f" Metadata saved: {MODELS_DIR}/model_metadata.json")
    return model_artifacts, metadata


if __name__ == "__main__":
    train_model()
