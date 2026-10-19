"""
Customer Churn Prediction System - Flask Backend
Handles ML inference, CSV uploads, outcome tracking, and data APIs.
"""

import os
import json
import uuid
import csv
import io
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ─── App Setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTCOMES_FILE = os.path.join(BASE_DIR, "models", "outcomes.json")
RESULTS_CACHE_FILE = os.path.join(BASE_DIR, "models", "last_results.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Feature Definitions ──────────────────────────────────────────────────────
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

# ─── Retention Recommendations ────────────────────────────────────────────────
RECOMMENDATIONS = {
    "SatisfactionScore": {
        "condition": lambda v: float(v) <= 2,
        "title": "Low Satisfaction Score",
        "action": "Schedule a personal call from customer success team. Offer service review and customized solution.",
        "priority": "critical"
    },
    "ContractType": {
        "condition": lambda v: str(v) in ["Month-to-Month", "0"],
        "title": "Month-to-Month Contract",
        "action": "Offer a discounted annual or biennial plan with added perks to lock in commitment.",
        "priority": "high"
    },
    "TenureMonths": {
        "condition": lambda v: float(v) < 12,
        "title": "New Customer (< 1 Year)",
        "action": "Enroll in a loyalty onboarding program with milestone rewards at 3, 6, and 12 months.",
        "priority": "high"
    },
    "NumSupportCalls": {
        "condition": lambda v: float(v) >= 5,
        "title": "High Support Call Volume",
        "action": "Assign a dedicated support rep. Conduct a root-cause service audit.",
        "priority": "high"
    },
    "LatePayments": {
        "condition": lambda v: float(v) >= 3,
        "title": "Multiple Late Payments",
        "action": "Offer flexible payment plans or payment date adjustment. Enable auto-pay incentive.",
        "priority": "medium"
    },
    "MonthlyCharges": {
        "condition": lambda v: float(v) > 90,
        "title": "High Monthly Charges",
        "action": "Review plan suitability. Offer a comparable plan at a reduced rate or add bundled benefits.",
        "priority": "medium"
    },
    "LastInteractionDays": {
        "condition": lambda v: float(v) > 180,
        "title": "Inactive for 6+ Months",
        "action": "Launch a re-engagement campaign with a personalized win-back offer.",
        "priority": "medium"
    },
    "ReferralsGiven": {
        "condition": lambda v: float(v) == 0,
        "title": "No Referrals Made",
        "action": "Introduce a referral rewards program to increase engagement and loyalty.",
        "priority": "low"
    },
    "OnlineSecurity": {
        "condition": lambda v: str(v) in ["No", "0"],
        "title": "No Online Security Service",
        "action": "Offer a free 3-month trial of the online security add-on.",
        "priority": "low"
    },
    "TechSupport": {
        "condition": lambda v: str(v) in ["No", "0"],
        "title": "No Tech Support Plan",
        "action": "Bundle tech support into the current plan at a promotional rate.",
        "priority": "low"
    }
}

# ─── Model Loading ─────────────────────────────────────────────────────────────
MODEL_ARTIFACTS = None
MODEL_METADATA = None

def load_model():
    global MODEL_ARTIFACTS, MODEL_METADATA
    model_path = os.path.join(MODELS_DIR, "churn_model.pkl")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    if not os.path.exists(model_path):
        print("  Model not found. Training now...")
        from model_trainer import train_model
        MODEL_ARTIFACTS, MODEL_METADATA = train_model()
    else:
        MODEL_ARTIFACTS = joblib.load(model_path)
        print(f"  Model loaded from {model_path}")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            MODEL_METADATA = json.load(f)
    return MODEL_ARTIFACTS, MODEL_METADATA

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_single(row_dict):
    """Preprocess a single customer dict for inference."""
    encoders = MODEL_ARTIFACTS["encoders"]
    feature_cols = MODEL_ARTIFACTS["feature_cols"]
    df = pd.DataFrame([row_dict])
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            df[col] = "Unknown"
        le = encoders.get(col)
        if le:
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
        else:
            df[col] = 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


def preprocess_batch(df_input):
    """Preprocess a DataFrame batch for inference."""
    encoders = MODEL_ARTIFACTS["encoders"]
    feature_cols = MODEL_ARTIFACTS["feature_cols"]
    df = df_input.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            df[col] = "Unknown"
        le = encoders.get(col)
        if le:
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
        else:
            df[col] = 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]

# ─── Risk Classification ───────────────────────────────────────────────────────
def classify_risk(prob):
    if prob >= 0.7:
        return "High", "#ef4444"
    elif prob >= 0.4:
        return "Medium", "#f59e0b"
    else:
        return "Low", "#22c55e"

# ─── Retention Recommendations Builder ───────────────────────────────────────
def build_recommendations(customer_dict):
    recs = []
    for feature, rule in RECOMMENDATIONS.items():
        val = customer_dict.get(feature)
        if val is not None:
            try:
                if rule["condition"](val):
                    recs.append({
                        "feature": feature,
                        "title": rule["title"],
                        "action": rule["action"],
                        "priority": rule["priority"]
                    })
            except Exception:
                pass
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    recs.sort(key=lambda x: priority_order.get(x["priority"], 99))
    return recs

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model-info", methods=["GET"])
def model_info():
    if MODEL_METADATA is None:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify(MODEL_METADATA)


@app.route("/api/predict/single", methods=["POST"])
def predict_single():
    """Predict churn for a single customer submitted as JSON."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    try:
        customer_id = data.get(ID_COL, f"CUST-{uuid.uuid4().hex[:6].upper()}")
        X = preprocess_single(data)
        model = MODEL_ARTIFACTS["model"]
        prob = float(model.predict_proba(X)[0][1])
        prediction = int(prob >= 0.5)
        risk_level, risk_color = classify_risk(prob)
        recs = build_recommendations(data)

        # Feature contributions (top 5 from importances)
        feature_imp = MODEL_METADATA.get("feature_importance", {})
        feature_descs = MODEL_METADATA.get("feature_descriptions", {})
        top_features = []
        for feat, imp in list(feature_imp.items())[:8]:
            val = data.get(feat, "N/A")
            top_features.append({
                "feature": feat,
                "description": feature_descs.get(feat, feat),
                "value": val,
                "importance": round(imp * 100, 1)
            })

        result = {
            "customer_id": customer_id,
            "churn_probability": round(prob, 4),
            "churn_probability_pct": round(prob * 100, 1),
            "prediction": prediction,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "recommendations": recs,
            "top_features": top_features,
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    """Accept a CSV file and return predictions for all customers."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {e}"}), 400

    # Validate required columns
    feature_cols = MODEL_ARTIFACTS["feature_cols"]
    missing = [c for c in feature_cols if c not in df.columns]
    if len(missing) > len(feature_cols) * 0.5:
        return jsonify({
            "error": "CSV is missing too many required columns",
            "missing_columns": missing,
            "required_columns": feature_cols
        }), 400

    try:
        ids = df[ID_COL].tolist() if ID_COL in df.columns else [f"CUST-{i}" for i in range(len(df))]
        actual_churn = df[TARGET_COL].tolist() if TARGET_COL in df.columns else [None] * len(df)

        X = preprocess_batch(df)
        model = MODEL_ARTIFACTS["model"]
        probs = model.predict_proba(X)[:, 1]
        feature_imp = MODEL_METADATA.get("feature_importance", {})
        feature_descs = MODEL_METADATA.get("feature_descriptions", {})

        results = []
        risk_counts = {"High": 0, "Medium": 0, "Low": 0}

        for idx, (cid, prob, actual) in enumerate(zip(ids, probs, actual_churn)):
            prob = float(prob)
            risk_level, risk_color = classify_risk(prob)
            risk_counts[risk_level] += 1
            row_data = df.iloc[idx].to_dict()
            recs = build_recommendations(row_data)

            top_features = []
            for feat, imp in list(feature_imp.items())[:5]:
                top_features.append({
                    "feature": feat,
                    "description": feature_descs.get(feat, feat),
                    "value": str(row_data.get(feat, "N/A")),
                    "importance": round(imp * 100, 1)
                })

            results.append({
                "customer_id": str(cid),
                "churn_probability": round(prob, 4),
                "churn_probability_pct": round(prob * 100, 1),
                "prediction": int(prob >= 0.5),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "actual_churn": int(actual) if actual is not None else None,
                "recommendations": recs,
                "top_features": top_features,
                "customer_data": {k: str(v) for k, v in row_data.items()
                                  if k not in [TARGET_COL, ID_COL]},
            })

        # Compute summary stats
        all_probs = [r["churn_probability"] for r in results]
        predicted_churns = sum(1 for r in results if r["prediction"] == 1)

        summary = {
            "total_customers": len(results),
            "predicted_churn": predicted_churns,
            "predicted_retain": len(results) - predicted_churns,
            "churn_rate_pct": round(predicted_churns / len(results) * 100, 1),
            "avg_churn_probability": round(np.mean(all_probs) * 100, 1),
            "risk_distribution": risk_counts
        }

        # Cache results
        with open(RESULTS_CACHE_FILE, "w") as f:
            json.dump({"results": results, "summary": summary}, f)

        return jsonify({"results": results, "summary": summary})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/download/results", methods=["GET"])
def download_results():
    """Download the last batch prediction results as CSV."""
    if not os.path.exists(RESULTS_CACHE_FILE):
        return jsonify({"error": "No results to download. Run a batch prediction first."}), 404

    with open(RESULTS_CACHE_FILE) as f:
        cached = json.load(f)

    results = cached.get("results", [])
    if not results:
        return jsonify({"error": "No results cached"}), 404

    output = io.StringIO()
    fieldnames = ["customer_id", "churn_probability_pct", "risk_level", "prediction", "actual_churn"]

    # Add customer data columns
    if results:
        extra_cols = list(results[0].get("customer_data", {}).keys())
        fieldnames = fieldnames + extra_cols

    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        row = {
            "customer_id": r["customer_id"],
            "churn_probability_pct": r["churn_probability_pct"],
            "risk_level": r["risk_level"],
            "prediction": r["prediction"],
            "actual_churn": r.get("actual_churn", "")
        }
        row.update(r.get("customer_data", {}))
        writer.writerow(row)

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )


@app.route("/api/outcomes/track", methods=["POST"])
def track_outcome():
    """Record an actual churn outcome for a customer."""
    data = request.get_json()
    customer_id = data.get("customer_id")
    actual_churn = data.get("actual_churn")
    predicted_churn = data.get("predicted_churn")
    churn_prob = data.get("churn_probability")

    if customer_id is None or actual_churn is None:
        return jsonify({"error": "customer_id and actual_churn are required"}), 400

    outcomes = []
    if os.path.exists(OUTCOMES_FILE):
        with open(OUTCOMES_FILE) as f:
            outcomes = json.load(f)

    # Update if exists
    for o in outcomes:
        if o["customer_id"] == str(customer_id):
            o["actual_churn"] = int(actual_churn)
            o["updated_at"] = datetime.now().isoformat()
            with open(OUTCOMES_FILE, "w") as f:
                json.dump(outcomes, f, indent=2)
            return jsonify({"message": "Outcome updated", "customer_id": customer_id})

    outcomes.append({
        "customer_id": str(customer_id),
        "actual_churn": int(actual_churn),
        "predicted_churn": predicted_churn,
        "churn_probability": churn_prob,
        "recorded_at": datetime.now().isoformat()
    })

    with open(OUTCOMES_FILE, "w") as f:
        json.dump(outcomes, f, indent=2)

    return jsonify({"message": "Outcome recorded", "customer_id": customer_id})


@app.route("/api/outcomes/performance", methods=["GET"])
def model_performance():
    """Return tracked model performance from actual outcomes."""
    if not os.path.exists(OUTCOMES_FILE):
        return jsonify({"tracked_outcomes": 0, "metrics": None})

    with open(OUTCOMES_FILE) as f:
        outcomes = json.load(f)

    if not outcomes:
        return jsonify({"tracked_outcomes": 0, "metrics": None})

    y_true = [o["actual_churn"] for o in outcomes]
    y_pred = [o.get("predicted_churn", 0) for o in outcomes if o.get("predicted_churn") is not None]

    metrics = None
    if len(y_pred) == len(y_true) and len(y_true) > 0:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        metrics = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "total_tracked": len(outcomes),
            "true_positives": sum(1 for a, p in zip(y_true, y_pred) if a == 1 and p == 1),
            "false_positives": sum(1 for a, p in zip(y_true, y_pred) if a == 0 and p == 1),
            "true_negatives": sum(1 for a, p in zip(y_true, y_pred) if a == 0 and p == 0),
            "false_negatives": sum(1 for a, p in zip(y_true, y_pred) if a == 1 and p == 0),
        }

    return jsonify({"tracked_outcomes": len(outcomes), "metrics": metrics, "outcomes": outcomes[-20:]})


@app.route("/api/dataset/sample", methods=["GET"])
def dataset_sample():
    """Return a sample of the training dataset."""
    dataset_path = os.path.join(BASE_DIR, "customer_churn_dataset.csv")
    df = pd.read_csv(dataset_path)
    sample = df.head(10).to_dict(orient="records")
    stats = {
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "churn_rate": round(df[TARGET_COL].mean() * 100, 1) if TARGET_COL in df.columns else None
    }
    return jsonify({"sample": sample, "stats": stats})


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n Starting Customer Churn Prediction System...")
    load_model()
    print(f" Dashboard: http://localhost:5000\n")
    app.run(debug=True, port=5000, host="0.0.0.0")
