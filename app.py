"""
app.py
------
Production-ready Flask REST API for Telco Customer Churn Prediction.
Loads serialized ML pipeline from model/churn_model.pkl and exposes:
  - GET  /         : API Info & Metadata
  - GET  /health   : Health Check
  - POST /predict  : Single customer or batch churn inference
"""

import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from pipeline_utils import load_pipeline, TelcoFeatureEngineer

app = Flask(__name__)

# Global model container
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'churn_model.pkl')
pipeline = None


def get_model():
    """Lazy load model pipeline on first request or startup."""
    global pipeline
    if pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run training first.")
        pipeline = load_pipeline(MODEL_PATH)
    return pipeline


# Required fields for validation
REQUIRED_FIELDS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]


def validate_customer_record(data, index=None):
    """
    Validates that a customer record contains all required fields
    and that numeric values are within sensible ranges.
    Returns (is_valid, error_message).
    """
    prefix = f"Record {index}: " if index is not None else ""
    
    if not isinstance(data, dict):
        return False, f"{prefix}Payload must be a JSON object."

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        return False, f"{prefix}Missing required fields: {', '.join(missing)}"

    # Validate numeric fields
    try:
        tenure = float(data['tenure'])
        if tenure < 0:
            return False, f"{prefix}'tenure' must be non-negative."
    except (ValueError, TypeError):
        return False, f"{prefix}'tenure' must be numeric."

    try:
        monthly = float(data['MonthlyCharges'])
        if monthly < 0:
            return False, f"{prefix}'MonthlyCharges' must be non-negative."
    except (ValueError, TypeError):
        return False, f"{prefix}'MonthlyCharges' must be numeric."

    try:
        # TotalCharges can be empty string or numeric
        tc_val = str(data['TotalCharges']).strip()
        if tc_val != '':
            tc = float(tc_val)
            if tc < 0:
                return False, f"{prefix}'TotalCharges' must be non-negative."
    except (ValueError, TypeError):
        return False, f"{prefix}'TotalCharges' must be numeric or blank string."

    return True, None


@app.route('/', methods=['GET'])
def index():
    """Root endpoint providing service metadata."""
    return jsonify({
        "service": "Telco Customer Churn Prediction API",
        "version": "1.0.0",
        "description": "Predict customer churn probability using an optimized Decision Tree pipeline.",
        "endpoints": {
            "GET /": "API overview",
            "GET /health": "Health check",
            "POST /predict": "Predict churn for customer payload(s)"
        }
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        _ = get_model()
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "model_path": MODEL_PATH
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Accepts single customer JSON or list of customer JSONs.
    Returns churn prediction ('Yes'/'No') and churn_probability (float).
    """
    if not request.is_json:
        return jsonify({
            "error": "Invalid Content-Type. Expected application/json."
        }), 400

    payload = request.get_json()
    if payload is None:
        return jsonify({
            "error": "Empty JSON payload received."
        }), 400

    try:
        model = get_model()
    except Exception as e:
        return jsonify({"error": f"Model loading error: {str(e)}"}), 500

    # Case 1: Single customer dictionary
    if isinstance(payload, dict):
        is_valid, error_msg = validate_customer_record(payload)
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        try:
            df_input = pd.DataFrame([payload])
            pred_code = model.predict(df_input)[0]
            churn_proba = float(model.predict_proba(df_input)[0, 1])

            prediction_label = "Yes" if pred_code == 1 else "No"

            response = {
                "prediction": prediction_label,
                "churn_probability": round(churn_proba, 4)
            }
            return jsonify(response), 200
        except Exception as e:
            return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    # Case 2: Batch list of customer dictionaries
    elif isinstance(payload, list):
        if len(payload) == 0:
            return jsonify({"error": "Payload array cannot be empty."}), 400

        for i, record in enumerate(payload):
            is_valid, error_msg = validate_customer_record(record, index=i)
            if not is_valid:
                return jsonify({"error": error_msg}), 400

        try:
            df_input = pd.DataFrame(payload)
            preds = model.predict(df_input)
            probas = model.predict_proba(df_input)[:, 1]

            results = [
                {
                    "prediction": "Yes" if p == 1 else "No",
                    "churn_probability": round(float(prob), 4)
                }
                for p, prob in zip(preds, probas)
            ]
            return jsonify({"predictions": results, "count": len(results)}), 200
        except Exception as e:
            return jsonify({"error": f"Batch inference failed: {str(e)}"}), 500

    else:
        return jsonify({
            "error": "Payload must be a JSON object or an array of JSON objects."
        }), 400


if __name__ == '__main__':
    # Pre-warm model on startup
    get_model()
    print("* Model pre-warmed and ready.")
    app.run(host='0.0.0.0', port=5000, debug=False)
