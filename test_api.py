"""
test_api.py
-----------
Automated test suite for the Telco Churn REST API (app.py).
Tests:
  1. Health check endpoint (GET /health)
  2. High-churn-risk customer prediction (POST /predict)
  3. Low-churn-risk customer prediction (POST /predict)
  4. Batch customer prediction (POST /predict)
  5. Error handling for missing fields and malformed payloads (HTTP 400)
"""

import sys
import time
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

# High-risk profile: Month-to-month, Fiber optic, Electronic check, low tenure, no support
HIGH_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.70,
    "TotalCharges": 151.65
}

# Low-risk profile: Two year, DSL, Bank transfer, tenure=62, bundled tech support
LOW_RISK_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 62,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 56.15,
    "TotalCharges": 3487.95
}

# Malformed customer missing required fields
MALFORMED_CUSTOMER = {
    "gender": "Female",
    "tenure": 10
}


def test_health():
    print("Testing GET /health...")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
    print("PASSED: Health check\n")


def test_high_risk_predict():
    print("Testing POST /predict (High Risk Customer)...")
    resp = requests.post(f"{BASE_URL}/predict", json=HIGH_RISK_CUSTOMER)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "churn_probability" in data
    assert data["prediction"] == "Yes"
    assert data["churn_probability"] > 0.50
    print("PASSED: High Risk Customer Prediction\n")


def test_low_risk_predict():
    print("Testing POST /predict (Low Risk Customer)...")
    resp = requests.post(f"{BASE_URL}/predict", json=LOW_RISK_CUSTOMER)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "churn_probability" in data
    assert data["prediction"] == "No"
    assert data["churn_probability"] < 0.50
    print("PASSED: Low Risk Customer Prediction\n")


def test_batch_predict():
    print("Testing POST /predict (Batch)...")
    batch_payload = [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]
    resp = requests.post(f"{BASE_URL}/predict", json=batch_payload)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    print("PASSED: Batch Prediction\n")


def test_malformed_input():
    print("Testing POST /predict (Malformed Missing Fields)...")
    resp = requests.post(f"{BASE_URL}/predict", json=MALFORMED_CUSTOMER)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 400
    assert "error" in resp.json()
    print("PASSED: Error handling for missing fields\n")

    print("Testing POST /predict (Negative Tenure)...")
    invalid_tenure = HIGH_RISK_CUSTOMER.copy()
    invalid_tenure["tenure"] = -5
    resp = requests.post(f"{BASE_URL}/predict", json=invalid_tenure)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.json()}")
    assert resp.status_code == 400
    print("PASSED: Error handling for negative tenure\n")


if __name__ == '__main__':
    print("Running API test suite...")
    test_health()
    test_high_risk_predict()
    test_low_risk_predict()
    test_batch_predict()
    test_malformed_input()
    print("ALL API TESTS COMPLETED SUCCESSFULLY!")
