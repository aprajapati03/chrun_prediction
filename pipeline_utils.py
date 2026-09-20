"""
pipeline_utils.py
-----------------
Preprocessing and Feature Engineering components for Telco Customer Churn Prediction.
Encapsulates transformers and pipelines for reproducible training, serialization,
and REST API inference.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Feature Engineering Transformer for Telco Customer Churn.
    Performs domain-specific feature derivation:
      1. Converts TotalCharges to numeric, cleanly imputing blank strings with 0.0.
      2. tenure_cohort: Bins tenure into distinct customer lifecycle stages.
      3. total_services: Aggregates count of all subscribed core & value-added services.
      4. avg_monthly_charges_diff: Difference between current MonthlyCharges and historical
         average monthly spend (TotalCharges / tenure), identifying billing shocks.
      5. is_long_term_contract: Flag indicating 1-year or 2-year contracted commitment.
    """
    def __init__(self):
        self.service_cols = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert dict or array to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        else:
            X = X.copy()

        # Clean numerical columns
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0.0)
        else:
            X['TotalCharges'] = 0.0

        if 'tenure' in X.columns:
            X['tenure'] = pd.to_numeric(X['tenure'], errors='coerce').fillna(0)
        else:
            X['tenure'] = 0

        if 'MonthlyCharges' in X.columns:
            X['MonthlyCharges'] = pd.to_numeric(X['MonthlyCharges'], errors='coerce').fillna(0.0)
        else:
            X['MonthlyCharges'] = 0.0

        # Feature 1: Tenure Cohort
        bins = [-1, 12, 24, 48, 72, np.inf]
        labels = ['0-12m', '12-24m', '24-48m', '48-72m', '72m+']
        X['tenure_cohort'] = pd.cut(X['tenure'], bins=bins, labels=labels).astype(str)

        # Feature 2: Total Services Count
        total_services = pd.Series(0, index=X.index)
        if 'PhoneService' in X.columns:
            total_services += (X['PhoneService'] == 'Yes').astype(int)
        if 'MultipleLines' in X.columns:
            total_services += (X['MultipleLines'] == 'Yes').astype(int)
        if 'InternetService' in X.columns:
            total_services += (X['InternetService'].isin(['DSL', 'Fiber optic'])).astype(int)

        for col in self.service_cols:
            if col in X.columns:
                total_services += (X[col] == 'Yes').astype(int)

        X['total_services'] = total_services

        # Feature 3: Difference from average monthly spend (billing shock detector)
        tenure_safe = X['tenure'].replace(0, 1)
        expected_monthly = X['TotalCharges'] / tenure_safe
        X['avg_monthly_charges_diff'] = X['MonthlyCharges'] - expected_monthly

        # Feature 4: Long-Term Contract Flag
        if 'Contract' in X.columns:
            X['is_long_term_contract'] = X['Contract'].apply(
                lambda val: 1 if val in ['One year', 'Two year'] else 0
            )
        else:
            X['is_long_term_contract'] = 0

        return X


def get_feature_lists():
    """Returns lists of categorical and numerical features used by the preprocessor."""
    categorical_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod',
        'tenure_cohort', 'is_long_term_contract'
    ]
    numeric_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'total_services', 'avg_monthly_charges_diff'
    ]
    return categorical_cols, numeric_cols


def create_preprocessor():
    """Creates a scikit-learn ColumnTransformer for categorical and numerical features."""
    categorical_cols, numeric_cols = get_feature_lists()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def build_full_pipeline(classifier):
    """
    Constructs a complete end-to-end Pipeline:
      Raw DataFrame -> TelcoFeatureEngineer -> ColumnTransformer -> Classifier
    Ensures ZERO data leakage during cross-validation and inference.
    """
    return Pipeline([
        ('feature_engineer', TelcoFeatureEngineer()),
        ('preprocessor', create_preprocessor()),
        ('classifier', classifier)
    ])


def save_pipeline(pipeline, filepath='model/churn_model.pkl'):
    """Serializes the pipeline to disk using joblib."""
    joblib.dump(pipeline, filepath)
    print(f"Pipeline successfully saved to {filepath}")


def load_pipeline(filepath='model/churn_model.pkl'):
    """Loads a serialized pipeline from disk."""
    return joblib.load(filepath)
