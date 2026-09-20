# Telco Customer Churn Prediction & REST API

An end-to-end machine learning project predicting customer churn using the **IBM Telco Customer Churn Dataset**. This repository covers the complete workflow: data hygiene, exploratory data analysis, feature engineering, Decision Tree modeling and comparison, evaluating why Recall takes priority from a business standpoint, and serving the final pipeline through a lightweight Flask REST API.

---

## 📋 Table of Contents
1. [Business Problem](#-business-problem)
2. [Project Architecture & Repository Structure](#-project-architecture--repository-structure)
3. [Setup & Installation](#-setup--installation)
4. [Jupyter Notebook Execution](#-jupyter-notebook-execution)
5. [Model Training & Serialization](#-model-training--serialization)
6. [REST API Documentation & Serving](#-rest-api-documentation--serving)
7. [Automated API Testing](#-automated-api-testing)
8. [Exploratory Data Analysis (EDA) Insights](#-exploratory-data-analysis-eda-insights)
9. [Feature Engineering](#-feature-engineering)
10. [Model Development, Comparison & Justification](#-model-development-comparison--justification)
11. [Business Decision: Precision vs. Recall Prioritization](#-business-decision-precision-vs-recall-prioritization)
12. [Model Interpretation & Top Predictors](#-model-interpretation--top-predictors)
13. [Retention Recommendations](#-retention-recommendations)

---

## 🎯 Business Problem
Customer churn directly erodes recurring subscription revenue. In telecommunications, acquiring a replacement customer generally costs **5 to 7 times more** than retaining an existing one. When accounts cancel unexpectedly, customer lifetime value (CLV) drops and marketing budgets have to work overtime just to keep subscriber counts flat.

The core goals of this project:
- Flag high-risk subscribers before they cancel.
- Identify the structural, financial, and service factors driving churn.
- Deliver an automated inference API returning both the classification (`"Yes"` / `"No"`) and continuous **`churn_probability`** so retention teams can prioritize personalized outreach.

---

## 📂 Project Architecture & Repository Structure

```
customer_churn_project/
│
├── data/
│   ├── TelcoCustomerChurn.csv                  # IBM Telco Customer Churn dataset (7,043 rows)
│   └── TelcoCustomerChurn - Data Dictionary.csv # Official data dictionary
│
├── notebook/
│   └── churn_analysis.ipynb                    # Executed Jupyter notebook (EDA, Models, Visuals)
│
├── model/
│   └── churn_model.pkl                         # Serialized pipeline (Preprocessing + Decision Tree)
│
├── app.py                                      # Flask REST API (POST /predict, GET /health)
├── pipeline_utils.py                           # Custom feature transformer & pipeline builder
├── test_api.py                                 # Automated integration tests for REST API
├── sample_request.json                         # Sample JSON payload for API testing
├── sample_response.json                        # Sample JSON response payload from API
├── requirements.txt                            # Python dependencies
├── .gitignore                                  # Git ignore rules
└── README.md                                   # Project documentation & run guide
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- `pip` package manager

### 2. Open the Workspace Directory
```bash
cd "customer_churn_project"
```

### 3. Create & Activate Virtual Environment (`venv`)

**On Windows (PowerShell):**
```powershell
# Create virtual environment named 'venv'
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```
> *Note:* If PowerShell script execution is restricted on the machine, activate using Command Prompt (`venv\Scripts\activate.bat`) or temporarily bypass policy for the current session via:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\venv\Scripts\Activate.ps1
> ```

**On macOS / Linux (bash/zsh):**
```bash
# Create virtual environment named 'venv'
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

Once activated, the terminal prompt displays `(venv)`.

### 4. Install Dependencies
With the virtual environment active, install the pinned packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📓 Jupyter Notebook Execution

The notebook [`notebook/churn_analysis.ipynb`](notebook/churn_analysis.ipynb) is self-contained and pre-executed with all figures and metrics in place.

To run it interactively with the virtual environment active:
```bash
jupyter notebook notebook/churn_analysis.ipynb
# or with jupyter lab:
jupyter lab notebook/churn_analysis.ipynb
```

*(Optional: To register this virtual environment as a named Jupyter kernel):*
```bash
python -m ipykernel install --user --name=churn_env --display-name="Python (churn_env)"
```

The notebook covers all 8 project stages:
1. Data Understanding & Hygiene (`TotalCharges` whitespace fix, duplicate verification).
2. Train/Test Stratified Split (70:30, `random_state=42`) with zero data leakage.
3. 6 detailed EDA visualizations with practical business takeaways.
4. Feature engineering and domain validation.
5. Decision Tree modeling (Baseline vs Regularized vs Hyperparameter Tuned via GridSearchCV).
6. Model evaluation and business justification for Recall prioritization.
7. Model interpretation (Feature importance and Decision Tree visualization).
8. Pipeline serialization and verification.

---

## 🚀 REST API Documentation & Serving

The Flask REST API lives in [`app.py`](app.py).

### Starting the API Server
Make sure the virtual environment `(venv)` is active, then start the app:
```bash
python app.py
```
The server binds to `http://127.0.0.1:5000`.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status, description, and available endpoints |
| `GET` | `/health` | Health check confirming service status and model load status |
| `POST` | `/predict` | Predicts customer churn and returns probability |

---

### Endpoint: `POST /predict`

#### Input Validation
The endpoint validates that all required customer fields are present, non-empty, and fall into sensible numeric ranges (non-negative tenure, non-negative charges). Missing or invalid fields return `HTTP 400 Bad Request` with an explicit error description.

#### Single Customer Sample Request
Test the endpoint using the payload in [`sample_request.json`](sample_request.json):

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @sample_request.json
```

**Payload:**
```json
{
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
```

**Response (`HTTP 200 OK`):**
```json
{
  "prediction": "Yes",
  "churn_probability": 0.8501
}
```

---

#### Low-Risk Customer Example

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

**Response (`HTTP 200 OK`):**
```json
{
  "prediction": "No",
  "churn_probability": 0.0293
}
```

---

#### Batch Prediction Support
The API also supports batch evaluation by passing an array of customer records:

```json
[
  { "gender": "Female", "SeniorCitizen": 0, "tenure": 2, ... },
  { "gender": "Male", "SeniorCitizen": 0, "tenure": 62, ... }
]
```
**Response:**
```json
{
  "count": 2,
  "predictions": [
    { "prediction": "Yes", "churn_probability": 0.8501 },
    { "prediction": "No", "churn_probability": 0.0293 }
  ]
}
```

---

## 🧪 Automated API Testing

A complete automated test suite is provided in [`test_api.py`](test_api.py).  
It tests the health check, high-risk profile, low-risk profile, batch queries, and malformed input handling.

```bash
# In terminal 1 (with venv activated), start the API:
python app.py

# In terminal 2 (with venv activated), run the test suite:
python test_api.py
```

Expected output:
```
Running API test suite...
Testing GET /health... -> Status Code: 200 (PASSED)
Testing POST /predict (High Risk Customer)... -> Status Code: 200, Churn Probability: 0.8501 (PASSED)
Testing POST /predict (Low Risk Customer)... -> Status Code: 200, Churn Probability: 0.0293 (PASSED)
Testing POST /predict (Batch)... -> Status Code: 200 (PASSED)
Testing POST /predict (Malformed Missing Fields)... -> Status Code: 400 (PASSED)
Testing POST /predict (Negative Tenure)... -> Status Code: 400 (PASSED)
ALL API TESTS COMPLETED SUCCESSFULLY!
```

---

## 📊 Exploratory Data Analysis (EDA) Insights

| Visualization | Main Pattern Discovered | Business Insight & Retention Action |
|---|---|---|
| **1. Churn Distribution** | 26.5% baseline churn rate (1,869 churners vs 5,174 active accounts). | Moderate class imbalance means a naive model predicting "No Churn" scores 73.5% accuracy while identifying zero at-risk customers. Precision and Recall metrics are essential. |
| **2. Contract Commitment** | Month-to-month contracts have **42.7% churn**, vs **11.3% for 1-year** and **2.8% for 2-year** plans. | Contract type is the single largest structural factor. Proactively incentivizing month-to-month subscribers to lock into 1- or 2-year commitments is an immediate win. |
| **3. Customer Tenure Hazard** | Churn hazard is heavily front-loaded in months **1–12**, peaking during months 1–3. | Implement a structured **90-Day Onboarding Journey** with regular check-ins to anchor new subscribers through the vulnerable trial window. |
| **4. Monthly Charges** | Median spend of churners ($79.65) is noticeably higher than non-churners ($64.40). | Churn density spikes between $70–$100/mo (mostly unbundled Fiber Optic plans). Price tier reviews and service bundling can help mitigate this. |
| **5. Value-Added Add-Ons** | Customers without **TechSupport** or **OnlineSecurity** churn at **>41.6%**, vs **14.6%–15.2%** for those who have them. | Technical support and security features act as strong retention anchors. Bundling these add-ons into base internet tiers helps keep customers sticky. |
| **6. Payment Friction** | Customers paying by **Electronic Check** churn at **45.3%**, vs **~15% for automated billing**. | Manual monthly check payments add recurring friction and cancellation decision points. Offering a small promotion for setting up auto-pay cuts churn significantly. |

---

## 🛠️ Feature Engineering

Four domain-specific features are engineered in [`pipeline_utils.py`](pipeline_utils.py) to give models clearer behavioral signals:

1. **`tenure_cohort`**:
   - *Definition:* Bins customer lifespan into cohorts: `0-12m`, `12-24m`, `24-48m`, `48-72m`, `72m+`.
   - *Business Rationale:* Maps continuous tenure into discrete customer hazard phases, capturing non-linear lifecycle vulnerability.
2. **`total_services`**:
   - *Definition:* Sum of active services (`Phone`, `MultipleLines`, `Internet`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`).
   - *Business Rationale:* Quantifies ecosystem stickiness. Customers with just 1 service churn at **33.3%**, while customers with 5+ services churn at **<15%**, showing how switching costs reduce churn.
3. **`avg_monthly_charges_diff`**:
   - *Definition:* Calculated as `MonthlyCharges - (TotalCharges / max(tenure, 1))`.
   - *Business Rationale:* Detects billing shocks or price creep where current charges suddenly outpace historical averages.
4. **`is_long_term_contract`**:
   - *Definition:* Binary flag (`1` for 1-year or 2-year contracts, `0` for month-to-month).
   - *Business Rationale:* Directly isolates contractual commitment.

---

## 📈 Model Development, Comparison & Justification

Three Decision Tree setups were evaluated on the **70% training** and **30% testing** stratified split (`random_state=42`), covering baseline unpruned, regularized with balanced class weights, and hyperparameter tuned via GridSearchCV:

### Performance Comparison Table (Test Set)

| Model Configuration | Train Acc | Test Acc | Test Precision | Test Recall | Test F1 Score | Test ROC-AUC |
|---|---|---|---|---|---|---|
| **1. DT Baseline (Unpruned, Default)** | 0.9980 | 0.7298 | 0.4913 | 0.5009 | 0.4960 | 0.6570 |
| **2. DT Regularized (Depth=5, Balanced)** | 0.7278 | 0.7080 | 0.4713 | **0.8182** | **0.5980** | **0.8309** |
| **3. DT GridSearch Optimized** | 0.7673 | 0.7435 | 0.5115 | 0.7504 | 0.6084 | 0.8189 |

### Confusion Matrix Comparison (Test Set, N = 2,113)

#### Baseline Unpruned Decision Tree:
- **True Negatives:** 1,261
- **False Positives:** 291
- **False Negatives (Missed Churners):** **280** (50% of all churners missed)
- **True Positives (Identified Churners):** 281

#### Selected Regularized Decision Tree (Balanced, Depth=5):
- **True Negatives:** 1,037
- **False Positives:** 515
- **False Negatives (Missed Churners):** **102** (Only 18% missed)
- **True Positives (Identified Churners):** **459** (Catches 81.8% of all churners)

### Model Selection Justification:
The unpruned baseline overfits heavily (99.8% train accuracy vs 73.0% test accuracy) and misses **280 churning customers**.  
Setting `max_depth=5` and `class_weight='balanced'` in the **Regularized Decision Tree** slashes False Negatives by **63.6%** (from 280 to 102) and achieves **81.82% Recall** on unseen data while keeping the tree simple and explainable.

---

## 🎯 Business Decision: Precision vs. Recall Prioritization

> ### **Verdict: Prioritize Recall (Sensitivity)**

For a telecommunications company seeking to prevent customer churn, **Recall is the priority over Precision**.

#### 1. Severe Asymmetry in Business Costs:
- **Cost of a False Negative (Type II Error — Critical Business Loss):**  
  The model predicts the customer will stay, but they cancel. Future recurring revenue and customer lifetime value (CLV) vanish. Replacing that subscriber requires substantial marketing and sales spend on Customer Acquisition Cost (CAC).
- **Cost of a False Positive (Type I Error — Minor Operational Cost):**  
  The model predicts the customer will churn, but they were planning to stay. The retention team intervenes with a proactive check-in email, customer appreciation perk, or promotional loyalty discount. The operational cost of this intervention is minor compared to customer replacement. Even for loyal customers, proactive appreciation often strengthens brand loyalty and customer satisfaction.

#### 2. Cost-Benefit Trade-Off:
Acquiring a replacement subscriber requires significantly greater investment (marketing campaigns, sales commissions, onboarding resources) compared to the marginal cost of a retention effort. Therefore, failing to identify an at-risk customer (False Negative) results in permanent, irrecoverable revenue loss, whereas extending a retention offer to a loyal customer (False Positive) represents only a minor, controllable operational expense. Capturing the maximum number of potential churners (**high Recall**) delivers far greater net business value.

#### 3. Operational Strategy:
The REST API returns continuous **`churn_probability`** alongside the label. This allows tiered operational intervention:
- **Top Decile (High probability):** Proactive outreach from experienced retention specialists with personalized intervention.
- **Middle Decile (Moderate probability):** Automated email/SMS promotional loyalty offers (such as auto-pay incentives or complimentary speed upgrades).
- **Lower Decile (Low probability):** Standard customer relationship management.

---

## 🔍 Model Interpretation & Top Predictors

### Gini Feature Importance Ranking (Top Features):
1. **`Contract_Month-to-month`**: Dominant root split. Month-to-month contracts have a 42.7% churn rate.
2. **`tenure` & `tenure_cohort_0-12m`**: Customer duration; new accounts in months 1–12 represent the highest hazard risk.
3. **`MonthlyCharges` & `TotalCharges`**: Financial friction; accounts paying >$70/month without bundled value churn rapidly.
4. **`InternetService_Fiber optic`**: Fiber optic subscribers show higher churn when unbundled from support.
5. **`PaymentMethod_Electronic check`**: Manual check payments add recurring churn friction.
6. **`TechSupport_No` / `OnlineSecurity_No`**: Absence of value-added security and support anchors.

---

## 💡 Retention Recommendations for Telecommunications Leadership

1. **Migrate Month-to-Month Contracts:** Run renewal campaigns offering a small discount for committing to a 1-year plan.
2. **First 90-Day VIP Onboarding:** Automate milestone check-ins, satisfaction surveys, and support assistance during the critical first 3 months.
3. **Bundle Value-Added Support:** Bundle 24/7 Tech Support and Cybersecurity into high-speed fiber tiers to establish customer stickiness.
4. **Auto-Pay Conversion Incentive:** Offer a one-time bill credit or promotional discount for switching from Electronic Check to Automatic Bank Transfer or Credit Card.
5. **Continuous API Integration:** Integrate the `POST /predict` endpoint into the CRM to score customer churn risk dynamically on every billing cycle or support contact.
