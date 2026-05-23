📉 Customer Churn Prediction
A machine learning project to predict customer churn for a telecommunications company using the IBM Telco Customer Churn dataset. The goal is to proactively identify at-risk customers and enable data-driven retention strategies.

📋 Table of Contents

Overview
Dataset
Project Workflow
Models Evaluated
Final Model
Results
Key Findings
Business Impact
Recommendations
Installation
Usage
Project Structure

Overview
Customer churn is one of the most critical challenges in the telecom industry. This project builds a classification model to identify customers who are likely to leave, so that proactive retention strategies (discounts, contract upgrades, personalized outreach) can be applied before the customer churns.

Key Insight: Missing a churner is more costly than incorrectly flagging a loyal customer. The model is therefore optimized for recall on the churn class.


Dataset

Source: IBM Sample Datasets — Telco Customer Churn
Rows: 7,043 customers (7,032 after cleaning)
Target column: Churn (Yes / No)
Class distribution: ~73.4% No Churn / ~26.6% Churn (imbalanced)

Feature Categories
CategoryFeaturesDemographicsgender, SeniorCitizen, Partner, DependentsServicesPhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMoviesAccount Infotenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

Project Workflow
1. Data Loading & Exploration
        ↓
2. Data Cleaning
   - Convert TotalCharges to numeric
   - Drop 11 rows with missing values
        ↓
3. Feature Engineering
   - One-hot encoding of categorical variables (drop_first=True)
   - Convert boolean columns to integer
        ↓
4. Train/Test Split  (75% / 25%, stratified)
        ↓
5. Feature Scaling  (StandardScaler)
        ↓
6. Model Training & Evaluation
   (KNN, Logistic Regression, Random Forest, XGBoost)
        ↓
7. Final Model Selection
   (Logistic Regression with class_weight='balanced')
        ↓
8. Threshold Tuning (0.35) & Final Evaluation

Models Evaluated
ModelAccuracyChurn RecallChurn F1NotesK-Nearest Neighbors74.6%0.510.52Weak churn recallLogistic Regression80.6%0.570.61Best baselineRandom Forest78.5%0.490.55Recall droppedXGBoost (RF)72.5%0.000.00❌ Failed — predicted only "No" due to class imbalance

XGBoost and Random Forest underperformed on the minority (churn) class without explicit imbalance handling.


Final Model
Logistic Regression with the following configuration:
pythonfrom sklearn.linear_model import LogisticRegression

final_model = LogisticRegression(
    class_weight='balanced',   # Handles class imbalance
    max_iter=1000,
    random_state=42
)
Threshold tuned to 0.35 (instead of the default 0.5) to prioritize recall for the churn class:
pythony_prob = final_model.predict_proba(x_test_sc)[:, 1]
y_pred_final = (y_prob >= 0.35).astype(int)

Results
Classification Report
              precision    recall  f1-score   support

           0       0.95      0.61      0.74      1275   ← Non-churn
           1       0.47      0.92      0.62       483   ← Churn

    accuracy                           0.69      1758
   macro avg       0.71      0.76      0.68      1758
weighted avg       0.82      0.69      0.71      1758
Confusion Matrix
Predicted →      No      Yes
Actual No   [ 774    501 ]   ← 501 false positives (acceptable)
Actual Yes  [  41    442 ]   ← only 41 churners missed ✅
Key Metrics
MetricValueChurn Recall92% ✅ROC-AUC Score0.856 ✅Churn F1-Score0.62Overall Accuracy69%

Accuracy is intentionally lower because the model trades non-churn precision for churn recall — the right trade-off for this business problem.


Key Findings
🚨 Features that INCREASE churn risk
FeatureCoefficientInsightTenure−1.23Short-tenure customers are highest riskMonthly Charges−0.71Higher charges drive dissatisfactionFiber Optic Internet+0.69Possibly costlier with more competitionTotal Charges+0.57Price-sensitive customers churn moreTwo-Year Contract−0.62Long contracts dramatically reduce churnStreaming Services+0.21 / +0.17Entertainment-only users switch easilyPaperless Billing+0.18Digitally flexible = less committedElectronic Check+0.16Less tied to auto-payment routines
🛡️ Features that REDUCE churn risk

Long tenure — the single strongest retention signal
Lower monthly charges — affordability drives loyalty
Long-term contracts (1-year or 2-year)
Online Security & Tech Support — satisfied, supported customers stay
Dependents & automatic credit card payments — stable household indicators


Business Impact

✅ Early identification of at-risk customers before they leave
✅ Enables targeted retention campaigns (discounts, upgrades, outreach)
✅ Reduces customer attrition and protects revenue
✅ Interpretable model — suitable for business decision-making without a black box


Recommendations

Target early-stage customers — churn risk is highest in the first few months of tenure
Offer contract upgrades — incentivize month-to-month customers to move to 1- or 2-year contracts
Bundle value-added services — tech support and security subscriptions significantly improve loyalty
Review fiber optic pricing — high churn among fiber users suggests a pricing or service quality issue
Continuously retrain the model — as customer behavior and offerings evolve, model performance should be monitored and updated


Installation
bash# Clone the repository
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Requirements
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn

Usage
bash# Run the notebook
jupyter notebook churn_prediction.ipynb

# Or run the script directly
python churn_prediction.py

Project Structure
customer-churn-prediction/
│
├── churn_prediction.ipynb     # Main Jupyter Notebook
├── churn_prediction.py        # Script version
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
