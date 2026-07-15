# 📉 Customer Churn Prediction Streamlit App

An end-to-end machine learning application that predicts whether a customer is likely to leave a company.

The application is built with **Python, Streamlit, Pandas, Scikit-learn, and Logistic Regression**.

## Why Logistic Regression?

Customer churn is a binary classification problem:

- `0` = Customer stays
- `1` = Customer churns

Logistic Regression is a suitable baseline model because it predicts the probability that a customer belongs to the churn class.

## Features

- Interactive customer data form
- Churn probability prediction
- Adjustable classification threshold
- Risk-level classification
- Retention recommendations
- Accuracy, Precision, Recall, F1 and ROC-AUC metrics
- Confusion matrix
- Classification report
- Logistic Regression coefficient chart
- Dataset overview and churn analysis
- Modular project structure

## Project Structure

```text
customer_churn_prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── customer_churn.csv
├── models/
└── src/
    ├── __init__.py
    ├── modeling.py
    └── visuals.py
```

## Input Features

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

## Machine Learning Workflow

```text
Load Dataset
      ↓
Separate Features and Target
      ↓
Handle Missing Values
      ↓
Scale Numerical Features
      ↓
Encode Categorical Features
      ↓
Train-Test Split
      ↓
Train Logistic Regression
      ↓
Evaluate Classification Metrics
      ↓
Predict Churn Probability
      ↓
Display Result in Streamlit
```

## Installation

### 1. Open the project folder

```bash
cd customer_churn_prediction
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application usually opens at:

```text
http://localhost:8501
```

## Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

## Future Improvements

- Add Random Forest and XGBoost comparison
- Add SMOTE for class imbalance
- Add customer CSV upload
- Save the trained model with Joblib
- Add SHAP explainability
- Add PDF prediction reports
- Add customer retention recommendations using an LLM
- Deploy on Streamlit Community Cloud

## Disclaimer

This project uses a synthetic dataset for learning and portfolio purposes. For business use, retrain the model using real customer data and validate it carefully.

## Author

**Harshpreet Kaur**
