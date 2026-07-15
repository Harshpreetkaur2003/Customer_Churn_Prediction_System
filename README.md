# 📉 Customer Churn Prediction using Logistic Regression

A machine learning web application built with **Python**, **Scikit-learn**, and **Streamlit** that predicts whether a customer is likely to **churn (leave the company)** based on customer demographics, account information, and service usage.

The application provides **real-time predictions**, **churn probability**, **risk assessment**, and **model performance metrics** through an interactive Streamlit dashboard.

---

# 🚀 Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses such as telecom companies, banks, insurance providers, and SaaS platforms.

This project predicts customer churn using **Logistic Regression**, helping organizations identify customers who are likely to leave so they can take proactive retention measures.

---

# 🎯 Objectives

- Predict customer churn using machine learning
- Build an interactive Streamlit dashboard
- Perform customer risk analysis
- Evaluate model performance
- Demonstrate an end-to-end ML workflow

---

# ✨ Features

### 📊 Interactive Dashboard

- Modern Streamlit UI
- Responsive layout
- Professional CSS styling

### 🤖 Machine Learning

- Logistic Regression
- Data preprocessing
- One-Hot Encoding
- Standard Scaling
- Missing value handling

### 📈 Model Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

### 🎯 Prediction Module

Users can enter:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

The application predicts:

- Customer will churn or not
- Churn probability
- Risk level
- Retention recommendation

---

# 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Frontend | Streamlit |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| IDE | VS Code |

---

# 📂 Project Structure

```text
customer_churn_prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── customer_churn.csv
│
├── models/
│
└── src/
    ├── __init__.py
    ├── modeling.py
    └── visuals.py
```

---

# 📊 Dataset Features

| Feature | Description |
|----------|-------------|
| Gender | Customer Gender |
| SeniorCitizen | Senior citizen status |
| Partner | Whether customer has a partner |
| Dependents | Whether customer has dependents |
| Tenure | Months with the company |
| PhoneService | Phone service availability |
| InternetService | Internet connection type |
| Contract | Contract duration |
| PaperlessBilling | Paperless billing status |
| PaymentMethod | Payment option |
| MonthlyCharges | Monthly bill amount |
| TotalCharges | Total amount paid |
| Churn | Target variable |

---

# 🧠 Machine Learning Algorithm

## Logistic Regression

This project uses **Logistic Regression**, a supervised machine learning algorithm designed for **binary classification problems**.

Since customer churn has only two possible outcomes:

- Stay (0)
- Churn (1)

Logistic Regression is a suitable choice.

Instead of predicting a continuous value, it predicts the **probability of churn**.

---

# 🔄 Workflow

```text
Load Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Handle Missing Values
        │
        ▼
Feature Encoding
        │
        ▼
Feature Scaling
        │
        ▼
Train-Test Split
        │
        ▼
Train Logistic Regression
        │
        ▼
Model Evaluation
        │
        ▼
Predict Customer Churn
        │
        ▼
Display Results in Streamlit
```

---

# 📈 Model Evaluation Metrics

The model is evaluated using:

### ✅ Accuracy

Percentage of correctly classified customers.

### ✅ Precision

How many predicted churn customers actually churned.

### ✅ Recall

How many actual churn customers were correctly identified.

### ✅ F1 Score

Balance between Precision and Recall.

### ✅ ROC-AUC

Measures the model's ability to distinguish between churn and non-churn customers.

---

# 🖥️ Streamlit Dashboard

## Prediction Page

- Customer Information Form
- Predict Churn Button
- Churn Probability
- Risk Level
- Retention Recommendation

---

## Model Performance

Displays:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report
- Feature Importance

---

## Dataset Insights

Displays:

- Dataset Preview
- Churn Distribution
- Customer Statistics
- Summary Statistics

---

# 📸 Sample Prediction

### Input

```text
Gender: Female
Senior Citizen: No
Partner: Yes
Dependents: No
Tenure: 10 Months
Contract: Month-to-Month
Monthly Charges: ₹85
```

### Output

```text
Prediction

⚠️ Customer is likely to Churn

Probability: 82.4%

Risk Level:
High Risk

Suggested Action:
Offer a loyalty discount
Recommend a long-term contract
Contact customer proactively
```

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
```

Move into the project folder:

```bash
cd customer_churn_prediction
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

---

# 📌 Future Enhancements

- Random Forest Classifier
- XGBoost Classifier
- LightGBM
- SHAP Explainability
- Customer Segmentation
- PDF Report Generation
- Email Alerts
- Model Deployment on Streamlit Cloud
- Customer CSV Upload
- Hyperparameter Tuning
- Feature Importance Dashboard
- LLM-powered Retention Suggestions

---

# 📚 Learning Outcomes

This project demonstrates:

- Data Cleaning
- Feature Engineering
- Data Preprocessing
- Logistic Regression
- Binary Classification
- Model Evaluation
- Streamlit Application Development
- Interactive Dashboard Design
- Machine Learning Deployment

---

# 👨‍💻 Author

**Harshpreet Kaur**

MCA | Data Analytics | Machine Learning | Python | Streamlit

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

Happy Coding! 🚀
