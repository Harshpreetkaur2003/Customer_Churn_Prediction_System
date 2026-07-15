import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.modeling import train_model, predict_customer
from src.visuals import churn_distribution_chart, feature_importance_chart

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #5b6d7d;
        margin-bottom: 6px;
    }

    .subtitle {
        font-size: 17px;
        color: #5b6d7d;
        margin-bottom: 24px;
    }

    .prediction-card {
        padding: 24px;
        border-radius: 18px;
        background: white;
        border: 1px solid #dfe7ef;
        box-shadow: 0 8px 24px rgba(20, 50, 80, 0.08);
        text-align: center;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #dfe7ef;
        padding: 14px;
        border-radius: 14px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.7rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

DATA_PATH = Path(__file__).parent / "data" / "customer_churn.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def get_model(data):
    return train_model(data)

data = load_data()
model, metrics, report_df, confusion_df, importance_df = get_model(data)

st.markdown('<div class="main-title">📉 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict whether a customer is likely to leave using Logistic Regression.</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(
    ["Predict Churn", "Model Performance", "Dataset & Insights"]
)

with tab1:
    st.subheader("Enter Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

    with col2:
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12,
            step=1
        )
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

    with col3:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
        )
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=500.0,
            value=70.0,
            step=1.0
        )
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=50000.0,
            value=840.0,
            step=10.0
        )

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    predict_button = st.button("Predict Customer Churn", type="primary")

    if predict_button:
        input_data = pd.DataFrame({
            "Gender": [gender],
            "SeniorCitizen": [senior],
            "Partner": [partner],
            "Dependents": [dependents],
            "Tenure": [tenure],
            "PhoneService": [phone_service],
            "InternetService": [internet_service],
            "Contract": [contract],
            "PaperlessBilling": [paperless],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        })

        prediction, probability = predict_customer(
            model,
            input_data,
            threshold=threshold
        )

        risk_level = (
            "High Risk" if probability >= 0.70
            else "Medium Risk" if probability >= 0.40
            else "Low Risk"
        )

        result_text = "Customer is likely to churn" if prediction == 1 else "Customer is likely to stay"
        result_icon = "⚠️" if prediction == 1 else "✅"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="prediction-card">
                <h2>{result_icon} {result_text}</h2>
                <h1>Churn Probability: {probability * 100:.1f}%</h1>
                <p><b>Risk Level:</b> {risk_level}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.warning(
                "Suggested retention actions: offer a long-term plan, provide a loyalty discount, "
                "review service issues, and contact the customer proactively."
            )
        else:
            st.success(
                "This customer currently appears stable. Continue engagement and monitor future changes."
            )

with tab2:
    st.subheader("Model Evaluation")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall", f"{metrics['recall']:.3f}")
    m4.metric("F1 Score", f"{metrics['f1']:.3f}")

    st.metric("ROC-AUC Score", f"{metrics['roc_auc']:.3f}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Confusion Matrix")
        st.dataframe(confusion_df, use_container_width=True)

    with col_b:
        st.markdown("#### Classification Report")
        st.dataframe(report_df, use_container_width=True)

    st.markdown("#### Important Features")
    fig = feature_importance_chart(importance_df)
    st.pyplot(fig)

with tab3:
    st.subheader("Dataset Overview")

    d1, d2, d3 = st.columns(3)
    d1.metric("Total Customers", len(data))
    d2.metric("Churned Customers", int((data["Churn"] == "Yes").sum()))
    d3.metric("Churn Rate", f"{(data['Churn'] == 'Yes').mean() * 100:.1f}%")

    st.markdown("#### Churn Distribution")
    fig2 = churn_distribution_chart(data)
    st.pyplot(fig2)

    st.markdown("#### Dataset Preview")
    st.dataframe(data.head(50), use_container_width=True)

    st.markdown("#### Summary Statistics")
    st.dataframe(data.describe(include="all"), use_container_width=True)
