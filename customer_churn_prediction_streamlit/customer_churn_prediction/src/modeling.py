import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_model(data: pd.DataFrame):
    X = data.drop(columns=["Churn"])
    y = data["Churn"].map({"No": 0, "Yes": 1})

    numeric_features = [
        "SeniorCitizen",
        "Tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    categorical_features = [
        "Gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="first")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }

    report = classification_report(
        y_test,
        predictions,
        target_names=["Stay", "Churn"],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose().round(3)

    matrix = confusion_matrix(y_test, predictions)
    confusion_df = pd.DataFrame(
        matrix,
        index=["Actual Stay", "Actual Churn"],
        columns=["Predicted Stay", "Predicted Churn"]
    )

    feature_names = list(numeric_features)

    encoder = (
        model.named_steps["preprocessor"]
        .named_transformers_["categorical"]
        .named_steps["onehot"]
    )

    categorical_names = encoder.get_feature_names_out(categorical_features)
    feature_names.extend(categorical_names.tolist())

    coefficients = model.named_steps["classifier"].coef_[0]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Importance": np.abs(coefficients)
    }).sort_values("Importance", ascending=False)

    return model, metrics, report_df, confusion_df, importance_df


def predict_customer(model, input_data: pd.DataFrame, threshold: float = 0.5):
    probability = float(model.predict_proba(input_data)[0, 1])
    prediction = int(probability >= threshold)
    return prediction, probability
