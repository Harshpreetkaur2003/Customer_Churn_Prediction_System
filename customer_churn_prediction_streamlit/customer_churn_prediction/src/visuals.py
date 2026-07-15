import matplotlib.pyplot as plt


def churn_distribution_chart(data):
    counts = data["Churn"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values)
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")

    for index, value in enumerate(counts.values):
        ax.text(index, value, str(value), ha="center", va="bottom")

    fig.tight_layout()
    return fig


def feature_importance_chart(importance_df):
    top_features = importance_df.head(12).sort_values(
        "Importance",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_features["Feature"], top_features["Coefficient"])
    ax.set_title("Top Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Feature")

    fig.tight_layout()
    return fig
