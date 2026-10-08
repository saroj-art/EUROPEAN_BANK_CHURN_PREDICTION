# ============================================================
# TEST THE SAVED CHURN MODEL
# ============================================================

import pandas as pd

from model import (
    load_artifacts,
    load_customer_data,
    predict_customer,
    score_dataset,
    get_feature_importance,
)


print("\n==========================================")
print("EUROPEAN BANK CHURN MODEL TEST")
print("==========================================\n")


# ------------------------------------------------------------
# 1. Load artifacts
# ------------------------------------------------------------

model, scaler, threshold, training_columns = (
    load_artifacts()
)

print("1. Model artifacts loaded successfully.")
print("   Model:", type(model).__name__)
print("   Scaler:", type(scaler).__name__)
print("   Threshold:", threshold)
print("   Features:", len(training_columns))


# ------------------------------------------------------------
# 2. Load customer data
# ------------------------------------------------------------

df = load_customer_data()

print("\n2. Dataset loaded successfully.")
print("   Rows:", len(df))
print("   Columns:", len(df.columns))


# ------------------------------------------------------------
# 3. Test one customer
# ------------------------------------------------------------

customer = {
    "Year": 2025,
    "CreditScore": 650,
    "Gender": "Female",
    "Age": 42,
    "Tenure": 5,
    "Balance": 100000,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 100000,
    "Geography": "France",
}


result = predict_customer(customer)

print("\n3. Single customer prediction")
print("--------------------------------")
print(
    f"Probability : "
    f"{result['percentage']:.2f}%"
)

print(
    f"Risk Band   : "
    f"{result['risk_band']}"
)

print(
    f"Prediction  : "
    f"{result['status']}"
)

print(
    f"Threshold   : "
    f"{result['threshold']:.2f}"
)


# ------------------------------------------------------------
# 4. Score complete dataset
# ------------------------------------------------------------

scored_df = score_dataset(df)

print("\n4. Full dataset prediction")
print("--------------------------------")

print(
    "Rows scored:",
    len(scored_df)
)

print(
    "Probability range:",
    round(
        scored_df["Churn_Probability"].min(),
        4
    ),
    "to",
    round(
        scored_df["Churn_Probability"].max(),
        4
    )
)

print("\nRisk distribution:")

print(
    scored_df["Risk_Band"].value_counts()
)


# ------------------------------------------------------------
# 5. Feature importance
# ------------------------------------------------------------

importance = get_feature_importance()

print("\n5. Top feature importance")
print("--------------------------------")

print(
    importance.head(10).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 6. Basic validation
# ------------------------------------------------------------

assert len(training_columns) == 16

assert (
    scored_df["Churn_Probability"]
    .between(0, 1)
    .all()
)

assert (
    scored_df["Predicted_Churn"]
    .isin([0, 1])
    .all()
)

print("\n==========================================")
print("ALL BASIC MODEL TESTS PASSED")
print("==========================================")