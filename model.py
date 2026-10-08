# ============================================================
# EUROPEAN BANK CUSTOMER CHURN - MODEL ENGINE
# ============================================================

from pathlib import Path
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

MODEL_PATH = MODELS_DIR / "final_gradient_boosting_model.pkl"
SCALER_PATH = MODELS_DIR / "final_scaler.pkl"
THRESHOLD_PATH = MODELS_DIR / "final_threshold.pkl"
COLUMNS_PATH = MODELS_DIR / "training_columns.pkl"


# ------------------------------------------------------------
# EXPECTED FEATURES FROM YOUR NOTEBOOK
# ------------------------------------------------------------

EXPECTED_FEATURES = [
    "Year",
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
    "Balance_Salary_Ratio",
    "Product_Density",
    "Engagement_Product_Score",
    "Age_Tenure_Interaction",
]


# ------------------------------------------------------------
# LOAD SAVED MODEL ARTIFACTS
# ------------------------------------------------------------

@lru_cache(maxsize=1)
def load_artifacts():
    """
    Load the trained Gradient Boosting model,
    scaler, threshold and training column names.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}"
        )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_PATH}"
        )

    if not COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Training columns file not found: {COLUMNS_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))
    training_columns = joblib.load(COLUMNS_PATH)

    training_columns = list(training_columns)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if len(training_columns) != 16:
        raise ValueError(
            f"Expected 16 training features, "
            f"but found {len(training_columns)}."
        )

    if set(training_columns) != set(EXPECTED_FEATURES):
        raise ValueError(
            "Training columns do not match the expected "
            "16-feature structure."
        )

    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != len(training_columns):
            raise ValueError(
                "Model feature count does not match "
                "training column count."
            )

    if hasattr(scaler, "n_features_in_"):
        if scaler.n_features_in_ != len(training_columns):
            raise ValueError(
                "Scaler feature count does not match "
                "training column count."
            )

    return model, scaler, threshold, training_columns


# ------------------------------------------------------------
# DATA FILE DISCOVERY
# ------------------------------------------------------------

def find_data_file():
    """
    Find the customer CSV inside the data folder.
    """

    data_dir = ROOT_DIR / "data"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data folder not found: {data_dir}"
        )

    # Look for CSV files inside data/
    csv_files = list(data_dir.glob("*.csv"))

    if len(csv_files) == 1:
        return csv_files[0]

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No CSV file found inside the data folder."
        )

    raise FileNotFoundError(
        "Multiple CSV files found inside data/. "
        "Please keep only the customer dataset there."
    )

# ------------------------------------------------------------
# LOAD CUSTOMER DATA
# ------------------------------------------------------------

@lru_cache(maxsize=1)
def load_customer_data():
    """
    Load the original customer dataset.
    """

    data_path = find_data_file()

    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("Customer dataset is empty.")

    return df


# ------------------------------------------------------------
# STANDARDIZE RAW DATASET
# ------------------------------------------------------------

def standardize_raw_data(df):
    """
    Prepare raw customer data before model feature engineering.
    """

    data = df.copy()

    # Rename notebook's unknown column for dashboard purposes.
    if "MyUnknownColumn" in data.columns:
        data = data.rename(
            columns={"MyUnknownColumn": "Customer_id"}
        )

    return data


# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------

def engineer_features(df):
    """
    Reproduce the same preprocessing used during model training.
    """

    df = df.copy()

    # -----------------------------
    # Gender encoding
    # Male = 1, Female = 0
    # -----------------------------
    if "Gender" in df.columns:
        df["Gender"] = (
            df["Gender"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "male": 1,
                "female": 0
            })
        )

    # -----------------------------
    # Geography encoding
    # France = baseline
    # -----------------------------
    if "Geography" in df.columns:
        df["Geography"] = (
            df["Geography"]
            .astype(str)
            .str.strip()
        )

        df = pd.get_dummies(
            df,
            columns=["Geography"],
            drop_first=True
        )

    # Make sure geography columns exist
    if "Geography_Germany" not in df.columns:
        df["Geography_Germany"] = 0

    if "Geography_Spain" not in df.columns:
        df["Geography_Spain"] = 0

    # Convert boolean dummy columns to integers
    df["Geography_Germany"] = df["Geography_Germany"].astype(int)
    df["Geography_Spain"] = df["Geography_Spain"].astype(int)

    # -----------------------------
    # Feature engineering
    # -----------------------------
    df["Balance_Salary_Ratio"] = (
        df["Balance"] / (df["EstimatedSalary"] + 1)
    )

    df["Product_Density"] = (
        df["NumOfProducts"] / (df["Tenure"] + 1)
    )

    df["Engagement_Product_Score"] = (
        df["IsActiveMember"] * df["NumOfProducts"]
    )

    df["Age_Tenure_Interaction"] = (
        df["Age"] * df["Tenure"]
    )

    return df


# ------------------------------------------------------------
# PREPARE MODEL INPUT
# ------------------------------------------------------------

def prepare_model_input(df):
    """
    Convert raw customer data into the exact
    16-feature structure expected by the model.
    """

    data = engineer_features(df)

    _, _, _, training_columns = load_artifacts()

    # Add any missing expected columns
    for column in training_columns:

        if column not in data.columns:
            data[column] = 0

    # Select ONLY training columns
    X = data[training_columns].copy()

    # Convert everything to numeric
    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Check missing values
    if X.isnull().any().any():

        missing_columns = X.columns[
            X.isnull().any()
        ].tolist()

        raise ValueError(
            "Missing or invalid values found in: "
            + ", ".join(missing_columns)
        )

    return X


# ------------------------------------------------------------
# SCALE FEATURES
# ------------------------------------------------------------

def scale_features(X):
    """
    Apply the exact scaler saved during training.
    """

    _, scaler, _, training_columns = load_artifacts()

    X = X[training_columns]

    X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=training_columns,
        index=X.index
    )

    return X_scaled


# ------------------------------------------------------------
# PREDICT CHURN PROBABILITY
# ------------------------------------------------------------

def predict_probability(df):
    """
    Return probability of churn for each input row.
    """

    model, _, _, _ = load_artifacts()

    X = prepare_model_input(df)

    X_scaled = scale_features(X)

    probabilities = model.predict_proba(
        X_scaled
    )[:, 1]

    return probabilities


# ------------------------------------------------------------
# RISK BAND
# ------------------------------------------------------------

def get_risk_band(probability):
    """
    Dashboard presentation bands.

    These bands are for visual interpretation only.
    The model classification threshold remains the
    saved 0.70 operating threshold.
    """

    if probability < 0.40:
        return "Low Risk"

    elif probability < 0.70:
        return "Medium Risk"

    return "High Risk"


# ------------------------------------------------------------
# PREDICT SINGLE CUSTOMER
# ------------------------------------------------------------

def predict_customer(customer_data):
    """
    Predict churn for one customer.

    customer_data should be a dictionary containing
    raw customer features.
    """

    df = pd.DataFrame([customer_data])

    probability = float(
        predict_probability(df)[0]
    )

    _, _, threshold, _ = load_artifacts()

    prediction = int(
        probability >= threshold
    )

    risk_band = get_risk_band(probability)

    return {
        "probability": probability,
        "percentage": probability * 100,
        "prediction": prediction,
        "status": (
            "Likely to Churn"
            if prediction == 1
            else "Likely to Stay"
        ),
        "risk_band": risk_band,
        "threshold": threshold,
    }


# ------------------------------------------------------------
# SCORE COMPLETE DATASET
# ------------------------------------------------------------

def score_dataset(df):
    """
    Add prediction probability, class and risk band
    to the original customer dataset.
    """

    data = standardize_raw_data(df)

    probabilities = predict_probability(data)

    _, _, threshold, _ = load_artifacts()

    data["Churn_Probability"] = probabilities

    data["Churn_Probability_Pct"] = (
        probabilities * 100
    )

    data["Predicted_Churn"] = (
        probabilities >= threshold
    ).astype(int)

    data["Risk_Band"] = [
        get_risk_band(probability)
        for probability in probabilities
    ]

    return data


# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

def get_feature_importance():
    """
    Return Gradient Boosting feature importance.
    """

    model, _, _, training_columns = load_artifacts()

    importance = pd.DataFrame(
        {
            "Feature": training_columns,
            "Importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)

    return importance


# ------------------------------------------------------------
# CUSTOMER RECOMMENDATIONS
# ------------------------------------------------------------

def get_recommendations(customer_data, probability):
    """
    Generate business recommendations based on
    the notebook's identified churn drivers.
    """

    recommendations = []

    age = customer_data["Age"]
    products = customer_data["NumOfProducts"]
    active = customer_data["IsActiveMember"]
    geography = customer_data["Geography"]

    # High-risk intervention
    if probability >= 0.70:
        recommendations.append(
            "Prioritize this customer for retention intervention."
        )

    # Age
    if age >= 40:
        recommendations.append(
            "Consider a personalized retention or loyalty program "
            "for this older customer segment."
        )

    # Activity
    if active == 0:
        recommendations.append(
            "Increase customer engagement through digital banking, "
            "personalized communication and account-usage incentives."
        )

    # Multiple products
    if products >= 3:
        recommendations.append(
            "Review the customer's multi-product relationship "
            "and consider a personalized product/service bundle."
        )

    # Germany
    if geography == "Germany":
        recommendations.append(
            "Consider a targeted regional retention strategy "
            "for the German customer segment."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Continue monitoring the customer and maintain "
            "regular engagement."
        )

    return recommendations


# ------------------------------------------------------------
# SAMPLE CUSTOMER
# ------------------------------------------------------------

def get_sample_customer(df, index=0):
    """
    Return one raw customer record for testing.
    """

    data = standardize_raw_data(df)

    if index < 0 or index >= len(data):
        raise IndexError("Customer index is out of range.")

    row = data.iloc[index]

    return {
        "Year": int(row["Year"]),
        "CreditScore": float(row["CreditScore"]),
        "Gender": row["Gender"],
        "Age": float(row["Age"]),
        "Tenure": float(row["Tenure"]),
        "Balance": float(row["Balance"]),
        "NumOfProducts": int(row["NumOfProducts"]),
        "HasCrCard": int(row["HasCrCard"]),
        "IsActiveMember": int(row["IsActiveMember"]),
        "EstimatedSalary": float(row["EstimatedSalary"]),
        "Geography": row["Geography"],
    }