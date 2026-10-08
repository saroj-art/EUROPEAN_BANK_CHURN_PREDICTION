# ============================================================
# EUROPEAN BANK CUSTOMER CHURN INTELLIGENCE DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from model import (
    load_artifacts,
    load_customer_data,
    predict_customer,
    score_dataset,
    get_feature_importance,
    get_recommendations,
    standardize_raw_data,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="European Bank Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 16px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .risk-high {
        padding: 18px;
        border-radius: 12px;
        background-color: rgba(220, 53, 69, 0.12);
        border: 1px solid rgba(220, 53, 69, 0.35);
        text-align: center;
    }

    .risk-medium {
        padding: 18px;
        border-radius: 12px;
        background-color: rgba(255, 193, 7, 0.12);
        border: 1px solid rgba(255, 193, 7, 0.35);
        text-align: center;
    }

    .risk-low {
        padding: 18px;
        border-radius: 12px;
        background-color: rgba(40, 167, 69, 0.12);
        border: 1px solid rgba(40, 167, 69, 0.35);
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA / MODEL
# ============================================================

@st.cache_resource
def get_model_artifacts():

    return load_artifacts()


@st.cache_data
def get_data():

    df = load_customer_data()

    return standardize_raw_data(df)


@st.cache_data
def get_scored_data():

    df = get_data()

    return score_dataset(df)


# Load
try:

    model, scaler, threshold, training_columns = (
        get_model_artifacts()
    )

    df = get_data()

    scored_df = get_scored_data()

except Exception as e:

    st.error(
        "Application initialization failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 Churn Intelligence")

st.sidebar.caption(
    "European Bank Customer Churn Prediction"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Customer Risk Calculator",
        "Portfolio Insights",
        "Feature Importance",
        "What-If Simulator",
        "Model Performance",
        "Business Recommendations",
    ],
)


st.sidebar.markdown("---")

st.sidebar.write(
    f"**Model:** Gradient Boosting"
)

st.sidebar.write(
    f"**Features:** {len(training_columns)}"
)

st.sidebar.write(
    f"**Operating Threshold:** {threshold:.2f}"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Prediction threshold is the saved operating "
    "threshold from the final notebook model."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def display_risk_result(result):

    probability = result["percentage"]
    risk_band = result["risk_band"]
    status = result["status"]

    if risk_band == "High Risk":

        css_class = "risk-high"

    elif risk_band == "Medium Risk":

        css_class = "risk-medium"

    else:

        css_class = "risk-low"

    st.markdown(
        f"""
        <div class="{css_class}">
            <h2>{probability:.2f}%</h2>
            <h3>{risk_band}</h3>
            <p>{status}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def add_age_group(data):

    bins = [
        17,
        30,
        40,
        50,
        60,
        100,
    ]

    labels = [
        "18–30",
        "31–40",
        "41–50",
        "51–60",
        "60+",
    ]

    data = data.copy()

    data["Age_Group"] = pd.cut(
        data["Age"],
        bins=bins,
        labels=labels,
    )

    return data


def recommendation_priority(probability):

    if probability >= 0.70:
        return "High Priority"

    elif probability >= 0.40:
        return "Medium Priority"

    return "Routine Monitoring"


# ============================================================
# PAGE 1 - EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.markdown(
        '<div class="main-title">'
        'European Bank Customer Churn Intelligence'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Predictive risk, customer segmentation and '
        'retention decision support'
        '</div>',
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    total_customers = len(scored_df)

    actual_churn_rate = (
        scored_df["Exited"].mean() * 100
    )

    avg_predicted_risk = (
        scored_df["Churn_Probability_Pct"].mean()
    )

    high_risk_customers = (
        scored_df["Predicted_Churn"]
        .sum()
    )

    high_risk_pct = (
        high_risk_customers /
        total_customers *
        100
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Total Customers",
        f"{total_customers:,}",
    )

    col2.metric(
        "Historical Churn Rate",
        f"{actual_churn_rate:.2f}%",
    )

    col3.metric(
        "Average Predicted Risk",
        f"{avg_predicted_risk:.2f}%",
    )

    col4.metric(
        "High-Risk Customers",
        f"{high_risk_customers:,}",
        f"{high_risk_pct:.2f}% of portfolio",
    )


    st.markdown("---")


    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        risk_counts = (
            scored_df["Risk_Band"]
            .value_counts()
            .reindex(
                [
                    "Low Risk",
                    "Medium Risk",
                    "High Risk",
                ]
            )
            .fillna(0)
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Band",
            "Customers",
        ]

        fig = px.bar(
            risk_counts,
            x="Risk Band",
            y="Customers",
            title="Customer Risk Distribution",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    with col2:

        fig = px.histogram(
            scored_df,
            x="Churn_Probability",
            nbins=20,
            range_x=[0, 1],
            title="Predicted Churn Probability Distribution",
        )

        fig.update_xaxes(
            tickformat=".0%",
            title="Predicted Churn Probability",
        )

        fig.update_yaxes(
            title="Number of Customers"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # BUSINESS SEGMENTS
    # --------------------------------------------------------

    data = add_age_group(scored_df)


    col1, col2 = st.columns(2)


    with col1:

        geo_summary = (
            data.groupby("Geography", observed=False)
            .agg(
                Customers=("Geography", "size"),
                Churn_Rate=(
                    "Exited",
                    "mean",
                ),
            )
            .reset_index()
        )

        geo_summary["Churn_Rate"] *= 100

        fig = px.bar(
            geo_summary,
            x="Geography",
            y="Churn_Rate",
            title="Historical Churn Rate by Geography",
            text_auto=".1f",
        )

        fig.update_yaxes(
            title="Churn Rate (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    with col2:

        age_summary = (
            data.groupby(
                "Age_Group",
                observed=False
            )
            .agg(
                Customers=("Age", "size"),
                Churn_Rate=(
                    "Exited",
                    "mean",
                ),
            )
            .reset_index()
        )

        age_summary["Churn_Rate"] *= 100

        fig = px.bar(
            age_summary,
            x="Age_Group",
            y="Churn_Rate",
            title="Historical Churn Rate by Age Group",
            text_auto=".1f",
        )

        fig.update_yaxes(
            title="Churn Rate (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # MAIN DRIVERS
    # --------------------------------------------------------

    st.subheader(
        "Top Model Drivers"
    )

    importance = get_feature_importance()

    top_features = importance.head(8)

    fig = px.bar(
        top_features.sort_values(
            "Importance",
            ascending=True,
        ),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Global Feature Importance",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    st.info(
        "The notebook's explainability analysis consistently "
        "identified Age, Number of Products, customer activity "
        "and Geography as important churn drivers."
    )


# ============================================================
# PAGE 2 - CUSTOMER RISK CALCULATOR
# ============================================================

elif page == "Customer Risk Calculator":

    st.title(
        "🎯 Customer Churn Risk Calculator"
    )

    st.write(
        "Enter customer information to estimate "
        "individual churn probability."
    )


    col1, col2 = st.columns(2)


    with col1:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650,
            step=1,
        )

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male",
            ],
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=40,
        )

        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=20,
            value=5,
        )

        geography = st.selectbox(
            "Geography",
            [
                "France",
                "Germany",
                "Spain",
            ],
        )


    with col2:

        balance = st.number_input(
            "Balance",
            min_value=0.0,
            max_value=1_000_000.0,
            value=100000.0,
            step=1000.0,
        )

        products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=1,
        )

        has_card = st.selectbox(
            "Has Credit Card",
            [
                "Yes",
                "No",
            ],
        )

        active_member = st.selectbox(
            "Is Active Member",
            [
                "Yes",
                "No",
            ],
        )

        salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            max_value=1_000_000.0,
            value=100000.0,
            step=1000.0,
        )


    calculate = st.button(
        "Calculate Churn Risk",
        type="primary",
        use_container_width=True,
    )


    if calculate:

        customer = {
            "Year": 2025,
            "CreditScore": credit_score,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": (
                1 if has_card == "Yes" else 0
            ),
            "IsActiveMember": (
                1
                if active_member == "Yes"
                else 0
            ),
            "EstimatedSalary": salary,
            "Geography": geography,
        }


        try:

            result = predict_customer(
                customer
            )

            st.markdown("---")

            st.subheader(
                "Prediction Result"
            )

            display_risk_result(result)


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Churn Probability",
                    f"{result['percentage']:.2f}%",
                )

            with col2:

                st.metric(
                    "Risk Band",
                    result["risk_band"],
                )

            with col3:

                st.metric(
                    "Classification",
                    result["status"],
                )


            st.subheader(
                "Suggested Business Actions"
            )

            recommendations = (
                get_recommendations(
                    customer,
                    result["probability"],
                )
            )

            for item in recommendations:

                st.write(
                    f"• {item}"
                )


            st.info(
                f"Operating threshold: "
                f"{result['threshold']:.2f}. "
                "A probability at or above this threshold "
                "is classified as likely churn."
            )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# ============================================================
# PAGE 3 - PORTFOLIO INSIGHTS
# ============================================================

elif page == "Portfolio Insights":

    st.title(
        "📊 Portfolio Churn Insights"
    )

    st.write(
        "Explore historical churn and model-predicted "
        "risk across customer segments."
    )


    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        selected_geo = st.multiselect(
            "Geography",
            sorted(
                scored_df["Geography"]
                .dropna()
                .unique()
            ),
            default=sorted(
                scored_df["Geography"]
                .dropna()
                .unique()
            ),
        )


    with col2:

        selected_gender = st.multiselect(
            "Gender",
            sorted(
                scored_df["Gender"]
                .dropna()
                .unique()
            ),
            default=sorted(
                scored_df["Gender"]
                .dropna()
                .unique()
            ),
        )


    with col3:

        selected_activity = st.multiselect(
            "Activity",
            [
                "Active",
                "Inactive",
            ],
            default=[
                "Active",
                "Inactive",
            ],
        )


    with col4:

        selected_risk = st.multiselect(
            "Risk Band",
            [
                "Low Risk",
                "Medium Risk",
                "High Risk",
            ],
            default=[
                "Low Risk",
                "Medium Risk",
                "High Risk",
            ],
        )


    filtered = scored_df.copy()


    filtered = filtered[
        filtered["Geography"].isin(
            selected_geo
        )
    ]

    filtered = filtered[
        filtered["Gender"].isin(
            selected_gender
        )
    ]

    filtered["Activity_Label"] = np.where(
        filtered["IsActiveMember"] == 1,
        "Active",
        "Inactive",
    )

    filtered = filtered[
        filtered["Activity_Label"].isin(
            selected_activity
        )
    ]

    filtered = filtered[
        filtered["Risk_Band"].isin(
            selected_risk
        )
    ]


    # --------------------------------------------------------
    # FILTERED KPIs
    # --------------------------------------------------------

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Filtered Customers",
        f"{len(filtered):,}",
    )

    c2.metric(
        "Historical Churn",
        f"{filtered['Exited'].mean() * 100:.2f}%",
    )

    c3.metric(
        "Average Predicted Risk",
        f"{filtered['Churn_Probability_Pct'].mean():.2f}%",
    )

    c4.metric(
        "High-Risk Customers",
        f"{(filtered['Predicted_Churn'] == 1).sum():,}",
    )


    # --------------------------------------------------------
    # PROBABILITY DISTRIBUTION
    # --------------------------------------------------------

    fig = px.histogram(
        filtered,
        x="Churn_Probability",
        nbins=25,
        range_x=[0, 1],
        title="Predicted Churn Probability Distribution",
    )

    fig.update_xaxes(
        tickformat=".0%",
        title="Predicted Churn Probability",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # CHURN BY PRODUCTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        product_summary = (
            filtered.groupby(
                "NumOfProducts"
            )
            .agg(
                Customers=(
                    "NumOfProducts",
                    "size",
                ),
                Churn_Rate=(
                    "Exited",
                    "mean",
                ),
            )
            .reset_index()
        )

        product_summary["Churn_Rate"] *= 100

        fig = px.bar(
            product_summary,
            x="NumOfProducts",
            y="Churn_Rate",
            text_auto=".1f",
            title="Historical Churn by Number of Products",
        )

        fig.update_yaxes(
            title="Churn Rate (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    with col2:

        activity_summary = (
            filtered.groupby(
                "Activity_Label"
            )
            .agg(
                Customers=(
                    "Activity_Label",
                    "size",
                ),
                Churn_Rate=(
                    "Exited",
                    "mean",
                ),
            )
            .reset_index()
        )

        activity_summary["Churn_Rate"] *= 100

        fig = px.bar(
            activity_summary,
            x="Activity_Label",
            y="Churn_Rate",
            text_auto=".1f",
            title="Historical Churn by Customer Activity",
        )

        fig.update_yaxes(
            title="Churn Rate (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # GEOGRAPHY
    # --------------------------------------------------------

    geo_summary = (
        filtered.groupby(
            "Geography"
        )
        .agg(
            Customers=(
                "Geography",
                "size",
            ),
            Churn_Rate=(
                "Exited",
                "mean",
            ),
            Avg_Risk=(
                "Churn_Probability",
                "mean",
            ),
        )
        .reset_index()
    )

    geo_summary["Churn_Rate"] *= 100
    geo_summary["Avg_Risk"] *= 100


    st.subheader(
        "Geography Risk Overview"
    )

    st.dataframe(
        geo_summary,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE 4 - FEATURE IMPORTANCE
# ============================================================

elif page == "Feature Importance":

    st.title(
        "🔍 Feature Importance Dashboard"
    )

    st.write(
        "Global feature importance from the final "
        "Gradient Boosting model."
    )


    importance = get_feature_importance()


    fig = px.bar(
        importance.head(12)
        .sort_values(
            "Importance",
            ascending=True,
        ),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Churn Prediction Drivers",
        text_auto=".3f",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    st.subheader(
        "Feature Importance Table"
    )

    display_importance = importance.copy()

    display_importance[
        "Importance"
    ] = display_importance[
        "Importance"
    ].round(4)

    st.dataframe(
        display_importance,
        use_container_width=True,
        hide_index=True,
    )


    st.markdown("---")


    st.subheader(
        "Business Interpretation"
    )


    business_drivers = {
        "Age":
            "Age is the strongest model feature. "
            "Use age as an important segmentation variable "
            "for retention campaigns.",

        "NumOfProducts":
            "Product count has a strong model contribution. "
            "Customers with multiple products should receive "
            "relationship-level monitoring.",

        "IsActiveMember":
            "Customer activity is an important behavioral "
            "signal. Inactive customers deserve additional "
            "engagement attention.",

        "Geography_Germany":
            "German geography is an important segment indicator "
            "and may justify region-specific retention analysis.",

        "Balance":
            "Balance contributes to risk but should not be "
            "used alone as a churn indicator.",
    }


    for feature, explanation in business_drivers.items():

        st.markdown(
            f"**{feature}**  \n"
            f"{explanation}"
        )


    st.info(
        "Feature importance describes how strongly the model "
        "uses a feature. It does not prove that the feature "
        "causes churn."
    )


# ============================================================
# PAGE 5 - WHAT-IF SIMULATOR
# ============================================================

elif page == "What-If Simulator":

    st.title(
        "🧪 What-If Churn Simulator"
    )

    st.write(
        "Change customer characteristics and observe "
        "how the model's churn probability changes."
    )


    # --------------------------------------------------------
    # SELECT CUSTOMER
    # --------------------------------------------------------

    raw = standardize_raw_data(
        get_data()
    )


    customer_ids = (
        raw["Customer_id"]
        .astype(str)
        .tolist()
    )


    selected_customer_id = st.selectbox(
        "Select Customer",
        customer_ids,
    )


    selected_row = raw[
        raw["Customer_id"].astype(str)
        == selected_customer_id
    ].iloc[0]


    baseline = {
        "Year": int(selected_row["Year"]),
        "CreditScore": float(
            selected_row["CreditScore"]
        ),
        "Gender": selected_row["Gender"],
        "Age": float(
            selected_row["Age"]
        ),
        "Tenure": float(
            selected_row["Tenure"]
        ),
        "Balance": float(
            selected_row["Balance"]
        ),
        "NumOfProducts": int(
            selected_row["NumOfProducts"]
        ),
        "HasCrCard": int(
            selected_row["HasCrCard"]
        ),
        "IsActiveMember": int(
            selected_row["IsActiveMember"]
        ),
        "EstimatedSalary": float(
            selected_row["EstimatedSalary"]
        ),
        "Geography": selected_row["Geography"],
    }


    baseline_result = predict_customer(
        baseline
    )


    st.subheader(
        "Current Customer"
    )


    c1, c2, c3 = st.columns(3)


    c1.metric(
        "Current Risk",
        f"{baseline_result['percentage']:.2f}%",
    )

    c2.metric(
        "Current Band",
        baseline_result["risk_band"],
    )

    c3.metric(
        "Current Status",
        baseline_result["status"],
    )


    st.markdown("---")


    st.subheader(
        "Adjust Scenario"
    )


    col1, col2 = st.columns(2)


    with col1:

        scenario_age = st.slider(
            "Age",
            min_value=18,
            max_value=100,
            value=int(
                baseline["Age"]
            ),
        )

        scenario_products = st.slider(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=int(
                baseline["NumOfProducts"]
            ),
        )

        scenario_active = st.selectbox(
            "Active Member",
            [
                "Yes",
                "No",
            ],
            index=(
                0
                if baseline["IsActiveMember"] == 1
                else 1
            ),
        )


    with col2:

        scenario_balance = st.number_input(
            "Balance",
            min_value=0.0,
            max_value=1_000_000.0,
            value=float(
                baseline["Balance"]
            ),
            step=1000.0,
        )

        scenario_tenure = st.slider(
            "Tenure",
            min_value=0,
            max_value=20,
            value=int(
                baseline["Tenure"]
            ),
        )

        scenario_geography = st.selectbox(
            "Geography",
            [
                "France",
                "Germany",
                "Spain",
            ],
            index=[
                "France",
                "Germany",
                "Spain",
            ].index(
                baseline["Geography"]
            ),
        )


    scenario = baseline.copy()

    scenario["Age"] = scenario_age
    scenario["NumOfProducts"] = (
        scenario_products
    )

    scenario["IsActiveMember"] = (
        1
        if scenario_active == "Yes"
        else 0
    )

    scenario["Balance"] = (
        scenario_balance
    )

    scenario["Tenure"] = (
        scenario_tenure
    )

    scenario["Geography"] = (
        scenario_geography
    )


    try:

        scenario_result = predict_customer(
            scenario
        )

        st.markdown("---")

        st.subheader(
            "Scenario Result"
        )


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "Baseline Risk",
            f"{baseline_result['percentage']:.2f}%",
        )

        c2.metric(
            "Scenario Risk",
            f"{scenario_result['percentage']:.2f}%",
        )

        change = (
            scenario_result["percentage"]
            -
            baseline_result["percentage"]
        )


        c3.metric(
            "Risk Change",
            f"{change:+.2f} pp",
        )


        if change < 0:

            st.success(
                "The scenario reduces the predicted "
                "churn probability."
            )

        elif change > 0:

            st.warning(
                "The scenario increases the predicted "
                "churn probability."
            )

        else:

            st.info(
                "The scenario produces the same "
                "predicted churn probability."
            )


        st.subheader(
            "Scenario Recommendation"
        )


        scenario_recommendations = (
            get_recommendations(
                scenario,
                scenario_result["probability"],
            )
        )


        for item in scenario_recommendations:

            st.write(
                f"• {item}"
            )


        # ----------------------------------------------------
        # SIMPLE COMPARISON
        # ----------------------------------------------------

        comparison = pd.DataFrame(
            {
                "Metric": [
                    "Age",
                    "Products",
                    "Active Member",
                    "Balance",
                    "Tenure",
                    "Geography",
                ],
                "Baseline": [
                    baseline["Age"],
                    baseline["NumOfProducts"],
                    (
                        "Yes"
                        if baseline["IsActiveMember"] == 1
                        else "No"
                    ),
                    baseline["Balance"],
                    baseline["Tenure"],
                    baseline["Geography"],
                ],
                "Scenario": [
                    scenario["Age"],
                    scenario["NumOfProducts"],
                    scenario_active,
                    scenario["Balance"],
                    scenario["Tenure"],
                    scenario["Geography"],
                ],
            }
        )


        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )


    except Exception as e:

        st.error(
            "What-if prediction failed."
        )

        st.exception(e)


# ============================================================
# PAGE 6 - MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title(
        "🤖 Model Performance"
    )

    st.write(
        "Performance of the final Gradient Boosting model "
        "selected in the notebook."
    )


    # --------------------------------------------------------
    # FINAL MODEL METRICS FROM NOTEBOOK
    # --------------------------------------------------------

    metrics = {
        "Accuracy": 0.8670,
        "Precision": 0.7439,
        "Recall": 0.5283,
        "F1 Score": 0.6178,
        "ROC-AUC": 0.8648,
    }


    c1, c2, c3, c4, c5 = st.columns(5)


    c1.metric(
        "Accuracy",
        "86.70%",
    )

    c2.metric(
        "Precision",
        "74.39%",
    )

    c3.metric(
        "Recall",
        "52.83%",
    )

    c4.metric(
        "F1 Score",
        "61.78%",
    )

    c5.metric(
        "ROC-AUC",
        "86.48%",
    )


    st.markdown("---")


    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "XGBoost",
                "Gradient Boosting",
            ],
            "Accuracy": [
                0.7145,
                0.7655,
                0.8445,
                0.8455,
                0.8310,
            ],
            "Precision": [
                0.3883,
                0.4365,
                0.6224,
                0.6476,
                0.5711,
            ],
            "Recall": [
                0.7002,
                0.5233,
                0.5995,
                0.5283,
                0.6806,
            ],
            "F1 Score": [
                0.4996,
                0.4760,
                0.6108,
                0.5819,
                0.6211,
            ],
            "ROC-AUC": [
                0.7757,
                0.6754,
                0.8470,
                0.8427,
                0.8648,
            ],
        }
    )


    st.subheader(
        "Model Comparison"
    )

    st.dataframe(
        comparison.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}",
                "ROC-AUC": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


    # ROC-AUC chart

    fig = px.bar(
        comparison.sort_values(
            "ROC-AUC"
        ),
        x="ROC-AUC",
        y="Model",
        orientation="h",
        title="ROC-AUC Comparison",
        text_auto=".3f",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    st.subheader(
        "Final Operating Threshold"
    )

    st.metric(
        "Threshold",
        f"{threshold:.2f}",
    )

    st.info(
        "The notebook reported an F1-maximizing threshold "
        "of 0.562 earlier, but the final saved operating "
        "threshold is 0.70. The deployed application uses "
        "the saved 0.70 threshold."
    )


    st.warning(
        "The final model has stronger precision at the "
        "0.70 threshold but lower recall. This means the "
        "operating point favors more targeted churn alerts."
    )


# ============================================================
# PAGE 7 - BUSINESS RECOMMENDATIONS
# ============================================================

elif page == "Business Recommendations":

    st.title(
        "💼 Business Recommendations"
    )

    st.write(
        "Retention recommendations derived from the "
        "analysis and explainability results in the notebook."
    )


    recommendations = [
        (
            "1. Focus on Older Customers",
            "Age was identified as the strongest model driver. "
            "Develop personalized retention programs and "
            "loyalty initiatives for older customers."
        ),
        (
            "2. Increase Customer Engagement",
            "Inactive customers showed stronger churn tendency. "
            "Promote digital banking usage, personalized "
            "notifications and engagement programs."
        ),
        (
            "3. Monitor Multiple-Product Customers",
            "Customers with multiple banking products showed "
            "elevated churn risk in the analysis. Review their "
            "relationship and consider customized bundles."
        ),
        (
            "4. Strengthen Retention in Germany",
            "German customers showed a higher churn tendency. "
            "Use regional customer analysis and targeted "
            "retention strategies."
        ),
        (
            "5. Prioritize High-Risk Customers",
            "Use model-generated churn probabilities to flag "
            "customers for proactive retention intervention."
        ),
    ]


    for title, description in recommendations:

        with st.expander(title):

            st.write(
                description
            )


    st.markdown("---")


    st.subheader(
        "Recommended Retention Workflow"
    )


    st.write(
        """
        1. Identify customers with high predicted churn probability.

        2. Prioritize customers above the operating threshold.

        3. Review the customer's main behavioral and demographic indicators.

        4. Select an appropriate retention action.

        5. Use the What-If Simulator to assess how customer
           engagement or product changes affect predicted risk.

        6. Track the customer's future behavior and retention outcome.
        """
    )


    st.info(
        "These recommendations are decision-support suggestions "
        "based on the project's observed patterns. They should "
        "not be treated as causal conclusions."
    )