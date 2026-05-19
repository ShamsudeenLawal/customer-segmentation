import pandas as pd
import streamlit as st

from utils import CustomerSegmentationModel


# =========================================================
# LOAD MODEL
# =========================================================
model = CustomerSegmentationModel(
    model_path="models/kmeans_model.joblib",
    label_path="models/cluster_labels.json"
)


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Customer Segmentation App",
    layout="wide"
)

st.title("🧠 Customer Segmentation App")

st.write(
    """
    Predict customer segments individually or upload CSV files
    for batch customer segmentation and campaign targeting.
    """
)


# =========================================================
# SIDEBAR MODE SELECTION
# =========================================================
mode = st.sidebar.radio(
    "Choose Prediction Mode",
    ["Single Prediction", "Batch Prediction"]
)


# =========================================================
# FEATURES
# =========================================================
FEATURE_COLUMNS = [
    "Age",
    "Income",
    "TotalSpending",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Recency"
]


# =========================================================
# SINGLE PREDICTION MODE
# =========================================================
if mode == "Single Prediction":

    st.subheader("Single Customer Prediction")

    customer_id = st.text_input(
        "Customer ID",
        value="001"
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        income = st.number_input(
            "Income",
            min_value=0,
            max_value=500000,
            value=50000
        )

        total_spending = st.number_input(
            "Total Spending",
            min_value=0,
            max_value=10000,
            value=1000
        )

        num_web_purchases = st.number_input(
            "Web Purchases",
            min_value=0,
            max_value=100,
            value=10
        )

    with col2:

        num_store_purchases = st.number_input(
            "Store Purchases",
            min_value=0,
            max_value=100,
            value=10
        )

        num_web_visits = st.number_input(
            "Monthly Web Visits",
            min_value=0,
            max_value=50,
            value=5
        )

        recency = st.number_input(
            "Recency",
            min_value=0,
            max_value=365,
            value=30
        )

    # Input dataframe
    input_data = pd.DataFrame({
        "ID": [customer_id],
        "Age": [age],
        "Income": [income],
        "TotalSpending": [total_spending],
        "NumWebPurchases": [num_web_purchases],
        "NumStorePurchases": [num_store_purchases],
        "NumWebVisitsMonth": [num_web_visits],
        "Recency": [recency]
    })

    if st.button("Predict Segment"):

        # Keep customer ID separately
        customer_ids = input_data["ID"]

        # Only prediction features
        prediction_features = input_data[FEATURE_COLUMNS]

        # Predict
        prediction = model.predict(prediction_features)

        # Reattach customer ID
        prediction.insert(0, "ID", customer_ids.values)

        cluster_name = prediction.iloc[0]["Cluster_Name"]

        st.success(f"Predicted Segment: {cluster_name}")

        st.write("### Prediction Result")
        st.dataframe(prediction)


# =========================================================
# BATCH PREDICTION MODE
# =========================================================
else:

    st.subheader("Batch Prediction")

    st.write(
        """
        Upload a CSV file containing customer IDs and feature columns
        for bulk segmentation and campaign targeting.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    st.markdown("### Required Columns")

    REQUIRED_COLUMNS = ["ID"] + FEATURE_COLUMNS

    st.code(
        ", ".join(REQUIRED_COLUMNS),
        language="text"
    )

    st.markdown("### Example CSV Format")

    example_df = pd.DataFrame({
        "ID": ["001", "002"],
        "Age": [35, 42],
        "Income": [50000, 80000],
        "TotalSpending": [1000, 2500],
        "NumWebPurchases": [10, 15],
        "NumStorePurchases": [5, 8],
        "NumWebVisitsMonth": [3, 5],
        "Recency": [20, 10]
    })

    st.dataframe(example_df)

    if uploaded_file is not None:

        try:
            batch_df = pd.read_csv(uploaded_file)

            st.write("### Uploaded Data")
            st.dataframe(batch_df.head(3))

            # Validate columns
            missing_cols = [
                col for col in REQUIRED_COLUMNS
                if col not in batch_df.columns
            ]

            if missing_cols:

                st.error(
                    f"Missing required columns: {missing_cols}"
                )

            else:

                # Preserve customer IDs
                customer_ids = batch_df["ID"]

                # Only model features
                prediction_features = batch_df[FEATURE_COLUMNS]

                # Predict
                predictions = model.predict(
                    prediction_features
                )

                # Add customer IDs back
                predictions.insert(
                    0,
                    "ID",
                    customer_ids.values
                )

                st.write("### Prediction Results")

                st.dataframe(predictions.head(3))

                # # Segment distribution
                # st.write("### Segment Distribution")

                # segment_counts = (
                #     predictions["Cluster_Name"]
                #     .value_counts()
                # )

                # st.bar_chart(segment_counts)

                # Download predictions
                csv = predictions.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="customer_segment_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(
                f"Error processing file: {e}"
            )