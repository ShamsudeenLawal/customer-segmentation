import os
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer


# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "data/customer_segmentation.csv"

MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/kmeans_model.joblib"
LABELS_PATH = f"{MODEL_DIR}/cluster_labels.json"

TEST_DATA_DIR = "data/test"
TEST_DATA_PATH = f"{TEST_DATA_DIR}/test_data.csv"

RANDOM_STATE = 42
N_CLUSTERS = 3

FEATURES = [
    "Age",
    "Income",
    "TotalSpending",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Recency"
]


# =========================================================
# 1. LOAD DATA
# =========================================================
def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV.
    """
    return pd.read_csv(path)


# =========================================================
# 2. CLEAN DATA
# =========================================================
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning operations.
    """
    df = df.copy()

    df["Dt_Customer"] = pd.to_datetime(
        df["Dt_Customer"],
        dayfirst=True
    )

    # remove missing rows
    df = df.dropna()

    return df


# =========================================================
# 3. FEATURE ENGINEERING
# =========================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create modeling features.
    """
    df = df.copy()

    current_year = pd.Timestamp.today().year

    # Customer age
    df["Age"] = current_year - df["Year_Birth"]

    # Total spending
    spend_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds"
    ]

    df["TotalSpending"] = df[spend_cols].sum(axis=1)

    return df


# =========================================================
# 4. TRAIN MODEL
# =========================================================
def train_model(
    df: pd.DataFrame,
    save_sample_data: bool = False
):
    """
    Train customer segmentation model.
    """

    X = df[["ID"] + FEATURES].copy()

    # save sample inference data
    if save_sample_data:

        os.makedirs(TEST_DATA_DIR, exist_ok=True)

        sample_data = X.head(20)

        sample_data.to_csv(
            TEST_DATA_PATH,
            index=False
        )

        print(f"\nSample test data saved to: {TEST_DATA_PATH}")

    feature_data = X[FEATURES]  # ONLY FEATURES for model

    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10))
    ])

    clusters = pipeline.fit_predict(feature_data)

    X["Cluster"] = clusters

    return pipeline, X

# =========================================================
# 5. CLUSTER PROFILES
# =========================================================
def get_cluster_profiles(df: pd.DataFrame):
    """
    Generate cluster statistics.
    """

    cluster_profile = df.groupby(
        "Cluster"
    ).mean(numeric_only=True)

    global_profile = df.mean(
        numeric_only=True
    )

    return cluster_profile, global_profile


# =========================================================
# 6. CLUSTER SCORING
# =========================================================
def compute_cluster_scores(profile, global_profile):

    value_score = (
        profile["TotalSpending"] /
        global_profile["TotalSpending"]

        +

        profile["Income"] /
        global_profile["Income"]

    ) / 2

    engagement_score = (
        global_profile["Recency"] /
        profile["Recency"]
    )

    digital_score = (
        profile["NumWebPurchases"] /
        (profile["NumStorePurchases"] + 1)
    )

    return {
        "value_score": value_score,
        "engagement_score": engagement_score,
        "digital_score": digital_score
    }


# =========================================================
# 7. ASSIGN CLUSTER LABELS
# =========================================================
def assign_cluster_names(
    cluster_profile,
    global_profile
):
    """
    Assign business-friendly cluster names.
    """

    labels = {}

    for cluster_id in cluster_profile.index:

        scores = compute_cluster_scores(
            cluster_profile.loc[cluster_id],
            global_profile
        )

        value = scores["value_score"]
        engagement = scores["engagement_score"]
        digital = scores["digital_score"]

        # -----------------------------
        # BUSINESS RULES
        # -----------------------------
        if value >= 1.2 and engagement >= 1.0:

            label = "High-Value Loyal Customers"

        elif value >= 1.2 and engagement < 1.0:

            label = "High-Value At-Risk Customers"

        elif value < 1.2 and digital >= 1.0:

            label = "Digital Growth Customers"

        else:

            label = "Low-Value Traditional Customers"

        labels[cluster_id] = label

    return labels


# =========================================================
# 8. SAVE ARTIFACTS
# =========================================================
def save_artifacts(model, labels):
    """
    Save trained model and cluster labels.
    """

    os.makedirs(MODEL_DIR, exist_ok=True)

    # save model
    joblib.dump(model, MODEL_PATH)

    # save labels
    with open(LABELS_PATH, "w") as f:

        json.dump(
            labels,
            f,
            indent=4
        )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Cluster labels saved to: {LABELS_PATH}")


# =========================================================
# 9. MAIN PIPELINE
# =========================================================
def main():

    print("\nLoading dataset...")

    df = load_data(DATA_PATH)

    print("Cleaning dataset...")
    df = clean_dataset(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Training clustering model...")
    model, df_clustered = train_model(
        df,
        save_sample_data=True
    )

    print("Generating cluster profiles...")
    cluster_profile, global_profile = get_cluster_profiles(
        df_clustered
    )

    print("Assigning cluster labels...")
    cluster_labels = assign_cluster_names(
        cluster_profile,
        global_profile
    )

    # attach names
    df_clustered["Cluster_Name"] = (
        df_clustered["Cluster"]
        .map(cluster_labels)
    )

    print("Saving artifacts...")
    save_artifacts(
        model,
        cluster_labels
    )

    print("\nCluster Labels:")
    print(cluster_labels)

    print("\nCluster Distribution:")
    print(
        df_clustered["Cluster_Name"]
        .value_counts()
    )

    print("\nSample Predictions:")
    print(df_clustered.head())


# =========================================================
# RUN SCRIPT
# =========================================================
if __name__ == "__main__":
    main()