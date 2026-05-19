
import json
import joblib
import pandas as pd


class CustomerSegmentationModel:
    def __init__(self, model_path, label_path):
        self.model = joblib.load(model_path)

        with open(label_path, "r") as f:
            self.labels = json.load(f)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Predict clusters
        clusters = self.model.predict(df)

        # Add predictions
        df["Cluster"] = clusters
        df["Cluster_Name"] = (
            df["Cluster"]
            .astype(str)
            .map(self.labels)
        )

        return df