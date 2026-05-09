import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class SarimaxPredictor:
    """CPR predictor for single (store_nbr, family) SARIMAX model.

    No disaggregation — predictions are direct forecasts for the target combo.
    """

    def __init__(self):
        self.model = None
        self.updated_model = None

    def load(self, artifacts_uri: str):
        """Load SARIMAX model and cure amnesia with training history."""
        logger.info(f"Loading artifacts from {artifacts_uri}")

        model_path = os.path.join(artifacts_uri, "model.pkl")
        local_model_path = "/tmp/model.pkl"

        # Download from GCS if needed
        if model_path.startswith("gs://"):
            import fsspec
            fs = fsspec.filesystem("gs")
            logger.info(f"Downloading model from {model_path}")
            fs.get(model_path, local_model_path)
            model_path = local_model_path

        # Load SARIMAX model
        from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
        self.model = SARIMAXResultsWrapper.load(model_path)
        logger.info("Model loaded.")

        # Cure amnesia: re-apply training history for the target store×family
        logger.info("Curing model amnesia...")
        BUCKET_URI = os.environ.get("BUCKET_URI", "gs://vertex-dump")
        TARGET_STORE_NBR = int(os.environ.get("TARGET_STORE_NBR", "1"))
        TARGET_FAMILY = os.environ.get("TARGET_FAMILY", "BEVERAGES")

        train_df = pd.read_csv(f"{BUCKET_URI}/sales_forecast/train.csv")
        oil_df = pd.read_csv(f"{BUCKET_URI}/sales_forecast/oil.csv")
        hol_df = pd.read_csv(f"{BUCKET_URI}/sales_forecast/holidays_events.csv")

        # Filter to target store×family
        filtered = train_df[
            (train_df["store_nbr"] == TARGET_STORE_NBR)
            & (train_df["family"] == TARGET_FAMILY)
        ].copy()
        filtered["date"] = pd.to_datetime(filtered["date"])
        filtered = filtered.sort_values("date")

        # Merge exog features
        oil_df["date"] = pd.to_datetime(oil_df["date"])
        oil_df.rename(columns={"dcoilwtico": "oil_price"}, inplace=True)
        oil_df["oil_price"] = oil_df["oil_price"].ffill().bfill()

        hol_df["date"] = pd.to_datetime(hol_df["date"])
        valid_holidays = hol_df[
            (hol_df["transferred"] == False) & (hol_df["type"] != "Work Day")
        ].copy()
        valid_holidays["is_holiday"] = 1
        holiday_flags = valid_holidays[["date", "is_holiday"]].drop_duplicates()

        history_df = filtered.merge(oil_df[["date", "oil_price"]], on="date", how="left")
        history_df = history_df.merge(holiday_flags, on="date", how="left")
        history_df["is_holiday"] = history_df["is_holiday"].fillna(0)
        history_df["oil_price"] = history_df["oil_price"].ffill().bfill()
        history_df["day_of_week"] = history_df["date"].dt.dayofweek

        exog_cols = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]
        self.updated_model = self.model.apply(
            endog=history_df["sales"].values,
            exog=history_df[exog_cols].values,
        )
        logger.info("Model amnesia cured!")

    def preprocess(self, prediction_input: dict) -> list:
        """Extract instances from Vertex AI request."""
        return prediction_input.get("instances", [])

    def predict(self, instances: list) -> list:
        """Forecast using exogenous features.

        Each instance is [onpromotion, oil_price, is_holiday, day_of_week].
        """
        import numpy as np
        steps = len(instances)
        exog = np.array(instances)

        forecast = self.updated_model.forecast(steps=steps, exog=exog)
        return [max(0, float(x)) for x in forecast]

    def postprocess(self, prediction_results: list) -> dict:
        """Format output for Vertex AI."""
        return {"predictions": prediction_results}
