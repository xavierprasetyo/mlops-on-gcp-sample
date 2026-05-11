import os
import pickle
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class SarimaxPredictor:
    """CPR predictor for single (store_nbr, family) SARIMAX model.

    Loads the pre-filtered training history from the model artifact directory
    to cure amnesia — no large GCS downloads at startup.
    """

    def __init__(self):
        self.model = None
        self.updated_model = None

    def load(self, artifacts_uri: str):
        """Load SARIMAX model and cure amnesia with bundled training history."""
        logger.info(f"Loading artifacts from {artifacts_uri}")

        model_path = os.path.join(artifacts_uri, "model.pkl")
        history_path = os.path.join(artifacts_uri, "train_history.csv")
        local_model_path = "/tmp/model.pkl"
        local_history_path = "/tmp/train_history.csv"

        # Download from GCS if needed
        if model_path.startswith("gs://"):
            import fsspec
            fs = fsspec.filesystem("gs")
            logger.info(f"Downloading model from {model_path}")
            fs.get(model_path, local_model_path)
            logger.info(f"Downloading training history from {history_path}")
            fs.get(history_path, local_history_path)
            model_path = local_model_path
            history_path = local_history_path

        # Load SARIMAX model
        logger.info(f"Loading SARIMAX model from {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Model loaded successfully.")

        # Cure amnesia using pre-saved training history
        logger.info("Curing model amnesia with bundled training history...")
        history_df = pd.read_csv(history_path)
        logger.info(f"Training history: {len(history_df)} rows")

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
