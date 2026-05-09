import os
import pickle
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class SarimaxPredictor:
    """Custom Prediction Routine for Seasonal ARIMAX with proportional disaggregation.

    At startup (load), this predictor:
      1. Loads the SARIMAX model from the artifact directory.
      2. Loads store×family proportions for disaggregation.
      3. Re-applies full training history to cure model amnesia.

    At prediction time (predict), it:
      1. Extracts unique forecast dates from the input instances.
      2. Forecasts aggregated daily totals using SARIMAX.
      3. Disaggregates totals to per-store-per-family predictions using proportions.
    """

    def __init__(self):
        self.model = None
        self.updated_model = None
        self.proportions = None
        self.daily_forecasts = None

    def load(self, artifacts_uri: str):
        """Load model, proportions, and cure amnesia with full history."""
        logger.info(f"Loading artifacts from {artifacts_uri}")

        # Use google.cloud.storage directly for all GCS operations
        # (fsspec/gcsfs auth may not work reliably in Vertex AI batch containers)
        from google.cloud import storage as gcs_storage

        def _download_gcs_file(gcs_uri: str, local_path: str) -> str:
            """Download a file from GCS to a local path. Returns the local path."""
            path = gcs_uri.replace("gs://", "")
            bucket_name = path.split("/")[0]
            blob_name = "/".join(path.split("/")[1:])
            client = gcs_storage.Client()
            bucket = client.bucket(bucket_name)
            bucket.blob(blob_name).download_to_filename(local_path)
            logger.info("Downloaded %s -> %s", gcs_uri, local_path)
            return local_path

        local_model_path = "/tmp/model.pkl"
        local_proportions_path = "/tmp/proportions.csv"

        if artifacts_uri.startswith("gs://"):
            model_gcs = artifacts_uri.rstrip("/") + "/model.pkl"
            proportions_gcs = artifacts_uri.rstrip("/") + "/proportions.csv"
            _download_gcs_file(model_gcs, local_model_path)
            _download_gcs_file(proportions_gcs, local_proportions_path)
            model_path = local_model_path
            proportions_path = local_proportions_path
        else:
            model_path = os.path.join(artifacts_uri, "model.pkl")
            proportions_path = os.path.join(artifacts_uri, "proportions.csv")

        # Load SARIMAX model
        logger.info("Loading SARIMAXResults...")
        from statsmodels.iolib.smpickle import load_pickle

        self.model = load_pickle(model_path)
        logger.info("SARIMAX model loaded successfully.")

        # Load proportions
        self.proportions = pd.read_csv(proportions_path)
        logger.info(f"Loaded {len(self.proportions)} store×family proportions")

        # Cure model amnesia: re-apply full training history
        logger.info("Curing model amnesia with training history...")
        BUCKET_URI = os.environ.get("BUCKET_URI", "gs://vertex-dump")

        train_df = pd.read_csv(_download_gcs_file(f"{BUCKET_URI}/sales_forecast/train.csv", "/tmp/train.csv"))
        oil_df = pd.read_csv(_download_gcs_file(f"{BUCKET_URI}/sales_forecast/oil.csv", "/tmp/oil.csv"))
        hol_df = pd.read_csv(_download_gcs_file(f"{BUCKET_URI}/sales_forecast/holidays_events.csv", "/tmp/holidays.csv"))

        # Aggregate and merge (same logic as preprocess component)
        daily_df = (
            train_df.groupby("date")
            .agg({"sales": "sum", "onpromotion": "sum"})
            .reset_index()
        )
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df.rename(columns={"sales": "total_cashflow"}, inplace=True)

        oil_df["date"] = pd.to_datetime(oil_df["date"])
        oil_df.rename(columns={"dcoilwtico": "oil_price"}, inplace=True)
        oil_df["oil_price"] = oil_df["oil_price"].ffill().bfill()

        hol_df["date"] = pd.to_datetime(hol_df["date"])
        valid_holidays = hol_df[
            (hol_df["transferred"] == False) & (hol_df["type"] != "Work Day")
        ].copy()
        valid_holidays["is_holiday"] = 1
        holiday_flags = valid_holidays[["date", "is_holiday"]].drop_duplicates()

        history_df = daily_df.merge(oil_df, on="date", how="left")
        history_df = history_df.merge(holiday_flags, on="date", how="left")
        history_df["is_holiday"] = history_df["is_holiday"].fillna(0)
        history_df["oil_price"] = history_df["oil_price"].ffill().bfill()
        history_df["day_of_week"] = history_df["date"].dt.dayofweek
        history_df = history_df.sort_values("date")

        exog_cols = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]

        last_15 = history_df[["date", "total_cashflow"]].tail(15)
        logger.info("Last 15 days of history used for amnesia cure:\n%s", last_15.to_string(index=False))

        self.updated_model = self.model.apply(
            endog=history_df["total_cashflow"].values,
            exog=history_df[exog_cols].values,
        )
        logger.info("Model amnesia cured!")

    def preprocess(self, prediction_input: dict) -> pd.DataFrame:
        """Parse the Vertex AI batch prediction request.

        Handles both list-format and dict-format instances from Vertex AI.
        CSV inputs are sent as dicts keyed by column name; all values arrive as
        strings and must be cast to proper numeric types.
        """
        instances = prediction_input.get("instances", [])
        columns = [
            "id", "date", "store_nbr", "family",
            "onpromotion", "oil_price", "is_holiday", "day_of_week",
        ]
        if instances and isinstance(instances[0], dict):
            df = pd.DataFrame(instances)
        else:
            df = pd.DataFrame(instances, columns=columns)

        # Ensure numeric columns are properly typed (CSV->JSON sends strings)
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype(int)
        df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").astype(int)
        df["onpromotion"] = pd.to_numeric(df["onpromotion"], errors="coerce").astype(float)
        df["oil_price"] = pd.to_numeric(df["oil_price"], errors="coerce").astype(float)
        df["is_holiday"] = pd.to_numeric(df["is_holiday"], errors="coerce").astype(float)
        df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce").astype(int)

        logger.info("Preprocessed data (last 15 rows):\n%s", df.tail(15).to_string(index=False))
        return df

    def predict(self, instances_df: pd.DataFrame) -> list:
        """Forecast aggregated daily totals then disaggregate to per-row predictions."""
        exog_cols = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]

        # Get unique dates and their aggregated exog features
        instances_df["date"] = pd.to_datetime(instances_df["date"])
        daily_exog = (
            instances_df.groupby("date")
            .agg({
                "onpromotion": "sum",
                "oil_price": "first",
                "is_holiday": "first",
                "day_of_week": "first",
            })
            .sort_index()
            .reset_index()
        )

        # Forecast aggregated daily totals
        steps = len(daily_exog)
        forecast = self.updated_model.forecast(
            steps=steps, exog=daily_exog[exog_cols].values
        )

        # Map daily forecast to dates
        daily_exog["predicted_total"] = [max(0, x) for x in forecast]
        date_to_total = dict(
            zip(daily_exog["date"], daily_exog["predicted_total"])
        )

        logger.info("Daily forecast results (last 15 days):\n%s", daily_exog[["date", "predicted_total"]].tail(15).to_string(index=False))

        # Disaggregate: multiply each day's total by each store×family proportion
        # Merge proportions onto instances
        merged = instances_df.merge(
            self.proportions,
            on=["store_nbr", "family"],
            how="left",
        )
        # Default to 0 for unknown store×family combos
        merged["proportion"] = merged["proportion"].fillna(0)

        # Calculate per-row prediction
        merged["sales"] = merged.apply(
            lambda row: date_to_total.get(row["date"], 0) * row["proportion"],
            axis=1,
        )
        merged["sales"] = merged["sales"].apply(lambda x: max(0, x))

        logger.info("Disaggregated sales (last 15 rows):%s", merged[["date", "store_nbr", "family", "sales"]].tail(15).to_string(index=False))

        # Convert date to string for JSON serialization
        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

        # Return the full merged DataFrame as a list of dicts
        output_cols = ["id", "date", "store_nbr", "family", "onpromotion",
                       "oil_price", "is_holiday", "day_of_week", "proportion", "sales"]
        return merged[output_cols].to_dict(orient="records")

    def postprocess(self, prediction_results: list) -> dict:
        """Format output for Vertex AI batch prediction."""
        return {"predictions": prediction_results}
