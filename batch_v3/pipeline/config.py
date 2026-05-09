# ==============================================================================
# Centralized Configuration for SARIMAX Cashflow Pipeline (v3)
# Single (store_nbr, family) model — no aggregation or disaggregation.
# ==============================================================================

PROJECT_ID = "project-sandbox-357505"
LOCATION = "asia-southeast2"
BUCKET_URI = "gs://vertex-dump"

MODEL_DISPLAY_NAME = "sarimax-single"

# GCS paths to dataset files
GCS_TRAIN_CSV = f"{BUCKET_URI}/sales_forecast/train.csv"
GCS_TEST_CSV = f"{BUCKET_URI}/sales_forecast/test.csv"
GCS_OIL_CSV = f"{BUCKET_URI}/sales_forecast/oil.csv"
GCS_HOLIDAYS_CSV = f"{BUCKET_URI}/sales_forecast/holidays_events.csv"

# Pipeline and batch prediction paths
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/sarimax_v3"
BATCH_INPUT_URI = f"{BUCKET_URI}/batch_predictions/sarimax_v3/input/"
BATCH_OUTPUT_DIR = f"{BUCKET_URI}/batch_predictions/sarimax_v3/output/"

FORECAST_HORIZON = 15

# Target store×family combination to model
TARGET_STORE_NBR = 1
TARGET_FAMILY = "BEVERAGES"

# Exogenous feature columns used by the model
EXOG_COLS = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]
