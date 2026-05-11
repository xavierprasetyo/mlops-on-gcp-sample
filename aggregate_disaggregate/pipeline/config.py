# ==============================================================================
# Centralized Configuration for SARIMAX Cashflow Pipeline
# ==============================================================================

PROJECT_ID = "project-sandbox-357505"
PROJECT_NUMBER = "1054726275594"
LOCATION = "asia-southeast2"
BUCKET_URI = "gs://vertex-dump"

MODEL_DISPLAY_NAME = "cashflow-sarimax"
EXPERIMENT_NAME = "sarimax-cashflow-aggregate"

# GCS paths to dataset files
GCS_TRAIN_CSV = f"{BUCKET_URI}/sales_forecast/train.csv"
GCS_TEST_CSV = f"{BUCKET_URI}/sales_forecast/test.csv"
GCS_OIL_CSV = f"{BUCKET_URI}/sales_forecast/oil.csv"
GCS_HOLIDAYS_CSV = f"{BUCKET_URI}/sales_forecast/holidays_events.csv"

# Pipeline and batch prediction paths
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/sarimax"
BATCH_INPUT_URI = f"{BUCKET_URI}/batch_predictions/sarimax/input/"
BATCH_OUTPUT_DIR = f"{BUCKET_URI}/batch_predictions/sarimax/output/"

FORECAST_HORIZON = 15

# Hyperparameter search space for SARIMAX
# Total combinations: 3×2×3 × 2×2×2 = 144
HYPERPARAM_SEARCH_SPACE = {
    "p": [0, 1, 2],
    "d": [0, 1],
    "q": [0, 1, 2],
    "P": [0, 1],
    "D": [0, 1],
    "Q": [0, 1],
    "s": 7,  # Fixed weekly seasonality
}

# Exogenous feature columns used by the model
EXOG_COLS = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]
