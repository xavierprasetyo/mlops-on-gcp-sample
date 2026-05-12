import pandas as pd
import numpy as np
from kfp import dsl, compiler
from google.cloud import aiplatform

# ==============================================================================
# 1. CONFIGURATION (UPDATE THESE WITH YOUR GCP DETAILS)
# ==============================================================================
PROJECT_ID = "project-sandbox-357505"        # <-- Update this
LOCATION = "asia-southeast2"                  # <-- Update this
BUCKET_URI = "gs://vertex-dump"      # <-- Update this

# Paths to where your Kaggle files are (or will be) stored in GCS
GCS_TRAIN_CSV = f"{BUCKET_URI}/sales_forecast/train.csv"
GCS_OIL_CSV = f"{BUCKET_URI}/sales_forecast/oil.csv"
GCS_HOLIDAYS_CSV = f"{BUCKET_URI}/sales_forecast/holidays_events.csv"

# NEW: Where your final production batch predictions will be delivered
BATCH_OUTPUT_URI = f"{BUCKET_URI}/batch_predictions/latest_forecast.csv"
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root"

# Shared logger name across all components for easy Cloud Logging filtering
CLOUD_LOGGER_NAME = "cashflow-pipeline"
EXPERIMENT_NAME = "sarimax-cashflow-monolith"

# ==============================================================================
# 2. KFP COMPONENTS
# ==============================================================================

@dsl.component(
    base_image="python:3.13",
    packages_to_install=["pandas", "fsspec", "gcsfs", "google-cloud-logging"]
)
def preprocess_and_prep_future(
    train_csv_uri: str,
    oil_csv_uri: str,
    holidays_csv_uri: str,
    forecast_horizon: int,
    project_id: str,
    logger_name: str,
    train_data: dsl.Output[dsl.Dataset],
    test_data: dsl.Output[dsl.Dataset],
    future_data: dsl.Output[dsl.Dataset] # <-- NEW: Calendar for true future predictions
):
    import pandas as pd
    import time
    import traceback
    from google.cloud import logging as cloud_logging

    # --- Cloud Logging Setup ---
    log_client = cloud_logging.Client(project=project_id)
    logger = log_client.logger(logger_name)
    step_name = "preprocess_and_prep_future"

    logger.log_struct({
        "step": step_name, "severity": "INFO", "event": "STEP_STARTED",
        "params": {
            "train_csv_uri": train_csv_uri,
            "oil_csv_uri": oil_csv_uri,
            "holidays_csv_uri": holidays_csv_uri,
            "forecast_horizon": forecast_horizon,
        }
    }, severity="INFO")
    t0 = time.time()

    try:
        # --- Read raw data ---
        logger.log_struct({"step": step_name, "event": "READING_DATA", "detail": "Loading train, oil, holidays CSVs from GCS"}, severity="INFO")
        train_df = pd.read_csv(train_csv_uri)
        oil_df = pd.read_csv(oil_csv_uri)
        hol_df = pd.read_csv(holidays_csv_uri)
        logger.log_struct({
            "step": step_name, "event": "DATA_LOADED",
            "train_shape": list(train_df.shape),
            "oil_shape": list(oil_df.shape),
            "holidays_shape": list(hol_df.shape),
            "train_columns": list(train_df.columns),
            "train_nulls": train_df.isnull().sum().to_dict(),
        }, severity="INFO")

        # 1. Aggregate Historical Cashflow
        daily_df = train_df.groupby('date').agg({'sales': 'sum', 'onpromotion': 'sum'}).reset_index()
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df.rename(columns={'sales': 'total_cashflow'}, inplace=True)
        logger.log_struct({
            "step": step_name, "event": "AGGREGATION_COMPLETE",
            "daily_shape": list(daily_df.shape),
            "date_range": [str(daily_df['date'].min()), str(daily_df['date'].max())],
            "cashflow_stats": {
                "mean": float(daily_df['total_cashflow'].mean()),
                "std": float(daily_df['total_cashflow'].std()),
                "min": float(daily_df['total_cashflow'].min()),
                "max": float(daily_df['total_cashflow'].max()),
            }
        }, severity="INFO")

        # 2. Clean External Calendars
        oil_df['date'] = pd.to_datetime(oil_df['date'])
        oil_df.rename(columns={'dcoilwtico': 'oil_price'}, inplace=True)
        oil_nulls_before = int(oil_df['oil_price'].isnull().sum())
        oil_df['oil_price'] = oil_df['oil_price'].ffill().bfill()

        hol_df['date'] = pd.to_datetime(hol_df['date'])
        valid_holidays = hol_df[(hol_df['transferred'] == False) & (hol_df['type'] != 'Work Day')].copy()
        valid_holidays['is_holiday'] = 1
        holiday_flags = valid_holidays[['date', 'is_holiday']].drop_duplicates()
        logger.log_struct({
            "step": step_name, "event": "CALENDARS_CLEANED",
            "oil_nulls_filled": oil_nulls_before,
            "valid_holidays_count": len(holiday_flags),
        }, severity="INFO")

        # 3. Merge History
        master_df = daily_df.merge(oil_df, on='date', how='left')
        master_df = master_df.merge(holiday_flags, on='date', how='left')
        master_df['is_holiday'] = master_df['is_holiday'].fillna(0)
        master_df['oil_price'] = master_df['oil_price'].ffill().bfill()
        logger.log_struct({
            "step": step_name, "event": "MERGE_COMPLETE",
            "master_shape": list(master_df.shape),
            "master_nulls": master_df.isnull().sum().to_dict(),
        }, severity="INFO")

        # 4. Split Train/Test
        master_df = master_df.sort_values('date')
        cutoff_date = master_df['date'].max() - pd.Timedelta(days=forecast_horizon)
        train_split = master_df[master_df['date'] <= cutoff_date]
        test_split = master_df[master_df['date'] > cutoff_date]
        logger.log_struct({
            "step": step_name, "event": "TRAIN_TEST_SPLIT",
            "cutoff_date": str(cutoff_date),
            "train_split_shape": list(train_split.shape),
            "test_split_shape": list(test_split.shape),
        }, severity="INFO")

        train_split.to_csv(train_data.path, index=False)
        test_split.to_csv(test_data.path, index=False)

        # 5. PREP THE ACTUAL FUTURE (For Batch Scoring)
        max_date = master_df['date'].max()
        future_dates = pd.date_range(start=max_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
        future_df = pd.DataFrame({'date': future_dates})
        future_df = future_df.merge(oil_df, on='date', how='left')
        future_df = future_df.merge(holiday_flags, on='date', how='left')
        future_df['is_holiday'] = future_df['is_holiday'].fillna(0)
        future_df['onpromotion'] = 0
        last_known_oil = master_df['oil_price'].iloc[-1]
        future_df['oil_price'] = future_df['oil_price'].fillna(last_known_oil)
        future_df.to_csv(future_data.path, index=False)

        logger.log_struct({
            "step": step_name, "event": "FUTURE_DATA_PREPARED",
            "future_shape": list(future_df.shape),
            "future_date_range": [str(future_df['date'].min()), str(future_df['date'].max())],
            "last_known_oil_price": float(last_known_oil),
        }, severity="INFO")

        elapsed = round(time.time() - t0, 2)
        logger.log_struct({"step": step_name, "event": "STEP_COMPLETED", "elapsed_seconds": elapsed}, severity="INFO")
        print(f"Preprocessing complete in {elapsed}s")

    except Exception as e:
        logger.log_struct({
            "step": step_name, "severity": "ERROR", "event": "STEP_FAILED",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }, severity="ERROR")
        raise


@dsl.component(
    base_image="python:3.13",
    packages_to_install=["pandas", "pmdarima", "joblib", "fsspec", "gcsfs", "google-cloud-logging"]
)
def tune_and_train_model(
    train_data: dsl.Input[dsl.Dataset],
    seasonal_period: int,
    project_id: str,
    logger_name: str,
    model_artifact: dsl.Output[dsl.Model],
    best_params: dsl.Output[dsl.Metrics]
):
    """Runs pmdarima auto_arima to find optimal Seasonal ARIMAX hyperparameters."""
    import pandas as pd
    import pmdarima as pm
    import joblib
    import time
    import traceback
    from google.cloud import logging as cloud_logging

    log_client = cloud_logging.Client(project=project_id)
    logger = log_client.logger(logger_name)
    step_name = "tune_and_train_model"

    logger.log_struct({
        "step": step_name, "event": "STEP_STARTED",
        "params": {"seasonal_period": seasonal_period, "train_data_path": train_data.path}
    }, severity="INFO")
    t0 = time.time()

    try:
        train_df = pd.read_csv(train_data.path)
        exog_cols = ['onpromotion', 'oil_price', 'is_holiday']
        logger.log_struct({
            "step": step_name, "event": "TRAINING_DATA_LOADED",
            "shape": list(train_df.shape),
            "columns": list(train_df.columns),
            "nulls": train_df.isnull().sum().to_dict(),
            "target_stats": {
                "mean": float(train_df['total_cashflow'].mean()),
                "std": float(train_df['total_cashflow'].std()),
            },
            "exog_sample": train_df[exog_cols].head(3).to_dict(orient="list"),
        }, severity="INFO")

        logger.log_struct({
            "step": step_name, "event": "AUTO_ARIMA_STARTING",
            "config": {"max_p": 7, "max_q": 3, "max_P": 2, "max_Q": 2, "max_d": 2, "max_D": 1, "ic": "aic"}
        }, severity="INFO")

        model = pm.auto_arima(
            y=train_df['total_cashflow'].values,
            X=train_df[exog_cols].values,
            seasonal=True,
            m=seasonal_period,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=True,
            max_p=7, max_q=3,
            max_P=2, max_Q=2,
            max_d=2, max_D=1,
            information_criterion='aic',
        )

        order = model.order
        seasonal_order = model.seasonal_order
        aic_val = float(model.aic())
        bic_val = float(model.bic())

        logger.log_struct({
            "step": step_name, "event": "AUTO_ARIMA_COMPLETE",
            "best_order": list(order),
            "best_seasonal_order": list(seasonal_order),
            "AIC": aic_val,
            "BIC": bic_val,
            "n_params": int(model.df_model()),
        }, severity="INFO")

        # Log discovered hyperparameters to Vertex AI Metrics
        best_params.log_metric("order_p", order[0])
        best_params.log_metric("order_d", order[1])
        best_params.log_metric("order_q", order[2])
        best_params.log_metric("seasonal_P", seasonal_order[0])
        best_params.log_metric("seasonal_D", seasonal_order[1])
        best_params.log_metric("seasonal_Q", seasonal_order[2])
        best_params.log_metric("seasonal_period", seasonal_order[3])
        best_params.log_metric("AIC", aic_val)
        best_params.log_metric("BIC", bic_val)

        joblib.dump(model, model_artifact.path)
        logger.log_struct({
            "step": step_name, "event": "MODEL_SAVED",
            "model_path": model_artifact.path,
        }, severity="INFO")

        elapsed = round(time.time() - t0, 2)
        logger.log_struct({"step": step_name, "event": "STEP_COMPLETED", "elapsed_seconds": elapsed}, severity="INFO")
        print(f"Training complete in {elapsed}s — Best: SARIMAX{order}x{seasonal_order}")

    except Exception as e:
        logger.log_struct({
            "step": step_name, "severity": "ERROR", "event": "STEP_FAILED",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }, severity="ERROR")
        raise


@dsl.component(
    base_image="python:3.13",
    packages_to_install=["pandas", "pmdarima", "joblib", "scikit-learn", "fsspec", "gcsfs", "google-cloud-logging"]
)
def evaluate_model(
    model_artifact: dsl.Input[dsl.Model],
    test_data: dsl.Input[dsl.Dataset],
    forecast_horizon: int,
    project_id: str,
    logger_name: str,
    metrics: dsl.Output[dsl.Metrics]
):
    """Loads the tuned Seasonal ARIMAX model and evaluates against the holdout set."""
    import pandas as pd
    import joblib
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import math
    import time
    import traceback
    from google.cloud import logging as cloud_logging

    log_client = cloud_logging.Client(project=project_id)
    logger = log_client.logger(logger_name)
    step_name = "evaluate_model"

    logger.log_struct({
        "step": step_name, "event": "STEP_STARTED",
        "params": {"forecast_horizon": forecast_horizon}
    }, severity="INFO")
    t0 = time.time()

    try:
        test_df = pd.read_csv(test_data.path)
        exog_cols = ['onpromotion', 'oil_price', 'is_holiday']

        logger.log_struct({
            "step": step_name, "event": "TEST_DATA_LOADED",
            "shape": list(test_df.shape),
            "nulls": test_df.isnull().sum().to_dict(),
            "actual_cashflow_stats": {
                "mean": float(test_df['total_cashflow'].mean()),
                "min": float(test_df['total_cashflow'].min()),
                "max": float(test_df['total_cashflow'].max()),
            }
        }, severity="INFO")

        model = joblib.load(model_artifact.path)
        logger.log_struct({
            "step": step_name, "event": "MODEL_LOADED",
            "model_order": list(model.order),
            "model_seasonal_order": list(model.seasonal_order),
        }, severity="INFO")

        forecast = model.predict(n_periods=forecast_horizon, X=test_df[exog_cols].values)
        logger.log_struct({
            "step": step_name, "event": "FORECAST_GENERATED",
            "forecast_length": len(forecast),
            "forecast_stats": {
                "mean": float(forecast.mean()),
                "min": float(forecast.min()),
                "max": float(forecast.max()),
            },
            "forecast_sample": [float(v) for v in forecast[:5]],
        }, severity="INFO")

        mae = mean_absolute_error(test_df['total_cashflow'], forecast)
        rmse = math.sqrt(mean_squared_error(test_df['total_cashflow'], forecast))

        logger.log_struct({
            "step": step_name, "event": "EVALUATION_METRICS",
            "MAE": float(mae),
            "RMSE": float(rmse),
        }, severity="INFO")

        metrics.log_metric("Mean Absolute Error", float(mae))
        metrics.log_metric("Root Mean Squared Error", float(rmse))

        elapsed = round(time.time() - t0, 2)
        logger.log_struct({"step": step_name, "event": "STEP_COMPLETED", "elapsed_seconds": elapsed}, severity="INFO")
        print(f"Evaluation complete in {elapsed}s — MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    except Exception as e:
        logger.log_struct({
            "step": step_name, "severity": "ERROR", "event": "STEP_FAILED",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }, severity="ERROR")
        raise


@dsl.component(
    base_image="python:3.13",
    packages_to_install=["pandas", "pmdarima", "joblib", "fsspec", "gcsfs", "google-cloud-logging"]
)
def batch_predict(
    model_artifact: dsl.Input[dsl.Model],
    future_data: dsl.Input[dsl.Dataset],
    forecast_horizon: int,
    batch_output_uri: str,
    project_id: str,
    logger_name: str,
):
    """The Batch Component: Generates true future forecasts and writes to GCS."""
    import pandas as pd
    import joblib
    import time
    import traceback
    from google.cloud import logging as cloud_logging

    log_client = cloud_logging.Client(project=project_id)
    logger = log_client.logger(logger_name)
    step_name = "batch_predict"

    logger.log_struct({
        "step": step_name, "event": "STEP_STARTED",
        "params": {"forecast_horizon": forecast_horizon, "batch_output_uri": batch_output_uri}
    }, severity="INFO")
    t0 = time.time()

    try:
        future_df = pd.read_csv(future_data.path)
        exog_cols = ['onpromotion', 'oil_price', 'is_holiday']

        logger.log_struct({
            "step": step_name, "event": "FUTURE_DATA_LOADED",
            "shape": list(future_df.shape),
            "columns": list(future_df.columns),
            "nulls": future_df.isnull().sum().to_dict(),
        }, severity="INFO")

        model = joblib.load(model_artifact.path)
        logger.log_struct({
            "step": step_name, "event": "MODEL_LOADED",
            "model_order": list(model.order),
            "model_seasonal_order": list(model.seasonal_order),
        }, severity="INFO")

        forecast = model.predict(n_periods=forecast_horizon, X=future_df[exog_cols].values)

        # Format predictions — floor at 0 to prevent negative cashflow
        future_df['predicted_cashflow'] = forecast
        negative_count = int((future_df['predicted_cashflow'] < 0).sum())
        future_df['predicted_cashflow'] = future_df['predicted_cashflow'].apply(lambda x: max(0, x))

        output_df = future_df[['date', 'predicted_cashflow', 'is_holiday', 'oil_price']]

        logger.log_struct({
            "step": step_name, "event": "PREDICTIONS_GENERATED",
            "forecast_length": len(forecast),
            "negative_values_floored": negative_count,
            "prediction_stats": {
                "mean": float(output_df['predicted_cashflow'].mean()),
                "min": float(output_df['predicted_cashflow'].min()),
                "max": float(output_df['predicted_cashflow'].max()),
            },
            "output_shape": list(output_df.shape),
            "output_preview": output_df.head(3).to_dict(orient="list"),
        }, severity="INFO")

        output_df.to_csv(batch_output_uri, index=False)
        logger.log_struct({
            "step": step_name, "event": "OUTPUT_WRITTEN",
            "output_uri": batch_output_uri,
            "rows_written": len(output_df),
        }, severity="INFO")

        elapsed = round(time.time() - t0, 2)
        logger.log_struct({"step": step_name, "event": "STEP_COMPLETED", "elapsed_seconds": elapsed}, severity="INFO")
        print(f"Batch prediction complete in {elapsed}s — {len(output_df)} rows written to {batch_output_uri}")

    except Exception as e:
        logger.log_struct({
            "step": step_name, "severity": "ERROR", "event": "STEP_FAILED",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }, severity="ERROR")
        raise


# ==============================================================================
# 3. PIPELINE DEFINITION
# ==============================================================================
@dsl.pipeline(
    name="cashflow-monolith",
    description="Auto-tunes Seasonal ARIMAX, evaluates, and deploys batch forecasts to GCS"
)
def kaggle_pipeline(
    train_uri: str = GCS_TRAIN_CSV,
    oil_uri: str = GCS_OIL_CSV,
    holidays_uri: str = GCS_HOLIDAYS_CSV,
    batch_output_uri: str = BATCH_OUTPUT_URI,
    forecast_horizon: int = 15,
    seasonal_period: int = 7,   # Weekly seasonality for daily aggregated data
    project_id: str = PROJECT_ID,
    logger_name: str = CLOUD_LOGGER_NAME,
):
    # Step 1: Preprocess and prepare future calendar
    prep_task = preprocess_and_prep_future(
        train_csv_uri=train_uri,
        oil_csv_uri=oil_uri,
        holidays_csv_uri=holidays_uri,
        forecast_horizon=forecast_horizon,
        project_id=project_id,
        logger_name=logger_name,
    )
    prep_task.set_cpu_limit("4")
    prep_task.set_memory_limit("16G")

    # Step 2: Hyperparameter Tuning + Training (auto_arima finds best SARIMAX order)
    train_task = tune_and_train_model(
        train_data=prep_task.outputs["train_data"],
        seasonal_period=seasonal_period,
        project_id=project_id,
        logger_name=logger_name,
    )
    train_task.set_cpu_limit("4")
    train_task.set_memory_limit("16G")

    # Step 3: Evaluate Model
    evaluate_task = evaluate_model(
        model_artifact=train_task.outputs["model_artifact"],
        test_data=prep_task.outputs["test_data"],
        forecast_horizon=forecast_horizon,
        project_id=project_id,
        logger_name=logger_name,
    )

    # Step 4: Batch Predict (Deploy predictions to final GCS Bucket)
    batch_task = batch_predict(
        model_artifact=train_task.outputs["model_artifact"],
        future_data=prep_task.outputs["future_data"],
        forecast_horizon=forecast_horizon,
        batch_output_uri=batch_output_uri,
        project_id=project_id,
        logger_name=logger_name,
    )

    # Best Practice: Only run the batch prediction if the evaluation step finishes
    batch_task.after(evaluate_task)

# ==============================================================================
# 4. COMPILE AND SUBMIT
# ==============================================================================
if __name__ == "__main__":
    import logging
    import sys

    # Local logging for the submission script itself
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
    log = logging.getLogger("pipeline-submit")

    try:
        from google.cloud import logging as cloud_logging
        cl_client = cloud_logging.Client(project=PROJECT_ID)
        cl_logger = cl_client.logger(CLOUD_LOGGER_NAME)
    except Exception:
        cl_logger = None

    def cloud_log(payload, severity="INFO"):
        """Log to both stdout and Cloud Logging."""
        log.info(str(payload))
        if cl_logger:
            cl_logger.log_struct(payload, severity=severity)

    cloud_log({"event": "PIPELINE_COMPILE_START", "pipeline_root": PIPELINE_ROOT})

    compiler_path = "batch_pipeline.yaml"
    compiler.Compiler().compile(pipeline_func=kaggle_pipeline, package_path=compiler_path)
    cloud_log({"event": "PIPELINE_COMPILED", "template_path": compiler_path})

    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
    cloud_log({"event": "VERTEX_AI_INITIALIZED", "project": PROJECT_ID, "location": LOCATION})

    pipeline_job = aiplatform.PipelineJob(
        display_name="cashflow-batch-scoring",
        template_path=compiler_path,
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False, # Set to False to ensure it generates a fresh forecast on every run
    )

    cloud_log({"event": "PIPELINE_SUBMITTING", "display_name": "cashflow-batch-scoring", "experiment": EXPERIMENT_NAME})
    pipeline_job.submit(experiment=EXPERIMENT_NAME)
    dashboard_uri = pipeline_job._dashboard_uri()
    cloud_log({"event": "PIPELINE_SUBMITTED", "dashboard_uri": dashboard_uri, "experiment": EXPERIMENT_NAME})
    print(f"\nPipeline submitted successfully (experiment={EXPERIMENT_NAME})!\nTrack your run visually here: {dashboard_uri}")