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

# ==============================================================================
# 2. KFP COMPONENTS
# ==============================================================================

@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "fsspec", "gcsfs"]
)
def preprocess_and_prep_future(
    train_csv_uri: str,
    oil_csv_uri: str,
    holidays_csv_uri: str,
    forecast_horizon: int,
    train_data: dsl.Output[dsl.Dataset],
    test_data: dsl.Output[dsl.Dataset],
    future_data: dsl.Output[dsl.Dataset] # <-- NEW: Calendar for true future predictions
):
    import pandas as pd
    
    print("Reading and aggregating data...")
    train_df = pd.read_csv(train_csv_uri)
    oil_df = pd.read_csv(oil_csv_uri)
    hol_df = pd.read_csv(holidays_csv_uri)
    
    # 1. Aggregate Historical Cashflow
    daily_df = train_df.groupby('date').agg({'sales': 'sum', 'onpromotion': 'sum'}).reset_index()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df.rename(columns={'sales': 'total_cashflow'}, inplace=True)
    
    # 2. Clean External Calendars
    oil_df['date'] = pd.to_datetime(oil_df['date'])
    oil_df.rename(columns={'dcoilwtico': 'oil_price'}, inplace=True)
    oil_df['oil_price'] = oil_df['oil_price'].ffill().bfill()
    
    hol_df['date'] = pd.to_datetime(hol_df['date'])
    valid_holidays = hol_df[(hol_df['transferred'] == False) & (hol_df['type'] != 'Work Day')].copy()
    valid_holidays['is_holiday'] = 1
    holiday_flags = valid_holidays[['date', 'is_holiday']].drop_duplicates()
    
    # 3. Merge History
    master_df = daily_df.merge(oil_df, on='date', how='left')
    master_df = master_df.merge(holiday_flags, on='date', how='left')
    master_df['is_holiday'] = master_df['is_holiday'].fillna(0)
    master_df['oil_price'] = master_df['oil_price'].ffill().bfill()
    
    # 4. Split Train/Test
    master_df = master_df.sort_values('date')
    cutoff_date = master_df['date'].max() - pd.Timedelta(days=forecast_horizon)
    
    train_split = master_df[master_df['date'] <= cutoff_date]
    test_split = master_df[master_df['date'] > cutoff_date]
    
    train_split.to_csv(train_data.path, index=False)
    test_split.to_csv(test_data.path, index=False)
    
    # 5. PREP THE ACTUAL FUTURE (For Batch Scoring)
    # Generate dates for the next N days *after* our historical dataset completely ends
    max_date = master_df['date'].max()
    future_dates = pd.date_range(start=max_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
    future_df = pd.DataFrame({'date': future_dates})
    
    # Attach known future exogenous variables to these future dates
    future_df = future_df.merge(oil_df, on='date', how='left')
    future_df = future_df.merge(holiday_flags, on='date', how='left')
    
    # Fill missing values for the unknown future
    future_df['is_holiday'] = future_df['is_holiday'].fillna(0)
    future_df['onpromotion'] = 0 # Assume zero planned marketing promotions for the future
    
    # Carry forward the last known oil price to the future
    last_known_oil = master_df['oil_price'].iloc[-1]
    future_df['oil_price'] = future_df['oil_price'].fillna(last_known_oil)
    
    future_df.to_csv(future_data.path, index=False)
    print("Preprocessing complete!")


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "statsmodels", "fsspec", "gcsfs"]
)
def train_model(
    train_data: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Output[dsl.Model] # <-- NEW: Saves model object natively
):
    """Trains ARIMAX and saves the serialized model artifact."""
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings("ignore")
    
    train_df = pd.read_csv(train_data.path)
    exog_cols = ['onpromotion', 'oil_price', 'is_holiday']
    
    print("Fitting ARIMAX model...")
    model = ARIMA(train_df['total_cashflow'].values, exog=train_df[exog_cols].values, order=(7, 1, 1))
    model_fit = model.fit()
    
    # Serialize and save the statsmodels object directly to the Vertex Artifact path
    print("Saving model artifact...")
    model_fit.save(model_artifact.path)


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "statsmodels", "scikit-learn", "fsspec", "gcsfs"]
)
def evaluate_model(
    model_artifact: dsl.Input[dsl.Model],
    test_data: dsl.Input[dsl.Dataset],
    forecast_horizon: int,
    metrics: dsl.Output[dsl.Metrics]
):
    """Loads the saved model and evaluates it against the holdout set."""
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMAResults
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import math
    import warnings
    warnings.filterwarnings("ignore")
    
    test_df = pd.read_csv(test_data.path)
    exog_cols = ['onpromotion', 'oil_price', 'is_holiday']
    
    print("Evaluating model...")
    # Load the saved model artifact
    model_fit = ARIMAResults.load(model_artifact.path)
    
    # Predict the holdout test period
    forecast = model_fit.forecast(steps=forecast_horizon, exog=test_df[exog_cols].values)
    
    mae = mean_absolute_error(test_df['total_cashflow'], forecast)
    rmse = math.sqrt(mean_squared_error(test_df['total_cashflow'], forecast))
    
    metrics.log_metric("Mean Absolute Error", float(mae))
    metrics.log_metric("Root Mean Squared Error", float(rmse))


@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "statsmodels", "fsspec", "gcsfs"]
)
def batch_predict(
    model_artifact: dsl.Input[dsl.Model],
    future_data: dsl.Input[dsl.Dataset],
    forecast_horizon: int,
    batch_output_uri: str
):
    """The Batch Component: Generates true future forecasts and writes to GCS."""
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMAResults
    import warnings
    warnings.filterwarnings("ignore")
    
    future_df = pd.read_csv(future_data.path)
    exog_cols = ['onpromotion', 'oil_price', 'is_holiday']
    
    print("Loading model for Batch Prediction...")
    # 1. Load the finalized model
    model_fit = ARIMAResults.load(model_artifact.path)
    
    print(f"Generating future forecast for {forecast_horizon} days...")
    # 2. Forecast the true future using our future calendar batch
    forecast = model_fit.forecast(steps=forecast_horizon, exog=future_df[exog_cols].values)
    
    # 3. Format predictions
    future_df['predicted_cashflow'] = forecast.values if hasattr(forecast, 'values') else forecast
    future_df['predicted_cashflow'] = future_df['predicted_cashflow'].apply(lambda x: max(0, x)) # Prevent negative cashflow
    
    output_df = future_df[['date', 'predicted_cashflow', 'is_holiday', 'oil_price']]
    
    # 4. Save results to the final destination Bucket
    print(f"Writing Batch Predictions to {batch_output_uri}")
    output_df.to_csv(batch_output_uri, index=False)


# ==============================================================================
# 3. PIPELINE DEFINITION
# ==============================================================================
@dsl.pipeline(
    name="kaggle-cashflow-batch-scoring",
    description="Trains ARIMAX, saves the model, and deploys batch forecasts to GCS"
)
def kaggle_pipeline(
    train_uri: str = GCS_TRAIN_CSV,
    oil_uri: str = GCS_OIL_CSV,
    holidays_uri: str = GCS_HOLIDAYS_CSV,
    batch_output_uri: str = BATCH_OUTPUT_URI,
    forecast_horizon: int = 15 
):
    # Step 1: Preprocess and prepare future calendar
    prep_task = preprocess_and_prep_future(
        train_csv_uri=train_uri,
        oil_csv_uri=oil_uri,
        holidays_csv_uri=holidays_uri,
        forecast_horizon=forecast_horizon
    )
    prep_task.set_cpu_limit("4")
    prep_task.set_memory_limit("16G")
    
    # Step 2: Train Model
    train_task = train_model(
        train_data=prep_task.outputs["train_data"]
    )
    
    # Step 3: Evaluate Model
    evaluate_task = evaluate_model(
        model_artifact=train_task.outputs["model_artifact"],
        test_data=prep_task.outputs["test_data"],
        forecast_horizon=forecast_horizon
    )
    
    # Step 4: Batch Predict (Deploy predictions to final GCS Bucket)
    # This natively passes the model from Step 2 into our Batch Prediction component
    batch_task = batch_predict(
        model_artifact=train_task.outputs["model_artifact"],
        future_data=prep_task.outputs["future_data"],
        forecast_horizon=forecast_horizon,
        batch_output_uri=batch_output_uri
    )
    
    # Best Practice: Only run the batch prediction if the evaluation step finishes
    batch_task.after(evaluate_task)

# ==============================================================================
# 4. COMPILE AND SUBMIT
# ==============================================================================
if __name__ == "__main__":
    print("Compiling pipeline...")
    compiler_path = "batch_pipeline.yaml"
    compiler.Compiler().compile(pipeline_func=kaggle_pipeline, package_path=compiler_path)

    print("Initializing Vertex AI SDK...")
    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)

    pipeline_job = aiplatform.PipelineJob(
        display_name="cashflow-batch-scoring",
        template_path=compiler_path,
        pipeline_root=PIPELINE_ROOT,
        enable_caching=False, # Set to False to ensure it generates a fresh forecast on every run
    )

    print("Submitting Pipeline to Vertex AI...")
    pipeline_job.submit()
    print(f"\nPipeline submitted successfully!\nTrack your run visually here: {pipeline_job._dashboard_uri()}")