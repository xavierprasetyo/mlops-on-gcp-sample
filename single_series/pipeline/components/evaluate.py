from kfp import dsl


@dsl.component(
    base_image="python:3.13",
    packages_to_install=["pandas", "statsmodels", "scikit-learn", "google-cloud-aiplatform"],
)
def evaluate_model(
    model_artifact: dsl.Input[dsl.Model],
    val_data: dsl.Input[dsl.Dataset],
    forecast_horizon: int,
    project: str,
    location: str,
    experiment_name: str,
    metrics: dsl.Output[dsl.Metrics],
):
    """Evaluate the trained SARIMAX model on the validation holdout.

    Logs MAE, RMSE, and MAPE metrics to both KFP Metrics artifact and
    Vertex AI Experiments for cross-run comparison.
    """
    import math
    import os
    import pandas as pd
    import warnings
    from google.cloud import aiplatform
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

    warnings.filterwarnings("ignore")

    val_df = pd.read_csv(val_data.path)
    exog_cols = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]

    model_path = os.path.join(model_artifact.path, "model.pkl")
    model_fit = SARIMAXResultsWrapper.load(model_path)

    steps = min(forecast_horizon, len(val_df))
    forecast = model_fit.forecast(steps=steps, exog=val_df[exog_cols].values[:steps])
    actuals = val_df["sales"].values[:steps]

    mae = mean_absolute_error(actuals, forecast)
    rmse = math.sqrt(mean_squared_error(actuals, forecast))

    nonzero_mask = actuals != 0
    if nonzero_mask.any():
        mape = (abs((actuals[nonzero_mask] - forecast[nonzero_mask]) / actuals[nonzero_mask])).mean() * 100
    else:
        mape = 0.0

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    # Log to KFP Metrics (existing behavior — visible in pipeline run UI)
    metrics.log_metric("MAE", float(mae))
    metrics.log_metric("RMSE", float(rmse))
    metrics.log_metric("MAPE", float(mape))

    # Log to Vertex AI Experiments (for cross-run comparison)
    aiplatform.init(
        project=project, location=location, experiment=experiment_name
    )
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    with aiplatform.start_run(f"final-evaluation-{ts}") as run:
        run.log_params({
            "forecast_horizon": forecast_horizon,
            "val_rows": len(val_df),
            "stage": "evaluation",
        })
        run.log_metrics({
            "val_mae": float(mae),
            "val_rmse": float(rmse),
            "val_mape": float(mape),
        })
