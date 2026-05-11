# Cashflow Batch Scoring Pipeline

An end-to-end MLOps pipeline built with **Kubeflow Pipelines (KFP)** and **Vertex AI** that automatically tunes a **Seasonal ARIMAX** model on Kaggle store sales data and generates batch cashflow forecasts delivered to Google Cloud Storage.

---

## Pipeline Overview

The pipeline aggregates daily store sales into a single cashflow time series, enriches it with exogenous signals (oil prices & holidays), runs **`pmdarima.auto_arima`** to find the best Seasonal ARIMAX hyperparameters `(p,d,q)(P,D,Q,s)` via AIC-minimizing stepwise search, evaluates the tuned model on a holdout set, and writes future forecasts to GCS.

```
┌─────────────────────────┐
│ 1. Preprocess &         │
│    Prep Future Calendar  │
└────┬──────────┬─────────┘
     │          │
     ▼          │
┌────────────────┐  │
│ 2. Auto-Tune   │  │
│  & Train SARIMAX│  │
└────┬───────────┘  │
     │          │
     ▼          ▼
┌─────────────────┐
│  3. Evaluate    │
│     Model       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Batch Predict│
│   → GCS Output  │
└─────────────────┘
```

### Pipeline Steps

| Step | Component | Description |
|------|-----------|-------------|
| 1 | **Preprocess & Prep Future** | Reads `train.csv`, `oil.csv`, and `holidays_events.csv` from GCS. Aggregates sales into daily cashflow, merges exogenous features (oil price, holiday flags, promotions), splits into train/test sets, and generates a future calendar for the forecast horizon. |
| 2 | **Tune & Train Model** | Runs `pmdarima.auto_arima` with stepwise search over `(p,d,q)(P,D,Q,s)` orders (AIC criterion, `m=7` weekly seasonality). Logs best hyperparameters, AIC, and BIC as Vertex Metrics. Saves the tuned model via `joblib`. |
| 3 | **Evaluate Model** | Loads the tuned model, forecasts over the holdout test period, and logs MAE and RMSE metrics to the Vertex AI Metrics store. |
| 4 | **Batch Predict** | Loads the tuned model, forecasts the true future using the prepared calendar, clamps predictions to non-negative values, and writes the results as a CSV to the configured GCS output path. |

### Output

The final forecast CSV is written to `gs://<BUCKET>/batch_predictions/latest_forecast.csv` and contains:

| Column | Description |
|--------|-------------|
| `date` | Forecast date |
| `predicted_cashflow` | Predicted daily cashflow (floored at 0) |
| `is_holiday` | Holiday flag for the date |
| `oil_price` | Oil price used as input |

---

## Prerequisites

- **Google Cloud Project** with the Vertex AI API enabled
- **GCS Bucket** containing the Kaggle [Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) dataset files:
  - `train.csv`
  - `oil.csv`
  - `holidays_events.csv`
- **Python 3.10+**
- Authenticated `gcloud` CLI (`gcloud auth application-default login`)

---

## Setup

1. **Clone the repo and navigate to the pipeline directory:**

   ```bash
   cd monolith_pipeline
   ```

2. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Update the configuration** at the top of `mlops.py`:

   ```python
   PROJECT_ID = "your-gcp-project-id"
   LOCATION   = "your-gcp-region"        # e.g. "asia-southeast2"
   BUCKET_URI = "gs://your-bucket-name"
   ```

4. **Upload your dataset to GCS** (if not already there):

   ```bash
   gsutil cp -r gs://vertex-dump/sales_forecast/ gs://your-bucket-name/sales_forecast/
   ```

---

## Running the Pipeline

Submit the pipeline to Vertex AI Pipelines:

```bash
python mlops.py
```

This will:

1. **Compile** the KFP pipeline to `batch_pipeline.yaml`
2. **Initialize** the Vertex AI SDK with your project settings
3. **Submit** the pipeline job to Vertex AI

After submission, a link to the Vertex AI Pipelines dashboard will be printed to the console where you can monitor the run visually.

---

## Pipeline Parameters

All parameters have defaults and can be overridden in the `kaggle_pipeline()` function:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_uri` | `gs://<BUCKET>/sales_forecast/train.csv` | GCS path to training data |
| `oil_uri` | `gs://<BUCKET>/sales_forecast/oil.csv` | GCS path to oil price data |
| `holidays_uri` | `gs://<BUCKET>/sales_forecast/holidays_events.csv` | GCS path to holidays data |
| `batch_output_uri` | `gs://<BUCKET>/batch_predictions/latest_forecast.csv` | GCS destination for forecast output |
| `forecast_horizon` | `15` | Number of days to forecast |
| `seasonal_period` | `7` | Seasonal period for SARIMAX (`m`); 7 = weekly seasonality |

---

## Project Structure

```
monolith_pipeline/
├── mlops.py               # Pipeline definition, components, and submission script
├── batch_pipeline.yaml    # Compiled KFP pipeline spec (auto-generated)
├── requirements.txt       # Python dependencies for submitting the pipeline
└── README.md              # This file
```
