# MLOps on GCP — SARIMAX Sales Forecasting

A collection of **Vertex AI MLOps pipelines** that forecast store sales using **Seasonal ARIMAX** models. The repo explores three progressively more production-ready approaches — from a single-file prototype to a fully containerised batch-prediction system — all built on the same [Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) dataset.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | [Kubeflow Pipelines (KFP) v2](https://www.kubeflow.org/docs/components/pipelines/) on Vertex AI Pipelines |
| **Training** | `pmdarima` · `statsmodels` SARIMAX |
| **Serving** | Vertex AI Batch Prediction · Custom Prediction Routines (CPR) via FastAPI |
| **Model Registry** | Vertex AI Model Registry |
| **Container Build** | Cloud Build · Artifact Registry |
| **Infrastructure** | Google Cloud Storage · BigQuery (qwiklabs only) |
| **Language** | Python 3.10+ |

---

## Repository Structure

```
vertex-mlops/
├── monolith_pipeline/         # Approach 1 — single-file KFP pipeline (prototype)
├── aggregate_disaggregate/    # Approach 2 — 2-stage pipeline with proportional disaggregation
├── single_series/             # Approach 3 — dedicated model per store×family
├── qwiklabs_pipeline/         # Qwiklabs lab: custom container training + batch predict (Scikit-learn)
├── dataset/                   # Kaggle Store Sales data (gitignored — download separately)
├── setup_gcs_bucket.sh        # One-time GCS bucket + IAM setup for the 3 SARIMAX pipelines
└── README.md                  # This file
```

Each pipeline directory has its own `README.md` with setup & run instructions.

---

## Dataset

All SARIMAX pipelines use the [Kaggle Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) dataset. Place the following files in `dataset/` or upload them to your GCS bucket:

| File | Description |
|---|---|
| `train.csv` | Daily sales per store × product family (~122 MB) |
| `test.csv` | 15-day forecast horizon |
| `oil.csv` | Daily oil prices (exogenous feature) |
| `holidays_events.csv` | Ecuador holiday calendar (exogenous feature) |
| `stores.csv` | Store metadata |
| `transactions.csv` | Daily transaction counts |
| `sample_submission.csv` | Submission template (`id, sales`) |

> **Note:** The `dataset/` directory is gitignored. Download the data from Kaggle or copy from `gs://vertex-dump/sales_forecast/`.

---

## Quick Start

```bash
# 1. Clone the repo
git clone git@github.com:xavierprasetyo/mlops-on-gcp-sample.git
cd mlops-on-gcp-sample

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Authenticate with GCP
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>

# 4. (SARIMAX pipelines only) Provision the GCS bucket & IAM roles
#    This is REQUIRED before running monolith_pipeline, aggregate_disaggregate, or single_series.
#    The qwiklabs_pipeline does NOT need this step.
./setup_gcs_bucket.sh --project <YOUR_PROJECT_ID> --bucket <BUCKET_NAME>

# 5. Pick an approach and follow its README
#    e.g. for the monolith pipeline:
cd monolith_pipeline
pip install -r requirements.txt
python mlops.py
```

> [!IMPORTANT]
> The three SARIMAX pipelines (`monolith_pipeline`, `aggregate_disaggregate`, `single_series`) require the GCS bucket and IAM permissions created by `setup_gcs_bucket.sh`. Run it **once** before executing any of these pipelines. The `qwiklabs_pipeline` manages its own infrastructure and does not need this step.

---

## Pipeline Comparison

A side-by-side comparison of the three model-building approaches in this repository: **monolith_pipeline**, **aggregate_disaggregate**, and **single_series**.

### High-Level Summary

| Dimension | `monolith_pipeline` | `aggregate_disaggregate` | `single_series` |
|---|---|---|---|
| **Philosophy** | All-in-one monolithic KFP pipeline | Modular 2-stage pipeline with CPR serving | Simplified single-case pipeline with CPR serving |
| **Modeling Strategy** | Aggregated daily cashflow (all stores) | Aggregated daily cashflow → disaggregate via proportions | Single (store, family) time series — no aggregation |
| **Hyperparameter Tuning** | `pmdarima.auto_arima` (stepwise) | Manual grid search over SARIMAX orders (AIC) | Manual grid search over SARIMAX orders (AIC) |
| **Training Library** | `pmdarima` | `statsmodels` (SARIMAX) | `statsmodels` (SARIMAX) |
| **Inference Method** | KFP component writes CSV to GCS | Vertex AI Batch Prediction via Custom Prediction Routine (CPR) | Vertex AI Batch Prediction via CPR |
| **Model Registry** | ❌ Not registered | ✅ Registered in Vertex AI Model Registry | ✅ Registered in Vertex AI Model Registry |
| **Online Endpoint** | ❌ None | ❌ None | ❌ None |

---

### Architecture Comparison

#### `monolith_pipeline` — Monolithic KFP Pipeline

```
Preprocess & Prep Future Calendar
    ├─→ Auto-Tune & Train SARIMAX (pmdarima)
    │       ├─→ Evaluate Model (MAE, RMSE)
    │       └─→ Batch Predict → GCS CSV
    └─ (future calendar) ─→ Batch Predict
```

- **Single file** (`mlops.py`) contains all 4 KFP components + pipeline definition + submission logic.
- Batch prediction is a **KFP component** that runs inside the pipeline — no external serving infrastructure.
- Output is a simple forecast CSV with `date, predicted_cashflow, is_holiday, oil_price`.

#### `aggregate_disaggregate` — 2-Stage Pipeline with Disaggregation

```
Stage 1 (KFP Pipeline):
    Cloud Build CPR container
        → Preprocess (aggregate + compute proportions)
            → Hyperparam Grid Search
                → Train SARIMAX (statsmodels)
                    → Evaluate (MAE, RMSE, MAPE)
                        → Register Model (+ proportions.csv)

Stage 2 (SDK Script):
    Prepare batch input (enrich test.csv with exog features)
        → Submit Vertex AI Batch Prediction Job
            → Post-process JSONL → id,sales CSV
```

- Training and inference are **decoupled** into separate stages.
- CPR container handles serving with a `preprocess → predict → postprocess` lifecycle.
- Proportions table is **bundled with the model artifact** for disaggregation at inference time.

#### `single_series` — Single-Case Pipeline

```
Stage 1 (KFP Pipeline):
    Cloud Build CPR container
        → Preprocess (FILTER to store_nbr=1, family=BEVERAGES)
            → Hyperparam Grid Search
                → Train SARIMAX (statsmodels)
                    → Evaluate (MAE, RMSE, MAPE)
                        → Register Model (model only, no proportions)

Stage 2 (SDK Script):
    Filter test.csv to target store×family
        → Submit Vertex AI Batch Prediction Job
            → Post-process JSONL → id,sales CSV
```

- Trains a **dedicated model for one (store_nbr, family) combination**.
- No aggregation or disaggregation — the model directly predicts sales for the target combination.
- Simplest CPR predictor (~100 lines vs ~209 lines in `aggregate_disaggregate`).

---

### Detailed Feature Comparison

#### 1. Data Preprocessing

| Feature | `monolith_pipeline` | `aggregate_disaggregate` | `single_series` |
|---|---|---|---|
| **Data granularity** | Aggregated (all stores → 1 daily series) | Aggregated (all stores → 1 daily series) | Filtered (single store×family series) |
| **Target variable** | `total_cashflow` (sum of all sales) | `total_cashflow` (sum of all sales) | `sales` (single store×family) |
| **Exogenous features** | `onpromotion`, `oil_price`, `is_holiday` | `onpromotion`, `oil_price`, `is_holiday`, `day_of_week` | `onpromotion`, `oil_price`, `is_holiday`, `day_of_week` |
| **Proportions table** | ❌ Not computed | ✅ Computed from last 30 days | ❌ Not needed |
| **Train/Val split** | Last N days holdout | Last N days holdout | Last N days holdout |
| **Future calendar prep** | ✅ Generated in preprocess | ❌ Handled by batch inference | ❌ Handled by batch inference |

#### 2. Hyperparameter Tuning

| Feature | `monolith_pipeline` | `aggregate_disaggregate` | `single_series` |
|---|---|---|---|
| **Method** | `pmdarima.auto_arima` (stepwise) | Manual grid search | Manual grid search |
| **Search space** | Automated stepwise (max p=7, q=3, P=2, Q=2) | Explicit grid: p∈{0,1,2}, d∈{0,1}, q∈{0,1,2}, P∈{0,1}, D∈{0,1}, Q∈{0,1} — 144 combos | Same grid as `aggregate_disaggregate` |
| **Criterion** | AIC | AIC | AIC |
| **Seasonal period** | `m=7` (weekly) | `s=7` (weekly, fixed) | `s=7` (weekly, fixed) |
| **Result passing** | Model object carries params | Best order returned as pipeline output artifact | Best order returned as pipeline output artifact |

#### 3. Model Training

| Feature | `monolith_pipeline` | `aggregate_disaggregate` | `single_series` |
|---|---|---|---|
| **Library** | `pmdarima` (`auto_arima` returns fitted model) | `statsmodels.tsa.statespace.SARIMAX` | `statsmodels.tsa.statespace.SARIMAX` |
| **Serialization** | `joblib.dump` | `pickle` (statsmodels native) | `pickle` (statsmodels native) |
| **Model artifact** | KFP `dsl.Model` (not registered) | KFP `dsl.Model` → registered to Vertex AI | KFP `dsl.Model` → registered to Vertex AI |
| **Amnesia cure** | N/A (model stays in pipeline memory) | CPR re-applies full training history at container startup | CPR re-applies filtered training history at container startup |

#### 4. Model Evaluation

| Feature | `monolith_pipeline` | `aggregate_disaggregate` | `single_series` |
|---|---|---|---|
| **Metrics** | MAE, RMSE | MAE, RMSE, MAPE | MAE, RMSE, MAPE |
| **Logging target** | Vertex AI Metrics (KFP) | Vertex AI Metrics (KFP) | Vertex AI Metrics (KFP) |

#### 5. Inference / Serving

| Feature | `monolith_pipeline` | `aggregate_disaggregate` | `single_series` |
|---|---|---|---|
| **Inference method** | In-pipeline KFP component | Vertex AI Batch Prediction (CPR container) | Vertex AI Batch Prediction (CPR container) |
| **Container** | None (runs in KFP component) | Custom Docker (FastAPI + uvicorn) | Custom Docker (FastAPI + uvicorn) |
| **Batch input format** | KFP artifact (CSV) | CSV uploaded to GCS → Vertex AI CSV format | CSV uploaded to GCS → Vertex AI CSV format |
| **Batch output format** | CSV (date, predicted_cashflow) | JSONL → post-processed to `id, sales` CSV | JSONL → post-processed to `id, sales` CSV |
| **Disaggregation** | ❌ Outputs only aggregated total | ✅ Proportional disaggregation (store×family) | ❌ Direct prediction (no disaggregation needed) |
| **Online endpoint** | ❌ | ❌ | ❌ |
| **Input columns** | `onpromotion`, `oil_price`, `is_holiday` | `id`, `date`, `store_nbr`, `family`, `onpromotion`, `oil_price`, `is_holiday`, `day_of_week` | `id`, `onpromotion`, `oil_price`, `is_holiday`, `day_of_week` |

---

### Project Structure Comparison

#### `monolith_pipeline/` — 5 files

```
monolith_pipeline/
├── mlops.py               # Everything: components + pipeline + submission
├── batch_pipeline.yaml    # Compiled KFP spec (auto-generated)
├── requirements.txt       # kfp, google-cloud-aiplatform
└── README.md
```

#### `aggregate_disaggregate/` — 13+ files

```
aggregate_disaggregate/
├── pipeline/
│   ├── config.py                  # Centralized config (incl. hyperparams)
│   ├── train_pipeline.py          # Stage 1: KFP pipeline
│   └── components/
│       ├── preprocess.py          # Aggregate + compute proportions
│       ├── hyperparam.py          # Grid search
│       ├── train.py               # Fit SARIMAX
│       ├── evaluate.py            # MAE/RMSE/MAPE
│       └── register.py           # Upload to Model Registry (+ proportions)
├── cpr_src/
│   ├── Dockerfile                 # FastAPI container
│   ├── requirements.txt
│   ├── main.py                    # FastAPI routes (/predict, /health)
│   └── predictor.py               # CPR lifecycle (load, preprocess, predict, postprocess)
├── batch_inference.py             # Stage 2: Batch predict + post-process
├── tests/
└── requirements.txt
```

#### `single_series/` — 12+ files

```
single_series/
├── pipeline/
│   ├── config.py                  # Config (incl. TARGET_STORE_NBR, TARGET_FAMILY)
│   ├── train_pipeline.py          # Stage 1: KFP pipeline
│   └── components/
│       ├── preprocess.py          # Filter to single store×family
│       ├── hyperparam.py          # Grid search
│       ├── train.py               # Fit SARIMAX
│       ├── evaluate.py            # MAE/RMSE/MAPE
│       └── register.py           # Upload to Model Registry (model only)
├── cpr_src/
│   ├── Dockerfile                 # FastAPI container
│   ├── requirements.txt
│   ├── main.py                    # FastAPI routes
│   └── predictor.py               # Simplified CPR (no disaggregation)
├── batch_inference.py             # Stage 2: Batch predict + post-process
└── tests/
```

---

### CPR Predictor Complexity

| Aspect | `aggregate_disaggregate` | `single_series` |
|---|---|---|
| **Lines of code** | ~209 | ~100 |
| **Artifacts loaded** | `model.pkl` + `proportions.csv` | `model.pkl` only |
| **GCS download method** | `google.cloud.storage` SDK (explicit) | `fsspec` (simpler) |
| **Predict output** | Full row dicts (`id, date, store_nbr, family, proportion, sales`) | Plain numeric list (`[sales_val, ...]`) |
| **Disaggregation** | ✅ Multiplies daily total × proportion per store×family | ❌ Returns forecast values directly |

---

### Trade-offs

| Approach | Pros | Cons |
|---|---|---|
| **`monolith_pipeline`** | Simplest to set up; single file; no containers needed; good for prototyping | No model registry; no serving infrastructure; output is aggregated only; tightly coupled |
| **`aggregate_disaggregate`** | Production-ready; disaggregates to store×family; batch prediction; model versioning | Complex architecture; CPR must cure model amnesia; proportional disaggregation introduces approximation error; large predictor codebase |
| **`single_series`** | Direct prediction per store×family; simplest CPR; most accurate for the target combo; clean separation | Only predicts one store×family combo; would need N pipelines for N combos; not scalable to all 1,782 store×family pairs |

---

### When to Use Each

- **`monolith_pipeline`**: Rapid prototyping, quick experiments, or when you only need an aggregated daily cashflow forecast and don't need serving infrastructure.
- **`aggregate_disaggregate`**: Production workloads where you need per-store, per-family predictions across all combinations, model versioning, and the ability to serve via online endpoints.
- **`single_series`**: When you care about accuracy for a **specific** store×family combination and want the simplest possible serving setup, or when the proportional disaggregation approach introduces too much error.

---

## Qwiklabs Pipeline

The `qwiklabs_pipeline/` directory contains a separate lab exercise adapted from the [Vertex AI Pipelines Qwiklabs course](https://www.cloudskillsboost.google/). It demonstrates:

- Creating a **managed Tabular Dataset** from BigQuery
- Running a **custom container training job** (Scikit-learn `DecisionTreeClassifier` on the UCI Dry Beans dataset)
- Executing a **batch prediction** job — all orchestrated via KFP v2

This pipeline is independent of the SARIMAX forecasting pipelines and serves as a learning reference for Vertex AI Pipelines fundamentals.

---

## License

This project is for educational and experimental purposes.
