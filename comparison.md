# Pipeline Comparison: Three Approaches to Building the SARIMAX Model

A side-by-side comparison of the three model-building approaches in this repository: **direct_pipeline**, **new_batch_pipeline**, and **batch_v3**.

---

## High-Level Summary

| Dimension | `direct_pipeline` | `new_batch_pipeline` | `batch_v3` |
|---|---|---|---|
| **Philosophy** | All-in-one monolithic KFP pipeline | Modular 2-stage pipeline with CPR serving | Simplified single-case pipeline with CPR serving |
| **Modeling Strategy** | Aggregated daily cashflow (all stores) | Aggregated daily cashflow → disaggregate via proportions | Single (store, family) time series — no aggregation |
| **Hyperparameter Tuning** | `pmdarima.auto_arima` (stepwise) | Manual grid search over SARIMAX orders (AIC) | Manual grid search over SARIMAX orders (AIC) |
| **Training Library** | `pmdarima` | `statsmodels` (SARIMAX) | `statsmodels` (SARIMAX) |
| **Inference Method** | KFP component writes CSV to GCS | Vertex AI Batch Prediction via Custom Prediction Routine (CPR) | Vertex AI Batch Prediction via CPR |
| **Model Registry** | ❌ Not registered | ✅ Registered in Vertex AI Model Registry | ✅ Registered in Vertex AI Model Registry |
| **Online Endpoint** | ❌ None | ✅ Supported (deploy_and_test.py) | ❌ None |

---

## Architecture Comparison

### `direct_pipeline` — Monolithic KFP Pipeline

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

### `new_batch_pipeline` — 2-Stage Pipeline with Disaggregation

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

### `batch_v3` — Single-Case Pipeline

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
- Simplest CPR predictor (~100 lines vs ~209 lines in `new_batch_pipeline`).

---

## Detailed Feature Comparison

### 1. Data Preprocessing

| Feature | `direct_pipeline` | `new_batch_pipeline` | `batch_v3` |
|---|---|---|---|
| **Data granularity** | Aggregated (all stores → 1 daily series) | Aggregated (all stores → 1 daily series) | Filtered (single store×family series) |
| **Target variable** | `total_cashflow` (sum of all sales) | `total_cashflow` (sum of all sales) | `sales` (single store×family) |
| **Exogenous features** | `onpromotion`, `oil_price`, `is_holiday` | `onpromotion`, `oil_price`, `is_holiday`, `day_of_week` | `onpromotion`, `oil_price`, `is_holiday`, `day_of_week` |
| **Proportions table** | ❌ Not computed | ✅ Computed from last 30 days | ❌ Not needed |
| **Train/Val split** | Last N days holdout | Last N days holdout | Last N days holdout |
| **Future calendar prep** | ✅ Generated in preprocess | ❌ Handled by batch inference | ❌ Handled by batch inference |

### 2. Hyperparameter Tuning

| Feature | `direct_pipeline` | `new_batch_pipeline` | `batch_v3` |
|---|---|---|---|
| **Method** | `pmdarima.auto_arima` (stepwise) | Manual grid search | Manual grid search |
| **Search space** | Automated stepwise (max p=7, q=3, P=2, Q=2) | Explicit grid: p∈{0,1,2}, d∈{0,1}, q∈{0,1,2}, P∈{0,1}, D∈{0,1}, Q∈{0,1} — 144 combos | Same grid as `new_batch_pipeline` |
| **Criterion** | AIC | AIC | AIC |
| **Seasonal period** | `m=7` (weekly) | `s=7` (weekly, fixed) | `s=7` (weekly, fixed) |
| **Result passing** | Model object carries params | Best order returned as pipeline output artifact | Best order returned as pipeline output artifact |

### 3. Model Training

| Feature | `direct_pipeline` | `new_batch_pipeline` | `batch_v3` |
|---|---|---|---|
| **Library** | `pmdarima` (`auto_arima` returns fitted model) | `statsmodels.tsa.statespace.SARIMAX` | `statsmodels.tsa.statespace.SARIMAX` |
| **Serialization** | `joblib.dump` | `pickle` (statsmodels native) | `pickle` (statsmodels native) |
| **Model artifact** | KFP `dsl.Model` (not registered) | KFP `dsl.Model` → registered to Vertex AI | KFP `dsl.Model` → registered to Vertex AI |
| **Amnesia cure** | N/A (model stays in pipeline memory) | CPR re-applies full training history at container startup | CPR re-applies filtered training history at container startup |

### 4. Model Evaluation

| Feature | `direct_pipeline` | `new_batch_pipeline` | `batch_v3` |
|---|---|---|---|
| **Metrics** | MAE, RMSE | MAE, RMSE, MAPE | MAE, RMSE, MAPE |
| **Logging target** | Vertex AI Metrics (KFP) | Vertex AI Metrics (KFP) | Vertex AI Metrics (KFP) |

### 5. Inference / Serving

| Feature | `direct_pipeline` | `new_batch_pipeline` | `batch_v3` |
|---|---|---|---|
| **Inference method** | In-pipeline KFP component | Vertex AI Batch Prediction (CPR container) | Vertex AI Batch Prediction (CPR container) |
| **Container** | None (runs in KFP component) | Custom Docker (FastAPI + uvicorn) | Custom Docker (FastAPI + uvicorn) |
| **Batch input format** | KFP artifact (CSV) | CSV uploaded to GCS → Vertex AI CSV format | CSV uploaded to GCS → Vertex AI CSV format |
| **Batch output format** | CSV (date, predicted_cashflow) | JSONL → post-processed to `id, sales` CSV | JSONL → post-processed to `id, sales` CSV |
| **Disaggregation** | ❌ Outputs only aggregated total | ✅ Proportional disaggregation (store×family) | ❌ Direct prediction (no disaggregation needed) |
| **Online endpoint** | ❌ | ✅ Via `deploy_and_test.py` | ❌ |
| **Input columns** | `onpromotion`, `oil_price`, `is_holiday` | `id`, `date`, `store_nbr`, `family`, `onpromotion`, `oil_price`, `is_holiday`, `day_of_week` | `id`, `onpromotion`, `oil_price`, `is_holiday`, `day_of_week` |

---

## Project Structure Comparison

### `direct_pipeline/` — 5 files

```
direct_pipeline/
├── mlops.py               # Everything: components + pipeline + submission
├── batch_pipeline.yaml    # Compiled KFP spec (auto-generated)
├── requirements.txt       # kfp, google-cloud-aiplatform
└── README.md
```

### `new_batch_pipeline/` — 14+ files

```
new_batch_pipeline/
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
├── deploy_and_test.py             # Build, deploy, and test online endpoint
├── tests/
└── requirements.txt
```

### `batch_v3/` — 12+ files

```
batch_v3/
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

## CPR Predictor Complexity

| Aspect | `new_batch_pipeline` | `batch_v3` |
|---|---|---|
| **Lines of code** | ~209 | ~100 |
| **Artifacts loaded** | `model.pkl` + `proportions.csv` | `model.pkl` only |
| **GCS download method** | `google.cloud.storage` SDK (explicit) | `fsspec` (simpler) |
| **Predict output** | Full row dicts (`id, date, store_nbr, family, proportion, sales`) | Plain numeric list (`[sales_val, ...]`) |
| **Disaggregation** | ✅ Multiplies daily total × proportion per store×family | ❌ Returns forecast values directly |

---

## Trade-offs

| Approach | Pros | Cons |
|---|---|---|
| **`direct_pipeline`** | Simplest to set up; single file; no containers needed; good for prototyping | No model registry; no serving infrastructure; output is aggregated only; tightly coupled |
| **`new_batch_pipeline`** | Production-ready; disaggregates to store×family; supports online + batch; model versioning | Complex architecture; CPR must cure model amnesia; proportional disaggregation introduces approximation error; large predictor codebase |
| **`batch_v3`** | Direct prediction per store×family; simplest CPR; most accurate for the target combo; clean separation | Only predicts one store×family combo; would need N pipelines for N combos; not scalable to all 1,782 store×family pairs |

---

## When to Use Each

- **`direct_pipeline`**: Rapid prototyping, quick experiments, or when you only need an aggregated daily cashflow forecast and don't need serving infrastructure.
- **`new_batch_pipeline`**: Production workloads where you need per-store, per-family predictions across all combinations, model versioning, and the ability to serve via online endpoints.
- **`batch_v3`**: When you care about accuracy for a **specific** store×family combination and want the simplest possible serving setup, or when the proportional disaggregation approach introduces too much error.
