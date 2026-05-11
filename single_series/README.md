# Cashflow SARIMAX Pipeline (Single Series)

Single store×family Seasonal ARIMAX pipeline for predicting daily cashflow (sales) on Vertex AI.

## Architecture

- **Stage 1** (KFP Pipeline): Preprocess (filter to target store×family) → Hyperparameter Search → Train SARIMAX → Evaluate → Register to Model Registry
- **Stage 2** (SDK Script): Filter & enrich test.csv → Vertex AI Batch Predict → Post-process to `id, sales` CSV

## Directory Structure

```
single_series/
├── pipeline/
│   ├── config.py                  # Configuration constants
│   ├── train_pipeline.py          # Stage 1: KFP pipeline
│   └── components/                # KFP pipeline components
│       ├── preprocess.py
│       ├── hyperparam.py
│       ├── train.py
│       ├── evaluate.py
│       └── register.py
├── cpr_src/                       # Custom Prediction Routine container
│   ├── Dockerfile
│   ├── main.py
│   ├── predictor.py
│   └── requirements.txt
├── batch_inference.py             # Stage 2: Batch inference script
├── tests/                         # Unit tests
└── README.md                      # This file
```

## Usage

### Prerequisites
```bash
pip install -r requirements.txt
gcloud auth login
gcloud config set project project-sandbox-357505
```

### Stage 1: Train & Register
```bash
# From the repo root directory
python -m single_series.pipeline.train_pipeline
```
This will:
1. Build the CPR container via Cloud Build
2. Compile and submit the KFP training pipeline to Vertex AI

### Stage 2: Batch Inference
```bash
python -m single_series.batch_inference
```
This will:
1. Filter `test.csv` to the target store×family combination
2. Enrich with exogenous features (oil prices, holidays)
3. Submit a batch prediction job via Vertex AI SDK
4. Post-process output to `id, sales` format

### Run Tests
```bash
python -m pytest single_series/tests/ -v
```

## Data
Datasets are stored in `gs://vertex-dump/sales_forecast/`:
- `train.csv` — Training data (store×family×date sales)
- `test.csv` — Test data (15-day forecast horizon)
- `oil.csv` — Daily oil prices (exogenous)
- `holidays_events.csv` — Holiday calendar (exogenous)
