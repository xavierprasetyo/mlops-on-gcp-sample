"""Stage 1: SARIMAX Training Pipeline — Compile & Submit to Vertex AI.

Pipeline flow:
    Preprocess → Hyperparameter Search → Train SARIMAX → Evaluate → Register

Usage:
    python -m new_batch_pipeline.pipeline.train_pipeline
"""
import os
import subprocess
import sys

# Ensure repo root is on sys.path so imports work whether invoked directly or via python -m
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kfp import dsl, compiler
from google.cloud import aiplatform

from new_batch_pipeline.pipeline.config import (
    PROJECT_ID,
    LOCATION,
    BUCKET_URI,
    MODEL_DISPLAY_NAME,
    GCS_TRAIN_CSV,
    GCS_OIL_CSV,
    GCS_HOLIDAYS_CSV,
    PIPELINE_ROOT,
    FORECAST_HORIZON,
)
from new_batch_pipeline.pipeline.components.preprocess import preprocess
from new_batch_pipeline.pipeline.components.hyperparam import hyperparam_search
from new_batch_pipeline.pipeline.components.train import train_model
from new_batch_pipeline.pipeline.components.evaluate import evaluate_model
from new_batch_pipeline.pipeline.components.register import register_model


@dsl.pipeline(
    name="cashflow-sarimax-training",
    description="Train Seasonal ARIMAX with hyperparameter search, evaluate, and register to Vertex AI Model Registry",
)
def training_pipeline(
    container_uri: str,
    train_uri: str = GCS_TRAIN_CSV,
    oil_uri: str = GCS_OIL_CSV,
    holidays_uri: str = GCS_HOLIDAYS_CSV,
    forecast_horizon: int = FORECAST_HORIZON,
):
    # Step 1: Preprocess — aggregate, merge exog, split, compute proportions
    prep_task = preprocess(
        train_csv_uri=train_uri,
        oil_csv_uri=oil_uri,
        holidays_csv_uri=holidays_uri,
        forecast_horizon=forecast_horizon,
    ).set_cpu_limit("4").set_memory_limit("16G")

    # Step 2: Hyperparameter search — grid search over SARIMAX orders using AIC
    search_task = hyperparam_search(
        train_data=prep_task.outputs["train_data"],
    ).set_cpu_limit("4").set_memory_limit("16G")

    # Step 3: Train — fit SARIMAX with best hyperparameters
    train_task = train_model(
        train_data=prep_task.outputs["train_data"],
        best_order=search_task.outputs["best_order"],
        best_seasonal_order=search_task.outputs["best_seasonal_order"],
    )

    # Step 4: Evaluate — forecast on validation set, log MAE/RMSE/MAPE
    evaluate_task = evaluate_model(
        model_artifact=train_task.outputs["model_artifact"],
        val_data=prep_task.outputs["val_data"],
        forecast_horizon=forecast_horizon,
    )

    # Step 5: Register — upload model + proportions to Vertex AI Model Registry
    register_task = register_model(
        project=PROJECT_ID,
        location=LOCATION,
        model_display_name=MODEL_DISPLAY_NAME,
        container_uri=container_uri,
        model_artifact=train_task.outputs["model_artifact"],
        proportions=prep_task.outputs["proportions"],
    )
    # Only register after evaluation completes successfully
    register_task.after(evaluate_task)


if __name__ == "__main__":
    # ================================================================
    # 1. Build the CPR Custom Container via Cloud Build
    # ================================================================
    cpr_image_uri = f"gcr.io/{PROJECT_ID}/cashflow-sarimax-cpr:latest"
    cpr_src_dir = "new_batch_pipeline/cpr_src"

    print("Building Custom Prediction Routine (CPR) container via Cloud Build...")
    try:
        subprocess.run(
            ["gcloud", "builds", "submit", "--tag", cpr_image_uri, cpr_src_dir],
            check=True,
        )
        print(f"Successfully built and pushed {cpr_image_uri}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to build CPR image. Error: {e}")
        sys.exit(1)

    # ================================================================
    # 2. Compile and submit the KFP pipeline
    # ================================================================
    pipeline_yaml = "sarimax_train.yaml"
    print("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=training_pipeline, package_path=pipeline_yaml
    )

    print("Submitting pipeline to Vertex AI...")
    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)

    job = aiplatform.PipelineJob(
        display_name="cashflow-sarimax-train-register",
        template_path=pipeline_yaml,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={"container_uri": cpr_image_uri},
        enable_caching=False,
    )
    job.submit()
    print(f"Pipeline submitted! Track at: {job._dashboard_uri()}")
