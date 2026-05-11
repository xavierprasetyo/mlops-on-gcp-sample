"""Stage 1: SARIMAX Training Pipeline (v3) — Single store×family model.

Pipeline flow:
    Preprocess (filter) → Hyperparameter Search → Train SARIMAX → Evaluate → Register

Usage:
    python -m batch_v3.pipeline.train_pipeline
"""
import os
import subprocess
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kfp import dsl, compiler
from google.cloud import aiplatform

from batch_v3.pipeline.config import (
    PROJECT_ID,
    LOCATION,
    BUCKET_URI,
    MODEL_DISPLAY_NAME,
    GCS_TRAIN_CSV,
    GCS_OIL_CSV,
    GCS_HOLIDAYS_CSV,
    PIPELINE_ROOT,
    FORECAST_HORIZON,
    TARGET_STORE_NBR,
    TARGET_FAMILY,
)
from batch_v3.pipeline.components.preprocess import preprocess
from batch_v3.pipeline.components.hyperparam import hyperparam_search
from batch_v3.pipeline.components.train import train_model
from batch_v3.pipeline.components.evaluate import evaluate_model
from batch_v3.pipeline.components.register import register_model


@dsl.pipeline(
    name="cashflow-sarimax-v3-training",
    description="Train Seasonal ARIMAX on a single store×family, evaluate, and register",
)
def training_pipeline(
    container_uri: str,
    train_uri: str = GCS_TRAIN_CSV,
    oil_uri: str = GCS_OIL_CSV,
    holidays_uri: str = GCS_HOLIDAYS_CSV,
    target_store_nbr: int = TARGET_STORE_NBR,
    target_family: str = TARGET_FAMILY,
    forecast_horizon: int = FORECAST_HORIZON,
):
    # Step 1: Preprocess — filter to target store×family, merge exog, split
    prep_task = preprocess(
        train_csv_uri=train_uri,
        oil_csv_uri=oil_uri,
        holidays_csv_uri=holidays_uri,
        target_store_nbr=target_store_nbr,
        target_family=target_family,
        forecast_horizon=forecast_horizon,
    ).set_cpu_limit("16").set_memory_limit("32G")

    # Step 2: Hyperparameter search
    search_task = hyperparam_search(
        train_data=prep_task.outputs["train_data"],
    ).set_cpu_limit("16").set_memory_limit("32G")

    # Step 3: Train SARIMAX with best hyperparameters
    train_task = train_model(
        train_data=prep_task.outputs["train_data"],
        best_order=search_task.outputs["best_order"],
        best_seasonal_order=search_task.outputs["best_seasonal_order"],
    )

    # Step 4: Evaluate on validation holdout
    evaluate_task = evaluate_model(
        model_artifact=train_task.outputs["model_artifact"],
        val_data=prep_task.outputs["val_data"],
        forecast_horizon=forecast_horizon,
    )

    # Step 5: Register to Vertex AI Model Registry (after evaluation)
    register_task = register_model(
        project=PROJECT_ID,
        location=LOCATION,
        model_display_name=MODEL_DISPLAY_NAME,
        container_uri=container_uri,
        bucket_uri=BUCKET_URI,
        target_store_nbr=target_store_nbr,
        target_family=target_family,
        model_artifact=train_task.outputs["model_artifact"],
    )
    register_task.after(evaluate_task)


if __name__ == "__main__":
    # 1. Build the CPR container
    cpr_image_uri = f"gcr.io/{PROJECT_ID}/cashflow-sarimax-v3-cpr:latest"
    cpr_src_dir = "batch_v3/cpr_src"

    print("Building CPR container via Cloud Build...")
    try:
        subprocess.run(
            ["gcloud", "builds", "submit", "--tag", cpr_image_uri, cpr_src_dir],
            check=True,
        )
        print(f"Built and pushed {cpr_image_uri}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to build CPR image. Error: {e}")
        sys.exit(1)

    # 2. Compile and submit pipeline
    pipeline_yaml = "sarimax_v3_train.yaml"
    print("Compiling pipeline...")
    compiler.Compiler().compile(
        pipeline_func=training_pipeline, package_path=pipeline_yaml
    )

    print("Submitting pipeline to Vertex AI...")
    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)

    job = aiplatform.PipelineJob(
        display_name="cashflow-sarimax-v3-train-register",
        template_path=pipeline_yaml,
        pipeline_root=PIPELINE_ROOT,
        parameter_values={"container_uri": cpr_image_uri},
        enable_caching=False,
    )
    job.submit()
    print(f"Pipeline submitted! Track at: {job._dashboard_uri()}")
