"""Stage 2: Batch Inference via Vertex AI SDK.

Reads test.csv from GCS, enriches with exogenous features, submits a batch
prediction job using the registered SARIMAX model, and post-processes the
output to match sample_submission.csv format (id, sales).

Usage:
    python batch_inference.py
"""
import csv
import json
import logging
import os
import sys

# Ensure repo root is on sys.path so imports work whether invoked as
# `python aggregate_disaggregate/batch_inference.py` or `python -m aggregate_disaggregate.batch_inference`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import google.cloud.logging
import pandas as pd
from google.cloud import aiplatform, storage

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure Python logging to emit to both Cloud Logging and the console."""
    client = google.cloud.logging.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    # Attach the Cloud Logging handler to the root logger so all log records
    # are forwarded to Cloud Logging.
    client.setup_logging(log_level=logging.INFO)

    # Ensure a console handler is present so we still see output locally.
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
    )
    logging.getLogger().addHandler(console)
    logger.info("Cloud Logging initialised for project %s", client.project)

from aggregate_disaggregate.pipeline.config import (
    PROJECT_ID,
    PROJECT_NUMBER,
    LOCATION,
    BUCKET_URI,
    MODEL_DISPLAY_NAME,
    EXPERIMENT_NAME,
    GCS_TEST_CSV,
    GCS_OIL_CSV,
    GCS_HOLIDAYS_CSV,
    BATCH_INPUT_URI,
    BATCH_OUTPUT_DIR,
)


def prepare_batch_input(
    test_csv_uri: str,
    oil_csv_uri: str,
    holidays_csv_uri: str,
    output_uri: str,
) -> pd.DataFrame:
    """Enrich test.csv with exogenous features and write to GCS as batch input.

    Returns the enriched DataFrame for reference (e.g., to extract IDs later).
    """
    logger.info("Preparing batch input from test.csv...")

    test_df = pd.read_csv(test_csv_uri)
    oil_df = pd.read_csv(oil_csv_uri)
    hol_df = pd.read_csv(holidays_csv_uri)

    # Clean oil prices
    oil_df["date"] = pd.to_datetime(oil_df["date"])
    oil_df.rename(columns={"dcoilwtico": "oil_price"}, inplace=True)
    oil_df["oil_price"] = oil_df["oil_price"].ffill().bfill()

    # Clean holidays
    hol_df["date"] = pd.to_datetime(hol_df["date"])
    valid_holidays = hol_df[
        (hol_df["transferred"] == False) & (hol_df["type"] != "Work Day")
    ].copy()
    valid_holidays["is_holiday"] = 1
    holiday_flags = valid_holidays[["date", "is_holiday"]].drop_duplicates()

    # Merge exogenous features onto test data
    test_df["date"] = pd.to_datetime(test_df["date"])
    test_df = test_df.merge(oil_df[["date", "oil_price"]], on="date", how="left")
    test_df = test_df.merge(holiday_flags, on="date", how="left")
    test_df["is_holiday"] = test_df["is_holiday"].fillna(0)
    test_df["oil_price"] = test_df["oil_price"].ffill().bfill()
    test_df["day_of_week"] = test_df["date"].dt.dayofweek

    # Format date back to string for CSV serialization
    test_df["date"] = test_df["date"].dt.strftime("%Y-%m-%d")

    # Select columns in the order expected by the CPR predictor
    instance_cols = [
        "id", "date", "store_nbr", "family",
        "onpromotion", "oil_price", "is_holiday", "day_of_week",
    ]
    instances_df = test_df[instance_cols]

    # Write to GCS — use QUOTE_NONNUMERIC so string columns (family, date)
    # are double-quoted, which Vertex AI's CSV parser requires.
    instances_df.to_csv(output_uri, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Batch input written to %s (%d rows)", output_uri, len(instances_df))
    logger.info("Last 10 instances: %s", instances_df.tail(10))
    return instances_df


def run_batch_prediction(input_uri: str) -> aiplatform.BatchPredictionJob:
    """Submit a batch prediction job using the registered model."""
    aiplatform.init(
        project=PROJECT_ID, location=LOCATION, experiment=EXPERIMENT_NAME
    )

    # Fetch the model and get its latest version
    models = aiplatform.Model.list(
        filter=f'display_name="{MODEL_DISPLAY_NAME}"',
        order_by="create_time desc",
    )
    if not models:
        raise ValueError(f"Model '{MODEL_DISPLAY_NAME}' not found in registry.")
    parent_model = models[0]
    # List all versions and pick the latest one
    registry = parent_model.versioning_registry
    versions = registry.list_versions()
    latest = versions[-1]  # list_versions returns ascending order
    model = registry.get_model(version=latest.version_id)
    logger.info("Using model: %s, version: %s", model.display_name, model.version_id)

    # Log batch inference run to Vertex AI Experiments
    import datetime
    run_id = f"batch-{model.version_id}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    aiplatform.start_run(run_id)
    aiplatform.log_params({
        "model_display_name": model.display_name,
        "model_version": model.version_id,
        "input_uri": input_uri,
        "machine_type": "n2d-standard-8",
        "stage": "batch_inference",
    })

    # Submit batch prediction
    logger.info("Submitting Vertex AI Batch Prediction Job...")
    batch_job = model.batch_predict(
        job_display_name="cashflow-sarimax-batch-inference",
        gcs_source=input_uri,
        instances_format="csv",
        gcs_destination_prefix=BATCH_OUTPUT_DIR,
        predictions_format="jsonl",
        machine_type="n2d-standard-8",
        starting_replica_count=1,
        max_replica_count=1,
        service_account=f"{PROJECT_NUMBER}-compute@developer.gserviceaccount.com",
        sync=False,
    )

    logger.info("Waiting for job resource creation...")
    batch_job.wait_for_resource_creation()
    logger.info("Job created: %s", batch_job.resource_name)

    aiplatform.log_params({
        "batch_job_resource": batch_job.resource_name,
    })

    return batch_job


def postprocess_output(batch_job: aiplatform.BatchPredictionJob, instances_df: pd.DataFrame):
    """Wait for batch job completion, download results, and format as id,sales CSV."""
    logger.info("Waiting for batch prediction job to complete...")

    try:
        batch_job.wait()
    except RuntimeError as exc:
        # The Vertex AI SDK raises RuntimeError directly when the job fails,
        # so we catch it here, log the details, and fetch container logs.
        logger.error("Batch prediction job FAILED: %s", exc)
        _fetch_job_logs(batch_job.resource_name)
        raise

    logger.info("Batch prediction complete! (state=%s)", batch_job.state)

    output_uri = batch_job.output_info.gcs_output_directory
    logger.info("Output directory: %s", output_uri)

    # Download prediction JSONL files from GCS output directory
    # Vertex AI writes predictions as prediction.results-XXXXX-of-XXXXX files
    storage_client = storage.Client(project=PROJECT_ID)

    # Parse bucket and prefix from output URI
    # output_uri format: gs://bucket/path/to/output/
    output_path = output_uri.replace("gs://", "")
    bucket_name = output_path.split("/")[0]
    prefix = "/".join(output_path.split("/")[1:])

    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    all_predictions = []
    all_raw_lines = []
    for blob in blobs:
        if "prediction.results" in blob.name:
            content = blob.download_as_text()
            raw_lines = [l for l in content.strip().split("\n") if l]
            all_raw_lines.extend(raw_lines)
            for line in content.strip().split("\n"):
                if line:
                    result = json.loads(line)
                    pred = result.get("prediction", result.get("predictions", None))
                    if isinstance(pred, list):
                        all_predictions.extend(pred)
                    else:
                        all_predictions.append(pred)

    logger.info("Last 10 raw JSONL lines:\n%s", "\n".join(all_raw_lines[-10:]))

    if len(all_predictions) != len(instances_df):
        logger.warning(
            "Got %d predictions but expected %d rows.",
            len(all_predictions),
            len(instances_df),
        )

    # Build submission DataFrame
    # Predictions now contain full row dicts with 'id', 'sales', etc.
    if all_predictions and isinstance(all_predictions[0], dict):
        submission_df = pd.DataFrame(all_predictions)[["id", "sales"]]
    else:
        # Fallback for plain numeric predictions
        submission_df = pd.DataFrame({
            "id": instances_df["id"].values[: len(all_predictions)],
            "sales": all_predictions,
        })
    # Ensure non-negative
    submission_df["sales"] = submission_df["sales"].apply(lambda x: max(0, x))

    # Write final output
    output_csv = BATCH_OUTPUT_DIR + "submission.csv"
    submission_df.to_csv(output_csv, index=False)
    logger.info("Submission CSV written to %s (%d rows)", output_csv, len(submission_df))

    # Log batch inference output metrics to Vertex AI Experiments
    try:
        aiplatform.log_metrics({
            "input_rows": len(instances_df),
            "output_rows": len(submission_df),
            "prediction_count": len(all_predictions),
        })
        aiplatform.end_run()
    except Exception as exc:
        logger.warning("Could not log batch metrics to experiment: %s", exc)

    return submission_df


def _fetch_job_logs(job_resource_name: str, max_entries: int = 50) -> None:
    """Query Cloud Logging for the batch prediction job's container logs."""
    try:
        log_client = google.cloud.logging.Client()
        # Vertex AI batch prediction logs use this resource filter.
        filtr = (
            f'resource.type="ml_job" '
            f'resource.labels.job_id="{job_resource_name.split("/")[-1]}"'
        )
        logger.info("Fetching container logs from Cloud Logging (filter: %s)...", filtr)
        entries = list(log_client.list_entries(filter_=filtr, max_results=max_entries))
        if not entries:
            logger.warning("No Cloud Logging entries found for this job. "
                           "Logs may take a few minutes to propagate.")
            return
        logger.info("--- Batch Job Container Logs (%d entries) ---", len(entries))
        for entry in entries:
            logger.info("  [%s] %s", entry.severity, entry.payload)
        logger.info("--- End of Container Logs ---")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch Cloud Logging entries: %s", exc)


if __name__ == "__main__":
    setup_logging()

    input_uri = BATCH_INPUT_URI + "instances.csv"

    # Step 1: Prepare batch input from test.csv
    instances_df = prepare_batch_input(
        GCS_TEST_CSV, GCS_OIL_CSV, GCS_HOLIDAYS_CSV, input_uri
    )

    # Step 2: Submit batch prediction job
    batch_job = run_batch_prediction(input_uri)

    # Step 3: Post-process output to submission format
    postprocess_output(batch_job, instances_df)
