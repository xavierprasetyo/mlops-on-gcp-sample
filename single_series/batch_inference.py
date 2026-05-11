"""Stage 2: Batch Inference (v3) — Single store×family prediction.

Filters test.csv to the target (store_nbr, family), enriches with exog features,
submits batch prediction via Vertex AI SDK, and outputs id,sales CSV.

Usage:
    python single_series/batch_inference.py
"""
import os
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from google.cloud import aiplatform, storage

from single_series.pipeline.config import (
    PROJECT_ID,
    LOCATION,
    MODEL_DISPLAY_NAME,
    GCS_TEST_CSV,
    GCS_OIL_CSV,
    GCS_HOLIDAYS_CSV,
    BATCH_INPUT_URI,
    BATCH_OUTPUT_DIR,
    TARGET_STORE_NBR,
    TARGET_FAMILY,
)


def prepare_batch_input(
    test_csv_uri: str,
    oil_csv_uri: str,
    holidays_csv_uri: str,
    output_uri: str,
) -> pd.DataFrame:
    """Filter test.csv to target store×family, enrich with exog, write to GCS."""
    print(f"Filtering test.csv to store_nbr={TARGET_STORE_NBR}, family={TARGET_FAMILY}")

    test_df = pd.read_csv(test_csv_uri)
    filtered = test_df[
        (test_df["store_nbr"] == TARGET_STORE_NBR)
        & (test_df["family"] == TARGET_FAMILY)
    ].copy()
    print(f"Filtered test rows: {len(filtered)}")

    # Clean oil prices
    oil_df = pd.read_csv(oil_csv_uri)
    oil_df["date"] = pd.to_datetime(oil_df["date"])
    oil_df.rename(columns={"dcoilwtico": "oil_price"}, inplace=True)
    oil_df["oil_price"] = oil_df["oil_price"].ffill().bfill()

    # Clean holidays
    hol_df = pd.read_csv(holidays_csv_uri)
    hol_df["date"] = pd.to_datetime(hol_df["date"])
    valid_holidays = hol_df[
        (hol_df["transferred"] == False) & (hol_df["type"] != "Work Day")
    ].copy()
    valid_holidays["is_holiday"] = 1
    holiday_flags = valid_holidays[["date", "is_holiday"]].drop_duplicates()

    # Merge exogenous features
    filtered["date"] = pd.to_datetime(filtered["date"])
    filtered = filtered.merge(oil_df[["date", "oil_price"]], on="date", how="left")
    filtered = filtered.merge(holiday_flags, on="date", how="left")
    filtered["is_holiday"] = filtered["is_holiday"].fillna(0)
    filtered["oil_price"] = filtered["oil_price"].ffill().bfill()
    filtered["day_of_week"] = filtered["date"].dt.dayofweek
    filtered = filtered.sort_values("date").reset_index(drop=True)

    # Write ONLY exog columns to batch input CSV — the CPR predict() expects
    # [onpromotion, oil_price, is_holiday, day_of_week] with no extra columns.
    # Track IDs separately for post-processing.
    exog_cols = ["onpromotion", "oil_price", "is_holiday", "day_of_week"]
    batch_df = filtered[exog_cols]
    batch_df.to_csv(output_uri, index=False)
    print(f"Batch input written to {output_uri} ({len(batch_df)} rows)")

    # Return full enriched df with IDs for post-processing
    return filtered[["id"] + exog_cols]


def run_batch_prediction(input_uri: str) -> aiplatform.BatchPredictionJob:
    """Submit batch prediction job using the registered model."""
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    models = aiplatform.Model.list(
        filter=f'display_name="{MODEL_DISPLAY_NAME}"',
        order_by="create_time desc",
    )
    if not models:
        raise ValueError(f"Model '{MODEL_DISPLAY_NAME}' not found in registry.")
    model = models[0]
    print(f"Using model: {model.display_name}, version: {model.version_id}")

    print("Submitting Vertex AI Batch Prediction Job...")
    batch_job = model.batch_predict(
        job_display_name="cashflow-sarimax-v3-batch-inference",
        gcs_source=input_uri,
        instances_format="csv",
        gcs_destination_prefix=BATCH_OUTPUT_DIR,
        predictions_format="jsonl",
        machine_type="n2-standard-16",
        starting_replica_count=1,
        max_replica_count=1,
        sync=False,
    )

    print("Waiting for job resource creation...")
    batch_job.wait_for_resource_creation()
    print(f"Job created: {batch_job.resource_name}")
    return batch_job


def postprocess_output(batch_job: aiplatform.BatchPredictionJob, instances_df: pd.DataFrame):
    """Wait for job, download results, format as id,sales CSV."""
    print("Waiting for batch prediction job to complete...")
    batch_job.wait()
    print("Batch prediction complete!")

    output_uri = batch_job.output_info.gcs_output_directory
    print(f"Output directory: {output_uri}")

    storage_client = storage.Client(project=PROJECT_ID)
    output_path = output_uri.replace("gs://", "")
    bucket_name = output_path.split("/")[0]
    prefix = "/".join(output_path.split("/")[1:])

    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    import json
    all_predictions = []
    for blob in blobs:
        if "prediction.results" in blob.name:
            content = blob.download_as_text()
            for line in content.strip().split("\n"):
                if line:
                    result = json.loads(line)
                    pred = result.get("prediction", result.get("predictions", None))
                    if isinstance(pred, list):
                        all_predictions.extend(pred)
                    else:
                        all_predictions.append(pred)

    if len(all_predictions) != len(instances_df):
        print(
            f"WARNING: Got {len(all_predictions)} predictions "
            f"but expected {len(instances_df)} rows."
        )

    # Build submission: strictly id, sales
    submission_df = pd.DataFrame({
        "id": instances_df["id"].values[: len(all_predictions)],
        "sales": all_predictions,
    })
    submission_df["sales"] = submission_df["sales"].apply(lambda x: max(0, x))

    output_csv = BATCH_OUTPUT_DIR + "submission.csv"
    submission_df.to_csv(output_csv, index=False)
    print(f"Submission CSV written to {output_csv} ({len(submission_df)} rows)")
    return submission_df


if __name__ == "__main__":
    input_uri = BATCH_INPUT_URI + "instances.csv"

    instances_df = prepare_batch_input(
        GCS_TEST_CSV, GCS_OIL_CSV, GCS_HOLIDAYS_CSV, input_uri
    )
    batch_job = run_batch_prediction(input_uri)
    postprocess_output(batch_job, instances_df)
