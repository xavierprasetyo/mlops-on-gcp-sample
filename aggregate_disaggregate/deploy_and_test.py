"""Build CPR image, deploy to Vertex AI endpoint, and test with dataset/test.csv.

Uses gcloud/curl for model & endpoint management (much faster than the Python SDK
in some environments), and the Python SDK only for the final predict call.

Usage:
    .venv/bin/python aggregate_disaggregate/deploy_and_test.py
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from google.cloud import aiplatform

from aggregate_disaggregate.pipeline.config import (
    PROJECT_ID,
    LOCATION,
    BUCKET_URI,
    MODEL_DISPLAY_NAME,
    GCS_OIL_CSV,
    GCS_HOLIDAYS_CSV,
)

API_BASE = f"https://{LOCATION}-aiplatform.googleapis.com/v1"
PARENT = f"projects/{PROJECT_ID}/locations/{LOCATION}"


def _get_token() -> str:
    return subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()


def _curl_json(method: str, url: str, data: dict | None = None) -> dict:
    """Make an authenticated REST call to Vertex AI API."""
    import tempfile

    token = _get_token()
    cmd = [
        "curl", "-s", "-X", method, url,
        "-H", f"Authorization: Bearer {token}",
        "-H", "Content-Type: application/json",
    ]
    tmp_file = None
    try:
        if data:
            tmp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            json.dump(data, tmp_file)
            tmp_file.close()
            cmd += ["-d", f"@{tmp_file.name}"]
        result = subprocess.check_output(cmd, text=True)
        return json.loads(result) if result.strip() else {}
    finally:
        if tmp_file:
            os.unlink(tmp_file.name)


# =========================================================================
# Step 1: Build container
# =========================================================================
def build_image(cpr_image_uri: str, cpr_src_dir: str) -> None:
    print("=" * 60)
    print("Step 1: Building CPR container image via Cloud Build...")
    print("=" * 60)
    subprocess.run(
        ["gcloud", "builds", "submit", "--tag", cpr_image_uri, cpr_src_dir],
        check=True,
    )
    print(f"✓ Built and pushed {cpr_image_uri}")


# =========================================================================
# Step 2: Upload new model version
# =========================================================================
def upload_model(cpr_image_uri: str) -> str:
    print("=" * 60)
    print("Step 2: Uploading new model version...")
    print("=" * 60)

    # Get existing model info
    model_id = "1759715583697354752"
    model_url = f"{API_BASE}/{PARENT}/models/{model_id}"
    model_info = _curl_json("GET", model_url)
    artifact_uri = model_info["artifactUri"]
    parent_model = model_info["name"]
    print(f"  Existing model: {parent_model}")
    print(f"  Artifact URI:   {artifact_uri}")

    # Upload as new version
    upload_url = f"{API_BASE}/{PARENT}/models:upload"
    body = {
        "model": {
            "displayName": MODEL_DISPLAY_NAME,
            "artifactUri": artifact_uri,
            "containerSpec": {
                "imageUri": cpr_image_uri,
                "predictRoute": "/predict",
                "healthRoute": "/health",
            },
            "description": "SARIMAX CPR — updated with logging",
        },
        "parentModel": parent_model,
    }
    resp = _curl_json("POST", upload_url, body)
    operation_name = resp.get("name", "")
    print(f"  Upload operation: {operation_name}")

    # Poll until done
    while True:
        op = _curl_json("GET", f"{API_BASE}/{operation_name}")
        if op.get("done"):
            model_name = op.get("response", {}).get("model", "")
            model_version = op.get("response", {}).get("modelVersionId", "")
            versioned_name = f"{model_name}@{model_version}"
            print(f"✓ Model uploaded: {versioned_name}")
            return versioned_name
        print("  ... waiting for model upload")
        time.sleep(10)


# =========================================================================
# Step 3: Create endpoint and deploy
# =========================================================================
def deploy_to_endpoint(model_name: str) -> str:
    print("=" * 60)
    print("Step 3: Creating endpoint and deploying model...")
    print("=" * 60)

    # Create endpoint
    ep_url = f"{API_BASE}/{PARENT}/endpoints"
    ep_resp = _curl_json("POST", ep_url, {
        "displayName": "cashflow-sarimax-test-endpoint",
    })
    ep_op_name = ep_resp.get("name", "")
    print(f"  Endpoint creation operation: {ep_op_name}")

    # Wait for endpoint
    endpoint_name = ""
    while True:
        op = _curl_json("GET", f"{API_BASE}/{ep_op_name}")
        if op.get("done"):
            endpoint_name = op.get("response", {}).get("name", "")
            print(f"✓ Endpoint created: {endpoint_name}")
            break
        print("  ... waiting for endpoint creation")
        time.sleep(5)

    # Deploy model to endpoint
    deploy_url = f"{API_BASE}/{endpoint_name}:deployModel"
    deploy_body = {
        "deployedModel": {
            "model": model_name,
            "displayName": "sarimax-deployed",
            "dedicatedResources": {
                "machineSpec": {"machineType": "n1-standard-4"},
                "minReplicaCount": 1,
                "maxReplicaCount": 1,
            },
        },
        "trafficSplit": {"0": 100},
    }
    deploy_resp = _curl_json("POST", deploy_url, deploy_body)
    deploy_op_name = deploy_resp.get("name", "")
    print(f"  Deploy operation: {deploy_op_name}")

    # Wait for deployment (this can take 10-15 min)
    while True:
        op = _curl_json("GET", f"{API_BASE}/{deploy_op_name}")
        if op.get("done"):
            if "error" in op:
                print(f"✗ Deploy failed: {op['error']}")
                sys.exit(1)
            print("✓ Model deployed to endpoint!")
            break
        print("  ... waiting for deployment (this may take 10-15 min)")
        time.sleep(30)

    return endpoint_name


# =========================================================================
# Step 4: Prepare test instances
# =========================================================================
def prepare_test_instances(test_csv_path: str) -> list[dict]:
    print("=" * 60)
    print(f"Step 4: Preparing test instances from {test_csv_path}...")
    print("=" * 60)

    test_df = pd.read_csv(test_csv_path)
    oil_df = pd.read_csv(GCS_OIL_CSV)
    hol_df = pd.read_csv(GCS_HOLIDAYS_CSV)

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

    # Merge exogenous features
    test_df["date"] = pd.to_datetime(test_df["date"])
    test_df = test_df.merge(oil_df[["date", "oil_price"]], on="date", how="left")
    test_df = test_df.merge(holiday_flags, on="date", how="left")
    test_df["is_holiday"] = test_df["is_holiday"].fillna(0)
    test_df["oil_price"] = test_df["oil_price"].ffill().bfill()
    test_df["day_of_week"] = test_df["date"].dt.dayofweek

    # Format date for serialization
    test_df["date"] = test_df["date"].dt.strftime("%Y-%m-%d")

    instance_cols = [
        "id", "date", "store_nbr", "family",
        "onpromotion", "oil_price", "is_holiday", "day_of_week",
    ]

    # Use 1 date only — this is an online endpoint test
    first_date = test_df["date"].iloc[0]
    single_day = test_df[test_df["date"] == first_date]

    instances = single_day[instance_cols].to_dict(orient="records")
    # Ensure all values are JSON-serializable native types
    for inst in instances:
        for k, v in inst.items():
            if hasattr(v, "item"):  # numpy scalar
                inst[k] = v.item()

    print(f"  Prepared {len(instances)} test instances for date: {first_date}")
    print(f"  Last 10 instances: {instances[-10:]}")
    return instances


# =========================================================================
# Step 5: Test endpoint
# =========================================================================
def test_endpoint(endpoint_name: str, instances: list[dict]) -> None:
    print("=" * 60)
    print(f"Step 5: Testing endpoint with {len(instances)} instances...")
    print("=" * 60)

    predict_url = f"{API_BASE}/{endpoint_name}:predict"
    resp = _curl_json("POST", predict_url, {"instances": instances})

    if "error" in resp:
        print(f"✗ Prediction error: {resp['error']}")
        return

    predictions = resp.get("predictions", [])
    print(f"\n  Received {len(predictions)} predictions")
    if predictions:
        # Predictions are now dicts with full row data
        if isinstance(predictions[0], dict):
            print(f"  Last 3 predictions: {predictions[-3:]}")
            nums = [float(p["sales"]) for p in predictions]
        else:
            print(f"  Last 10: {predictions[-10:]}")
            nums = [float(p) for p in predictions]
        print(f"  Min: {min(nums):.2f}, Max: {max(nums):.2f}, Mean: {sum(nums)/len(nums):.2f}")


# =========================================================================
# Main
# =========================================================================
def main():
    cpr_image_uri = f"gcr.io/{PROJECT_ID}/cashflow-sarimax-cpr:latest"
    cpr_src_dir = os.path.join(os.path.dirname(__file__), "cpr_src")
    test_csv_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "test.csv")

    skip_build = "--skip-build" in sys.argv
    skip_upload = "--skip-upload" in sys.argv

    # Step 1: Build the container
    if skip_build:
        print("=" * 60)
        print("Step 1: SKIPPED (--skip-build)")
        print("=" * 60)
    else:
        build_image(cpr_image_uri, cpr_src_dir)

    # Step 2: Upload new model version
    if skip_upload:
        print("=" * 60)
        print("Step 2: SKIPPED (--skip-upload)")
        print("=" * 60)
        # Fetch the latest version of the existing model
        model_id = "1759715583697354752"
        model_url = f"{API_BASE}/{PARENT}/models/{model_id}"
        model_info = _curl_json("GET", model_url)
        latest_version = model_info.get("versionId", "1")
        model_name = f"{model_info['name']}@{latest_version}"
        print(f"  Using existing model: {model_name}")
    else:
        model_name = upload_model(cpr_image_uri)

    # Step 3: Deploy to endpoint
    if skip_upload:
        print("=" * 60)
        print("Step 3: SKIPPED (--skip-upload)")
        print("=" * 60)
        # Use the existing endpoint
        endpoint_name = f"{PARENT}/endpoints/5876454243858120704"
        print(f"  Using existing endpoint: {endpoint_name}")
    else:
        endpoint_name = deploy_to_endpoint(model_name)

    # Step 4: Prepare test data
    instances = prepare_test_instances(test_csv_path)

    # Step 5: Test it
    test_endpoint(endpoint_name, instances)

    print("\n" + "=" * 60)
    print("Done! Endpoint:")
    print(f"  {endpoint_name}")
    print("=" * 60)
    if not skip_upload:
        print("\nCleanup (when done):")
        print(f"  gcloud ai endpoints undeploy-model {endpoint_name.split('/')[-1]} \\")
        print(f"    --region={LOCATION} --deployed-model-id=<DEPLOYED_MODEL_ID>")
        print(f"  gcloud ai endpoints delete {endpoint_name.split('/')[-1]} --region={LOCATION}")


if __name__ == "__main__":
    main()
