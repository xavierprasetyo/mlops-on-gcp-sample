#!/usr/bin/env bash
# =============================================================================
# build_training_image.sh
#
# Builds and pushes the custom training container image to Artifact Registry.
# This script is decoupled from the Jupyter notebook so the image build can be
# run independently (e.g. from CI/CD or a local terminal).
#
# Usage:
#   ./build_training_image.sh \
#       --project  <GCP_PROJECT_ID> \
#       --region   <GCP_REGION> \
#       --bucket   <GCS_BUCKET_NAME_WITHOUT_gs://> \
#       [--repo    <AR_REPO_NAME>]   \
#       [--image   <IMAGE_NAME>]     \
#       [--version <IMAGE_TAG>]
#
# Example:
#   ./build_training_image.sh \
#       --project my-project \
#       --region us-central1 \
#       --bucket my-project-bucket
# =============================================================================

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
REPO_NAME="beans-model-trainer"
IMAGE_NAME="scikit-beans"
VERSION="v1"
PROJECT_ID=""
REGION=""
BUCKET=""

# ─── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --project)  PROJECT_ID="$2";  shift 2 ;;
    --region)   REGION="$2";      shift 2 ;;
    --bucket)   BUCKET="$2";      shift 2 ;;
    --repo)     REPO_NAME="$2";   shift 2 ;;
    --image)    IMAGE_NAME="$2";  shift 2 ;;
    --version)  VERSION="$2";     shift 2 ;;
    *) echo "❌ Unknown flag: $1"; exit 1 ;;
  esac
done

if [[ -z "$PROJECT_ID" || -z "$REGION" || -z "$BUCKET" ]]; then
  echo "❌ --project, --region, and --bucket are required."
  echo "   Usage: $0 --project <PROJECT> --region <REGION> --bucket <BUCKET>"
  exit 1
fi

REPO_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"
FULL_IMAGE="${REPO_URI}/${IMAGE_NAME}:${VERSION}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/traincontainer"

echo "============================================================"
echo " Training Image Builder"
echo "============================================================"
echo " Project:    ${PROJECT_ID}"
echo " Region:     ${REGION}"
echo " Bucket:     ${BUCKET}"
echo " Repository: ${REPO_URI}"
echo " Image:      ${FULL_IMAGE}"
echo "============================================================"

# ─── 1. Scaffold the build directory ─────────────────────────────────────────
echo ""
echo "▶ Step 1/6: Scaffolding ${BUILD_DIR}/"
mkdir -p "${BUILD_DIR}/trainer"

# ─── 2. Write the Dockerfile ─────────────────────────────────────────────────
echo "▶ Step 2/6: Writing Dockerfile"
cat > "${BUILD_DIR}/Dockerfile" <<'DOCKERFILE'
FROM us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0
WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

RUN pip install \
  scikit-learn==1.2 \
  'protobuf>=3.9.2,<3.20' \
  google-cloud-bigquery \
  google-cloud-storage \
  joblib \
  pandas \
  db_dtypes

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train"]
DOCKERFILE

# ─── 3. Write the training script ────────────────────────────────────────────
echo "▶ Step 3/6: Writing trainer/train.py (bucket=${BUCKET})"
cat > "${BUILD_DIR}/trainer/train.py" <<TRAINPY
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from google.cloud import bigquery
from google.cloud import storage
from joblib import dump

import os
import pandas as pd

bqclient = bigquery.Client()
storage_client = storage.Client()

def download_table(bq_table_uri: str):
    prefix = "bq://"
    if bq_table_uri.startswith(prefix):
        bq_table_uri = bq_table_uri[len(prefix):]

    table = bigquery.TableReference.from_string(bq_table_uri)
    rows = bqclient.list_rows(
        table,
    )
    return rows.to_dataframe(create_bqstorage_client=False)

# These environment variables are from Vertex AI managed datasets
training_data_uri = os.environ["AIP_TRAINING_DATA_URI"]
test_data_uri = os.environ["AIP_TEST_DATA_URI"]

# Download data into Pandas DataFrames, split into train / test
df = download_table(training_data_uri)
test_df = download_table(test_data_uri)
labels = df.pop("Class").tolist()
data = df.values.tolist()
test_labels = test_df.pop("Class").tolist()
test_data = test_df.values.tolist()

skmodel = DecisionTreeClassifier()
skmodel.fit(data, labels)
score = skmodel.score(test_data, test_labels)
print('accuracy is:',score)

# Save the model to a local file
dump(skmodel, "model.joblib")

# Upload the saved model file to GCS
bucket = storage_client.get_bucket("${BUCKET}")
model_directory = os.environ["AIP_MODEL_DIR"]
storage_path = os.path.join(model_directory, "model.joblib")
blob = storage.blob.Blob.from_string(storage_path, client=storage_client)
blob.upload_from_filename("model.joblib")
TRAINPY

# ─── 4. Create the Artifact Registry repository (idempotent) ─────────────────
echo "▶ Step 4/6: Ensuring Artifact Registry repo exists"
gcloud artifacts repositories create "${REPO_NAME}" \
  --location="${REGION}" \
  --repository-format=docker \
  2>/dev/null || echo "  (repository already exists – skipping)"

# ─── 5. Configure docker auth & build ────────────────────────────────────────
echo "▶ Step 5/6: Building container image"
echo y | gcloud auth configure-docker "${REGION}-docker.pkg.dev" 2>/dev/null
docker build "${BUILD_DIR}" -t "${FULL_IMAGE}"

# ─── 6. Push to Artifact Registry ────────────────────────────────────────────
echo "▶ Step 6/6: Pushing ${FULL_IMAGE}"
docker push "${FULL_IMAGE}"

echo ""
echo "============================================================"
echo " ✅  Image pushed: ${FULL_IMAGE}"
echo "============================================================"
