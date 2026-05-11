#!/usr/bin/env bash
# =============================================================================
# setup_gcs_bucket.sh
#
# Creates a GCS bucket in the project's default region and grants the
# Compute Engine default service account the following IAM roles:
#   - roles/storage.admin     (full control over GCS objects & buckets)
#   - roles/logging.logWriter  (write logs to Cloud Logging)
#
# Usage:
#   ./setup_gcs_bucket.sh \
#       --project <GCP_PROJECT_ID> \
#       --bucket  <BUCKET_NAME_WITHOUT_gs://>
#
# Example:
#   ./setup_gcs_bucket.sh \
#       --project my-project \
#       --bucket  my-project-ml-bucket
# =============================================================================

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
PROJECT_ID=""
BUCKET=""

# ─── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --project)  PROJECT_ID="$2";  shift 2 ;;
    --bucket)   BUCKET="$2";      shift 2 ;;
    *) echo "❌ Unknown flag: $1"; exit 1 ;;
  esac
done

if [[ -z "$PROJECT_ID" || -z "$BUCKET" ]]; then
  echo "❌ --project and --bucket are required."
  echo "   Usage: $0 --project <PROJECT> --bucket <BUCKET>"
  exit 1
fi

# ─── Resolve default region & Compute Engine default SA ──────────────────────
echo ""
echo "============================================================"
echo " GCS Bucket Setup & IAM Configuration"
echo "============================================================"

# Get the default compute region from project metadata
DEFAULT_REGION=$(gcloud config get compute/region 2>/dev/null || true)
if [[ -z "$DEFAULT_REGION" ]]; then
  DEFAULT_REGION=$(gcloud compute project-info describe \
    --project="$PROJECT_ID" \
    --format="value(commonInstanceMetadata.items.filter(key='google-compute-default-region').firstof(value))" \
    2>/dev/null || true)
fi
if [[ -z "$DEFAULT_REGION" ]]; then
  DEFAULT_REGION="us-central1"
  echo "⚠️  No default region found – falling back to ${DEFAULT_REGION}"
fi

# Derive the Compute Engine default service account
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo " Project:    ${PROJECT_ID}"
echo " Project #:  ${PROJECT_NUMBER}"
echo " Region:     ${DEFAULT_REGION}"
echo " Bucket:     gs://${BUCKET}"
echo " Compute SA: ${COMPUTE_SA}"
echo "============================================================"

# ─── 1. Create the GCS bucket ───────────────────────────────────────────────
echo ""
echo "▶ Step 1/3: Creating GCS bucket gs://${BUCKET}"
if gsutil ls -b "gs://${BUCKET}" &>/dev/null; then
  echo "  (bucket already exists – skipping)"
else
  gsutil mb -p "$PROJECT_ID" -l "$DEFAULT_REGION" "gs://${BUCKET}"
  echo "  ✅ Bucket created"
fi

# ─── 2. Grant Storage Admin to Compute Engine default SA ─────────────────────
echo ""
echo "▶ Step 2/3: Granting roles/storage.admin to ${COMPUTE_SA}"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/storage.admin" \
  --condition=None \
  --quiet
echo "  ✅ roles/storage.admin granted"

# ─── 3. Grant Log Writer to Compute Engine default SA ────────────────────────
echo ""
echo "▶ Step 3/3: Granting roles/logging.logWriter to ${COMPUTE_SA}"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/logging.logWriter" \
  --condition=None \
  --quiet
echo "  ✅ roles/logging.logWriter granted"

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " ✅  Setup complete!"
echo ""
echo "  Bucket:  gs://${BUCKET}  (${DEFAULT_REGION})"
echo "  SA:      ${COMPUTE_SA}"
echo "  Roles:   storage.admin, logging.logWriter"
echo "============================================================"
