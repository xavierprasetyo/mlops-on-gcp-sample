import os
import sys
import logging
import traceback
from fastapi import FastAPI, Request

from predictor import SarimaxPredictor

# Configure logging to BOTH stdout and stderr for Vertex AI Cloud Logging
# Vertex AI may capture either stream depending on the runtime
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Vertex AI injects AIP_STORAGE_URI pointing to the model artifact directory
AIP_STORAGE_URI = os.environ.get("AIP_STORAGE_URI", "/models")

predictor = SarimaxPredictor()

try:
    logger.info(f"Starting up CPR container. AIP_STORAGE_URI: {AIP_STORAGE_URI}")
    logger.info("Python version: %s", sys.version)
    logger.info("Environment: GOOGLE_CLOUD_PROJECT=%s", os.environ.get('GOOGLE_CLOUD_PROJECT', 'NOT SET'))
    # Log artifact directory contents for debugging
    if os.path.isdir(AIP_STORAGE_URI):
        logger.info("Artifact directory contents: %s", os.listdir(AIP_STORAGE_URI))
    else:
        logger.info("AIP_STORAGE_URI is not a local directory (will download from GCS)")
    predictor.load(AIP_STORAGE_URI)
    logger.info("Predictor loaded successfully.")
except Exception as e:
    error_msg = f"Failed to load predictor during startup: {str(e)}\n{traceback.format_exc()}"
    # Include environment variables for debugging
    env_dump = "\n".join(f"  {k}={v}" for k, v in sorted(os.environ.items()))
    full_error = f"{error_msg}\n\nEnvironment variables:\n{env_dump}"
    logger.error(full_error)
    print(full_error, file=sys.stderr)
    sys.stderr.flush()

    # Write crash log to GCS for post-mortem debugging
    try:
        from google.cloud import storage as gcs_storage
        client = gcs_storage.Client()
        bucket = client.bucket("vertex-dump")
        blob = bucket.blob("batch_predictions/sarimax/crash_log.txt")
        blob.upload_from_string(full_error)
        logger.info("Crash log written to gs://vertex-dump/batch_predictions/sarimax/crash_log.txt")
    except Exception as upload_err:
        logger.error(f"Could not write crash log to GCS: {upload_err}")

    # Flush all handlers so Cloud Logging captures the error before exit
    for handler in logging.root.handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(1)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
        instances = body.get("instances", [])
        logger.info("Received %d instances. Type: %s. Sample: %s",
                     len(instances),
                     type(instances[0]).__name__ if instances else "empty",
                     str(instances[0])[:200] if instances else "empty")
        instances_df = predictor.preprocess(body)
        logger.info("Preprocessed DataFrame shape: %s, dtypes: %s",
                     instances_df.shape, instances_df.dtypes.to_dict())
        predictions = predictor.predict(instances_df)
        logger.info("Returning %d predictions for %d instances",
                     len(predictions), len(instances))
        result = predictor.postprocess(predictions)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
