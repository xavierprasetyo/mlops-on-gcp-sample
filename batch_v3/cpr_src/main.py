import os
import sys
import logging
import traceback
from fastapi import FastAPI, Request

from predictor import SarimaxPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = FastAPI()

AIP_STORAGE_URI = os.environ.get("AIP_STORAGE_URI", "/models")
predictor = SarimaxPredictor()

try:
    logger.info(f"Starting CPR container. AIP_STORAGE_URI: {AIP_STORAGE_URI}")
    predictor.load(AIP_STORAGE_URI)
    logger.info("Predictor loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load predictor: {str(e)}")
    logger.error(traceback.format_exc())
    import time
    time.sleep(2)
    sys.exit(1)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
        instances = predictor.preprocess(body)
        predictions = predictor.predict(instances)
        return predictor.postprocess(predictions)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
