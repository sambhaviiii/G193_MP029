import logging
import json
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

# Load trained model
model = joblib.load(MODEL_PATH)


# Configure JSON logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Warehouse Picking Time API")


# Request schema
class PredictionRequest(BaseModel):
    items_in_order: int = Field(ge=1)
    aisle_distance_m: float = Field(ge=0)
    picker_exp_months: int = Field(ge=0)
    shelf_height_m: float = Field(ge=0, le=4)
    shift_hour: int = Field(ge=0, le=23)
    order_weight_kg: float = Field(ge=0)
    zone_type: str
    shift: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):

    # Convert request into a DataFrame
    input_data = pd.DataFrame([request.model_dump()])

    # Generate prediction
    prediction = model.predict(input_data)[0]

    # JSON logging
    logger.info(json.dumps({
        "event": "prediction",
        "predicted_pick_seconds": round(float(prediction), 2),
        "shift": request.shift,
        "zone_type": request.zone_type
    }))

    return {
        "predicted_pick_seconds": round(float(prediction), 2)
    }