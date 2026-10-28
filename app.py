import os
import json
import pickle
from typing import List
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="Air Quality Index (AQI) Prediction REST API",
    description="MLOps Experiment 5: Production-grade REST API serving a trained Machine Learning model for Air Quality Index (US AQI) prediction.",
    version="1.0.0"
)

# Load Trained Model and Metadata
MODEL_PATH = "aqi_model.pkl"
METADATA_PATH = "model_metadata.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please run 'python train_model.py' first.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

metadata = {}
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

FEATURE_COLUMNS = metadata.get("features", [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone"
])


# Input Schema Definition
class AirQualityInput(BaseModel):
    pm10: float = Field(..., ge=0.0, description="Particulate Matter <= 10 µm (µg/m³)", example=85.0)
    pm2_5: float = Field(..., ge=0.0, description="Fine Particulate Matter <= 2.5 µm (µg/m³)", example=72.5)
    carbon_monoxide: float = Field(..., ge=0.0, description="Carbon Monoxide concentration (µg/m³)", example=340.0)
    nitrogen_dioxide: float = Field(..., ge=0.0, description="Nitrogen Dioxide concentration (µg/m³)", example=24.5)
    sulphur_dioxide: float = Field(..., ge=0.0, description="Sulphur Dioxide concentration (µg/m³)", example=8.2)
    ozone: float = Field(..., ge=0.0, description="Surface Ozone concentration (µg/m³)", example=35.0)


# Response Schema Definition
class PredictionResponse(BaseModel):
    predicted_aqi: float
    aqi_category: str
    health_advisory: str
    dominant_pollutant: str
    input_features: dict


# Helper Function: US EPA Air Quality Index Breakdown & Health Advisory
def categorize_aqi(aqi_value: float):
    if aqi_value <= 50:
        return "Good", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi_value <= 100:
        return "Moderate", "Air quality is acceptable. Sensitive individuals may experience minor symptoms."
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "Sensitive groups should reduce prolonged outdoor exertion."
    elif aqi_value <= 200:
        return "Unhealthy", "Everyone may begin to experience health effects; sensitive groups may experience more serious effects."
    elif aqi_value <= 300:
        return "Very Unhealthy", "Health alert: The risk of health effects is increased for the entire population."
    else:
        return "Hazardous", "Health warning of emergency conditions: Avoid all outdoor physical activity."


# Endpoints
@app.get("/", tags=["General"])
def root():
    """
    Root endpoint: Provides API status, metadata, and documentation URL.
    """
    return {
        "message": "Air Quality Index (AQI) REST API is running",
        "experiment": "Experiment 5 - REST API for Machine Learning Model",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "model_info": "/model-info",
            "predict": "POST /predict",
            "predict_batch": "POST /predict-batch"
        }
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """
    Health check endpoint for container/orchestrator liveliness probes.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "algorithm": metadata.get("algorithm", "RandomForestRegressor")
    }


@app.get("/model-info", tags=["Monitoring"])
def model_info():
    """
    Returns trained model specifications, feature dependencies, and validation metrics.
    """
    return metadata


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_aqi(data: AirQualityInput):
    """
    Perform real-time Air Quality Index (AQI) inference for a single atmospheric measurement.
    """
    try:
        features_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        input_dict = {
            "pm10": [data.pm10],
            "pm2_5": [data.pm2_5],
            "carbon_monoxide": [data.carbon_monoxide],
            "nitrogen_dioxide": [data.nitrogen_dioxide],
            "sulphur_dioxide": [data.sulphur_dioxide],
            "ozone": [data.ozone]
        }
        input_df = pd.DataFrame(input_dict)[FEATURE_COLUMNS]

        prediction = float(model.predict(input_df)[0])
        predicted_aqi = max(0.0, round(prediction, 2))
        category, advisory = categorize_aqi(predicted_aqi)

        dominant = "PM2.5" if data.pm2_5 > (data.pm10 * 0.5) else "PM10"

        return PredictionResponse(
            predicted_aqi=predicted_aqi,
            aqi_category=category,
            health_advisory=advisory,
            dominant_pollutant=dominant,
            input_features=features_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict-batch", tags=["Inference"])
def predict_batch(records: List[AirQualityInput]):
    """
    Perform batch inference on multiple sensor readings.
    """
    results = []
    for record in records:
        pred_res = predict_aqi(record)
        results.append(pred_res)
    return {
        "total_records": len(results),
        "predictions": results
    }
