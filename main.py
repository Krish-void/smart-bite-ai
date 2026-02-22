from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
from contextlib import asynccontextmanager

# Define models
class RecommendRequest(BaseModel):
    weather: str = Field(..., json_schema_extra={"example": "sunny"})
    time_hour: int = Field(..., ge=0, le=23, json_schema_extra={"example": 14})
    city: str = Field(..., json_schema_extra={"example": "Mumbai"})

class RecommendResponse(BaseModel):
    recommended_dish: str
    confidence: float

# Global variables for model and encoder
model = None
encoder = None

def get_time_segment(h):
    if 5 <= h < 11: return 'Morning'
    if 11 <= h < 16: return 'Lunch'
    if 16 <= h < 19: return 'Tea-Time'
    if 19 <= h < 23: return 'Dinner'
    return 'Late-Night'

def get_weather_type(w):
    w_clean = str(w).strip()
    if w_clean in ['Sunny', 'Sandstorms', 'Windy']: return 'Hot/Dry'
    if w_clean in ['Stormy', 'Cloudy', 'Fog']: return 'Cool/Rainy'
    return 'Normal'

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoder
    model_path = 'model/food_recommender_model.joblib'
    encoder_path = 'model/encoder.joblib'
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
    yield

app = FastAPI(
    title="Smart Food Recommender API",
    description="A heavyweight API that provides real-data personalized food recommendations.",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Smart Food Recommender API (Intelligence Enhanced)!",
        "endpoints": {"health": "/health", "recommend": "/recommend (POST)", "docs": "/docs"}
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_version": "2.0.0", "model_loaded": model is not None}

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 1. Feature Engineering (Must match training)
        hour_sin = np.sin(2 * np.pi * request.time_hour / 24)
        hour_cos = np.cos(2 * np.pi * request.time_hour / 24)
        time_segment = get_time_segment(request.time_hour)
        weather_type = get_weather_type(request.weather)
        
        # 2. Encoding
        features = pd.DataFrame([{
            'weather': request.weather,
            'city': request.city,
            'time_segment': time_segment,
            'weather_type': weather_type
        }])
        X_encoded = encoder.transform(features)
        
        # 3. Predict
        X = np.hstack([X_encoded, [[hour_sin, hour_cos]]])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = float(np.max(probabilities))
        
        return RecommendResponse(recommended_dish=prediction, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
