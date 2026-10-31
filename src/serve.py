from pathlib import Path
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'model.joblib'
FEATURES = ["wait_time_min", "application_completeness", "service_load", "repeat_visits", "travel_distance_km", "staff_ratio", "response_delay_hours", "neighborhood_index", "weather_impact", "request_complexity"]
def get_bundle():
    if not MODEL_PATH.exists(): raise RuntimeError('Train first: python src/train.py --limit 20000')
    return joblib.load(MODEL_PATH)
def predict(values):
    bundle = get_bundle()
    row = {k: float(values.get(k, 0.0)) for k in FEATURES}
    return float(bundle['model'].predict(pd.DataFrame([row], columns=FEATURES))[0])

from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI(title='AI/ML Starter API')
class Request(BaseModel):
    features: dict[str, float]
@app.get('/health')
def health(): return {'status':'ok','model_ready':MODEL_PATH.exists()}
@app.post('/predict')
def predict_route(request: Request):
    return {'prediction': predict(request.features)}
