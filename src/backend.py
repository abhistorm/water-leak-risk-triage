"""Inference API and local browser GUI for this AI/ML starter."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
CONFIG = __import__('json').loads((ROOT / 'project.json').read_text(encoding='utf-8'))
MODEL_PATH = ROOT / 'models' / 'model.joblib'
if not MODEL_PATH.exists():
    raise RuntimeError('Trained model not found. Run: python src/train.py --limit 5000')
BUNDLE = joblib.load(MODEL_PATH)
MODEL = BUNDLE['model']
FEATURES = list(BUNDLE['features'])
TASK_TYPE = BUNDLE['task_type']
app = FastAPI(title=CONFIG['name'], version='1.0.0', description='Synthetic-data educational ML demo')


def checked_features(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise HTTPException(422, 'features must be a JSON object')
    missing = [name for name in FEATURES if name not in payload]
    if missing:
        raise HTTPException(422, f'Missing feature values: {", ".join(missing)}')
    result: dict[str, float] = {}
    for name in FEATURES:
        try:
            value = float(payload[name])
        except (TypeError, ValueError):
            raise HTTPException(422, f'{name} must be numeric')
        if not math.isfinite(value):
            raise HTTPException(422, f'{name} must be finite')
        result[name] = value
    return result


def raw_prediction(values: dict[str, float]):
    frame = pd.DataFrame([{name: values[name] for name in FEATURES}], columns=FEATURES)
    value = MODEL.predict(frame)[0]
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, (np.integer, int)):
        value = int(value)
    elif isinstance(value, (np.floating, float)):
        value = float(value)
    probabilities: dict[str, float] = {}
    if TASK_TYPE == 'classification' and hasattr(MODEL, 'predict_proba'):
        raw = MODEL.predict_proba(frame)[0]
        classes = getattr(MODEL, 'classes_', range(len(raw)))
        probabilities = {str(c.item() if hasattr(c, 'item') else c): float(p) for c, p in zip(classes, raw)}
    return value, probabilities


def prediction_result(values: dict[str, float]) -> dict[str, Any]:
    prediction, probabilities = raw_prediction(values)
    confidence = max(probabilities.values()) if probabilities else None
    return {
        'project_id': CONFIG['id'], 'project': CONFIG['name'], 'task_type': TASK_TYPE,
        'target': CONFIG['target'], 'prediction': prediction, 'probabilities': probabilities,
        'confidence': confidence, 'model': MODEL.__class__.__name__,
        'synthetic_demo': True,
    }


@app.get('/api/health')
def health():
    return {'status': 'ok', 'model_ready': True, 'project_id': CONFIG['id']}


@app.get('/api/metadata')
def metadata():
    return CONFIG


@app.get('/api/model')
def model_info():
    return {
        'project': CONFIG['name'], 'algorithm': MODEL.__class__.__name__,
        'task_type': TASK_TYPE, 'artifact': 'models/model.joblib',
        'training': CONFIG.get('model_training', {}), 'features': FEATURES,
        'synthetic_demo': True,
    }


@app.post('/api/predict')
def predict(payload: dict[str, Any]):
    values = checked_features(payload.get('features'))
    return prediction_result(values)


@app.post('/api/batch')
def batch_predict(payload: dict[str, Any]):
    rows = payload.get('rows')
    if not isinstance(rows, list) or not rows:
        raise HTTPException(422, 'rows must be a non-empty array')
    if len(rows) > 100:
        raise HTTPException(413, 'Batch limit is 100 rows per request')
    return {'project_id': CONFIG['id'], 'count': len(rows), 'results': [prediction_result(checked_features(row)) for row in rows]}


@app.post('/api/compare')
def compare(payload: dict[str, Any]):
    base = checked_features(payload.get('base'))
    scenarios = payload.get('scenarios')
    if not isinstance(scenarios, list) or not scenarios or len(scenarios) > 10:
        raise HTTPException(422, 'scenarios must contain 1–10 entries')
    results = [{'label': 'Baseline', **prediction_result(base)}]
    for index, scenario in enumerate(scenarios, start=1):
        if not isinstance(scenario, dict):
            raise HTTPException(422, f'Scenario {index} must be an object')
        label = str(scenario.get('label') or f'Scenario {index}')[:80]
        values = checked_features(scenario.get('features'))
        results.append({'label': label, **prediction_result(values)})
    return {'project_id': CONFIG['id'], 'results': results}


@app.post('/api/explain')
def explain(payload: dict[str, Any]):
    values = checked_features(payload.get('features'))
    base, _ = raw_prediction(values)
    if not isinstance(base, (float, int)):
        base = float(base)
    impacts = []
    step = float(payload.get('step', 0.5))
    step = min(max(abs(step), 0.05), 2.0)
    for feature in FEATURES:
        low = dict(values); high = dict(values)
        low[feature] -= step; high[feature] += step
        low_pred, _ = raw_prediction(low); high_pred, _ = raw_prediction(high)
        try:
            delta = float(high_pred) - float(low_pred)
            impacts.append({'feature': feature, 'effect': abs(delta), 'direction': 'up' if delta > 0 else 'down' if delta < 0 else 'flat'})
        except (TypeError, ValueError):
            impacts.append({'feature': feature, 'effect': 0.0, 'direction': 'categorical'})
    scale = max((entry['effect'] for entry in impacts), default=0.0) or 1.0
    for entry in impacts:
        entry['relative_effect'] = entry['effect'] / scale
    impacts.sort(key=lambda entry: entry['effect'], reverse=True)
    return {'project_id': CONFIG['id'], 'baseline_prediction': base, 'step': step, 'drivers': impacts, 'method': 'local ±step perturbation', 'synthetic_demo': True}


@app.get('/')
def gui():
    return FileResponse(ROOT / 'frontend' / 'index.html')


@app.get('/README.md')
def project_readme():
    return FileResponse(ROOT / 'README.md', media_type='text/markdown')

app.mount('/', StaticFiles(directory=ROOT / 'frontend'), name='frontend')
