"""Load the trained model and make a single prediction from JSON."""
import argparse, json
from pathlib import Path
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('request', nargs='?', default=str(ROOT / 'data' / 'sample_request.json'))
    ap.add_argument('--model', default=str(ROOT / 'models' / 'model.joblib'))
    args = ap.parse_args()
    bundle = joblib.load(args.model)
    payload = json.loads(Path(args.request).read_text(encoding='utf-8'))
    values = payload.get('features', payload)
    row = {k: float(values.get(k, 0.0)) for k in bundle['features']}
    x = pd.DataFrame([row], columns=bundle['features'])
    prediction = bundle['model'].predict(x)[0]
    print(json.dumps({'project': bundle['project'], 'target': bundle['target'], 'prediction': float(prediction)}, indent=2))
if __name__ == '__main__': main()
