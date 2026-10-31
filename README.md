# Water Leak Risk Triage

**Project 096 of 100 · Public services & civic systems · Classification**

A standalone, runnable AI/ML project with its own saved fitted model, prediction API, and browser GUI. Its primary workflow is **Water Leak Risk Triage · Input quality check**. The feature fields, target, model, and project-specific control are configured for this use case.

## Included model and stack

- Saved fitted model: `models/model.joblib` (`KNeighborsClassifier`).
- Model code: `src/train.py` (retrain) and `src/predict.py` (CLI inference).
- Backend: FastAPI in `src/backend.py`, including health, metadata, model-info, prediction, batch, scenario comparison, and local sensitivity endpoints.
- Frontend/GUI: `frontend/index.html`, `frontend/app.js`, and `frontend/styles.css`.
- Optional technology focus: **Weights & Biases experiment tracking**. The local app works without that optional integration.

## Run the GUI and API

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-app.txt
python -m uvicorn src.backend:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` for the project GUI. The API docs are at `http://127.0.0.1:8000/docs`; health is `/api/health` and project-specific predictions use `POST /api/predict` with `{"features": {"<feature-name>": 0.0}}`.

To retrain and replace the included artifact, run `python src/train.py --limit 20000`. For a CLI prediction, run `python src/predict.py data/sample_request.json`.

## Model and data notes

The included model was actually fitted on a bounded sample of **5,000 synthetic rows** from this project's dataset, with 4,000 fit rows and 1,000 holdout rows. The saved artifact is ready for inference; it is not just an algorithm name or an untrained architecture.

Holdout metrics from this demonstration fit:
- `accuracy`: 0.8910
- `macro_f1`: 0.8910
- `roc_auc`: 0.9601

`data/dataset.csv` contains 449,367 synthetic rows and is sized to keep the whole project directory near 40 MB. The feature values are generated standardized signals for software demos, not measured real-world units or observations. Outputs are not validated for clinical, financial, safety, hiring, or other consequential decisions. Replace the synthetic data and retrain/evaluate before any real use.
