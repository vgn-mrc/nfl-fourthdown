Backend service for model inference

Overview
- FastAPI app exposing endpoints to run inference for three models:
  - wp (win probability for GO/FG/PUNT)
  - comp (component probabilities: fg_make, first_down conversion)
  - coach (coach policy classifier: GO, FIELD_GOAL, PUNT)

Artifacts
- Expected under `notebooks/artifacts_wp`, `notebooks/artifacts_comp`, `notebooks/artifacts_coach`.
- Each directory should contain:
  - `preprocess.joblib` (sklearn transformer)
  - `preprocess_feature_names.json` (transformed feature names)
  - `feature_meta.json` (feature lists, class order)
  - `*.best.weights.h5` (TensorFlow/Keras weights)

Important: These folders are currently git-ignored in `.gitignore` at repo root. To deploy on Hugging Face, either:
- Remove those ignore lines and commit the artifacts, or
- Copy artifacts into a tracked path and update `backend/app/main.py` paths accordingly.

Run locally
- Ensure project dependencies include `fastapi`, `uvicorn`, `tensorflow`, `pandas`, `scikit-learn`.
- Start server:
  - `uvicorn backend.app.main:app --reload --port 8000`

Endpoints
- GET `/health` → basic readiness
- GET `/metadata` → model and feature info
- POST `/predict/wp`
- POST `/predict/comp`
- POST `/predict/coach`
- POST `/predict/all`

Notes
- The API accepts raw, human-friendly inputs and derives model features to match training.
- Categorical bucketing and flags mirror the notebook logic.
 - On Spaces, default port is `$PORT`. The included `Procfile` uses it automatically.
