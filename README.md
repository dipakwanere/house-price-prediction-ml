# House Price Prediction API (FastAPI) — Kaggle House Prices

FastAPI service for house price prediction using the Kaggle competition dataset
`House Prices: Advanced Regression Techniques`.

## Features
- Train a regression model and save it to `artifacts/model.joblib`
- FastAPI app with:
  - `GET /health` health check
  - `POST /predict` (JSON body: `{ "features": { ...raw kaggle columns... } }`)
  - `POST /admin/reload-model` to reload the artifact
  - `GET /frontend/index.html` info page
- Typed request/response models via Pydantic

## Project layout
```
.
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ schemas.py
│  └─ service/
│     ├─ __init__.py
│     └─ model.py
├─ scripts/
│  ├─ download_kaggle.py
│  ├─ train.py
│  └─ evaluate.py
├─ artifacts/  # created after training
├─ notebooks/
│  └─ ca_housing_dataset_EDA.ipynb
├─ requirements.txt
├─ data/
│  └─ raw/  # created by download step (train.csv, test.csv, data_description.txt)
└─ README.md
```

## Setup (Windows, PowerShell)
```powershell
# 1) Create and activate a virtual environment (recommended)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Configure Kaggle API (one-time)
# Place kaggle.json at %USERPROFILE%\.kaggle\kaggle.json (Windows)
# or C:\Users\<you>\.kaggle\kaggle.json and ensure 600 permissions

# 4) Download competition data
python scripts/download_kaggle.py

# 5) Train and save the model
python scripts/train.py

# 4) Run the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API
- Health
```http
GET http://localhost:8000/health
```

- Predict (POST JSON)
```http
POST http://localhost:8000/predict
Content-Type: application/json

{
  "features": {
    "MSSubClass": 60,
    "MSZoning": "RL",
    "LotFrontage": 65,
    "LotArea": 8450,
    "Street": "Pave",
    "Alley": null,
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "OverallQual": 7,
    "OverallCond": 5,
    "YearBuilt": 2003,
    "YearRemodAdd": 2003,
    "RoofStyle": "Gable",
    "Neighborhood": "CollgCr",
    "1stFlrSF": 856,
    "2ndFlrSF": 854,
    "GrLivArea": 1710,
    "FullBath": 2,
    "BedroomAbvGr": 3
    // ... any other raw columns are accepted; missing values are imputed
  }
}
```
Response: `{ "price": 215000.0 }`

- Reload model
```http
POST http://localhost:8000/admin/reload-model
```

Note: The model predicts median house value in USD (scaled from the dataset's target units).

## Re-train with different model settings
Edit `scripts/train.py` (pipeline or estimator params), rerun training, then restart the API.

## Notes
- Ensure `artifacts/model.joblib` exists before hitting `/predict`. If not, run the training step.
- Input schema: the API accepts a flexible `features` dict; preprocessing handles missing values and categories.
- EDA notebook: open `notebooks/ca_housing_dataset_EDA.ipynb` for detailed exploration and column explanations.

## What has been done
- Switched to Kaggle House Prices dataset
- Added Kaggle download script and instructions
- Implemented feature engineering with imputation + scaling + OHE
- Refactored training/evaluation scripts
- Updated API to `POST /predict` with JSON
- Expanded EDA notebook with column descriptions and diagnostics

## Current status
- Model code: Implemented for Kaggle dataset
- Training: Pending until you run download + train commands above
- Feature engineering: Implemented in `app/service/feature_engineering.py`
- Evaluation: Cross-validation metrics in `scripts/evaluate.py`; prints and saves `artifacts/metrics.json`

## What has been done
- Virtual environment setup and activation
- Dependencies pinned in `requirements.txt`
- Training script `scripts/train.py` implemented (loads scikit-learn California Housing, builds pipeline with `StandardScaler` + `RandomForestRegressor`, trains, evaluates, saves to `artifacts/model.joblib`)
- FastAPI application in `app/main.py` with health check, predict, and admin reload
- Model service utilities in `app/service/model.py`
- EDA notebook `notebooks/ca_housing_dataset_EDA.ipynb`
- `.gitignore` added for common Python/IDE/artifacts

## Current status
- Model code: Implemented
- Training: Pending until you run `python scripts/train.py` (creates `artifacts/model.joblib`). If `artifacts/model.joblib` exists, training has been completed already.
- Feature engineering: Minimal (standardization via `StandardScaler` in the pipeline). No domain feature crafting beyond the original dataset features.
- Evaluation: Basic MAE and R² printed in `scripts/train.py`. For deeper analysis, see the EDA notebook.
