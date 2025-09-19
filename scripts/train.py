import os
import sys
from pathlib import Path

# Ensure project root is on sys.path to import `app.*`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.service.feature_engineering import build_preprocessing_pipeline


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
RAW_COLUMNS_PATH = ARTIFACTS_DIR / "raw_feature_columns.json"
DATA_DIR = Path("data") / "raw"


def load_kaggle_house_prices() -> tuple[pd.DataFrame, pd.Series]:
    train_csv = DATA_DIR / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(
            f"{train_csv} not found. Run: python scripts/download_kaggle.py"
        )
    df = pd.read_csv(train_csv)
    y = df["SalePrice"].astype(float)
    X = df.drop(columns=["SalePrice"])
    return X, y


def main() -> None:
    print("[Train] Loading Kaggle House Prices dataset (train.csv)...")
    X, y = load_kaggle_house_prices()
    full_df = X.copy()
    full_df["SalePrice"] = y

    print("[Train] Building preprocessing pipeline (impute+scale+one-hot)...")
    preprocessing = build_preprocessing_pipeline(full_df)

    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[("preprocess", preprocessing), ("model", model)])

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[Train] Training RandomForestRegressor (n=500)...")
    pipeline.fit(X_train, y_train)

    print("[Train] Evaluating on hold-out validation set...")
    y_pred = pipeline.predict(X_valid)
    mae = float(mean_absolute_error(y_valid, y_pred))
    r2 = float(r2_score(y_valid, y_pred))
    print(f"[Train] Hold-out metrics -> MAE: {mae:,.0f} USD | R2: {r2:.3f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[Train] Saved model to {MODEL_PATH}")
    # Save expected raw feature columns for inference alignment
    import json

    RAW_COLUMNS_PATH.write_text(json.dumps(list(X.columns), indent=2))
    print(f"[Train] Saved raw feature columns to {RAW_COLUMNS_PATH}")


if __name__ == "__main__":
	main()
