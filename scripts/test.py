"""Quick validation script.

Runs a tiny train/validate cycle and single prediction to ensure imports and
pipelines work end-to-end with the Kaggle dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.service.feature_engineering import build_preprocessing_pipeline


def main() -> None:
    data_dir = ROOT / "data" / "raw"
    train_csv = data_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(
            f"{train_csv} not found. Run: python scripts/download_kaggle.py"
        )

    df = pd.read_csv(train_csv)
    y = df["SalePrice"].astype(float)
    X = df.drop(columns=["SalePrice"])

    pre = build_preprocessing_pipeline(df)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_va)
    mae = float(mean_absolute_error(y_va, preds))
    r2 = float(r2_score(y_va, preds))
    print({"mae": mae, "r2": r2, "n_valid": len(y_va)})

    # Single-row inference check
    sample = X_va.iloc[[0]].to_dict(orient="records")[0]
    single_pred = float(pipe.predict(pd.DataFrame([sample]))[0])
    print({"single_pred": single_pred})


if __name__ == "__main__":
    main()


