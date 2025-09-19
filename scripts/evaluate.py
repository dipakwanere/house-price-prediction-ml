from pathlib import Path
import sys

# Ensure project root is on sys.path to import `app.*`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from app.service.feature_engineering import build_preprocessing_pipeline


ARTIFACTS_DIR = Path("artifacts")
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
REPORT_PATH = ARTIFACTS_DIR / "metrics_report.txt"
DATA_DIR = Path("data") / "raw"


def load_train() -> pd.DataFrame:
    train_csv = DATA_DIR / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(
            f"{train_csv} not found. Run: python scripts/download_kaggle.py"
        )
    return pd.read_csv(train_csv)


def main() -> None:
    df = load_train()
    y = df["SalePrice"].astype(float)
    X = df.drop(columns=["SalePrice"])

    preprocessing = build_preprocessing_pipeline(df)
    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[("preprocess", preprocessing), ("model", model)])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # Negative MAE; take absolute and average
    neg_mae_scores = cross_val_score(
        pipeline, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
    )
    mae_scores = -neg_mae_scores

    # R2 via manual CV loop
    r2_scores = []
    for train_idx, valid_idx in cv.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_va)
        r2_scores.append(r2_score(y_va, preds))

    cv_mae_mean = float(np.mean(mae_scores))
    cv_mae_std = float(np.std(mae_scores))
    cv_r2_mean = float(np.mean(r2_scores))
    cv_r2_std = float(np.std(r2_scores))

    metrics = {
        "cv_mae_mean": cv_mae_mean,
        "cv_mae_std": cv_mae_std,
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,
        "n_splits": 5,
    }

    analysis_lines = [
        "Cross-Validation Evaluation (5-fold)",
        "- Metric: MAE (lower is better). Interpreted as average absolute dollars error.",
        f"- cv_mae_mean: {cv_mae_mean:,.2f}",
        f"- cv_mae_std:  {cv_mae_std:,.2f}",
        "- Metric: R^2 (higher is better). Proportion of variance explained (0..1+).",
        f"- cv_r2_mean:  {cv_r2_mean:.3f}",
        f"- cv_r2_std:   {cv_r2_std:.3f}",
    ]
    verdict = (
        "Result: Strong baseline."
        if cv_r2_mean >= 0.80
        else "Result: Reasonable baseline; consider feature engineering/model tuning."
    )
    analysis_lines.append(verdict)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    REPORT_PATH.write_text("\n".join(analysis_lines))
    print(json.dumps(metrics, indent=2))
    print("\n" + "\n".join(analysis_lines))


if __name__ == "__main__":
    main()


