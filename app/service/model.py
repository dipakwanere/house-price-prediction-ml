import os
import json
import threading
from typing import Any, Dict

import joblib
import pandas as pd

_MODEL_LOCK = threading.RLock()
_MODEL: Any = None
_RAW_FEATURE_COLUMNS: list[str] | None = None
_DEFAULT_MODEL_PATH = os.path.join("artifacts", "model.joblib")
_DEFAULT_COLUMNS_PATH = os.path.join("artifacts", "raw_feature_columns.json")


def get_model_path() -> str:
	return os.getenv("MODEL_PATH", _DEFAULT_MODEL_PATH)


def load_model(model_path: str | None = None, columns_path: str | None = None) -> None:
    """Load the trained model artifact and expected raw columns (thread-safe)."""
    global _MODEL, _RAW_FEATURE_COLUMNS
    path = model_path or get_model_path()
    cols_path = columns_path or _DEFAULT_COLUMNS_PATH
    with _MODEL_LOCK:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artifact not found at: {path}. Train the model first.")
        _MODEL = joblib.load(path)
        if os.path.exists(cols_path):
            with open(cols_path, "r", encoding="utf-8") as f:
                _RAW_FEATURE_COLUMNS = json.load(f)


def is_model_loaded() -> bool:
	with _MODEL_LOCK:
		return _MODEL is not None


def predict(features: Dict[str, Any]) -> float:
    """Predict SalePrice from raw Kaggle features (single record).

    Missing columns are added with NaN and handled by the preprocessing pipeline.
    """
    with _MODEL_LOCK:
        if _MODEL is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        # Create single-row DataFrame
        input_df = pd.DataFrame([features])

        # Reindex to expected raw columns if known
        global _RAW_FEATURE_COLUMNS
        if _RAW_FEATURE_COLUMNS:
            input_df = input_df.reindex(columns=_RAW_FEATURE_COLUMNS)

        pred = _MODEL.predict(input_df)[0]
        return float(pred)
