from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessing_pipeline(train_df: pd.DataFrame) -> ColumnTransformer:
    """Create a ColumnTransformer that imputes and encodes Kaggle House Prices features.

    - Numerical: median impute + StandardScaler
    - Categorical: most_frequent impute + OneHotEncoder(handle_unknown="ignore")
    """

    target_col = "SalePrice"
    feature_df = train_df.drop(columns=[target_col], errors="ignore")

    # Identify numeric vs categorical columns
    numeric_cols: List[str] = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols: List[str] = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessing


