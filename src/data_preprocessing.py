from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the Telco CSV and fail with an actionable error when absent."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Download the Kaggle Telco CSV "
            "and place it in data/raw/."
        )
    return pd.read_csv(csv_path)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco data and encode the target column."""
    cleaned = data.copy()
    required = {TARGET_COLUMN}
    missing = required.difference(cleaned.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(
            cleaned["TotalCharges"], errors="coerce"
        )
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    if cleaned[TARGET_COLUMN].isna().any():
        raise ValueError("Churn must contain only 'Yes' and 'No' values.")
    return cleaned.drop(columns=[ID_COLUMN], errors="ignore")


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing.")
    return data.drop(columns=[TARGET_COLUMN]), data[TARGET_COLUMN].astype(int)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
