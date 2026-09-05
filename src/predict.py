from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering import add_features


def load_model(path: str | Path):
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run 'python -m src.train' first."
        )
    return joblib.load(model_path)


def predict(model, customer: dict) -> tuple[int, float]:
    frame = add_features(pd.DataFrame([customer])).drop(
        columns=["customerID"], errors="ignore"
    )
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])
    return prediction, probability
