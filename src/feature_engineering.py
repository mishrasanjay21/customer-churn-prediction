import pandas as pd


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add stable numeric features without changing the raw input columns."""
    engineered = data.copy()
    if {"TotalCharges", "tenure"}.issubset(engineered.columns):
        tenure = engineered["tenure"].replace(0, 1)
        engineered["AvgMonthlySpend"] = engineered["TotalCharges"] / tenure
    if "tenure" in engineered.columns:
        engineered["TenureGroup"] = pd.cut(
            engineered["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12", "12-24", "24-48", "48-72"],
        )
    return engineered
