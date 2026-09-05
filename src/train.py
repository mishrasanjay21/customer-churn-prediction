import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_preprocessing import build_preprocessor, clean_data, load_data, split_features_target
from src.feature_engineering import add_features


def make_pipeline(model, features):
    return Pipeline([("preprocessor", build_preprocessor(features)), ("model", model)])


def evaluate(model, features, target):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "roc_auc": roc_auc_score(target, probabilities),
    }


def train(dataset_path: str | Path, model_path: str | Path, metrics_path: str | Path):
    data = add_features(clean_data(load_data(dataset_path)))
    features, target = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=3000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
        ),
    }
    results = {}
    fitted_models = {}
    for name, estimator in models.items():
        pipeline = make_pipeline(estimator, x_train)
        pipeline.fit(x_train, y_train)
        fitted_models[name] = pipeline
        results[name] = evaluate(pipeline, x_test, y_test)

    tuning_pipeline = make_pipeline(
        XGBClassifier(eval_metric="logloss", random_state=42), x_train
    )
    search = GridSearchCV(
        tuning_pipeline,
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.05, 0.1],
        },
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    results["tuned_xgboost"] = evaluate(search.best_estimator_, x_test, y_test)

    output_model = Path(model_path)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, output_model)
    output_metrics = Path(metrics_path)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train customer churn models.")
    parser.add_argument("--data", default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--model", default="models/churn_model.pkl")
    parser.add_argument("--metrics", default="models/metrics.json")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.model, args.metrics), indent=2))


if __name__ == "__main__":
    main()
