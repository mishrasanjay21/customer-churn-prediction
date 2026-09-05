# Customer Churn Prediction

## Project layout

```text
customer-churn-prediction/
├── data/
│   ├── raw/          # Original dataset files
│   └── processed/    # Generated cleaned datasets
├── models/           # Trained model artifacts
├── notebooks/        # Exploration notebooks
├── src/              # Training and inference code
├── tests/            # Automated tests
├── requirements.txt
└── venv/             # Local virtual environment (ignored by Git)
```

## Setup on Windows

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the customer churn CSV file in `data/raw/` before running the training
pipeline. The training package compares Logistic Regression, Random Forest,
and XGBoost, then tunes and saves the best XGBoost pipeline.

## Train the model

```powershell
python -m src.train
```

The command writes `models/churn_model.pkl` and `models/metrics.json`.

## Run the API

```powershell
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger UI. `/health` works before a
model exists; `/predict` returns `503` until training has been completed.
