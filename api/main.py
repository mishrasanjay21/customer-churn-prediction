from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import load_model, predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
model = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model
    if MODEL_PATH.exists():
        model = load_model(MODEL_PATH)
    yield


app = FastAPI(
    title="Customer Churn Prediction API", version="1.0.0", lifespan=lifespan
)

class CustomerData(BaseModel):
    gender: str | None = None
    SeniorCitizen: int | None = None
    Partner: str | None = None
    Dependents: str | None = None
    tenure: float | None = None
    PhoneService: str | None = None
    MultipleLines: str | None = None
    InternetService: str | None = None
    OnlineSecurity: str | None = None
    OnlineBackup: str | None = None
    DeviceProtection: str | None = None
    TechSupport: str | None = None
    StreamingTV: str | None = None
    StreamingMovies: str | None = None
    Contract: str | None = None
    PaperlessBilling: str | None = None
    PaymentMethod: str | None = None
    MonthlyCharges: float | None = None
    TotalCharges: float | None = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is unavailable. Run 'python -m src.train' first.",
        )
    values = customer.model_dump() if hasattr(customer, "model_dump") else customer.dict()
    prediction, probability = predict(model, values)
    return {"churn_prediction": prediction, "churn_probability": probability}
