from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import load_model, predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
model = None
customers_db = {}


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


@app.get("/customers")
def list_customers():
    return list(customers_db.values())


@app.get("/customers/{customer_id}")
def get_customer(customer_id: int):
    customer = customers_db.get(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@app.post("/customers")
def create_customer(customer: CustomerData):
    customer_id = max(customers_db, default=0) + 1
    customer_data = customer.model_dump(exclude_none=True)
    customer_data["id"] = customer_id
    customers_db[customer_id] = customer_data
    return customer_data


@app.put("/customers/{customer_id}")
def update_customer(customer_id: int, customer: CustomerData):
    if customer_id not in customers_db:
        raise HTTPException(status_code=404, detail="Customer not found")

    update_data = customer.model_dump(exclude_none=True)
    customers_db[customer_id].update(update_data)
    customers_db[customer_id]["id"] = customer_id
    return customers_db[customer_id]


@app.delete("/customers/{customer_id}")
def delete_customer(customer_id: int):
    if customer_id not in customers_db:
        raise HTTPException(status_code=404, detail="Customer not found")
    del customers_db[customer_id]
    return {"detail": "Customer deleted successfully"}


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
