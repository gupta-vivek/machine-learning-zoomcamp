import pickle
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any, Literal
from pydantic import BaseModel, conint, confloat

# request


class Customer(BaseModel):
    model_config = {"extra": "forbid"}
    
    gender: Literal["male", "female"]
    seniorcitizen: int
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal["electronic_check", "mailed_check",
                           "bank_transfer_(automatic)", "credit_card_(automatic)"]
    tenure: int
    monthlycharges: float
    totalcharges: float


class PredictionResponse(BaseModel):
    churn_probability: float
    churn: bool


app = FastAPI(title="churn prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

datapoint = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}


def predict_single(customer:Customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer:Customer) -> PredictionResponse:
    result = predict_single(customer.model_dump())
    # result = predict_single(customer)

    return {
        "churn_probability": result,
        "churn": bool(result >= 0.5)
    }


# a = predict_single(datapoint)
# print(f"Churn probability: {a:.3f}")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
