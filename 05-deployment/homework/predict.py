import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


class ClientData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


app = FastAPI(title="homework prediction")

with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

datapoint = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}


def predict_client(x):
    return pipeline.predict_proba(x)[0, 1]


@app.post("/predict")
def predict(data: ClientData):
    print("Received data:", data, type(data))
    prediction = predict_client(data.model_dump())
    return {'prediction': prediction}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9696)
