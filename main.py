import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("models/model.joblib")
vec = joblib.load("models/vectorizer.joblib")


class Ticket(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"message": "Hello, Ticket Classifier"}


@app.post("/predict")
def predict(ticket: Ticket):
    features = vec.transform([ticket.text])
    result = model.predict(features)[0]
    return {"category": str(result)}