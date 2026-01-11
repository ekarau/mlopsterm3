import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI(title="Course Completion Prediction API")

MODEL_PATH = "models/model.pkl"
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    pred = model.predict(df)[0]
    return {"prediction": str(pred)}
