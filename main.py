from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("model.pkl")

class Features(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(features: Features):
    prediction = model.predict([features.features])[0]
    probability = model.predict_proba([features.features])[0][0]
    return {"prediction": int(prediction), "probability": float(probability)}
