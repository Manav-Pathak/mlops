from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"msg": "ML Model API Running"}

@app.post("/predict")
def predict(features: list[float]):
    prediction = model.predict([features])
    return {"prediction": prediction.tolist()}