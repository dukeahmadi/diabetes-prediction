from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# بار گذاری مدل
with open("model_lgbm_diabetes.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# مدل داده ورودی با Pydantic
class Features(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "API is running!"}

@app.post("/predict")
def predict(data: Features):
    arr = np.array(data.features).reshape(1, -1)
    prediction = model.predict(arr)
    return {"prediction": int(prediction[0])}
