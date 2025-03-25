from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json

# Load the trained model
model = joblib.load("fishing_zone_model.pkl")

# Store detected fishing locations
fishing_zones = []

# Define input schema
class ShipData(BaseModel):
    distance_from_shore: float
    distance_from_port: float
    speed: float
    course: float
    lat: float
    lon: float

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Fishing Zone Detection API is Running"}

@app.post("/predict")
def predict(data: ShipData):
    try:
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)[0]

        if prediction == 1:
            fishing_zones.append({"lat": data.lat, "lon": data.lon})

        return {"is_fishing": int(prediction)}

    except Exception as e:
        return {"error": str(e)}

@app.get("/fishing_zones")
def get_fishing_zones():
    return {"fishing_zones": fishing_zones}
