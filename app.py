from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("fishing_zone_model.pkl")

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
    """
    Input: JSON object with ship's details (distance_from_shore, distance_from_port, speed, course, lat, lon)
    Output: Fishing prediction (1: Fishing, 0: Not Fishing)
    """
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])  # Convert Pydantic model to dictionary
        
        # Predict
        prediction = model.predict(df)[0]

        return {"is_fishing": int(prediction)}

    except Exception as e:
        return {"error": str(e)}
