from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Create FastAPI application
app = FastAPI(
    title="House Price Prediction API",
    description="REST API for predicting house prices",
    version="1.0"
)

# Load trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)


# Input data format
class HouseData(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    age: int


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


@app.post("/predict")
def predict_price(house: HouseData):

    input_data = [[
        house.area,
        house.bedrooms,
        house.bathrooms,
        house.age
    ]]

    prediction = model.predict(input_data)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }