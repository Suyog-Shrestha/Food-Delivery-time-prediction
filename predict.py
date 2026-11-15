from typing import Literal

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = "models/food_delivery_time_model.json"

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)
FEATURES = model.get_booster().feature_names

app = FastAPI(title="Food Delivery Time Predictor")


class Order(BaseModel):
    delivery_person_age: int = Field(25, ge=18, le=50)
    delivery_person_ratings: float = Field(4.5, ge=1, le=5)
    vehicle_condition: int = Field(2, ge=0, le=3)
    multiple_deliveries: int = Field(0, ge=0, le=3)
    distance_km: float = Field(5.0, ge=1, le=30)
    cook_diff: int = Field(10, ge=5, le=300, description="Prep time in minutes")
    order_hour: int = Field(19, ge=0, le=23)
    weather: Literal["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"] = "Sunny"
    traffic: Literal["Low", "Medium", "High", "Jam"] = "Medium"
    order_type: Literal["Buffet", "Drinks", "Meal", "Snack"] = "Meal"
    vehicle_type: Literal["bicycle", "electric_scooter", "motorcycle", "scooter"] = "motorcycle"
    festival: Literal["No", "Yes"] = "No"
    city: Literal["Urban", "Semi-Urban", "Metropolitian"] = "Urban"


def to_features(o: Order) -> pd.DataFrame:
    row = {
        "delivery_person_age": o.delivery_person_age,
        "delivery_person_ratings": o.delivery_person_ratings,
        "vehicle_condition": o.vehicle_condition,
        "multiple_deliveries": o.multiple_deliveries,
        "distance": o.distance_km / 111,  # model was trained on degrees
        "cook_diff": o.cook_diff,
        "order_hour": o.order_hour,
        "weatherconditions_conditions Fog": int(o.weather == "Fog"),
        "weatherconditions_conditions Sandstorms": int(o.weather == "Sandstorms"),
        "weatherconditions_conditions Stormy": int(o.weather == "Stormy"),
        "weatherconditions_conditions Sunny": int(o.weather == "Sunny"),
        "weatherconditions_conditions Windy": int(o.weather == "Windy"),
        "road_traffic_density_Jam": int(o.traffic == "Jam"),
        "road_traffic_density_Low": int(o.traffic == "Low"),
        "road_traffic_density_Medium": int(o.traffic == "Medium"),
        "type_of_order_Drinks": int(o.order_type == "Drinks"),
        "type_of_order_Meal": int(o.order_type == "Meal"),
        "type_of_order_Snack": int(o.order_type == "Snack"),
        "type_of_vehicle_motorcycle": int(o.vehicle_type == "motorcycle"),
        "type_of_vehicle_scooter": int(o.vehicle_type == "scooter"),
        "festival_No": int(o.festival == "No"),
        "festival_Yes": int(o.festival == "Yes"),
        "city_NaN": 0,
        "city_Semi-Urban": int(o.city == "Semi-Urban"),
        "city_Urban": int(o.city == "Urban"),
    }
    return pd.DataFrame([row])[FEATURES]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(order: Order):
    pred = float(model.predict(to_features(order))[0])
    return {"predicted_delivery_time_minutes": round(pred, 1)}
