"""Train the food delivery time model.

Usage (from repo root):
    python train.py                      # uses data/train.csv
    python train.py --data path/to.csv

Writes models/food_delivery_time_model.json (the file predict.py and app.py load).
Dataset: Kaggle "Food Delivery Dataset" (train.csv only; Time_taken(min) is the target).
"""
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

SEED = 42

# The exact 25 columns (and order) predict.py / app.py send to the model.
FEATURES = [
    "delivery_person_age", "delivery_person_ratings", "vehicle_condition",
    "multiple_deliveries", "distance", "cook_diff", "order_hour",
    "weatherconditions_conditions Fog", "weatherconditions_conditions Sandstorms",
    "weatherconditions_conditions Stormy", "weatherconditions_conditions Sunny",
    "weatherconditions_conditions Windy",
    "road_traffic_density_Jam", "road_traffic_density_Low", "road_traffic_density_Medium",
    "type_of_order_Drinks", "type_of_order_Meal", "type_of_order_Snack",
    "type_of_vehicle_motorcycle", "type_of_vehicle_scooter",
    "festival_No", "festival_Yes",
    "city_NaN", "city_Semi-Urban", "city_Urban",
]


def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(exclude="number").columns:
        df[c] = df[c].astype(str).str.strip()
    df = df.replace("NaN", np.nan)

    df["target"] = df["Time_taken(min)"].str.extract(r"(\d+)").astype(float)
    for c in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # time features (minutes between order and pickup; % 1440 fixes orders that cross midnight)
    ordered = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce")
    picked = pd.to_datetime(df["Time_Order_picked"], format="%H:%M:%S", errors="coerce")
    df["cook_diff"] = ((picked - ordered).dt.total_seconds() / 60) % 1440
    df["order_hour"] = ordered.dt.hour

    # distance in degrees (some rows have negative coordinates -> use abs)
    for c in ["Restaurant_latitude", "Restaurant_longitude",
              "Delivery_location_latitude", "Delivery_location_longitude"]:
        df[c] = df[c].abs()
    df["distance"] = np.sqrt(
        (df["Restaurant_latitude"] - df["Delivery_location_latitude"]) ** 2
        + (df["Restaurant_longitude"] - df["Delivery_location_longitude"]) ** 2
    )

    df["City"] = df["City"].fillna("NaN")  # kept as its own category -> city_NaN
    df = df.dropna(subset=["Delivery_person_Age", "Delivery_person_Ratings", "order_hour",
                           "Road_traffic_density", "multiple_deliveries", "Festival"])
    df = df[df["Weatherconditions"] != "conditions NaN"]
    return df.reset_index(drop=True)


def build_features(df):
    df = df.rename(columns={
        "Delivery_person_Age": "delivery_person_age",
        "Delivery_person_Ratings": "delivery_person_ratings",
        "Vehicle_condition": "vehicle_condition",
    })
    dummies = pd.get_dummies(
        df[["Weatherconditions", "Road_traffic_density", "Type_of_order",
            "Type_of_vehicle", "Festival", "City"]], dtype=int)
    dummies.columns = [c.replace("Weatherconditions_", "weatherconditions_")
                        .replace("Road_traffic_density_", "road_traffic_density_")
                        .replace("Type_of_order_", "type_of_order_")
                        .replace("Type_of_vehicle_", "type_of_vehicle_")
                        .replace("Festival_", "festival_")
                        .replace("City_", "city_") for c in dummies.columns]
    X = pd.concat([df[["delivery_person_age", "delivery_person_ratings", "vehicle_condition",
                       "multiple_deliveries", "distance", "cook_diff", "order_hour"]], dummies], axis=1)
    return X.reindex(columns=FEATURES, fill_value=0)  # baseline categories are dropped here


def evaluate(y_true, y_pred):
    return {"RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--out", default="models/food_delivery_time_model.json")
    args = ap.parse_args()

    df = load_and_clean(args.data)
    X, y = build_features(df), df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED)
    model.fit(X_tr, y_tr)
    print("Test metrics:", {k: round(v, 3) for k, v in evaluate(y_te, model.predict(X_te)).items()})
    model.save_model(args.out)
    print("Saved", args.out)
