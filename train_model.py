import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load Dataset from Experiment 4
data_path = os.path.join("data", "air_quality_data.csv")
if not os.path.exists(data_path):
    # Fallback to sibling Experiment 4 directory if run from root
    data_path = os.path.join("..", "DataIngestionMLOPS", "artifacts", "processed_data.csv")

print(f"Loading Air Quality dataset from: {data_path}")
df = pd.read_csv(data_path)

# 2. Select Features and Target
# Input features: atmospheric pollutant concentrations (µg/m³)
features = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone"
]
target = "us_aqi"

X = df[features]
y = df[target]

print(f"Total dataset records: {len(df)}")
print(f"Features: {features}")
print(f"Target: {target}")

# 3. Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Model
# RandomForestRegressor captures non-linear atmospheric interactions
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# 5. Evaluate Model
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_r2 = float(r2_score(y_train, train_preds))
test_r2 = float(r2_score(y_test, test_preds))
test_mae = float(mean_absolute_error(y_test, test_preds))
test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))

print("\n--- Model Evaluation ---")
print(f"Train R² Score : {train_r2:.4f}")
print(f"Test R² Score  : {test_r2:.4f}")
print(f"Test MAE       : {test_mae:.4f}")
print(f"Test RMSE      : {test_rmse:.4f}")

# 6. Save Model Artifact (.pkl)
model_path = "aqi_model.pkl"
with open(model_path, "wb") as file:
    pickle.dump(model, file)
print(f"\nTrained model successfully saved to '{model_path}'")

# 7. Save MLOps Metadata (.json)
metadata = {
    "model_name": "Air Quality Index (AQI) Predictor",
    "algorithm": "RandomForestRegressor",
    "version": "1.0.0",
    "experiment": "Experiment 5 - REST API for ML Model",
    "features": features,
    "target": target,
    "dataset_records": int(len(df)),
    "train_records": int(len(X_train)),
    "test_records": int(len(X_test)),
    "metrics": {
        "train_r2": round(train_r2, 4),
        "test_r2": round(test_r2, 4),
        "test_mae": round(test_mae, 4),
        "test_rmse": round(test_rmse, 4)
    }
}

metadata_path = "model_metadata.json"
with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4)
print(f"Model metadata successfully saved to '{metadata_path}'")
