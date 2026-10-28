# EXPERIMENT NO. 05: REST API TO SERVE A MACHINE LEARNING MODEL

---

## 1. EXPERIMENT DETAILS
* **Subject:** Machine Learning Operations (MLOps)
* **Experiment No:** 05
* **Title:** Developing and Deploying a REST API to Serve a Machine Learning Model
* **Faculty:** Prof. Swapnil Pawar
* **Domain / Problem Statement:** Air Quality Index (AQI) Prediction REST API (Derived from Experiment 4)
* **Frameworks & Tools:** FastAPI, Uvicorn, Scikit-Learn, Pydantic, Python 3.11

---

## 2. AIM & OBJECTIVES

### Aim:
To design, implement, deploy, and validate a production-ready REST API using FastAPI that serves a Machine Learning model trained on the atmospheric pollutant dataset engineered during Experiment 04.

### Objectives:
1. Re-use the preprocessed and feature-engineered dataset from Experiment 04 (*Air Quality Index Prediction*).
2. Train a non-linear regression model (`RandomForestRegressor`) and serialize it into a deployable artifact (`aqi_model.pkl`).
3. Develop high-throughput REST API endpoints (`GET /`, `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict-batch`) using FastAPI.
4. Enforce strict type validation and request body constraints using Pydantic models.
5. Provide automatic OpenAPI documentation via interactive Swagger UI (`/docs`).
6. Validate endpoint performance and prediction accuracy across multiple atmospheric conditions using automated client tests (`test_api.py`).

---

## 3. SOFTWARE & HARDWARE REQUIREMENTS

* **Operating System:** Windows 10/11, Linux, or macOS
* **Runtime:** Python 3.10+ (Tested on Python 3.11.9)
* **Core Libraries:**
  * `fastapi`: Modern, fast web framework for building APIs with Python.
  * `uvicorn`: Lightning-fast ASGI web server implementation.
  * `pydantic`: Data validation and settings management using Python type hints.
  * `scikit-learn`: Machine learning model training, evaluation, and serialization.
  * `pandas` & `numpy`: Tabular data processing and numerical transformations.
  * `requests`: Automated HTTP client for end-to-end integration tests.

---

## 4. ARCHITECTURE & SYSTEM WORKFLOW

```
┌─────────────────────────────────────────────────────────────┐
│             Experiment 04 Feature Pipeline                  │
│       (data/air_quality_data.csv: 288 hourly records)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Model Training & Serialization (train_model.py)   │
│  - Features: PM10, PM2.5, CO, NO2, SO2, O3                  │
│  - Target: us_aqi (Numeric US Air Quality Index)            │
│  - Train/Test Split: 80% Train (230) / 20% Test (58)        │
│  - Model: RandomForestRegressor(n_estimators=100)           │
│  - Artifacts Generated: aqi_model.pkl, model_metadata.json  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: REST API Deployment with ASGI Server (app.py)     │
│  - Framework: FastAPI + Uvicorn                             │
│  - Endpoints:                                               │
│    * GET  /              -> API metadata and route listing  │
│    * GET  /health        -> Health check & liveliness probe │
│    * GET  /model-info    -> Model metrics & hyperparameters │
│    * POST /predict       -> Real-time AQI prediction        │
│    * POST /predict-batch -> High-throughput batch inference│
│    * GET  /docs          -> Interactive Swagger UI          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Client Verification & Validation (test_api.py)    │
│  - Validates API responses against HTTP status 200 OK       │
│  - Tests scenarios: Rural/Clean, Moderate, Severe Pollution │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. PROBLEM STATEMENT & DATASET MAPPING

Per the assignment guidelines, the problem statement was selected from **Experiment 04** rather than the baseline House Price example:

### Input Features:
1. `pm10`: Coarse particulate matter concentration $\le 10\ \mu\text{m}\ (\mu\text{g}/\text{m}^3)$
2. `pm2_5`: Fine particulate matter concentration $\le 2.5\ \mu\text{m}\ (\mu\text{g}/\text{m}^3)$
3. `carbon_monoxide`: CO concentration $(\mu\text{g}/\text{m}^3)$
4. `nitrogen_dioxide`: $\text{NO}_2$ concentration $(\mu\text{g}/\text{m}^3)$
5. `sulphur_dioxide`: $\text{SO}_2$ concentration $(\mu\text{g}/\text{m}^3)$
6. `ozone`: Surface $\text{O}_3$ concentration $(\mu\text{g}/\text{m}^3)$

### Target Output:
* `predicted_aqi`: Numeric US Air Quality Index (0 to 500+)
* `aqi_category`: Standard EPA Air Quality Tier:
  * **0 – 50:** Good
  * **51 – 100:** Moderate
  * **101 – 150:** Unhealthy for Sensitive Groups
  * **151 – 200:** Unhealthy
  * **201 – 300:** Very Unhealthy
  * **301+:** Hazardous
* `health_advisory`: Actionable public health guidance based on the predicted AQI level.

---

## 6. MODEL TRAINING & EVALUATION

Executing `train_model.py` fits a `RandomForestRegressor` and produces the following metrics on unseen test data:

* **Total Dataset Size:** 288 records
* **Training Set:** 230 records (80%)
* **Testing Set:** 58 records (20%)
* **Train $R^2$ Score:** `0.9703`
* **Test $R^2$ Score:** `0.7838`
* **Test MAE:** `16.0310`
* **Test RMSE:** `20.9604`

---

## 7. REST API ENDPOINTS SPECIFICATION

| HTTP Method | Path | Description | Sample Status |
|---|---|---|---|
| `GET` | `/` | Root information, service version, available endpoints | `200 OK` |
| `GET` | `/health` | Liveliness check for container orchestrators | `200 OK` |
| `GET` | `/model-info` | Metadata, hyperparameters, and test metrics | `200 OK` |
| `POST` | `/predict` | Real-time single sensor inference | `200 OK` |
| `POST` | `/predict-batch` | High-throughput batch inference | `200 OK` |
| `GET` | `/docs` | Interactive Swagger UI documentation | `200 OK` |

---

## 8. STEP-BY-STEP EXECUTION GUIDE

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Train the ML Model
```powershell
python train_model.py
```
*Output:*
```
Loading Air Quality dataset from: data\air_quality_data.csv
Total dataset records: 288
Features: ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']
Target: us_aqi

--- Model Evaluation ---
Train R² Score : 0.9703
Test R² Score  : 0.7838
Test MAE       : 16.0310
Test RMSE      : 20.9604

Trained model successfully saved to 'aqi_model.pkl'
Model metadata successfully saved to 'model_metadata.json'
```

### Step 3: Run the REST API Server
```powershell
uvicorn app:app --reload --port 8000
```
The server binds to `http://127.0.0.1:8000`.

### Step 4: Access Interactive Swagger UI
Open your browser and navigate to:
```
http://127.0.0.1:8000/docs
```
You can inspect schemas and test endpoints directly from the browser interface.

### Step 5: Execute Automated Verification Tests
In a second terminal window, run:
```powershell
python test_api.py
```

*Verification Output:*
```
Connected to live FastAPI server at http://127.0.0.1:8000!

=================================================================
1. Testing GET / (Root Endpoint)
Status Code: 200
{
  "message": "Air Quality Index (AQI) REST API is running",
  "experiment": "Experiment 5 - REST API for Machine Learning Model",
  "version": "1.0.0",
  ...
}

=================================================================
2. Testing GET /health (Health Check)
Status Code: 200
{
  "status": "healthy",
  "model_loaded": true,
  "algorithm": "RandomForestRegressor"
}

=================================================================
4. Testing POST /predict (Single Prediction Scenarios)

>> Scenario: Clean Air / Rural Environment
Status: 200
Predicted AQI : 99.57
Category      : Moderate
Advisory      : Air quality is acceptable. Sensitive individuals may experience minor symptoms.
Dominant      : PM2.5

>> Scenario: Moderate City Traffic
Status: 200
Predicted AQI : 131.87
Category      : Unhealthy for Sensitive Groups
Advisory      : Sensitive groups should reduce prolonged outdoor exertion.
Dominant      : PM2.5

>> Scenario: Severe Industrial Smog / High Pollution
Status: 200
Predicted AQI : 204.66
Category      : Very Unhealthy
Advisory      : Health alert: The risk of health effects is increased for the entire population.
Dominant      : PM2.5
```

---

## 9. MLOps PRINCIPLES DEMONSTRATED

1. **Decoupled Training & Serving:** Model training logic (`train_model.py`) is completely decoupled from online serving (`app.py`), enabling independent update cycles.
2. **Schema Enforcement & Input Sanitization:** Pydantic models prevent malformed inputs, missing features, and invalid negative concentrations before requests reach the model.
3. **Model Metadata Versioning:** Evaluation metrics and feature lists are preserved in `model_metadata.json` for model governance and lineage tracking.
4. **Health Check Probes:** `/health` endpoint enables integration with Docker, Kubernetes, and cloud load balancers.
5. **Interactive Contract Documentation:** FastAPI automatically generates OpenAPI 3.0 schemas and Swagger UI for cross-team collaboration.

---

## 10. CONCLUSION

Experiment 05 successfully developed and deployed a production-ready REST API using FastAPI to serve an Air Quality Index (AQI) machine learning model derived from Experiment 04. The implementation fulfills all requirements specified by the instructor, includes robust input validation, batch processing, comprehensive automated tests, and interactive API documentation.

