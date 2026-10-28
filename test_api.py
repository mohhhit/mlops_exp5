import json
import requests
from app import app, predict_aqi, root, health_check, model_info, AirQualityInput

SERVER_URL = "http://127.0.0.1:8000"

# Sample test cases representing different atmospheric pollution profiles
TEST_CASES = [
    {
        "name": "Clean Air / Rural Environment",
        "payload": {
            "pm10": 15.0,
            "pm2_5": 8.0,
            "carbon_monoxide": 180.0,
            "nitrogen_dioxide": 6.5,
            "sulphur_dioxide": 2.0,
            "ozone": 20.0
        }
    },
    {
        "name": "Moderate City Traffic",
        "payload": {
            "pm10": 45.0,
            "pm2_5": 28.0,
            "carbon_monoxide": 260.0,
            "nitrogen_dioxide": 15.0,
            "sulphur_dioxide": 4.5,
            "ozone": 38.0
        }
    },
    {
        "name": "Severe Industrial Smog / High Pollution",
        "payload": {
            "pm10": 120.0,
            "pm2_5": 95.0,
            "carbon_monoxide": 450.0,
            "nitrogen_dioxide": 35.0,
            "sulphur_dioxide": 14.0,
            "ozone": 60.0
        }
    }
]


def test_via_http():
    """Test endpoints using real HTTP requests against running uvicorn server."""
    print("=" * 65)
    print("1. Testing GET / (Root Endpoint)")
    r = requests.get(f"{SERVER_URL}/")
    print(f"Status Code: {r.status_code}")
    print(json.dumps(r.json(), indent=2))

    print("\n" + "=" * 65)
    print("2. Testing GET /health (Health Check)")
    r = requests.get(f"{SERVER_URL}/health")
    print(f"Status Code: {r.status_code}")
    print(json.dumps(r.json(), indent=2))

    print("\n" + "=" * 65)
    print("3. Testing GET /model-info (Model Metadata)")
    r = requests.get(f"{SERVER_URL}/model-info")
    print(f"Status Code: {r.status_code}")
    print(json.dumps(r.json(), indent=2))

    print("\n" + "=" * 65)
    print("4. Testing POST /predict (Single Prediction Scenarios)")
    for tc in TEST_CASES:
        print(f"\n>> Scenario: {tc['name']}")
        res = requests.post(f"{SERVER_URL}/predict", json=tc["payload"])
        print(f"Status: {res.status_code}")
        body = res.json()
        print(f"Predicted AQI : {body['predicted_aqi']}")
        print(f"Category      : {body['aqi_category']}")
        print(f"Advisory      : {body['health_advisory']}")
        print(f"Dominant      : {body['dominant_pollutant']}")

    print("\n" + "=" * 65)
    print("5. Testing POST /predict-batch (Batch Prediction)")
    batch_payload = [tc["payload"] for tc in TEST_CASES]
    res = requests.post(f"{SERVER_URL}/predict-batch", json=batch_payload)
    print(f"Status: {res.status_code}")
    print(f"Records Processed: {res.json()['total_records']}")
    print("Batch test successfully passed!")


def test_offline_unit():
    """Test endpoint logic directly without requiring live server."""
    print("=" * 65)
    print("RUNNING OFFLINE ENDPOINT UNIT TESTS")
    print("=" * 65)

    print("\n[Test 1] Root Endpoint:")
    print(root())

    print("\n[Test 2] Health Endpoint:")
    print(health_check())

    print("\n[Test 3] Model Info Endpoint:")
    print(model_info())

    print("\n[Test 4] Prediction Scenarios:")
    for tc in TEST_CASES:
        print(f"\n>> Scenario: {tc['name']}")
        input_obj = AirQualityInput(**tc["payload"])
        res = predict_aqi(input_obj)
        print(f"Predicted AQI : {res.predicted_aqi}")
        print(f"Category      : {res.aqi_category}")
        print(f"Advisory      : {res.health_advisory}")
        print(f"Dominant      : {res.dominant_pollutant}")


if __name__ == "__main__":
    is_live = False
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=1.0)
        if r.status_code == 200:
            is_live = True
    except Exception:
        is_live = False

    if is_live:
        print(f"Connected to live FastAPI server at {SERVER_URL}!\n")
        test_via_http()
    else:
        print(f"No active server on {SERVER_URL}. Running local verification tests:\n")
        test_offline_unit()
        print("\n" + "=" * 65)
        print("To test against a live HTTP server, run in another terminal:")
        print("    uvicorn app:app --reload")
        print("and then re-run:")
        print("    python test_api.py")
        print("=" * 65)

