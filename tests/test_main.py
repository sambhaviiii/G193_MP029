from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_INPUT = {
    "items_in_order": 5,
    "aisle_distance_m": 120.5,
    "picker_exp_months": 12,
    "shelf_height_m": 2.5,
    "shift_hour": 10,
    "order_weight_kg": 15.2,
    "zone_type": "ambient",
    "shift": "Morning"
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200
    assert "predicted_pick_seconds" in response.json()
    assert response.json()["predicted_pick_seconds"] > 0


def test_shelf_height_validation():
    data = VALID_INPUT.copy()
    data["shelf_height_m"] = 5.0

    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_negative_weight_validation():
    data = VALID_INPUT.copy()
    data["order_weight_kg"] = -1

    response = client.post("/predict", json=data)
    assert response.status_code == 422