from fastapi.testclient import TestClient

from Part4_API.app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_by_customer_id():
    response = client.post("/predict", json={"customer_id": "CUST00001"})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["predicted_class"] in [0, 1]
    assert body["risk_level"] in ["low", "medium", "high"]
    assert body["risk_explanation"]


def test_batch_predict():
    response = client.post(
        "/batch_predict",
        json={"records": [{"customer_id": "CUST00001"}, {"customer_id": "CUST00002"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
