from pathlib import Path

from fastapi.testclient import TestClient

from api import main

def test_health_without_model(monkeypatch):
    monkeypatch.setattr(main, "MODEL_PATH", Path("missing-model-for-test.pkl"))
    main.model = None
    with TestClient(main.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}

def test_predict_without_model_returns_service_unavailable(monkeypatch):
    monkeypatch.setattr(main, "MODEL_PATH", Path("missing-model-for-test.pkl"))
    main.model = None
    with TestClient(main.app) as client:
        response = client.post("/predict", json={"tenure": 12})
    assert response.status_code == 503
    assert "python -m src.train" in response.json()["detail"]


def test_customer_crud_flow(monkeypatch):
    monkeypatch.setattr(main, "MODEL_PATH", Path("missing-model-for-test.pkl"))
    main.model = None
    main.customers_db.clear()

    with TestClient(main.app) as client:
        response = client.get("/customers")
        assert response.status_code == 200
        assert response.json() == []

        create_response = client.post("/customers", json={"gender": "Male", "tenure": 12})
        assert create_response.status_code == 200
        created_customer = create_response.json()
        assert created_customer["id"] == 1
        assert created_customer["gender"] == "Male"

        get_response = client.get("/customers/1")
        assert get_response.status_code == 200
        assert get_response.json()["gender"] == "Male"

        update_response = client.put("/customers/1", json={"gender": "Female", "tenure": 18})
        assert update_response.status_code == 200
        assert update_response.json()["gender"] == "Female"
        assert update_response.json()["tenure"] == 18

        delete_response = client.delete("/customers/1")
        assert delete_response.status_code == 200
        assert delete_response.json()["detail"] == "Customer deleted successfully"

        missing_response = client.get("/customers/1")
        assert missing_response.status_code == 404
