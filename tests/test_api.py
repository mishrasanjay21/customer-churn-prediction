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
