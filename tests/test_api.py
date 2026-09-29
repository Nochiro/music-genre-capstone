from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["models_loaded"]["cnn"] is True
    assert data["models_loaded"]["lstm"] is True
    assert data["models_loaded"]["resnet"] is True


def test_predict_requires_audio():
    response = client.post(
        "/predict",
        data={"model": "resnet"},
    )

    assert response.status_code == 422
