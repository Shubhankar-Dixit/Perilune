from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_predict_with_object_id() -> None:
    response = client.post("/api/predict", json={"objectId": "KOI-0001", "dryRun": True})
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in {"planet-candidate", "false-positive"}
    assert "features" in payload


def test_search_transits_endpoint() -> None:
    times = [float(i) for i in range(10)]
    flux = [1.0] * 10
    response = client.post(
        "/api/search-transits",
        json={"times": times, "flux": flux, "dryRun": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "result" in payload
    assert "best_period_days" in payload["result"]


def test_predict_rejects_length_mismatch() -> None:
    response = client.post(
        "/api/predict",
        json={"times": [0.0, 1.0], "flux": [1.0], "dryRun": True},
    )
    assert response.status_code == 422
