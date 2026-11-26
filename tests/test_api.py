import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import app.main as app_main
import app.monitor as app_monitor
from app.schemas import TransactionRequest
from src.config import FEATURE_COLS, PCA_COLS
from src.features import build_preprocessor


def _sample_payload(**overrides) -> dict:
    payload = {"Time": 120.0, "Amount": 45.0, **{col: 0.0 for col in PCA_COLS}}
    payload.update(overrides)
    return payload


def test_schema_rejects_negative_amount():
    with pytest.raises(ValidationError):
        TransactionRequest(**_sample_payload(Amount=-5.0))


def test_api_health_predict_and_stats(tmp_path, monkeypatch):
    # 1. Build a tiny dummy Pipeline so the test never touches mlflow.db
    X_dummy = pd.DataFrame([_sample_payload(V14=-5.0), _sample_payload(V14=0.5)], columns=FEATURE_COLS)
    y_dummy = np.array([1, 0])
    dummy_pipeline = Pipeline([
        ("prep", build_preprocessor("robust", False)),
        ("clf", LogisticRegression()),
    ]).fit(X_dummy, y_dummy)

    # 2. Redirect model loader + log path to isolated test doubles
    monkeypatch.setattr(app_main, "load_champion_model", lambda: (dummy_pipeline, 0.5, "test-v1"))
    monkeypatch.setattr(app_monitor, "PREDICTION_LOG_PATH", tmp_path / "predictions.jsonl")

    # 3. `with TestClient(app)` triggers FastAPI lifespan startup!
    with TestClient(app_main.app) as client:
        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json()["model_version"] == "test-v1"

        pred_res = client.post("/predict", json=_sample_payload(V14=-5.0))
        assert pred_res.status_code == 200
        body = pred_res.json()
        assert {"is_fraud", "fraud_probability", "threshold", "model_version"} <= body.keys()

        bad_res = client.post("/predict", json=_sample_payload(Amount=-1.0))
        assert bad_res.status_code == 422

        stats_res = client.get("/stats")
        assert stats_res.status_code == 200
        assert stats_res.json()["total_predictions"] == 1
