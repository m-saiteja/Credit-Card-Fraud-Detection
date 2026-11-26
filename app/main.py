import time
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.schemas import TransactionRequest, PredictionResponse
from app.model_loader import load_champion_model
from app.monitor import log_prediction, get_live_stats

from src.config import (
    REGISTERED_MODEL_NAME,
    CHAMPION_ALIAS,
    FEATURE_COLS,
    setup_logger
)

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, threshold, version = load_champion_model()
    app.state.model = model
    app.state.threshold = threshold
    app.state.version = version
    yield


app = FastAPI(title='Credit Card Fraud Detection API', lifespan=lifespan)

@app.get("/health")
def health():
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {
        'status': 'healthy',
        'model_name': REGISTERED_MODEL_NAME,
        'alias': CHAMPION_ALIAS,
        'model_version': app.state.version,
        'threshold': app.state.threshold
    }


@app.post("/predict")
def predict(request: TransactionRequest) -> PredictionResponse:
    start = time.perf_counter()
    df = pd.DataFrame([request.model_dump()], columns=FEATURE_COLS)
    prob = float(app.state.model.predict_proba(df)[0, 1])
    is_fraud = (prob >= app.state.threshold)
    latency_ms = (time.perf_counter() - start) * 1000
    log_prediction(
        features_dict=request.model_dump(),
        fraud_probability=prob,
        is_fraud=is_fraud,
        threshold=app.state.threshold,
        model_version=app.state.version,
        latency_ms=latency_ms
    )
    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=prob,
        threshold=app.state.threshold,
        model_version=app.state.version
    )


@app.get("/stats")
def get_stats():
    return get_live_stats()
