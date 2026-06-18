from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "Part3_Model" / "model.pkl"
SNAPSHOT_PATH = ROOT / "dataset" / "rfm_modeling_snapshot.csv"

app = FastAPI(title="Churn Prediction API", version="1.0.0")
artifact = joblib.load(MODEL_PATH)
snapshot = pd.read_csv(SNAPSHOT_PATH)
FEATURE_COLUMNS = artifact["feature_columns"]
THRESHOLD = artifact["threshold"]


class PredictRequest(BaseModel):
    customer_id: str | None = Field(default=None)
    features: dict[str, Any] | None = Field(default=None)


class BatchPredictRequest(BaseModel):
    records: list[PredictRequest]


def risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "high"
    if probability >= 0.40:
        return "medium"
    return "low"


def load_features(request: PredictRequest) -> tuple[str | None, pd.DataFrame]:
    if request.features:
        row = pd.DataFrame([request.features])
        missing = [col for col in FEATURE_COLUMNS if col not in row.columns]
        if missing:
            raise HTTPException(status_code=400, detail={"missing_features": missing})
        return request.customer_id, row[FEATURE_COLUMNS]
    if request.customer_id:
        rows = snapshot.loc[snapshot["customer_id"] == request.customer_id]
        if rows.empty:
            raise HTTPException(status_code=404, detail="customer_id not found")
        return request.customer_id, rows[FEATURE_COLUMNS]
    raise HTTPException(status_code=400, detail="Provide customer_id or features")


def explain(row: pd.Series, probability: float) -> str:
    drivers = []
    if row.get("recency_days", 0) >= 90:
        drivers.append(f"recency {int(row['recency_days'])} days")
    if row.get("last_visit_days_ago", 0) >= 14:
        drivers.append(f"last visit {int(row['last_visit_days_ago'])} days ago")
    if row.get("negative_ticket_rate_90d", 0) > 0:
        drivers.append(f"negative ticket rate {row['negative_ticket_rate_90d']:.2f}")
    if row.get("abandoned_carts_30d", 0) > 0:
        drivers.append(f"{int(row['abandoned_carts_30d'])} abandoned carts")
    if row.get("campaign_clicks_30d", 0) > 0:
        drivers.append(f"{int(row['campaign_clicks_30d'])} campaign clicks")
    if not drivers:
        drivers.append("model score from pre-snapshot RFM, support, and web activity")
    return f"{risk_level(probability).title()} risk based on " + ", ".join(drivers[:4]) + "."


def score_one(request: PredictRequest) -> dict[str, Any]:
    customer_id, X = load_features(request)
    probability = float(artifact["pipeline"].predict_proba(X)[0, 1])
    predicted_class = int(probability >= THRESHOLD)
    row = X.iloc[0]
    return {
        "customer_id": customer_id,
        "churn_probability": round(probability, 6),
        "predicted_class": predicted_class,
        "risk_level": risk_level(probability),
        "risk_explanation": explain(row, probability),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "snapshot_date": artifact["snapshot_date"],
        "model_type": artifact["model_type"],
        "threshold": THRESHOLD,
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    return score_one(request)


@app.post("/batch_predict")
def batch_predict(request: BatchPredictRequest) -> dict[str, Any]:
    return {"predictions": [score_one(record) for record in request.records]}
