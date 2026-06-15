from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from model_rf import load_model, predict

app = FastAPI()

model = load_model("models/model_rf.pkl")


class FeaturePayload(BaseModel):
    account_age_days: float = Field(..., ge=0)
    repos_per_year: float = Field(..., ge=0)
    followers_following_ratio: float = Field(..., ge=0)
    active_repo_ratio: float = Field(..., ge=0, le=1)
    inactive_repo_ratio: float = Field(..., ge=0, le=1)
    avg_stars_per_repo: float = Field(..., ge=0)
    avg_forks_per_repo: float = Field(..., ge=0)
    repo_activity_density: float = Field(..., ge=0)
    repository_maintenance_ratio: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    churned: bool
    churn_probability: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(payload: FeaturePayload) -> Any:
    features = payload.dict()
    result = predict(model, features)
    return result
