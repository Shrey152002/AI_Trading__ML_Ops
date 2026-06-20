"""Pydantic request/response models for the AutoPortfolio API."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RecommendationRequest(BaseModel):
    portfolio_id: str
    current_holdings: dict[str, float] = Field(default_factory=dict)
    risk_appetite: Literal["conservative", "moderate", "aggressive"]
    capital: float = Field(gt=0)
    sector_constraints: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    recommended_allocation: dict[str, float]
    expected_return: float
    expected_volatility: float
    confidence: float
    explanation: str
    model_version: str
    timestamp: str


class PortfolioStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    portfolio_id: str
    model_version: str
    algorithm: str
    last_training_date: Optional[str]
    sharpe: float
    drift_score: float
    data_freshness_hours: float


class AllocationHistoryEntry(BaseModel):
    date: str
    allocation: dict[str, float]
    realized_return: Optional[float]


class PortfolioHistoryResponse(BaseModel):
    portfolio_id: str
    history: list[AllocationHistoryEntry]


class PipelineRunRequest(BaseModel):
    portfolio_id: str
    total_timesteps: int = 100_000
    n_trials: int = 20


class PipelineRunResponse(BaseModel):
    portfolio_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    portfolios: list[str]
    mlflow: str
