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


class PortfolioConfigResponse(BaseModel):
    name: str
    display_name: str
    tickers: list[str]
    risk_appetite: Literal["conservative", "moderate", "aggressive"]
    capital: float
    rebalance_frequency: str


class PortfolioListResponse(BaseModel):
    portfolios: list[PortfolioConfigResponse]


class PipelineRunStep(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: str


class PipelineRunRecord(BaseModel):
    portfolio: str
    started_at: str
    steps: list[PipelineRunStep]
    status: str
    finished_at: Optional[str] = None


class PipelineRunsResponse(BaseModel):
    portfolio_id: str
    runs: list[PipelineRunRecord]


class PipelineLogEntry(BaseModel):
    seq: int
    timestamp: str
    level: str
    logger: str
    message: str


class PipelineLogsResponse(BaseModel):
    entries: list[PipelineLogEntry]
