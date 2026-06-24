"""FastAPI application: a thin HTTP layer over inference.service and scheduler.pipeline.

No model loading, feature reads, or prediction logic lives here — see inference/service.py.
"""
import logging
import os
import time

import mlflow
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from api.schemas import (
    HealthResponse,
    PipelineLogsResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    PipelineRunsResponse,
    PortfolioConfigResponse,
    PortfolioHistoryResponse,
    PortfolioListResponse,
    PortfolioStatusResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from config import get_portfolio, list_portfolio_names
from inference import service as inference_service
from monitoring.log_buffer import get_recent_logs, install_log_capture
from monitoring.metrics import api_prediction_latency_seconds, api_requests_total, export_metrics
from scheduler.pipeline import read_pipeline_runs, run_nightly_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
install_log_capture()

app = FastAPI(title="AutoPortfolio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:3001").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_and_metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    api_requests_total.labels(endpoint=request.url.path, status=str(response.status_code)).inc()
    logger.info(
        "%s %s -> %d (%.3fs)", request.method, request.url.path, response.status_code, duration
    )
    return response


@app.post("/portfolio/recommendation", response_model=RecommendationResponse)
def get_recommendation(request: RecommendationRequest):
    with api_prediction_latency_seconds.time():
        try:
            result = inference_service.get_recommendation(request.portfolio_id, request.current_holdings)
        except (KeyError, LookupError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return RecommendationResponse(**result)


@app.get("/portfolio/{portfolio_id}/status", response_model=PortfolioStatusResponse)
def get_portfolio_status(portfolio_id: str):
    try:
        result = inference_service.get_status(portfolio_id)
    except (KeyError, LookupError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PortfolioStatusResponse(**result)


@app.get("/portfolio/{portfolio_id}/history", response_model=PortfolioHistoryResponse)
def get_portfolio_history(portfolio_id: str):
    try:
        history = inference_service.get_history(portfolio_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PortfolioHistoryResponse(portfolio_id=portfolio_id, history=history)


@app.post("/pipeline/run", response_model=PipelineRunResponse)
def trigger_pipeline_run(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    try:
        get_portfolio(request.portfolio_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    background_tasks.add_task(
        run_nightly_pipeline,
        request.portfolio_id,
        total_timesteps=request.total_timesteps,
        n_trials=request.n_trials,
    )
    return PipelineRunResponse(
        portfolio_id=request.portfolio_id,
        status="scheduled",
        message="Nightly pipeline run started in the background.",
    )


@app.get("/portfolios", response_model=PortfolioListResponse)
def list_portfolios():
    configs = [get_portfolio(name) for name in list_portfolio_names()]
    return PortfolioListResponse(
        portfolios=[
            PortfolioConfigResponse(
                name=c["name"],
                display_name=c["display_name"],
                tickers=c["tickers"],
                risk_appetite=c["risk_appetite"],
                capital=c["capital"],
                rebalance_frequency=c["rebalance_frequency"],
            )
            for c in configs
        ]
    )


@app.get("/pipeline/runs/{portfolio_id}", response_model=PipelineRunsResponse)
def get_pipeline_runs(portfolio_id: str):
    try:
        get_portfolio(portfolio_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PipelineRunsResponse(portfolio_id=portfolio_id, runs=read_pipeline_runs(portfolio_id))


@app.get("/pipeline/logs", response_model=PipelineLogsResponse)
def get_pipeline_logs(since: int = 0):
    return PipelineLogsResponse(entries=get_recent_logs(since))


@app.get("/health", response_model=HealthResponse)
def health():
    mlflow_status = "connected"
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.search_experiments(max_results=1)
    except Exception:  # noqa: BLE001
        mlflow_status = "disconnected"

    return HealthResponse(status="ok", portfolios=list_portfolio_names(), mlflow=mlflow_status)


@app.get("/metrics")
def metrics():
    return Response(content=export_metrics(), media_type="text/plain; version=0.0.4")
