"""Wraps the MLflow Model Registry: versioning, promotion, and production model retrieval."""
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def _model_name(portfolio_name: str) -> str:
    return f"autoportfolio-{portfolio_name}"


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    return MlflowClient()


def register_model(run_id: str, portfolio_name: str, metrics: dict) -> int:
    """Registers the model artifact produced by `run_id` as a new version for this portfolio."""
    client = _client()
    model_name = _model_name(portfolio_name)

    try:
        client.create_registered_model(model_name)
    except MlflowException:
        pass  # already exists

    run = client.get_run(run_id)
    algorithm = run.data.tags.get("algorithm", "unknown")

    model_uri = f"runs:/{run_id}/model"
    mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)

    client.set_model_version_tag(model_name, mv.version, "algorithm", algorithm)
    client.set_model_version_tag(model_name, mv.version, "portfolio", portfolio_name)
    for key, value in metrics.items():
        client.set_model_version_tag(model_name, mv.version, key, str(value))
    client.set_model_version_tag(model_name, mv.version, "registered_at", datetime.utcnow().isoformat())

    logger.info("Registered %s v%s for portfolio %s (algorithm=%s)", model_name, mv.version, portfolio_name, algorithm)
    return int(mv.version)


def promote_to_production(model_name: str, version: int) -> None:
    """Promotes a model version to Production, automatically archiving the prior Production version."""
    client = _client()
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production", archive_existing_versions=True
    )
    logger.info("Promoted %s v%s to Production", model_name, version)


def load_agent_from_run(run_id: str, algorithm: str):
    """Downloads and loads the SB3 model artifact logged under a specific MLflow run."""
    from agents import AGENT_REGISTRY  # local import avoids a hard dependency at module load time

    if algorithm not in AGENT_REGISTRY:
        raise ValueError(f"Unknown algorithm '{algorithm}' for run {run_id}")

    client = _client()
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_dir = client.download_artifacts(run_id, "model", tmp_dir)
        model_files = list(Path(local_dir).glob("*.zip"))
        if not model_files:
            raise FileNotFoundError(f"No model .zip artifact found under {local_dir}")

        agent = AGENT_REGISTRY[algorithm]()
        agent.load(str(model_files[0]))
    return agent


def get_production_model(portfolio_name: str):
    """Loads the Production model + metadata for a portfolio.

    Returns (agent, metadata) where agent is a loaded BaseAgent subclass instance.
    """
    client = _client()
    model_name = _model_name(portfolio_name)
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except MlflowException as exc:
        raise LookupError(f"No model registered for portfolio '{portfolio_name}': {exc}") from exc
    if not versions:
        raise LookupError(f"No Production model registered for portfolio '{portfolio_name}'")

    mv = versions[0]
    tags = mv.tags
    algorithm = tags.get("algorithm")
    agent = load_agent_from_run(mv.run_id, algorithm)

    metadata = {
        "model_name": model_name,
        "version": mv.version,
        "algorithm": algorithm,
        "stage": mv.current_stage,
        "run_id": mv.run_id,
        "tags": tags,
    }
    return agent, metadata


def archive_old_versions(portfolio_name: str, keep_last_n: int = 3) -> list[int]:
    """Archives all non-Production versions beyond the most recent `keep_last_n`."""
    client = _client()
    model_name = _model_name(portfolio_name)
    versions = client.search_model_versions(f"name='{model_name}'")
    versions = sorted(versions, key=lambda v: int(v.version), reverse=True)

    archived = []
    non_production = [v for v in versions if v.current_stage != "Production"]
    for v in non_production[keep_last_n:]:
        client.transition_model_version_stage(name=model_name, version=v.version, stage="Archived")
        archived.append(int(v.version))

    logger.info("Archived %d old versions for %s: %s", len(archived), model_name, archived)
    return archived


def list_versions(portfolio_name: str) -> pd.DataFrame:
    """Returns a DataFrame with version, algorithm, Sharpe, registration date, and stage."""
    client = _client()
    model_name = _model_name(portfolio_name)
    versions = client.search_model_versions(f"name='{model_name}'")

    rows = []
    for v in versions:
        rows.append(
            {
                "version": int(v.version),
                "algorithm": v.tags.get("algorithm", "unknown"),
                "sharpe": float(v.tags.get("sharpe", "nan")) if v.tags.get("sharpe") else None,
                "registered_at": v.tags.get("registered_at"),
                "status": v.current_stage,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("version", ascending=False).reset_index(drop=True)
    return df
