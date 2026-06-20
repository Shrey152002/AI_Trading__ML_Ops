"""Nightly orchestration: ingest -> validate -> features -> drift check -> train -> promote -> report."""
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import get_portfolio, list_portfolio_names
from data.ingestion import download_portfolio_data, load_cached_portfolio_data
from data.validation import DataValidationError, validate_portfolio_data
from drift.detector import check_drift
from feature_store.store import compute_and_store
from registry.model_registry import (
    archive_old_versions,
    get_production_model,
    load_agent_from_run,
    promote_to_production,
    register_model,
)
from reports.generator import generate_report
from training.benchmark import run_episode
from training.env_utils import build_envs
from training.trainer import train_portfolio

logger = logging.getLogger(__name__)

PIPELINE_RUNS_LOG = Path(__file__).resolve().parent / "pipeline_runs.json"


def _append_pipeline_run(record: dict) -> None:
    runs = []
    if PIPELINE_RUNS_LOG.exists():
        try:
            runs = json.loads(PIPELINE_RUNS_LOG.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            runs = []
    runs.append(record)
    PIPELINE_RUNS_LOG.write_text(json.dumps(runs, indent=2), encoding="utf-8")


def read_pipeline_runs(portfolio_name: str, limit: int = 10) -> list[dict]:
    """Returns up to `limit` most recent pipeline runs for portfolio_name, newest first."""
    if not PIPELINE_RUNS_LOG.exists():
        return []
    try:
        runs = json.loads(PIPELINE_RUNS_LOG.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    matching = [r for r in runs if r.get("portfolio") == portfolio_name]
    return list(reversed(matching))[:limit]


def run_nightly_pipeline(
    portfolio_name: str,
    total_timesteps: int = 100_000,
    n_trials: int = 20,
    lookback_years: int = 3,
) -> dict:
    """Runs the full nightly lifecycle for one portfolio. Returns a status record."""
    started_at = datetime.now(timezone.utc)
    record = {"portfolio": portfolio_name, "started_at": started_at.isoformat(), "steps": []}

    portfolio = get_portfolio(portfolio_name)

    # 1. Ingest
    end = started_at
    start = end - timedelta(days=365 * lookback_years)
    ingestion_result = download_portfolio_data(
        portfolio_name, portfolio["tickers"], start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
    )
    record["steps"].append({"step": "ingest", "summary": ingestion_result.to_summary()})

    # 2. Validate — abort with alert if it fails
    cached = load_cached_portfolio_data(portfolio_name, portfolio["tickers"])
    try:
        validation_report = validate_portfolio_data(portfolio_name, cached)
        record["steps"].append({"step": "validate", "status": "passed"})
    except DataValidationError as exc:
        logger.critical("ALERT: data validation failed for %s: %s", portfolio_name, exc)
        record["steps"].append({"step": "validate", "status": "failed", "report": exc.report})
        record["status"] = "aborted"
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        _append_pipeline_run(record)
        return record

    # 3. Features — computed and versioned in the feature store
    feature_metadata = compute_and_store(portfolio_name, cached)
    record["steps"].append({"step": "features", "status": "generated", "version": feature_metadata["version"]})

    # 4. Drift check — skip retraining if the current production model is healthy
    needs_retrain = True
    try:
        needs_retrain = check_drift(portfolio_name)
        record["steps"].append({"step": "drift_check", "needs_retrain": needs_retrain})
    except LookupError:
        logger.info("No production model yet for %s — forcing initial training", portfolio_name)
        record["steps"].append({"step": "drift_check", "needs_retrain": True, "reason": "no_production_model"})

    if not needs_retrain:
        record["status"] = "skipped_healthy"
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        _append_pipeline_run(record)
        return record

    # 5. Train all 4 agents
    train_result = train_portfolio(portfolio_name, total_timesteps=total_timesteps, n_trials=n_trials)
    record["steps"].append({"step": "train", "winner": train_result["winner_algorithm"], "metrics": train_result["winner_metrics"]})

    # 6. Benchmark + promote
    previous_metrics = None
    try:
        _, prev_metadata = get_production_model(portfolio_name)
        previous_metrics = {k: float(prev_metadata["tags"].get(k, 0.0) or 0.0) for k in ("sharpe", "max_drawdown", "total_return")}
    except LookupError:
        pass

    version = register_model(train_result["winner_run_id"], portfolio_name, train_result["winner_metrics"])
    model_name = f"autoportfolio-{portfolio_name}"

    should_promote = previous_metrics is None or train_result["winner_metrics"]["sharpe"] > previous_metrics["sharpe"]
    if should_promote:
        promote_to_production(model_name, version)
        archive_old_versions(portfolio_name, keep_last_n=3)
    record["steps"].append({"step": "promote", "promoted": should_promote, "version": version})

    # 7. Evaluation report — always reflects the new candidate, regardless of promotion outcome
    envs = build_envs(portfolio_name)
    candidate_agent = load_agent_from_run(train_result["winner_run_id"], train_result["winner_algorithm"])
    test_returns = run_episode(candidate_agent, envs["test"])
    report_path = generate_report(
        portfolio_name=portfolio_name,
        benchmark_rows=train_result["ranking"],
        best_algorithm=train_result["winner_algorithm"],
        new_metrics=train_result["winner_metrics"],
        test_returns=test_returns,
        previous_metrics=previous_metrics,
    )
    record["steps"].append({"step": "report", "path": str(report_path)})

    record["status"] = "completed"
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    _append_pipeline_run(record)
    return record


def run_all_portfolios(parallel: bool = False, max_workers: int = 4, **kwargs) -> dict[str, dict]:
    """Runs run_nightly_pipeline for every portfolio defined in config/portfolios.yaml."""
    names = list_portfolio_names()
    results = {}

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_nightly_pipeline, name, **kwargs): name for name in names}
            for future in futures:
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:  # noqa: BLE001 - one portfolio failing shouldn't stop the rest
                    logger.error("Pipeline failed for %s: %s", name, exc)
                    results[name] = {"portfolio": name, "status": "error", "error": str(exc)}
    else:
        for name in names:
            try:
                results[name] = run_nightly_pipeline(name, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.error("Pipeline failed for %s: %s", name, exc)
                results[name] = {"portfolio": name, "status": "error", "error": str(exc)}

    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        result = run_nightly_pipeline(sys.argv[1], total_timesteps=20_000, n_trials=5)
        print(json.dumps(result, indent=2, default=str))
    else:
        results = run_all_portfolios(total_timesteps=20_000, n_trials=5)
        print(json.dumps(results, indent=2, default=str))
