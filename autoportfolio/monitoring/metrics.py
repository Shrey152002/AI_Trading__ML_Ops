"""Prometheus metric definitions shared by the API and the nightly pipeline."""
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry()

portfolio_sharpe_ratio = Gauge(
    "portfolio_sharpe_ratio", "Latest benchmark Sharpe ratio for the production model", ["portfolio"], registry=REGISTRY
)
portfolio_daily_return = Gauge(
    "portfolio_daily_return", "Most recent realized daily portfolio return", ["portfolio"], registry=REGISTRY
)
api_prediction_latency_seconds = Histogram(
    "api_prediction_latency_seconds", "Latency of /portfolio/recommendation inference calls", registry=REGISTRY
)
api_requests_total = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "status"], registry=REGISTRY
)
model_drift_score = Gauge(
    "model_drift_score", "Ratio of live rolling Sharpe to benchmark Sharpe", ["portfolio"], registry=REGISTRY
)
data_freshness_hours = Gauge(
    "data_freshness_hours", "Hours since the most recent cached data point", ["portfolio"], registry=REGISTRY
)
training_runs_total = Counter(
    "training_runs_total", "Total training runs", ["portfolio", "algorithm", "outcome"], registry=REGISTRY
)


def export_metrics() -> bytes:
    """Renders all registered metrics in Prometheus text exposition format."""
    return generate_latest(REGISTRY)
