"""Generates a self-contained HTML evaluation report after every training run."""
import base64
import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

logger = logging.getLogger(__name__)

REPORTS_ROOT = Path(__file__).resolve().parent

REPORT_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AutoPortfolio Report — {{ portfolio }} — {{ date }}</title>
<style>
  body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #1a1a1a; }
  h1 { font-size: 22px; }
  h2 { font-size: 17px; margin-top: 32px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; }
  th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; font-size: 14px; }
  th { background: #f5f5f5; text-align: center; }
  td:first-child, th:first-child { text-align: left; }
  .winner { background: #e8f5e9; font-weight: 600; }
  .promote { color: #1b5e20; font-weight: 700; }
  .keep { color: #b71c1c; font-weight: 700; }
  .meta { color: #555; font-size: 14px; }
  pre.equity { background: #fafafa; border: 1px solid #ddd; padding: 8px; overflow-x: auto; font-size: 11px; }
</style>
</head>
<body>
  <h1>AutoPortfolio Evaluation Report</h1>
  <p class="meta">
    Portfolio: <strong>{{ portfolio }}</strong> &nbsp;|&nbsp;
    Training date: <strong>{{ date }}</strong> &nbsp;|&nbsp;
    Best algorithm: <strong>{{ best_algorithm }}</strong>
  </p>

  <h2>Benchmark Results</h2>
  <table>
    <tr><th>Algorithm</th><th>Sharpe</th><th>Max Drawdown</th><th>Total Return</th><th>Calmar</th><th>Volatility</th></tr>
    {% for row in benchmark_rows %}
    <tr {% if row.algorithm == best_algorithm %}class="winner"{% endif %}>
      <td>{{ row.algorithm }}</td>
      <td>{{ '%.4f'|format(row.sharpe) }}</td>
      <td>{{ '%.4f'|format(row.max_drawdown) }}</td>
      <td>{{ '%.4f'|format(row.total_return) }}</td>
      <td>{{ '%.4f'|format(row.calmar) }}</td>
      <td>{{ '%.4f'|format(row.volatility) }}</td>
    </tr>
    {% endfor %}
  </table>

  <h2>Production Comparison</h2>
  {% if previous_metrics %}
  <table>
    <tr><th></th><th>Sharpe</th><th>Max Drawdown</th><th>Total Return</th></tr>
    <tr><td>Previous Production</td><td>{{ '%.4f'|format(previous_metrics.sharpe) }}</td><td>{{ '%.4f'|format(previous_metrics.max_drawdown) }}</td><td>{{ '%.4f'|format(previous_metrics.total_return) }}</td></tr>
    <tr><td>New Candidate ({{ best_algorithm }})</td><td>{{ '%.4f'|format(new_metrics.sharpe) }}</td><td>{{ '%.4f'|format(new_metrics.max_drawdown) }}</td><td>{{ '%.4f'|format(new_metrics.total_return) }}</td></tr>
  </table>
  {% else %}
  <p class="meta">No previous production model on record — this will be the first deployment.</p>
  {% endif %}
  <p>Recommendation: <span class="{{ 'promote' if recommendation == 'Promote' else 'keep' }}">{{ recommendation }}</span></p>

  <h2>Equity Curve ({{ best_algorithm }}, test window)</h2>
  {% if equity_curve_png %}
  <img src="data:image/png;base64,{{ equity_curve_png }}" alt="equity curve" style="max-width:100%;">
  {% else %}
  <pre class="equity">{{ equity_curve_ascii }}</pre>
  {% endif %}

  <h2>Drift History (last 30 days)</h2>
  {% if drift_rows %}
  <table>
    <tr><th>Timestamp</th><th>Live Sharpe</th><th>Benchmark Sharpe</th><th>Threshold</th><th>Action</th></tr>
    {% for row in drift_rows %}
    <tr>
      <td>{{ row.timestamp }}</td>
      <td>{{ row.live_value }}</td>
      <td>{{ row.benchmark_value }}</td>
      <td>{{ row.threshold }}</td>
      <td>{{ row.action }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <p class="meta">No drift events recorded in the last 30 days.</p>
  {% endif %}
</body>
</html>
"""
)


def _equity_curve_png(returns: np.ndarray) -> Optional[str]:
    if len(returns) == 0:
        return None
    equity = np.cumprod(1.0 + returns)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(equity, color="#1565c0", linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Portfolio value (normalized)")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _equity_curve_ascii(returns: np.ndarray, width: int = 60, height: int = 12) -> str:
    if len(returns) == 0:
        return "(no returns to plot)"
    equity = np.cumprod(1.0 + returns)
    sampled = np.interp(np.linspace(0, len(equity) - 1, width), np.arange(len(equity)), equity)
    lo, hi = sampled.min(), sampled.max()
    span = max(hi - lo, 1e-9)
    rows = []
    for level in range(height, 0, -1):
        threshold = lo + (level / height) * span
        rows.append("".join("*" if v >= threshold else " " for v in sampled))
    return "\n".join(rows)


def _recent_drift_rows(drift_log_path: Path, days: int = 30) -> list[dict]:
    if not drift_log_path.exists():
        return []
    df = pd.read_csv(drift_log_path)
    if df.empty:
        return []
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=days)
    cutoff = cutoff.replace(tzinfo=None) if df["timestamp"].dt.tz is None else cutoff
    recent = df[df["timestamp"] >= cutoff].sort_values("timestamp", ascending=False)
    return recent.to_dict(orient="records")


def generate_report(
    portfolio_name: str,
    benchmark_rows: list[dict],
    best_algorithm: str,
    new_metrics: dict,
    test_returns: np.ndarray,
    previous_metrics: Optional[dict] = None,
    drift_log_path: Optional[Path] = None,
) -> Path:
    """Renders and writes reports/<portfolio>_<date>.html, returning the written path."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if previous_metrics is None:
        recommendation = "Promote"
    else:
        recommendation = "Promote" if new_metrics["sharpe"] > previous_metrics["sharpe"] else "Keep Current"

    png = _equity_curve_png(test_returns)
    drift_path = drift_log_path or (Path(__file__).resolve().parents[1] / "drift" / "drift_log.csv")
    drift_rows = _recent_drift_rows(drift_path)

    html = REPORT_TEMPLATE.render(
        portfolio=portfolio_name,
        date=date_str,
        best_algorithm=best_algorithm,
        benchmark_rows=benchmark_rows,
        previous_metrics=previous_metrics,
        new_metrics=new_metrics,
        recommendation=recommendation,
        equity_curve_png=png,
        equity_curve_ascii=_equity_curve_ascii(test_returns) if png is None else None,
        drift_rows=drift_rows,
    )

    out_path = REPORTS_ROOT / f"{portfolio_name}_{date_str}.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("Wrote evaluation report to %s", out_path)
    return out_path
