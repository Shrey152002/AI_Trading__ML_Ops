"""Data quality gate. Nothing reaches feature engineering or training without passing this."""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

VALIDATED_ROOT = Path(__file__).resolve().parent / "validated"
EXPECTED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

MAX_MISSING_RATIO = 0.05
MAX_DAILY_RETURN_ABS = 0.50
MAX_STALENESS_DAYS = 2


class DataValidationError(Exception):
    """Raised when one or more tickers fail a hard data quality check."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


@dataclass
class TickerCheckResult:
    ticker: str
    passed: bool
    missing_ratio: float = 0.0
    has_non_positive_price: bool = False
    staleness_days: float = 0.0
    anomaly_days: list = field(default_factory=list)
    schema_ok: bool = True
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "passed": self.passed,
            "missing_ratio": round(self.missing_ratio, 4),
            "has_non_positive_price": self.has_non_positive_price,
            "staleness_days": round(self.staleness_days, 2),
            "anomaly_days": self.anomaly_days,
            "schema_ok": self.schema_ok,
            "errors": self.errors,
        }


def _check_schema(df: pd.DataFrame, ticker: str, result: TickerCheckResult) -> None:
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        result.schema_ok = False
        result.errors.append(f"Missing expected columns: {missing_cols}")


def _check_missing_rows(
    df: pd.DataFrame, result: TickerCheckResult, reference_index: Optional[pd.DatetimeIndex] = None
) -> None:
    """Flags rows missing relative to a reference trading calendar.

    Defaults to a Mon-Fri business-day calendar when no reference is supplied (used by
    single-ticker unit tests). In production, validate_portfolio_data passes the union of
    trading dates observed across the whole portfolio, since exchange holiday calendars
    (e.g. NSE) don't align with a generic business-day calendar and would otherwise produce
    a false-positive missing ratio on every ticker uniformly.
    """
    if len(df) == 0:
        result.missing_ratio = 1.0
        result.errors.append("No rows present")
        return
    if reference_index is None:
        reference_index = pd.date_range(df.index.min(), df.index.max(), freq="B")
    else:
        reference_index = reference_index[
            (reference_index >= df.index.min()) & (reference_index <= df.index.max())
        ]
    missing = len(reference_index.difference(df.index))
    result.missing_ratio = max(missing, 0) / max(len(reference_index), 1)
    if result.missing_ratio > MAX_MISSING_RATIO:
        result.errors.append(
            f"Missing ratio {result.missing_ratio:.2%} exceeds {MAX_MISSING_RATIO:.0%} threshold"
        )


def _check_positive_prices(df: pd.DataFrame, result: TickerCheckResult) -> None:
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if not price_cols:
        return
    if (df[price_cols] <= 0).any().any():
        result.has_non_positive_price = True
        result.errors.append("Found non-positive price values")


def _check_return_anomalies(df: pd.DataFrame, result: TickerCheckResult) -> None:
    if "Close" not in df.columns or len(df) < 2:
        return
    daily_return = df["Close"].pct_change()
    anomalies = daily_return[daily_return.abs() > MAX_DAILY_RETURN_ABS]
    result.anomaly_days = [d.strftime("%Y-%m-%d") for d in anomalies.index]
    # Anomalies are flagged, not dropped, and do not fail validation on their own.


def _check_freshness(df: pd.DataFrame, result: TickerCheckResult) -> None:
    if len(df) == 0:
        return
    last_date = df.index.max()
    if last_date.tzinfo is None:
        last_date = last_date.tz_localize(timezone.utc)
    now = datetime.now(timezone.utc)
    staleness = (now - last_date.to_pydatetime()).total_seconds() / 86400
    result.staleness_days = staleness
    if staleness > MAX_STALENESS_DAYS:
        result.errors.append(
            f"Data is {staleness:.1f} days stale, exceeds {MAX_STALENESS_DAYS} day threshold"
        )


def validate_ticker(
    ticker: str,
    df: pd.DataFrame,
    check_freshness: bool = True,
    reference_index: Optional[pd.DatetimeIndex] = None,
) -> TickerCheckResult:
    result = TickerCheckResult(ticker=ticker, passed=True)
    _check_schema(df, ticker, result)
    if result.schema_ok:
        _check_missing_rows(df, result, reference_index=reference_index)
        _check_positive_prices(df, result)
        _check_return_anomalies(df, result)
        if check_freshness:
            _check_freshness(df, result)
    result.passed = len(result.errors) == 0
    return result


def validate_portfolio_data(
    portfolio_name: str,
    data: dict[str, pd.DataFrame],
    check_freshness: bool = True,
) -> dict:
    """Validates every ticker's DataFrame. Raises DataValidationError if any ticker fails.

    On success, writes validation_report.json to data/validated/<portfolio_name>/.
    """
    reference_index = None
    if len(data) > 1:
        reference_index = pd.DatetimeIndex(sorted(set().union(*(df.index for df in data.values()))))

    results = [
        validate_ticker(t, df, check_freshness=check_freshness, reference_index=reference_index)
        for t, df in data.items()
    ]
    failed = [r for r in results if not r.passed]

    report = {
        "portfolio": portfolio_name,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "tickers_checked": len(results),
        "tickers_passed": len(results) - len(failed),
        "tickers_failed": len(failed),
        "results": [r.to_dict() for r in results],
    }

    if failed:
        logger.error("Validation failed for %s: %s", portfolio_name, [r.ticker for r in failed])
        raise DataValidationError(
            f"Validation failed for portfolio '{portfolio_name}': "
            f"{len(failed)}/{len(results)} tickers failed checks "
            f"({[r.ticker for r in failed]})",
            report=report,
        )

    out_dir = VALIDATED_ROOT / portfolio_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Validation passed for %s: %d/%d tickers", portfolio_name, len(results), len(results))
    return report


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import get_portfolio
    from data.ingestion import load_cached_portfolio_data

    name = sys.argv[1] if len(sys.argv) > 1 else "banking"
    portfolio = get_portfolio(name)
    cached = load_cached_portfolio_data(name, portfolio["tickers"])
    try:
        report = validate_portfolio_data(name, cached)
        print(json.dumps(report, indent=2))
    except DataValidationError as e:
        print(json.dumps(e.report, indent=2))
        raise
