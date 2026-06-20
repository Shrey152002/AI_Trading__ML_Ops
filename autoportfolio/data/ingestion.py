"""Downloads and caches OHLCV price data per portfolio using yfinance."""
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent / "raw"
EXPECTED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class IngestionResult:
    def __init__(self, portfolio_name: str):
        self.portfolio_name = portfolio_name
        self.downloaded: dict[str, pd.DataFrame] = {}
        self.missing_tickers: list[str] = []
        self.freshness: dict[str, datetime] = {}

    @property
    def download_count(self) -> int:
        return len(self.downloaded)

    def to_summary(self) -> dict:
        return {
            "portfolio": self.portfolio_name,
            "tickers_requested": self.download_count + len(self.missing_tickers),
            "tickers_downloaded": self.download_count,
            "missing_tickers": self.missing_tickers,
            "freshness": {t: ts.isoformat() for t, ts in self.freshness.items()},
        }


def _cache_path(portfolio_name: str) -> Path:
    path = DATA_ROOT / portfolio_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_portfolio_data(
    portfolio_name: str,
    tickers: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
) -> IngestionResult:
    """Downloads OHLCV data for every ticker in a portfolio and caches it as Parquet.

    Either (start, end) or period (e.g. "5y") must be provided to yfinance.
    """
    result = IngestionResult(portfolio_name)
    cache_dir = _cache_path(portfolio_name)

    for ticker in tickers:
        try:
            if period:
                df = yf.download(ticker, period=period, auto_adjust=False, progress=False)
            else:
                df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df is None or df.empty:
                logger.warning("No data returned for %s", ticker)
                result.missing_tickers.append(ticker)
                continue

            missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
            if missing_cols:
                logger.warning("%s missing columns %s", ticker, missing_cols)
                result.missing_tickers.append(ticker)
                continue

            df = df[EXPECTED_COLUMNS].copy()
            df.index.name = "Date"

            out_path = cache_dir / f"{ticker.replace('.', '_')}.parquet"
            df.to_parquet(out_path)

            result.downloaded[ticker] = df
            result.freshness[ticker] = df.index.max().to_pydatetime()
            logger.info("Downloaded %d rows for %s -> %s", len(df), ticker, out_path)

        except Exception as exc:  # noqa: BLE001 - network/provider errors are expected & logged
            logger.error("Failed to download %s: %s", ticker, exc)
            result.missing_tickers.append(ticker)

    logger.info(
        "Ingestion complete for %s: %d/%d tickers downloaded, missing=%s",
        portfolio_name,
        result.download_count,
        result.download_count + len(result.missing_tickers),
        result.missing_tickers,
    )
    return result


def load_cached_portfolio_data(portfolio_name: str, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Loads previously cached Parquet files for a portfolio without re-downloading."""
    cache_dir = _cache_path(portfolio_name)
    data = {}
    for ticker in tickers:
        path = cache_dir / f"{ticker.replace('.', '_')}.parquet"
        if path.exists():
            data[ticker] = pd.read_parquet(path)
        else:
            logger.warning("No cached data for %s at %s", ticker, path)
    return data


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import get_portfolio

    name = sys.argv[1] if len(sys.argv) > 1 else "banking"
    portfolio = get_portfolio(name)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * 3)
    res = download_portfolio_data(
        name, portfolio["tickers"], start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
    )
    print(res.to_summary())
