"""Versioned feature store: separates feature computation from feature consumption.

`data/features.py` owns the computation (turning raw OHLCV into engineered features). This
module owns storage, versioning, and the two read paths consumers actually need:
  - a full (n_days, n_tickers, n_features) tensor, for training
  - a single latest-day observation row, for low-latency inference
Both training and inference read through here so they can never silently diverge on what
feature set a model was trained vs served against.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data.features import FEATURE_NAMES, build_feature_tensor

logger = logging.getLogger(__name__)

STORE_ROOT = Path(__file__).resolve().parents[1] / "data" / "feature_store"
PREDICTION_LOG_LIMIT = 90


def _portfolio_dir(portfolio_name: str) -> Path:
    path = STORE_ROOT / portfolio_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_path(portfolio_name: str) -> Path:
    return _portfolio_dir(portfolio_name) / "manifest.json"


def _read_manifest(portfolio_name: str) -> dict:
    path = _manifest_path(portfolio_name)
    if not path.exists():
        return {"latest": None, "versions": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(portfolio_name: str, manifest: dict) -> None:
    _manifest_path(portfolio_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _next_version(manifest: dict) -> str:
    existing = [int(v["version"].lstrip("v")) for v in manifest["versions"]]
    return f"v{max(existing, default=0) + 1}"


def compute_and_store(portfolio_name: str, raw_data: dict[str, pd.DataFrame]) -> dict:
    """Computes features from raw OHLCV data and persists a new versioned snapshot.

    Returns the metadata dict for the newly written version.
    """
    tensor, dates, tickers = build_feature_tensor(portfolio_name, raw_data, save=False)
    n_days, n_tickers, n_features = tensor.shape

    manifest = _read_manifest(portfolio_name)
    version = _next_version(manifest)
    version_dir = _portfolio_dir(portfolio_name) / version
    version_dir.mkdir(parents=True, exist_ok=True)

    dates_repeated = np.repeat(np.asarray(dates), n_tickers)
    tickers_tiled = np.tile(tickers, n_days)
    long_df = pd.DataFrame(tensor.reshape(n_days * n_tickers, n_features), columns=FEATURE_NAMES)
    long_df.insert(0, "ticker", tickers_tiled)
    long_df.insert(0, "date", dates_repeated)
    long_df.to_parquet(version_dir / "features.parquet", index=False)

    metadata = {
        "version": version,
        "portfolio": portfolio_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_days": n_days,
        "n_tickers": n_tickers,
        "tickers": tickers,
        "feature_names": FEATURE_NAMES,
        "start_date": pd.Timestamp(dates[0]).isoformat(),
        "end_date": pd.Timestamp(dates[-1]).isoformat(),
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    manifest["versions"].append(metadata)
    manifest["latest"] = version
    _write_manifest(portfolio_name, manifest)

    logger.info("Stored feature_store version %s for %s: tensor=%s", version, portfolio_name, tensor.shape)
    return metadata


def _latest_version_or_raise(portfolio_name: str) -> str:
    manifest = _read_manifest(portfolio_name)
    if not manifest["latest"]:
        raise LookupError(f"No feature_store versions found for portfolio '{portfolio_name}'")
    return manifest["latest"]


def _load_version(portfolio_name: str, version: str) -> tuple[pd.DataFrame, dict]:
    version_dir = _portfolio_dir(portfolio_name) / version
    df = pd.read_parquet(version_dir / "features.parquet")
    metadata = json.loads((version_dir / "metadata.json").read_text(encoding="utf-8"))
    return df, metadata


def get_latest_for_training(portfolio_name: str) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Returns the (n_days, n_tickers, n_features) tensor for the latest stored version."""
    version = _latest_version_or_raise(portfolio_name)
    df, metadata = _load_version(portfolio_name, version)

    tickers = metadata["tickers"]
    dates_sorted = sorted(df["date"].unique())
    index = pd.MultiIndex.from_product([dates_sorted, tickers], names=["date", "ticker"])
    wide = df.set_index(["date", "ticker"]).reindex(index)

    n_days, n_tickers, n_features = len(dates_sorted), len(tickers), len(FEATURE_NAMES)
    tensor = wide[FEATURE_NAMES].to_numpy(dtype=np.float32).reshape(n_days, n_tickers, n_features)
    tensor = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    return tensor, np.array(dates_sorted), tickers, metadata


def get_latest_observation(portfolio_name: str) -> tuple[np.ndarray, list[str], dict]:
    """Returns just the most recent date's flattened feature row, for low-latency inference."""
    version = _latest_version_or_raise(portfolio_name)
    df, metadata = _load_version(portfolio_name, version)

    tickers = metadata["tickers"]
    last_date = df["date"].max()
    subset = df[df["date"] == last_date].set_index("ticker").reindex(tickers)
    obs_row = subset[FEATURE_NAMES].to_numpy(dtype=np.float32).flatten()
    obs_row = np.nan_to_num(obs_row, nan=0.0, posinf=0.0, neginf=0.0)

    return obs_row, tickers, metadata


def list_versions(portfolio_name: str) -> pd.DataFrame:
    manifest = _read_manifest(portfolio_name)
    df = pd.DataFrame(manifest["versions"])
    if not df.empty:
        df = df.sort_values("version", ascending=False).reset_index(drop=True)
    return df


def _predictions_path(portfolio_name: str) -> Path:
    path = _portfolio_dir(portfolio_name) / "predictions"
    path.mkdir(parents=True, exist_ok=True)
    return path / "predictions.jsonl"


def log_prediction(portfolio_name: str, allocation: dict, timestamp: str, realized_return: float = None) -> None:
    """Appends a prediction record. Feeds the feature store so future retraining or drift
    analysis can read back what the model actually recommended, not just what happened.
    """
    path = _predictions_path(portfolio_name)
    record = {"date": timestamp, "allocation": allocation, "realized_return": realized_return}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_predictions(portfolio_name: str, limit: int = PREDICTION_LOG_LIMIT) -> list[dict]:
    path = _predictions_path(portfolio_name)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:]]
