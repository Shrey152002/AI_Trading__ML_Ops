"""Feature engineering: turns validated per-ticker OHLCV data into a model-ready tensor.

This module is the computation engine only. `feature_store/store.py` wraps it for
versioned storage and is what training/inference actually read from; the save/load
helpers here remain for standalone CLI use (`python -m data.features <portfolio>`).
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)

FEATURES_ROOT = Path(__file__).resolve().parent / "features"

FEATURE_NAMES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "rsi_14",
    "macd_signal",
    "volume_zscore",
    "avg_corr_20d",
]
ROLLING_WINDOW = 20
CORR_WINDOW = 20


def _per_ticker_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame:
    feats = pd.DataFrame(index=close.index)
    feats["return_1d"] = close.pct_change()
    feats["return_5d"] = close.pct_change(5)
    feats["return_20d"] = close.pct_change(20)
    feats["volatility_20d"] = feats["return_1d"].rolling(ROLLING_WINDOW).std()
    feats["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd = ta.trend.MACD(close=close)
    feats["macd_signal"] = macd.macd() - macd.macd_signal()

    vol_mean = volume.rolling(ROLLING_WINDOW).mean()
    vol_std = volume.rolling(ROLLING_WINDOW).std()
    feats["volume_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    return feats


def build_feature_tensor(
    portfolio_name: str,
    data: dict[str, pd.DataFrame],
    save: bool = True,
) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """Builds a (n_days, n_tickers, n_features) tensor aligned on the common date index.

    Also computes and persists the rolling pairwise correlation matrix separately
    (n_days, n_tickers, n_tickers) since it doesn't collapse into a per-ticker scalar.
    """
    tickers = list(data.keys())
    if not tickers:
        raise ValueError("No ticker data provided for feature engineering")

    common_index = None
    for df in data.values():
        idx = df.index
        common_index = idx if common_index is None else common_index.intersection(idx)
    common_index = common_index.sort_values()

    returns = pd.DataFrame(
        {t: data[t]["Close"].reindex(common_index).pct_change() for t in tickers}
    )
    rolling_corr = returns.rolling(CORR_WINDOW).corr()

    per_ticker_feats = {}
    avg_corr_by_ticker = {t: pd.Series(index=common_index, dtype=float) for t in tickers}

    for date in common_index:
        try:
            corr_slice = rolling_corr.loc[date]
        except KeyError:
            continue
        for t in tickers:
            if t in corr_slice.index:
                others = corr_slice.loc[t].drop(labels=[t], errors="ignore")
                avg_corr_by_ticker[t].loc[date] = others.mean()

    for t in tickers:
        close = data[t]["Close"].reindex(common_index)
        volume = data[t]["Volume"].reindex(common_index)
        feats = _per_ticker_features(close, volume)
        feats["avg_corr_20d"] = avg_corr_by_ticker[t]
        per_ticker_feats[t] = feats[FEATURE_NAMES]

    n_days = len(common_index)
    n_tickers = len(tickers)
    n_features = len(FEATURE_NAMES)
    tensor = np.zeros((n_days, n_tickers, n_features), dtype=np.float32)
    for i, t in enumerate(tickers):
        tensor[:, i, :] = per_ticker_feats[t].to_numpy()

    tensor = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    corr_tensor = np.zeros((n_days, n_tickers, n_tickers), dtype=np.float32)
    for day_idx, date in enumerate(common_index):
        try:
            corr_tensor[day_idx] = rolling_corr.loc[date].reindex(index=tickers, columns=tickers).to_numpy()
        except KeyError:
            continue
    corr_tensor = np.nan_to_num(corr_tensor, nan=0.0)

    if save:
        out_dir = FEATURES_ROOT / portfolio_name
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "features.npy", tensor)
        np.save(out_dir / "correlation.npy", corr_tensor)
        np.save(out_dir / "dates.npy", common_index.values)
        with open(out_dir / "tickers.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(tickers))
        logger.info(
            "Saved feature tensor %s for %s (tickers=%d, days=%d)",
            tensor.shape,
            portfolio_name,
            n_tickers,
            n_days,
        )

    return tensor, common_index, tickers


def load_feature_tensor(portfolio_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    in_dir = FEATURES_ROOT / portfolio_name
    tensor = np.load(in_dir / "features.npy")
    dates = np.load(in_dir / "dates.npy", allow_pickle=True)
    with open(in_dir / "tickers.txt", "r", encoding="utf-8") as f:
        tickers = f.read().splitlines()
    return tensor, dates, tickers


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import get_portfolio
    from data.ingestion import load_cached_portfolio_data

    name = sys.argv[1] if len(sys.argv) > 1 else "banking"
    portfolio = get_portfolio(name)
    cached = load_cached_portfolio_data(name, portfolio["tickers"])
    tensor, dates, tickers = build_feature_tensor(name, cached)
    print(f"tensor shape={tensor.shape}, tickers={tickers}, first_date={dates[0]}, last_date={dates[-1]}")
