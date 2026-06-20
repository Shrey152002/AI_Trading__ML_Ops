from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).resolve().parent
PORTFOLIOS_PATH = CONFIG_DIR / "portfolios.yaml"


def load_portfolios_config() -> dict:
    with open(PORTFOLIOS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_portfolio(name: str) -> dict:
    cfg = load_portfolios_config()
    portfolios = cfg["portfolios"]
    if name not in portfolios:
        raise KeyError(f"Unknown portfolio '{name}'. Available: {list(portfolios.keys())}")
    portfolio = dict(portfolios[name])
    portfolio["name"] = name
    portfolio["defaults"] = cfg.get("defaults", {})
    return portfolio


def list_portfolio_names() -> list:
    return list(load_portfolios_config()["portfolios"].keys())
