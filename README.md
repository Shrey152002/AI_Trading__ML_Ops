# AI_Trading__ML_Ops

This repo contains **AutoPortfolio** — a production-grade MLOps platform for deep
reinforcement learning portfolio management. **It is not a trading bot.** It doesn't place
orders or talk to a broker; it researches, trains, benchmarks, registers, monitors, and serves
portfolio *allocation recommendations*.

The platform lives in [`autoportfolio/`](autoportfolio/) — see
[`autoportfolio/README.md`](autoportfolio/README.md) for the full architecture, setup
instructions, and API docs.

[`stocks_trading.ipynb`](stocks_trading.ipynb) is the original exploratory notebook this
platform evolved from — kept for history, fully superseded by `autoportfolio/`.

## Quick links

- [Architecture & full docs](autoportfolio/README.md)
- [Portfolio definitions](autoportfolio/config/portfolios.yaml)
- [Data versioning (DVC)](autoportfolio/docs/data_versioning.md)
- [Tests](autoportfolio/tests/)
