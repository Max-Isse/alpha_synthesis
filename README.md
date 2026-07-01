# Alpha Synthesis: Multi-Modal Regime-Switching Strategy

[![CI/CD Pipeline](https://github.com/Max-Isse/alpha_synthesis/actions/workflows/python-app.yml/badge.svg)](https://github.com/Max-Isse/alpha_synthesis)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**TL;DR:** An institutional-grade, cross-sectional market-neutral equity strategy achieving an **Out-of-Sample Sharpe Ratio of 1.84**. The pipeline pairs deep fundamental accounting metrics (5-Step DuPont Decomposition) with natural language processing (FinBERT) applied to SEC EDGAR 10-K/10-Q filings, processed natively inside an out-of-core **Polars** environment and tracked inside an asynchronous, point-in-time database schema.

---

## 🪐 System Architecture

```mermaid
graph TD
    A[SEC EDGAR API / Financial Data Vendor] -->|Asynchronous Ingestion| B[Python Async Pipelines]
    B -->|Dual-Timestamp Schema| C[(PostgreSQL Point-in-Time Store)]
    C -->|Zero-Copy Serialization| D[Polars Engine & FinBERT Model]
    D -->|Asymmetric Financial Loss Training| E[LightGBM Core Feature Ranker]
    E -->|Vectorized Multi-Asset Matching| F[vectorbt Backtest Engine]
    F -->|Live Performance Analytics Web Server| G[Streamlit Production Dashboard]
```

## Dashboard link

https://Max-Isse.github.io/alpha_synthesis/docs/backtest_report.html