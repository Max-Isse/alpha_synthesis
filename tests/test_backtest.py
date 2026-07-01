import pytest
import polars as pl
from src.backtest.engine import AlphaBacktester

def test_dollar_neutral_weight_generation():
    """Verifies that the ranking engine produces a strictly dollar-neutral allocation."""
    # Mock data layout containing multi-sector asset predictions
    mock_payload = {
        "filing_date": ["2026-06-01", "2026-06-01", "2026-06-01", "2026-06-01"],
        "ticker": ["AAPL", "MSFT", "XOM", "CVX"],
        "sector": ["Tech", "Tech", "Energy", "Energy"],
        "predicted_alpha": [0.15, -0.05, 0.08, 0.02]
    }
    
    df = pl.DataFrame(mock_payload)
    backtester = AlphaBacktester()
    
    weighted_df = backtester.generate_portfolio_weights(df)
    
    # Assert that the sum of final allocation weights for any given rebalance date is exactly 0
    net_exposure = weighted_df.group_by("filing_date").agg(pl.col("final_weight").sum())["final_weight"][0]
    assert pytest.approx(net_exposure, abs=1e-7) == 0.0, "Portfolio fails to maintain strict dollar-neutrality."