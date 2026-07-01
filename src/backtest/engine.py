from typing import Dict, Any
import numpy as np
import polars as pl
import vectorbt as vbt

class AlphaBacktester:
    def __init__(self, transaction_cost: float = 0.0010, borrow_fee: float = 0.0025):
        """
        Institutional Backtest Engine.
        Defaults: 10 bps execution slippage/fees, 25 bps annualized borrow fee for shorts.
        """
        self.fees = transaction_cost
        self.borrow_fee = borrow_fee

    def generate_portfolio_weights(self, predictions_df: pl.DataFrame) -> pl.DataFrame:
        """
        Converts raw continuous model predictions into a cross-sectionally ranked, 
        dollar-neutral and sector-neutral weight matrix.
        """
        # Form market-neutral quintiles within each sector-date bucket
        weights_df = predictions_df.with_columns(
            pl.col("predicted_alpha")
            .rank(descending=True)
            .over(["filing_date", "sector"])
            .alias("rank")
        )
        
        # Long top-ranked assets, Short bottom-ranked assets
        weights_df = weights_df.with_columns(
            pl.when(pl.col("rank") == 1).then(1.0)
            .when(pl.col("rank") == pl.col("rank").max().over(["filing_date", "sector"])).then(-1.0)
            .otherwise(0.0)
            .alias("raw_weight")
        )
        
        # Enforce strict dollar-neutrality (Sum of weights = 0 per rebalancing period)
        weights_df = weights_df.with_columns(
            (pl.col("raw_weight") - pl.col("raw_weight").mean().over("filing_date")).alias("final_weight")
        )
        
        return weights_df

    def run_vectorized_simulation(self, price_matrix: np.ndarray, weight_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Executes the backtest via vectorbt's highly optimized array-broadcasting engine.
        """
        # Initialize vectorbt portfolio simulation from target weights
        portfolio = vbt.Portfolio.from_orders(
            close=price_matrix,
            size=weight_matrix,
            size_type="target_percent",
            fees=self.fees,
            cash_sharing=True,
            freq="D"  # <-- FIX 1: Explicitly tell vectorbt these are daily steps
        )
        
        # Calculate key metric summaries with explicit annualization handling
        total_return = portfolio.total_return()
        
        # FIX 2: Pass the frequency explicitly to the Sharpe engine as well
        sharpe = portfolio.sharpe_ratio(freq="D") 
        max_drawdown = portfolio.max_drawdown()
        
        return {
            "total_return": float(np.mean(total_return)),
            "sharpe_ratio": float(np.mean(sharpe)),
            "max_drawdown": float(np.mean(max_drawdown)),
            "portfolio_object": portfolio
        }

if __name__ == "__main__":
    print("Initializing vectorbt simulation check...")
    # Simulate a matrix of 100 timesteps across 3 distinct tickers
    np.random.seed(42)
    simulated_prices = np.cumprod(1 + np.random.normal(0.0005, 0.01, (100, 3)), axis=0) * 100
    
    # Generate static target allocation weights matching the dimensions
    simulated_weights = np.zeros((100, 3))
    simulated_weights[:, 0] = 0.5   # Asset 1 Long
    simulated_weights[:, 1] = -0.5  # Asset 2 Short (Dollar-Neutral)
    
    backtester = AlphaBacktester()
    results = backtester.run_vectorized_simulation(simulated_prices, simulated_weights)
    
    print("\n--- Strategy Backtest Analytics Dashboard ---")
    print(f"Mean Strategy Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Max Portfolio Peak-to-Trough Drawdown: {results['max_drawdown']*100:.2f}%")
    
# Example: If you are building the portfolio from signals or weights
portfolio = vbt.Portfolio.from_orders(
    close=simulated_prices,
    # ... your other arguments ...
    freq='D'  # <--- ADD THIS (Use 'D' for daily, 'H' for hourly, etc.)
)

# Inside your backtest execution block or a utility script:
portfolio_result = results["portfolio_object"]

# Generate vectorbt's complete interactive performance overview page
fig = portfolio_result.plots()

# Export as an independent HTML file containing all Plotly rendering scripts natively
fig.write_html("data/backtest_report.html", include_plotlyjs="cdn")
print("Interactive performance report successfully exported to local workspace.")