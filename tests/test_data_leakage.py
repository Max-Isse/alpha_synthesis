import pytest
import polars as pl
import duckdb
from datetime import date

@pytest.fixture
def mock_analytical_db():
    """Sets up an in-memory DuckDB instance representing our historical store."""
    conn = duckdb.connect(database=':memory:')
    
    # Create the mirror table
    conn.execute("""
        CREATE TABLE fundamental_pit (
            ticker VARCHAR,
            fiscal_period VARCHAR,
            period_end_date DATE,
            filing_date DATE,
            net_income DOUBLE,
            revenue DOUBLE
        );
    """)
    
    # Insert a real-world scenario: Apple Inc. Q1 2026
    # Period ends Dec 27, 2025, but isn't public until Jan 30, 2026.
    conn.execute("""
        INSERT INTO fundamental_pit VALUES 
        ('AAPL', 'Q12026', '2025-12-27', '2026-01-30', 33916000000.0, 119575000000.0);
    """)
    return conn

def test_prevent_look_ahead_bias(mock_analytical_db):
    """
    CRITICAL TEST: Ensures that if an algorithm queries data *before* the filing date,
    the financial metrics return NULL or are completely invisible.
    """
    # Define our trading/backtest simulation date (Before filing, after period end)
    simulation_date = "2026-01-15"
    
    # Query database using standard point-in-time window logic
    query = f"""
        SELECT * FROM fundamental_pit
        WHERE filing_date <= '{simulation_date}';
    """
    
    # Convert result to Polars DataFrame for evaluation
    df = pl.from_arrow(mock_analytical_db.query(query).to_arrow_table())
    
    # ASSERTION: The dataframe must be completely empty for this ticker on this date
    assert len(df) == 0, f"DATA LEAKAGE DETECTED: Data visible on {simulation_date} before public filing date."

def test_data_availability_post_filing(mock_analytical_db):
    """Ensures data becomes correctly visible exactly on or after the filing date."""
    simulation_date = "2026-02-01" # Post filing date
    
    query = f"""
        SELECT net_income FROM fundamental_pit
        WHERE filing_date <= '{simulation_date}' AND ticker = 'AAPL';
    """
    
    df = pl.from_arrow(mock_analytical_db.query(query).to_arrow_table())
    
    assert len(df) == 1
    assert df["net_income"][0] == 33916000000.0