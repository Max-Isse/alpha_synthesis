-- Core table tracking companies and preventing survivorship bias by tracking active status
CREATE TABLE IF NOT EXISTS tickers (
    ticker_id SERIAL PRIMARY KEY,
    ticker VARCHAR(12) UNIQUE NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Point-In-Time Fundamental Financials Table
CREATE TABLE IF NOT EXISTS fundamental_pit (
    record_id SERIAL PRIMARY KEY,
    ticker_id INT REFERENCES tickers(ticker_id),
    fiscal_period VARCHAR(6) NOT NULL, -- e.g., 'FY2025', 'Q12026'
    period_end_date DATE NOT NULL,
    filing_date DATE NOT NULL,         -- CRITICAL: Prevent Look-Ahead Bias
    net_income NUMERIC(18, 2),
    pretax_income NUMERIC(18, 2),
    ebit NUMERIC(18, 2),
    revenue NUMERIC(18, 2),
    total_assets NUMERIC(18, 2),
    total_equity NUMERIC(18, 2),
    cash_and_equiv NUMERIC(18, 2),
    operating_accruals NUMERIC(18, 2),
    inserted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexing strategies for high-frequency window queries during backtests
CREATE INDEX IF NOT EXISTS idx_pit_lookup ON fundamental_pit (filing_date, ticker_id);