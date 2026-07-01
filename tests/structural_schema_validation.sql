SELECT 
    t.ticker,
    f.fiscal_period,
    f.period_end_date,
    f.filing_date,
    (f.filing_date - f.period_end_date) AS days_of_information_lag,
    f.net_income
FROM fundamental_pit f
JOIN tickers t ON f.ticker_id = t.ticker_id
ORDER BY f.filing_date DESC
LIMIT 5;