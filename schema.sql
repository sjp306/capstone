-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Table for quantitative market data (price, rates, fx)
CREATE TABLE IF NOT EXISTS market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    price NUMERIC NOT NULL,
    volume BIGINT,
    metric_type VARCHAR(20) NOT NULL -- 'equity', 'fx', 'rate'
);

-- Create hypertable for market_data
SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);

-- Index for symbol-based queries
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data (symbol, time DESC);

-- Table for news and sentiment
CREATE TABLE IF NOT EXISTS news_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    sentiment_score NUMERIC,
    url VARCHAR(512) NOT NULL,
    UNIQUE (url, time)
);

-- Create hypertable for news_data
SELECT create_hypertable('news_data', 'time', if_not_exists => TRUE);

-- Index for sentiment and symbol queries
CREATE INDEX IF NOT EXISTS idx_news_data_symbol_time ON news_data (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_news_data_sentiment ON news_data (sentiment_score);

-- Table for pipeline observability
CREATE TABLE IF NOT EXISTS pipeline_logs (
    id SERIAL PRIMARY KEY,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL, -- 'running', 'success', 'failed'
    market_records_count INTEGER DEFAULT 0,
    news_records_count INTEGER DEFAULT 0,
    error_message TEXT
);
