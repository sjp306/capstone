-- 1) 뉴스 기사 원문 테이블
CREATE TABLE IF NOT EXISTS news_articles (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    source VARCHAR(100),
    title TEXT NOT NULL,
    description TEXT,
    content TEXT,
    url TEXT UNIQUE,
    published_at TIMESTAMP NOT NULL,
    collected_at TIMESTAMP DEFAULT NOW()
);

-- 2) 뉴스 감성 분석 결과 테이블
CREATE TABLE IF NOT EXISTS news_sentiment (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT REFERENCES news_articles(id) ON DELETE CASCADE,
    sentiment_label VARCHAR(20),   -- positive / neutral / negative
    sentiment_score FLOAT,         -- -1.0 ~ 1.0 (TextBlob) 또는 FinBERT 스코어
    confidence FLOAT,
    analyzed_at TIMESTAMP DEFAULT NOW()
);

-- 3) 주가/지수 데이터 테이블
CREATE TABLE IF NOT EXISTS market_prices (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    UNIQUE (ticker, trade_date)
);

-- 4) 거시 경제 지표 테이블
CREATE TABLE IF NOT EXISTS macro_indicators (
    id BIGSERIAL PRIMARY KEY,
    indicator_name VARCHAR(50) NOT NULL,
    indicator_date DATE NOT NULL,
    value FLOAT NOT NULL,
    UNIQUE (indicator_name, indicator_date)
);

-- 5) 일자별 뉴스 집계 테이블 (분석 편의용)
CREATE TABLE IF NOT EXISTS daily_news_summary (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    summary_date DATE NOT NULL,
    article_count INT,
    avg_sentiment FLOAT,
    positive_ratio FLOAT,
    negative_ratio FLOAT,
    UNIQUE (ticker, summary_date)
);

-- 인덱스 추가 (조회 성능 최적화)
CREATE INDEX IF NOT EXISTS idx_news_articles_ticker_date ON news_articles(ticker, published_at);
CREATE INDEX IF NOT EXISTS idx_market_prices_ticker_date ON market_prices(ticker, trade_date);
